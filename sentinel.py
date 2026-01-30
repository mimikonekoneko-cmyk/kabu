#!/usr/bin/env python3
# SENTINEL v32.0 - CRITICAL FIXES
# 
# FIXES:
# 1. Entry price calculation (was 6x off)
# 2. Macro environment display (VIX, 10Y, Market trend)
# 3. Scoring adjustment (more CORE signals)

import os
import time
import logging
import warnings
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("SENTINEL")

# ---------------------------
# CONFIG
# ---------------------------
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

INITIAL_CAPITAL_JPY = 350_000
TRADING_RATIO = 0.75
RISK_PER_TRADE = 0.02
MAX_POSITION_SIZE_RATIO = 0.25
ATR_STOP_MULT = 2.0

STAGE1_MIN_VOLUME_USD = 1_000_000
STAGE2_MAX_TIGHTNESS = 2.5
STAGE3_MIN_SCORE = 60  # 45 → 60 (SECONDARY以上のみ)

BATCH_SIZE = 50
BATCH_SLEEP_TIME = 60
MAX_WORKERS = 5
UNIVERSE_FILE = "rakuten_universe.txt"

# ---------------------------
# MACRO ANALYZER
# ---------------------------
class MacroAnalyzer:
    @staticmethod
    def get_environment():
        """Get market macro environment"""
        try:
            # VIX
            vix_data = yf.download("^VIX", period="5d", progress=False)
            vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else 20.0
            
            # 10Y Treasury
            tnx_data = yf.download("^TNX", period="5d", progress=False)
            treasury_10y = float(tnx_data['Close'].iloc[-1]) if not tnx_data.empty else 4.5
            
            # SPY trend
            spy_data = yf.download("SPY", period="400d", progress=False)
            if not spy_data.empty and len(spy_data) >= 200:
                spy_close = spy_data['Close']
                current = float(spy_close.iloc[-1])
                ma200 = float(spy_close.rolling(200).mean().iloc[-1])
                dist_pct = ((current - ma200) / ma200) * 100
                trend = "Bull" if current > ma200 else "Bear"
                trend_str = f"{trend} ({dist_pct:+.1f}%)"
            else:
                trend_str = "Unknown"
            
            # FX
            fx_data = yf.download("JPY=X", period="5d", progress=False)
            fx_rate = float(fx_data['Close'].iloc[-1]) if not fx_data.empty else 153.0
            
            # Environment labels
            rate_env = '⚠️ELEVATED' if treasury_10y > 4.0 else '✅LOW_RATE'
            vol_env = '✅LOW_VOL' if vix < 20 else '⚠️ELEVATED'
            
            return {
                'vix': vix,
                'treasury_10y': treasury_10y,
                'market_trend': trend_str,
                'fx_rate': fx_rate,
                'rate_env': rate_env,
                'vol_env': vol_env
            }
        except Exception as e:
            logger.warning(f"Macro fetch failed: {e}")
            return {
                'vix': 20.0,
                'treasury_10y': 4.5,
                'market_trend': 'Unknown',
                'fx_rate': 153.0,
                'rate_env': 'UNKNOWN',
                'vol_env': 'UNKNOWN'
            }


# ---------------------------
# MATH ENGINE
# ---------------------------
class MathEngine:
    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()


# ---------------------------
# STAGE 1
# ---------------------------
class Stage1Filter:
    @staticmethod
    def screen_single(symbol: str) -> Optional[Dict]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="30d")
            
            if df.empty or len(df) < 20:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            
            current_price = df['close'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()
            volume_usd = avg_volume * current_price
            
            if volume_usd < STAGE1_MIN_VOLUME_USD:
                return None
                
            return {
                'symbol': symbol,
                'price': current_price,
                'volume_usd': volume_usd
            }
        except Exception as e:
            logger.debug(f"Stage 1 error for {symbol}: {e}")
            return None


# ---------------------------
# STAGE 2 - FIXED VCP LOGIC
# ---------------------------
class Stage2Filter:
    @staticmethod
    def analyze_single(s1_data: Dict) -> Optional[Dict]:
        symbol = s1_data['symbol']
        try:
            df = yf.download(symbol, period="1y", progress=False, timeout=15)
            if df.empty or len(df) < 100:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            h, l, c, v = df['high'], df['low'], df['close'], df['volume']
            
            # ATR
            atr = MathEngine.calculate_atr(df).iloc[-1]
            
            # CRITICAL FIX: VCP with recent data only
            current_price = float(c.iloc[-1])
            
            # Use last 20 days for high/low
            recent_20d = df.tail(20)
            recent_high = float(recent_20d['high'].max())
            recent_low = float(recent_20d['low'].min())
            
            # Sanity check: if recent_high is way above current (>30%), cap it
            if recent_high > current_price * 1.30:
                logger.debug(f"{symbol}: Capping anomalous high ${recent_high:.2f} → ${current_price*1.10:.2f}")
                recent_high = current_price * 1.10
            
            # Tightness
            tightness = (recent_high - recent_low) / atr if atr > 0 else 99
            
            if tightness > STAGE2_MAX_TIGHTNESS:
                return None
            
            # VCP Maturity Scoring
            maturity = 0
            signals = []
            
            if tightness < 1.0:
                maturity += 40
                signals.append("極度収縮")
            elif tightness < 1.5:
                maturity += 30
                signals.append("強収縮")
            elif tightness < 2.0:
                maturity += 20
                signals.append("収縮中")
            else:
                maturity += 10
                signals.append("軽度収縮")
            
            vol_ma50 = v.rolling(50).mean().iloc[-1]
            if v.iloc[-1] < vol_ma50 * 0.8:
                maturity += 30
                signals.append("出来高縮小")
            
            lows_10d_ago = l.iloc[-20:-10].min() if len(l) >= 20 else l.iloc[0]
            lows_recent = l.iloc[-10:].min()
            if lows_recent > lows_10d_ago:
                maturity += 30
                signals.append("切上中")
            
            # Entry = recent_high + 0.2% (VCP breakout)
            pivot = recent_high * 1.002
            stop = pivot - (atr * ATR_STOP_MULT)
            
            # Final sanity check
            if pivot > current_price * 1.15:
                logger.warning(f"{symbol}: Pivot ${pivot:.2f} too far from current ${current_price:.2f}, adjusting")
                pivot = current_price * 1.02
                stop = pivot - (atr * ATR_STOP_MULT)
            
            return {
                **s1_data,
                'vcp_maturity': maturity,
                'tightness': float(tightness),
                'vcp_signals': signals,
                'pivot': pivot,
                'stop': stop,
                'atr': float(atr),
                'current_price': current_price  # Store for validation
            }
        except Exception as e:
            logger.debug(f"Stage 2 error for {symbol}: {e}")
            return None


# ---------------------------
# STAGE 3
# ---------------------------
class Stage3Analyzer:
    @staticmethod
    def analyze_single(s2_data: Dict, fx_rate: float, budget_usd: float) -> Optional[Dict]:
        symbol = s2_data['symbol']
        try:
            ticker = yf.Ticker(symbol)
            mod = 0
            alerts = []
            
            # Insider
            try:
                insider = ticker.insider_transactions
                if insider is not None and not insider.empty:
                    cutoff = datetime.now() - timedelta(days=90)
                    recent = insider[insider.index >= cutoff]
                    if not recent.empty:
                        buys = recent[recent['Shares'] > 0]['Shares'].sum()
                        sells = abs(recent[recent['Shares'] < 0]['Shares'].sum())
                        ratio = sells / max(buys, 1)
                        if ratio > 5:
                            mod -= 20
                            alerts.append(f"Insider売{ratio:.1f}x")
                        elif ratio < 0.2 and buys > 0:
                            mod += 15
                            alerts.append("Insider買")
            except:
                pass
            
            # Options
            pc_ratio = 1.0
            try:
                options = ticker.options
                if options:
                    chain = ticker.option_chain(options[0])
                    calls = chain.calls['volume'].sum()
                    puts = chain.puts['volume'].sum()
                    pc_ratio = puts / max(calls, 1)
                    if pc_ratio > 1.5:
                        mod -= 15
                        alerts.append(f"P/C {pc_ratio:.2f} 弱気")
                    elif pc_ratio < 0.7:
                        mod += 10
                        alerts.append(f"P/C {pc_ratio:.2f} 強気")
            except:
                pass
            
            # Scoring
            tech_score = s2_data['vcp_maturity']
            inst_score = 30 + mod
            
            pivot = s2_data['pivot']
            stop = s2_data['stop']
            risk_amt = pivot - stop
            
            # RR assumes 2:1 target
            potential_reward = risk_amt * 2
            rr_ratio = 2.0 if risk_amt > 0 else 0
            rr_score = min(30, int(rr_ratio * 10))
            
            # ADJUSTED: Give more weight to VCP (tech)
            total_score = (tech_score * 0.5) + (inst_score * 0.25) + (rr_score * 0.25)
            
            if total_score < STAGE3_MIN_SCORE:
                return None
            
            # Position sizing
            risk_usd = budget_usd * RISK_PER_TRADE
            shares = int(risk_usd / risk_amt) if risk_amt > 0 else 0
            pos_usd = min(shares * pivot, budget_usd * MAX_POSITION_SIZE_RATIO)
            
            # Metadata
            try:
                info = ticker.info
            except:
                info = {}
            
            quality = {
                'total_score': int(total_score),
                'tech_score': int(tech_score),
                'inst_score': int(inst_score),
                'rr_score': int(rr_score),
                'tier': 'CORE' if total_score >= 75 else 'SECONDARY' if total_score >= 60 else 'WATCH',
                'alerts': alerts
            }
            
            return {
                **s2_data,
                'quality': quality,
                'pos_usd': pos_usd,
                'shares': int(pos_usd / pivot) if pivot > 0 else 0,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A')
            }
        except Exception as e:
            logger.debug(f"Stage 3 error for {symbol}: {e}")
            return None


# ---------------------------
# REPORTING
# ---------------------------
def generate_report(final_picks, s1_count, s2_count, total, elapsed, macro):
    lines = []
    lines.append("="*50)
    lines.append("SENTINEL v32.0 - CRITICAL FIXES")
    lines.append("="*50)
    lines.append(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    lines.append("")
    
    # MACRO INFO
    lines.append(f"Market: {macro['market_trend']} | VIX: {macro['vix']:.1f} | FX: ¥{macro['fx_rate']:.2f}")
    lines.append(f"10Y: {macro['treasury_10y']:.2f}% | {macro['rate_env']} {macro['vol_env']}")
    lines.append("")
    
    lines.append("【FILTERING RESULTS】")
    lines.append(f"Input:    {total} stocks")
    lines.append(f"Stage 1:  {s1_count} passed ({s1_count/max(1,total)*100:.1f}%)")
    lines.append(f"Stage 2:  {s2_count} passed ({s2_count/max(1,s1_count)*100:.1f}%)")
    lines.append(f"Stage 3:  {len(final_picks)} final picks")
    lines.append(f"Time:     {elapsed/60:.1f} minutes")
    lines.append("")
    
    if not final_picks:
        lines.append("No candidates.")
        return '\n'.join(lines)
    
    core = [p for p in final_picks if p['quality']['tier'] == 'CORE']
    secondary = [p for p in final_picks if p['quality']['tier'] == 'SECONDARY']
    
    lines.append("【PRIORITY SIGNALS】")
    lines.append(f"🔥 CORE (75+):      {len(core)} signals")
    lines.append(f"⚡ SECONDARY (60+): {len(secondary)} signals")
    lines.append("")
    
    if core:
        lines.append("🔥 CORE - IMMEDIATE CONSIDERATION")
        for i, p in enumerate(core[:10], 1):
            q = p['quality']
            discount = ((p['pivot'] - p['current_price']) / p['current_price']) * 100
            lines.append(f"\n[{i}] {p['symbol']} {q['total_score']}/100 | VCP:{p['vcp_maturity']}%")
            lines.append(f"    Tech:{q['tech_score']} RR:{q['rr_score']} Inst:{q['inst_score']}")
            lines.append(f"    Current: ${p['current_price']:.2f} | Entry: ${p['pivot']:.2f} (+{discount:.1f}%) | Stop: ${p['stop']:.2f}")
            lines.append(f"    Size: ${p['pos_usd']:.0f} ({p['shares']} sh) | Sector: {p['sector']}")
            lines.append(f"    {', '.join(p['vcp_signals'])}")
            if q['alerts']:
                lines.append(f"    ⚠️  {' | '.join(q['alerts'])}")
    
    if secondary:
        lines.append("\n⚡ SECONDARY")
        for i, p in enumerate(secondary[:10], 1):
            q = p['quality']
            discount = ((p['pivot'] - p['current_price']) / p['current_price']) * 100
            lines.append(f"[{i}] {p['symbol']} {q['total_score']}/100 | Current: ${p['current_price']:.2f} | Entry: ${p['pivot']:.2f} (+{discount:.1f}%) | {p['sector']}")
    
    lines.append("\n" + "="*50)
    return '\n'.join(lines)


# ---------------------------
# MAIN
# ---------------------------
class SentinelPipeline:
    def __init__(self):
        self.macro = {}
        self.budget_usd = 0.0
    
    def run(self):
        start = time.time()
        
        # Get macro
        logger.info("Fetching macro environment...")
        self.macro = MacroAnalyzer.get_environment()
        self.budget_usd = (INITIAL_CAPITAL_JPY * TRADING_RATIO) / self.macro['fx_rate']
        
        logger.info(f"Environment: VIX={self.macro['vix']:.1f}, 10Y={self.macro['treasury_10y']:.2f}%, FX={self.macro['fx_rate']:.1f}")
        
        # Load universe
        if not os.path.exists(UNIVERSE_FILE):
            logger.error(f"{UNIVERSE_FILE} not found")
            return
        
        with open(UNIVERSE_FILE, 'r') as f:
            universe = [l.strip().upper() for l in f if l.strip() and not l[0].isdigit() and '.' not in l]
        
        total = len(universe)
        logger.info(f"Loaded {total} symbols")
        
        # Diagnostic
        logger.info("Testing first 3 symbols...")
        for sym in universe[:3]:
            res = Stage1Filter.screen_single(sym)
            logger.info(f"  {sym}: {'PASS' if res else 'FAIL'}")
        
        # Pipeline
        s1_results = []
        s2_results = []
        final_picks = []
        
        for i in range(0, total, BATCH_SIZE):
            batch = universe[i:i+BATCH_SIZE]
            logger.info(f"Batch {i//BATCH_SIZE + 1}: {i} to {min(i+BATCH_SIZE, total)}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # S1
                s1_futures = [executor.submit(Stage1Filter.screen_single, sym) for sym in batch]
                batch_s1 = [f.result() for f in s1_futures if f.result()]
                s1_results.extend(batch_s1)
                
                # S2
                s2_futures = [executor.submit(Stage2Filter.analyze_single, data) for data in batch_s1]
                batch_s2 = [f.result() for f in s2_futures if f.result()]
                s2_results.extend(batch_s2)
                
                # S3
                s3_futures = [executor.submit(Stage3Analyzer.analyze_single, data, self.macro['fx_rate'], self.budget_usd) for data in batch_s2]
                batch_s3 = [f.result() for f in s3_futures if f.result()]
                final_picks.extend(batch_s3)
            
            if i + BATCH_SIZE < total:
                time.sleep(BATCH_SLEEP_TIME)
        
        final_picks.sort(key=lambda x: x['quality']['total_score'], reverse=True)
        
        elapsed = time.time() - start
        report = generate_report(final_picks, len(s1_results), len(s2_results), total, elapsed, self.macro)
        
        print("\n" + report)
        
        if ACCESS_TOKEN and USER_ID:
            self._send_line(report)
    
    def _send_line(self, msg):
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": msg}]}
        try:
            requests.post(url, headers=headers, json=payload, timeout=10)
            logger.info("LINE sent")
        except Exception as e:
            logger.error(f"LINE failed: {e}")

if __name__ == "__main__":
    SentinelPipeline().run()
