#!/usr/bin/env python3
# SENTINEL v28.0 UNIVERSE - Alpaca Integration
# 3-Stage Filtering System for 2,500+ stocks
# Stage 1: Quick Screen (2500 → 500) - 3 min
# Stage 2: VCP Analysis (500 → 100) - 5 min  
# Stage 3: Institutional (100 → 10-20) - 5 min
# Total: ~13 minutes
#
# Requirements: pip install alpaca-trade-api pandas numpy yfinance requests beautifulsoup4
# Usage: python sentinel_v28_universe.py

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import concurrent.futures
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import alpaca_trade_api as tradeapi
import warnings

warnings.filterwarnings('ignore')

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("SENTINEL")

# ---------------------------
# CONFIG
# ---------------------------
# Alpaca API (Paper Trading - Free)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or "YOUR_API_KEY"
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET") or "YOUR_API_SECRET"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading (free)

# LINE notification (optional)
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

# Trading parameters
INITIAL_CAPITAL_JPY = 350_000
TRADING_RATIO = 0.75
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25
MIN_POSITION_USD = 500

# Filtering thresholds
STAGE1_MIN_PRICE = 10.0
STAGE1_MAX_PRICE = 1000.0
STAGE1_MIN_VOLUME_USD = 10_000_000
STAGE1_OUTPUT_TARGET = 500

STAGE2_MAX_TIGHTNESS = 2.5
STAGE2_MIN_VCP_MATURITY = 40
STAGE2_OUTPUT_TARGET = 100

STAGE3_MIN_SCORE = 60  # SECONDARY tier or above

# Performance
MAX_WORKERS = 50  # Parallel threads
REQUEST_DELAY = 0.05  # 50ms between requests (stay under 200/min limit)

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# Universe file (user provides)
UNIVERSE_FILE = "rakuten_universe.txt"  # One ticker per line

# ---------------------------
# Alpaca API Client
# ---------------------------
class AlpacaClient:
    """Alpaca API wrapper for market data"""
    
    def __init__(self):
        try:
            self.api = tradeapi.REST(
                ALPACA_API_KEY,
                ALPACA_API_SECRET,
                base_url=ALPACA_BASE_URL,
                api_version='v2'
            )
            logger.info("Alpaca API connected successfully")
        except Exception as e:
            logger.error(f"Alpaca API connection failed: {e}")
            self.api = None
    
    def get_bars(self, symbol: str, days: int = 250) -> Optional[pd.DataFrame]:
        """Get historical bars for a symbol"""
        try:
            if not self.api:
                return None
            
            end = datetime.now()
            start = end - timedelta(days=days)
            
            bars = self.api.get_bars(
                symbol,
                '1Day',
                start=start.isoformat(),
                end=end.isoformat(),
                limit=days,
                adjustment='all'  # Adjust for splits/dividends
            ).df
            
            if bars.empty:
                return None
            
            # Flatten multi-index if present
            if isinstance(bars.columns, pd.MultiIndex):
                bars.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in bars.columns]
            
            # Standardize column names
            bars.columns = [col.lower() for col in bars.columns]
            
            # Extract symbol if in columns
            if 'symbol' in bars.columns:
                bars = bars[bars['symbol'] == symbol]
            
            return bars
            
        except Exception as e:
            logger.debug(f"Failed to get bars for {symbol}: {e}")
            return None


# ---------------------------
# Stage 1: Quick Screening
# ---------------------------
class Stage1Filter:
    """Fast screening based on price, volume, and basic trend"""
    
    def __init__(self, alpaca_client: AlpacaClient):
        self.alpaca = alpaca_client
    
    def screen_single(self, symbol: str) -> Optional[Dict]:
        """
        Quick screen for a single symbol
        
        Criteria:
        - Price: $10-$1000
        - Volume: $10M+ daily
        - Trend: Price > MA50 or MA50 > MA200
        """
        try:
            bars = self.alpaca.get_bars(symbol, days=250)
            
            if bars is None or len(bars) < 200:
                return None
            
            # Extract columns (handle different naming)
            close = bars['close'] if 'close' in bars.columns else bars.iloc[:, 3]
            volume = bars['volume'] if 'volume' in bars.columns else bars.iloc[:, 4]
            
            close = close.astype(float)
            volume = volume.astype(float)
            
            # Current metrics
            current_price = float(close.iloc[-1])
            avg_volume_usd = float((volume * close).tail(20).mean())
            
            # Price filter
            if current_price < STAGE1_MIN_PRICE or current_price > STAGE1_MAX_PRICE:
                return None
            
            # Volume filter
            if avg_volume_usd < STAGE1_MIN_VOLUME_USD:
                return None
            
            # Trend filter
            ma50 = float(close.rolling(50, min_periods=25).mean().iloc[-1])
            ma200 = float(close.rolling(200, min_periods=100).mean().iloc[-1]) if len(close) >= 100 else None
            
            # Accept if: price > MA50 OR MA50 > MA200
            trend_ok = current_price > ma50
            if ma200 is not None:
                trend_ok = trend_ok or ma50 > ma200
            
            if not trend_ok:
                return None
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume_usd': avg_volume_usd,
                'ma50': ma50,
                'ma200': ma200,
                'stage': 'STAGE1_PASS'
            }
            
        except Exception as e:
            logger.debug(f"Stage1 error for {symbol}: {e}")
            return None
    
    def screen_batch(self, symbols: List[str]) -> List[Dict]:
        """Screen multiple symbols in parallel"""
        results = []
        
        logger.info(f"Stage 1: Screening {len(symbols)} symbols...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.screen_single, sym): sym for sym in symbols}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                
                # Rate limiting
                time.sleep(REQUEST_DELAY)
        
        logger.info(f"Stage 1 complete: {len(results)} / {len(symbols)} passed")
        return results


# ---------------------------
# Stage 2: VCP Analysis
# ---------------------------
class Stage2Filter:
    """Detailed VCP pattern analysis"""
    
    def __init__(self, alpaca_client: AlpacaClient):
        self.alpaca = alpaca_client
    
    def analyze_single(self, symbol: str) -> Optional[Dict]:
        """
        VCP analysis for a single symbol
        
        Criteria:
        - Tightness < 2.5
        - VCP maturity >= 40%
        - Volume contraction
        - Higher lows pattern
        """
        try:
            bars = self.alpaca.get_bars(symbol, days=250)
            
            if bars is None or len(bars) < 60:
                return None
            
            # Extract OHLCV
            high = bars['high'] if 'high' in bars.columns else bars.iloc[:, 1]
            low = bars['low'] if 'low' in bars.columns else bars.iloc[:, 2]
            close = bars['close'] if 'close' in bars.columns else bars.iloc[:, 3]
            volume = bars['volume'] if 'volume' in bars.columns else bars.iloc[:, 4]
            
            high = high.astype(float)
            low = low.astype(float)
            close = close.astype(float)
            volume = volume.astype(float)
            
            # ATR calculation
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            
            atr14 = tr.rolling(14, min_periods=7).mean().iloc[-1]
            
            if pd.isna(atr14) or atr14 <= 0:
                return None
            
            # Tightness (5-day range / ATR)
            recent_high = float(high.iloc[-5:].max())
            recent_low = float(low.iloc[-5:].min())
            tightness = (recent_high - recent_low) / atr14
            
            if tightness > STAGE2_MAX_TIGHTNESS:
                return None
            
            # VCP Maturity calculation
            maturity = 0
            signals = []
            
            # 1. Volatility Contraction (40 pts)
            if tightness < 1.0:
                maturity += 40
                signals.append("極度収縮")
            elif tightness < 1.5:
                maturity += 30
                signals.append("強収縮")
            elif tightness < 2.0:
                maturity += 20
                signals.append("収縮中")
            elif tightness < 2.5:
                maturity += 10
                signals.append("軽度収縮")
            
            # 2. Volume Drying (20 pts)
            vol_avg = float(volume.rolling(50, min_periods=25).mean().iloc[-1])
            if volume.iloc[-1] < vol_avg:
                maturity += 20
                signals.append("出来高縮小")
            
            # 3. Higher Lows (30 pts)
            lows_5d = close.rolling(5, min_periods=3).min()
            if len(lows_5d) >= 10:
                if lows_5d.iloc[-1] > lows_5d.iloc[-10]:
                    maturity += 20
                    signals.append("切上中")
            
            # 4. MA Structure (10 pts)
            ma50 = close.rolling(50, min_periods=25).mean().iloc[-1]
            ma200 = close.rolling(200, min_periods=100).mean().iloc[-1] if len(close) >= 100 else None
            
            if ma200 is not None and ma50 > ma200:
                maturity += 10
                signals.append("MA整列")
            
            if maturity < STAGE2_MIN_VCP_MATURITY:
                return None
            
            # Pivot calculation
            pivot = float(high.iloc[-5:].max() * 1.002)
            stop = pivot - (atr14 * ATR_STOP_MULT)
            
            return {
                'symbol': symbol,
                'tightness': float(tightness),
                'vcp_maturity': maturity,
                'vcp_signals': signals,
                'price': float(close.iloc[-1]),
                'pivot': pivot,
                'stop': stop,
                'atr': float(atr14),
                'volume': float(volume.iloc[-1]),
                'stage': 'STAGE2_PASS'
            }
            
        except Exception as e:
            logger.debug(f"Stage2 error for {symbol}: {e}")
            return None
    
    def analyze_batch(self, stage1_results: List[Dict]) -> List[Dict]:
        """Analyze multiple symbols in parallel"""
        symbols = [r['symbol'] for r in stage1_results]
        results = []
        
        logger.info(f"Stage 2: Analyzing {len(symbols)} symbols for VCP...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.analyze_single, sym): sym for sym in symbols}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                
                time.sleep(REQUEST_DELAY)
        
        logger.info(f"Stage 2 complete: {len(results)} / {len(symbols)} passed")
        return results


# ---------------------------
# Stage 3: Institutional Analysis
# (Reuse v27 logic with Yahoo Finance)
# ---------------------------

class InsiderTracker:
    @staticmethod
    def get_insider_activity(ticker, days=30):
        try:
            cache_file = CACHE_DIR / f"insider_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            stock = yf.Ticker(ticker)
            insider_trades = stock.insider_transactions
            
            if insider_trades is None or insider_trades.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent = insider_trades[insider_trades.index >= cutoff_date]
            
            if recent.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}
            
            buy_shares = recent[recent['Shares'] > 0]['Shares'].sum()
            sell_shares = abs(recent[recent['Shares'] < 0]['Shares'].sum())
            ratio = sell_shares / max(buy_shares, 1)
            
            if ratio > 5:
                signal = '🚨SELL'
            elif ratio > 2:
                signal = '⚠️CAUTION'
            elif ratio < 0.5:
                signal = '✅BUY'
            else:
                signal = 'NEUTRAL'
            
            result = {'buy_shares': int(buy_shares), 'sell_shares': int(sell_shares), 'ratio': float(ratio), 'signal': signal}
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            return result
        except Exception:
            return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}


class OptionFlowAnalyzer:
    @staticmethod
    def get_put_call_ratio(ticker):
        try:
            stock = yf.Ticker(ticker)
            exp_dates = stock.options
            if not exp_dates:
                return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}
            
            opt = stock.option_chain(exp_dates[0])
            calls = opt.calls
            puts = opt.puts
            
            if calls.empty or puts.empty:
                return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}
            
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            ratio = put_volume / max(call_volume, 1)
            
            if ratio > 1.5:
                signal = '🐻BEARISH'
            elif ratio < 0.7:
                signal = '🐂BULLISH'
            else:
                signal = 'NEUTRAL'
            
            return {'put_call_ratio': float(ratio), 'signal': signal}
        except Exception:
            return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}


class InstitutionalAnalyzer:
    @staticmethod
    def analyze(ticker):
        signals = {}
        alerts = []
        risk_score = 0
        
        insider = InsiderTracker.get_insider_activity(ticker)
        signals['insider'] = insider
        if insider['signal'] == '🚨SELL':
            alerts.append(f"Insider売{insider['ratio']:.1f}x")
            risk_score += 30
        elif insider['signal'] == '✅BUY':
            risk_score -= 10
        
        options = OptionFlowAnalyzer.get_put_call_ratio(ticker)
        signals['options'] = options
        if options['signal'] == '🐻BEARISH':
            alerts.append(f"P/C{options['put_call_ratio']:.2f}")
            risk_score += 15
        elif options['signal'] == '🐂BULLISH':
            risk_score -= 10
        
        if risk_score > 60:
            overall = '🚨HIGH_RISK'
        elif risk_score > 30:
            overall = '⚠️CAUTION'
        elif risk_score < 0:
            overall = '✅LOW_RISK'
        else:
            overall = 'NEUTRAL'
        
        return {'signals': signals, 'alerts': alerts, 'risk_score': risk_score, 'overall': overall}


class VCPAnalyzer:
    @staticmethod
    def calculate_stage(maturity):
        if maturity >= 85:
            return "🔥爆発直前"
        elif maturity >= 70:
            return "⚡初動圏"
        elif maturity >= 50:
            return "👁形成中"
        else:
            return "⏳準備段階"


class SignalQuality:
    @staticmethod
    def calculate_comprehensive_score(vcp_maturity, winrate, ev, risk_score):
        # Tech Score (0-40) from VCP maturity
        tech_score = min(vcp_maturity * 0.4, 40)
        
        # RR Score (0-30)
        rr_score = 0
        if ev > 0.6 and winrate > 0.5:
            rr_score = 30
        elif ev > 0.4 and winrate > 0.45:
            rr_score = 25
        elif ev > 0.3 and winrate > 0.42:
            rr_score = 20
        elif ev > 0.2 and winrate > 0.40:
            rr_score = 15
        elif ev > 0.1 and winrate > 0.35:
            rr_score = 10
        elif ev > 0:
            rr_score = 5
        
        # Institutional Score (0-30)
        if risk_score < 0:
            inst_score = 30
        elif risk_score < 20:
            inst_score = 25
        elif risk_score < 40:
            inst_score = 20
        else:
            inst_score = max(0, 20 - risk_score // 10)
        
        total = tech_score + rr_score + inst_score
        
        if total >= 75:
            tier = 'CORE'
            tier_emoji = '🔥'
        elif total >= 60:
            tier = 'SECONDARY'
            tier_emoji = '⚡'
        elif total >= 45:
            tier = 'WATCH'
            tier_emoji = '👁'
        else:
            tier = 'AVOID'
            tier_emoji = '❌'
        
        return {
            'total_score': int(total),
            'tech_score': int(tech_score),
            'rr_score': int(rr_score),
            'inst_score': int(inst_score),
            'tier': tier,
            'tier_emoji': tier_emoji
        }


class Stage3Analyzer:
    """Full institutional analysis with backtest"""
    
    @staticmethod
    def analyze_single(stage2_result: Dict) -> Optional[Dict]:
        """Run full analysis on a Stage 2 candidate"""
        try:
            symbol = stage2_result['symbol']
            
            # Simple backtest (placeholder - use realistic win rate estimation)
            # In production, implement actual backtest with historical data
            winrate = 0.50  # Default assumption
            ev = 0.40  # Default assumption
            
            # Institutional analysis
            inst = InstitutionalAnalyzer.analyze(symbol)
            
            # Calculate quality score
            quality = SignalQuality.calculate_comprehensive_score(
                stage2_result['vcp_maturity'],
                winrate,
                ev,
                inst['risk_score']
            )
            
            if quality['total_score'] < STAGE3_MIN_SCORE:
                return None
            
            return {
                'symbol': symbol,
                'price': stage2_result['price'],
                'pivot': stage2_result['pivot'],
                'stop': stage2_result['stop'],
                'tightness': stage2_result['tightness'],
                'vcp_maturity': stage2_result['vcp_maturity'],
                'vcp_signals': stage2_result['vcp_signals'],
                'vcp_stage': VCPAnalyzer.calculate_stage(stage2_result['vcp_maturity']),
                'quality': quality,
                'institutional': inst,
                'winrate': winrate * 100,
                'ev': ev,
                'stage': 'FINAL_PICK'
            }
            
        except Exception as e:
            logger.debug(f"Stage3 error for {stage2_result['symbol']}: {e}")
            return None
    
    @staticmethod
    def analyze_batch(stage2_results: List[Dict]) -> List[Dict]:
        """Analyze all Stage 2 candidates"""
        results = []
        
        logger.info(f"Stage 3: Full analysis on {len(stage2_results)} candidates...")
        
        for candidate in stage2_results:
            result = Stage3Analyzer.analyze_single(candidate)
            if result:
                results.append(result)
            time.sleep(0.1)  # Rate limit for Yahoo Finance
        
        # Sort by quality score
        results = sorted(results, key=lambda x: x['quality']['total_score'], reverse=True)
        
        logger.info(f"Stage 3 complete: {len(results)} final picks")
        return results


# ---------------------------
# Universe Management
# ---------------------------
def load_universe(filepath: str) -> List[str]:
    """Load ticker symbols from file"""
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        return symbols
    except FileNotFoundError:
        logger.error(f"Universe file not found: {filepath}")
        logger.info("Creating sample universe file...")
        
        # Create sample file
        sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
        with open(filepath, 'w') as f:
            f.write('\n'.join(sample_symbols))
        
        logger.info(f"Created {filepath} with {len(sample_symbols)} sample symbols")
        return sample_symbols


# ---------------------------
# Reporting
# ---------------------------
def generate_report(final_picks: List[Dict], stage1_count: int, stage2_count: int) -> str:
    """Generate final report"""
    
    lines = []
    lines.append("="*50)
    lines.append("SENTINEL v28.0 UNIVERSE - Alpaca Integration")
    lines.append("3-Stage Filtering: 2,500+ → 10-20 picks")
    lines.append("="*50)
    lines.append(datetime.now().strftime("%m/%d %H:%M"))
    lines.append("")
    
    lines.append("【FILTERING RESULTS】")
    lines.append(f"Stage 1 (Quick Screen):  {stage1_count} passed")
    lines.append(f"Stage 2 (VCP Analysis):  {stage2_count} passed")
    lines.append(f"Stage 3 (Institutional): {len(final_picks)} final picks")
    lines.append("")
    
    if not final_picks:
        lines.append("No candidates met all criteria.")
        return '\n'.join(lines)
    
    # Group by tier
    core = [p for p in final_picks if p['quality']['tier'] == 'CORE']
    secondary = [p for p in final_picks if p['quality']['tier'] == 'SECONDARY']
    
    lines.append("【PRIORITY SIGNALS】")
    lines.append(f"🔥 CORE (75+):      {len(core)} signals")
    lines.append(f"⚡ SECONDARY (60+): {len(secondary)} signals")
    lines.append("")
    
    if core:
        lines.append("🔥 CORE - IMMEDIATE CONSIDERATION")
        for i, pick in enumerate(core[:5], 1):
            q = pick['quality']
            inst = pick['institutional']
            
            lines.append(f"\n[{i}] {pick['symbol']} {q['total_score']}/100 | VCP:{pick['vcp_maturity']}% {pick['vcp_stage']}")
            lines.append(f"    Tech:{q['tech_score']} RR:{q['rr_score']} Inst:{q['inst_score']} | Risk:{inst['risk_score']}")
            lines.append(f"    Price: ${pick['price']:.2f} | Entry: ${pick['pivot']:.2f} | Stop: ${pick['stop']:.2f}")
            lines.append(f"    Tightness: {pick['tightness']:.2f} | Signals: {', '.join(pick['vcp_signals'])}")
            
            if inst['alerts']:
                lines.append(f"    ⚠️  {' | '.join(inst['alerts'])}")
    
    if secondary:
        lines.append("\n⚡ SECONDARY - CONDITIONAL WATCH")
        for i, pick in enumerate(secondary[:5], 1):
            q = pick['quality']
            lines.append(f"[{i}] {pick['symbol']} {q['total_score']}/100 | VCP:{pick['vcp_maturity']}% {pick['vcp_stage']}")
            lines.append(f"    Price: ${pick['price']:.2f} | Entry: ${pick['pivot']:.2f}")
    
    lines.append("\n" + "="*50)
    lines.append("【PHILOSOPHY】")
    lines.append("✓ 3-Stage filtering maximizes signal quality")
    lines.append("✓ Alpaca API enables 2,500+ stock universe")
    lines.append("✓ Catch institutional accumulation BEFORE news")
    lines.append("="*50)
    
    return '\n'.join(lines)


def send_line(msg):
    """Send LINE notification"""
    if not ACCESS_TOKEN or not USER_ID:
        return
    
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages":[{"type":"text", "text":msg}]}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("LINE notification sent")
    except Exception as e:
        logger.warning(f"LINE notification failed: {e}")


# ---------------------------
# Main Pipeline
# ---------------------------
def run_pipeline():
    """Execute 3-stage filtering pipeline"""
    
    start_time = time.time()
    
    # Initialize Alpaca
    alpaca = AlpacaClient()
    if not alpaca.api:
        logger.error("Alpaca API not available - cannot proceed")
        return
    
    # Load universe
    universe = load_universe(UNIVERSE_FILE)
    logger.info(f"Universe size: {len(universe)} stocks")
    
    # Stage 1: Quick Screen
    stage1 = Stage1Filter(alpaca)
    stage1_results = stage1.screen_batch(universe)
    
    if not stage1_results:
        logger.warning("No stocks passed Stage 1")
        return
    
    # Stage 2: VCP Analysis
    stage2 = Stage2Filter(alpaca)
    stage2_results = stage2.analyze_batch(stage1_results)
    
    if not stage2_results:
        logger.warning("No stocks passed Stage 2")
        return
    
    # Stage 3: Institutional Analysis
    final_picks = Stage3Analyzer.analyze_batch(stage2_results)
    
    # Generate report
    report = generate_report(final_picks, len(stage1_results), len(stage2_results))
    
    # Output
    print("\n" + report)
    logger.info(report)
    
    # Send notification
    send_line(report)
    
    # Timing
    elapsed = time.time() - start_time
    logger.info(f"Total execution time: {elapsed:.1f} seconds")
    
    return final_picks


if __name__ == "__main__":
    logger.info("SENTINEL v28.0 UNIVERSE starting...")
    results = run_pipeline()
    
    if results:
        logger.info(f"Pipeline complete: {len(results)} final picks")
    else:
        logger.warning("Pipeline complete: No final picks")
