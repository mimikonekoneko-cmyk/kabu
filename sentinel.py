#!/usr/bin/env python3
# SENTINEL PRO v2.1 - ENHANCED DISPLAY & EXIT STRATEGY
# -----------------------------------------------------------------------------
# 改良点:
# 1. 表示数増加: TOP 10 ACTION + 10 WATCH
# 2. 出口戦略: 3段階ターゲット（保守/中立/強気）
# 3. 期待リターン計算
# -----------------------------------------------------------------------------

import os
import time
import logging
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'CAPITAL_JPY': 350_000,          # 運用資金
    'MAX_POSITIONS': 4,              # 推奨購入数
    'DISPLAY_ACTION': 10,            # ACTION表示数
    'DISPLAY_WATCH': 10,             # WATCH表示数
    'ACCOUNT_RISK_PCT': 0.015,       # 1トレード許容リスク (1.5%)
    'MIN_RS_RATING': 70,             # RSスコア下限
    'MIN_VCP_SCORE': 50,             # VCPスコア下限
    'MIN_PROFIT_FACTOR': 1.2,        # バックテストPF下限
    'STOP_LOSS_ATR': 2.0,            # ATRストップ倍率
    'MAX_TIGHTNESS': 2.5,            # VCP収縮度上限
    # 出口戦略（R倍率）
    'TARGET_CONSERVATIVE': 1.5,      # 保守的（50%利確推奨）
    'TARGET_MODERATE': 2.5,          # 中立（メイン目標）
    'TARGET_AGGRESSIVE': 4.0,        # 強気（残り50%）
}

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("SENTINEL_PRO")

CACHE_DIR = Path("./cache_pro_v21")
CACHE_DIR.mkdir(exist_ok=True)

# ==========================================
# TICKER UNIVERSE (125+ Tickers)
# ==========================================
TICKERS = [
    # === TOP PERFORMERS / Core ===
    'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'MU', 'QCOM', 'MRVL', 'LRCX', 'AMAT',
    'KLAC', 'ADI', 'ON', 'SMCI', 'ARM', 'MPWR', 'TER',
    # === Space / Defense / New Core ===
    'RKLB', 'ASTS', 'PLTR', 'AERO',
    # === Mega Tech / Cloud / Ads ===
    'MSFT', 'GOOGL', 'GOOG', 'META', 'AAPL', 'AMZN', 'NFLX', 'CRM', 'NOW',
    'SNOW', 'ADBE', 'INTU', 'ORCL', 'SAP',
    # === Growth Retail / Consumer ===
    'COST', 'WMT', 'TSLA', 'SBUX', 'NKE', 'MELI', 'BABA', 'CVNA', 'MTN',
    # === Biotech / Healthcare ===
    'LLY', 'ABBV', 'REGN', 'VRTX', 'NVO', 'BSX', 'HOLX', 'OMER', 'DVAX',
    'RARE', 'RIGL', 'KOD', 'TARS', 'ORKA', 'DSGN',
    # === Fintech / Crypto ===
    'MA', 'V', 'COIN', 'MSTR', 'HOOD', 'PAY', 'MDLN',
    # === New Discoveries / Volume Trend ===
    'COHR', 'ACN', 'ETN', 'SPOT', 'RDDT', 'RBLX', 'CEVA', 'FFIV',
    'DAKT', 'ITRN', 'TBLA', 'CHA', 'EPAC', 'DJT', 'TV', 'SEM',
    'SCVL', 'INBX', 'CCOI', 'NMAX', 'HY', 'AVR', 'PRSU', 'WBTN',
    'ASTE', 'FULC',
    # === Priority List ===
    'SNDK', 'WDC', 'STX', 'GEV', 'APH', 'TXN', 'PG', 'UBER',
    'BE', 'LITE', 'IBM', 'CLS', 'CSCO', 'APLD', 'ANET', 'NET',
    'GLW', 'PANW', 'CRWD', 'NBIS', 'RCL', 'ONDS', 'IONQ', 'ROP',
    'PM', 'PEP', 'KO',
    # === ETFs (Market Check) ===
    'SPY', 'QQQ', 'IWM'
]
TICKERS = list(set(TICKERS))

# ==========================================
# DATA ENGINE
# ==========================================
class DataEngine:
    @staticmethod
    def get_data(ticker, period="700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"
        
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 200:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            
            required = ['Close', 'High', 'Low', 'Volume']
            if not all(c in df.columns for c in required):
                return None
            
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            
            return df
        except Exception as e:
            return None

# ==========================================
# VCP ANALYZER
# ==========================================
class VCPAnalyzer:
    @staticmethod
    def calculate_vcp_score(df):
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            tr = pd.concat([
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=7).mean().iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                return {'score': 0, 'tightness': 999, 'stage': 'INVALID', 'atr': 0}
            
            recent_high = high.iloc[-10:].max()
            recent_low = low.iloc[-10:].min()
            tightness = (recent_high - recent_low) / atr
            
            if tightness > CONFIG['MAX_TIGHTNESS']:
                return {'score': 0, 'tightness': tightness, 'stage': 'LOOSE', 'atr': atr}
            
            score = 0
            signals = []
            
            if tightness < 0.8:
                score += 40; signals.append("極度収縮")
            elif tightness < 1.2:
                score += 30; signals.append("強収縮")
            elif tightness < 1.8:
                score += 20; signals.append("収縮中")
            
            vol_ma = volume.rolling(50, min_periods=10).mean().iloc[-1]
            if volume.iloc[-1] < vol_ma * 0.8:
                score += 20; signals.append("Vol枯渇")
            
            curr = close.iloc[-1]
            ma50 = close.rolling(50, min_periods=10).mean().iloc[-1]
            ma200 = close.rolling(200, min_periods=50).mean().iloc[-1]
            
            if curr > ma50 > ma200:
                score += 20; signals.append("MA整列")
            elif curr > ma50:
                score += 10; signals.append("MA形成中")
            
            mom5 = close.rolling(5, min_periods=3).mean().iloc[-1]
            mom20 = close.rolling(20, min_periods=10).mean().iloc[-1]
            if (mom5 / mom20) > 1.02:
                score += 20; signals.append("モメンタム+")
            
            if score >= 85: stage = "🔥爆発直前"
            elif score >= 70: stage = "⚡初動圏"
            elif score >= 50: stage = "👁形成中"
            else: stage = "準備段階"
            
            return {
                'score': score,
                'tightness': tightness,
                'stage': stage,
                'signals': signals,
                'atr': atr
            }
            
        except Exception as e:
            return {'score': 0, 'tightness': 999, 'stage': 'ERROR', 'atr': 0}

# ==========================================
# RS ANALYZER
# ==========================================
class RSAnalyzer:
    @staticmethod
    def calculate_rs_rating(ticker_df, benchmark_df):
        try:
            if benchmark_df is None: return 50
            
            common_idx = ticker_df.index.intersection(benchmark_df.index)
            if len(common_idx) < 100: return 50
            
            t_close = ticker_df.loc[common_idx, 'Close']
            s_close = benchmark_df.loc[common_idx, 'Close']
            
            periods = {'3mo': 63, '6mo': 126, '9mo': 189, '12mo': 252}
            weights = {'3mo': 0.4, '6mo': 0.2, '9mo': 0.2, '12mo': 0.2}
            
            rs_score = 0
            valid_periods = 0
            
            for period, days in periods.items():
                if len(t_close) > days:
                    t_ret = (t_close.iloc[-1] - t_close.iloc[-days]) / t_close.iloc[-days]
                    s_ret = (s_close.iloc[-1] - s_close.iloc[-days]) / s_close.iloc[-days]
                    rel_perf = t_ret - s_ret
                    rs_score += rel_perf * weights[period]
                    valid_periods += 1
            
            if valid_periods == 0: return 50
            
            # 修正: *50に変更（より現実的な分布）
            normalized = min(99, max(1, int(50 + (rs_score * 50))))
            return normalized
            
        except Exception:
            return 50

# ==========================================
# BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_v28_backtest(df):
        if len(df) < 200:
            return {'pf': 0, 'winrate': 0}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=7).mean()
        
        reward_mult = 2.5
        
        trades = []
        wins = 0
        losses = 0
        
        for i in range(200, len(df) - 30):
            try:
                pivot = high.iloc[i-10:i].max() * 1.002
                
                if high.iloc[i] < pivot:
                    continue
                
                ma50 = close.rolling(50, min_periods=10).mean().iloc[i]
                if close.iloc[i] < ma50 * 0.95:
                    continue
                
                entry = pivot
                stop = entry - (atr.iloc[i] * CONFIG['STOP_LOSS_ATR'])
                target = entry + (atr.iloc[i] * CONFIG['STOP_LOSS_ATR'] * reward_mult)
                
                outcome_pnl = 0
                for j in range(i+1, min(i+31, len(df))):
                    if high.iloc[j] >= target:
                        outcome_pnl = reward_mult
                        wins += 1
                        break
                    if low.iloc[j] <= stop:
                        outcome_pnl = -1.0
                        losses += 1
                        break
                    if j == min(i+30, len(df)-1):
                        pnl = (close.iloc[j] - entry) / (entry - stop)
                        outcome_pnl = pnl
                        if pnl > 0: wins += 1
                        else: losses += 1
                
                trades.append(outcome_pnl)
                        
            except Exception:
                continue
        
        if not trades:
            return {'pf': 0, 'winrate': 0}
        
        total_wins = sum([t for t in trades if t > 0])
        total_losses = abs(sum([t for t in trades if t < 0]))
        
        pf = (total_wins / total_losses) if total_losses > 0 else 10.0
        winrate = (wins / len(trades)) * 100
        
        return {
            'pf': pf,
            'winrate': winrate
        }

# ==========================================
# MAIN ANALYZER
# ==========================================
def analyze_full_universe():
    print(f"🚀 SENTINEL PRO v2.1 - Scanning {len(TICKERS)} tickers...")
    
    spy_df = DataEngine.get_data('SPY', period="400d")
    if spy_df is None:
        return "❌ Market data unavailable"
    
    curr = spy_df['Close'].iloc[-1]
    ma200 = spy_df['Close'].rolling(200).mean().iloc[-1]
    
    if curr < ma200:
        return "🔴 BEAR MARKET DETECTED\nSENTINEL PRO停止中\nキャッシュ100%推奨"
    
    candidates = []
    
    for ticker in TICKERS:
        if ticker in ['SPY', 'QQQ', 'IWM']: continue
        
        df = DataEngine.get_data(ticker)
        if df is None: continue
        
        vcp = VCPAnalyzer.calculate_vcp_score(df)
        if vcp['score'] < CONFIG['MIN_VCP_SCORE']:
            continue
            
        rs = RSAnalyzer.calculate_rs_rating(df, spy_df)
        if rs < CONFIG['MIN_RS_RATING']:
            continue
            
        bt = BacktestEngine.run_v28_backtest(df)
        if bt['pf'] < CONFIG['MIN_PROFIT_FACTOR']:
            continue
            
        curr_price = df['Close'].iloc[-1]
        pivot = df['High'].iloc[-10:].max() * 1.002
        stop = pivot - (vcp['atr'] * CONFIG['STOP_LOSS_ATR'])
        
        # 出口戦略: 3段階ターゲット
        risk = pivot - stop
        target_conservative = pivot + (risk * CONFIG['TARGET_CONSERVATIVE'])
        target_moderate = pivot + (risk * CONFIG['TARGET_MODERATE'])
        target_aggressive = pivot + (risk * CONFIG['TARGET_AGGRESSIVE'])
        
        risk_amt_jpy = CONFIG['CAPITAL_JPY'] * CONFIG['ACCOUNT_RISK_PCT']
        risk_amt_usd = risk_amt_jpy / 150
        
        risk_per_share = pivot - stop
        shares = int(risk_amt_usd / risk_per_share) if risk_per_share > 0 else 0
        
        dist_pct = ((curr_price - pivot) / pivot) * 100
        
        if -1 <= dist_pct < 2:
            status = "🔥 ACTION"
        elif -4 < dist_pct < -1:
            status = "👀 WATCH"
        else:
            status = "WAIT"
            
        if status != "WAIT":
            candidates.append({
                'ticker': ticker,
                'status': status,
                'stage': vcp['stage'],
                'vcp': vcp['score'],
                'rs': rs,
                'pf': bt['pf'],
                'winrate': bt['winrate'],
                'entry': pivot,
                'stop': stop,
                'target_cons': target_conservative,
                'target_mod': target_moderate,
                'target_agg': target_aggressive,
                'current': curr_price,
                'shares': shares,
                'cost_usd': shares * pivot,
                'signals': ",".join(vcp['signals'])
            })
            
    # ソート
    candidates.sort(key=lambda x: (1 if "ACTION" in x['status'] else 0, x['vcp'], x['pf']), reverse=True)
    
    # 分類
    action_list = [c for c in candidates if c['status'] == "🔥 ACTION"]
    watch_list = [c for c in candidates if c['status'] == "👀 WATCH"]
    
    # レポート生成
    report = []
    report.append("=" * 50)
    report.append("🛡 SENTINEL PRO v2.1 (Enhanced)")
    report.append("=" * 50)
    report.append(f"Market: 🟢 Bullish (SPY > MA200)")
    report.append(f"Scanned: {len(TICKERS)}")
    report.append(f"🔥 ACTION: {len(action_list)} | 👀 WATCH: {len(watch_list)}")
    report.append("=" * 50)
    
    # TOP推奨
    if action_list:
        report.append(f"\n🎯 TOP {CONFIG['MAX_POSITIONS']}推奨:")
        report.append("-" * 50)
        
        for idx, p in enumerate(action_list[:CONFIG['MAX_POSITIONS']], 1):
            jpy_cost = p['cost_usd'] * 150
            risk_pct = ((p['entry'] - p['stop']) / p['entry']) * 100
            cons_pct = ((p['target_cons'] - p['entry']) / p['entry']) * 100
            mod_pct = ((p['target_mod'] - p['entry']) / p['entry']) * 100
            agg_pct = ((p['target_agg'] - p['entry']) / p['entry']) * 100
            
            report.append(f"\n{idx}. 💎 {p['ticker']} ({p['stage']})")
            report.append(f"   VCP:{p['vcp']} | RS:{p['rs']} | PF:{p['pf']:.2f} | WR:{p['winrate']:.0f}%")
            report.append(f"   現在価格: ${p['current']:.2f}")
            report.append(f"   📍 Entry: ${p['entry']:.2f}")
            report.append(f"   🛑 Stop:  ${p['stop']:.2f} ({risk_pct:.1f}%)")
            report.append(f"   🎯 T1: ${p['target_cons']:.2f} (+{cons_pct:.1f}%) ← 50%利確")
            report.append(f"   🎯 T2: ${p['target_mod']:.2f} (+{mod_pct:.1f}%) ← メイン")
            report.append(f"   🎯 T3: ${p['target_agg']:.2f} (+{agg_pct:.1f}%) ← 残り")
            report.append(f"   📦 推奨: {p['shares']}株 (¥{jpy_cost/10000:.1f}万)")
            report.append(f"   💡 {p['signals']}")
    
    # ACTION全表示
    if len(action_list) > CONFIG['MAX_POSITIONS']:
        report.append(f"\n\n🔥 ACTION候補 (残り{len(action_list) - CONFIG['MAX_POSITIONS']}銘柄):")
        report.append("-" * 50)
        
        for idx, p in enumerate(action_list[CONFIG['MAX_POSITIONS']:CONFIG['DISPLAY_ACTION']], CONFIG['MAX_POSITIONS']+1):
            mod_pct = ((p['target_mod'] - p['entry']) / p['entry']) * 100
            report.append(f"{idx}. {p['ticker']:6} VCP:{p['vcp']:2} RS:{p['rs']:2} PF:{p['pf']:.2f} "
                         f"Entry:${p['entry']:.2f} T2:${p['target_mod']:.2f}(+{mod_pct:.1f}%)")
    
    # WATCH表示
    if watch_list:
        report.append(f"\n\n👀 WATCH (押し目待ち {min(len(watch_list), CONFIG['DISPLAY_WATCH'])}銘柄):")
        report.append("-" * 50)
        
        for idx, p in enumerate(watch_list[:CONFIG['DISPLAY_WATCH']], 1):
            dist_pct = ((p['current'] - p['entry']) / p['entry']) * 100
            mod_pct = ((p['target_mod'] - p['entry']) / p['entry']) * 100
            report.append(f"{idx}. {p['ticker']:6} VCP:{p['vcp']:2} RS:{p['rs']:2} PF:{p['pf']:.2f} "
                         f"Now:${p['current']:.2f} Entry:${p['entry']:.2f}({dist_pct:+.1f}%) "
                         f"T2:+{mod_pct:.0f}%")
    
    if not action_list and not watch_list:
        report.append("\n⚠️ 現在、基準を満たすセットアップはありません")
    
    report.append("\n" + "=" * 50)
    report.append("出口戦略:")
    report.append("T1到達 → 50%利確（リスク回収）")
    report.append("T2到達 → 残り半分利確 or トレーリングストップ")
    report.append("T3到達 → 完全利確 or 長期ホールド判断")
    report.append("=" * 50)
    
    return "\n".join(report)

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID:
        print(msg)
        return
    
    # LINE文字数制限対策（5000文字以下に分割）
    MAX_LEN = 4800
    if len(msg) <= MAX_LEN:
        messages = [msg]
    else:
        lines = msg.split('\n')
        messages = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 < MAX_LEN:
                current += line + '\n'
            else:
                if current:
                    messages.append(current)
                current = line + '\n'
        if current:
            messages.append(current)
    
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"}
    
    for msg_part in messages:
        payload = {"to": USER_ID, "messages":[{"type":"text", "text":msg_part}]}
        try:
            requests.post(url, headers=headers, json=payload, timeout=10)
            time.sleep(1)  # API制限対策
        except:
            pass

if __name__ == "__main__":
    result = analyze_full_universe()
    send_line(result)
    print(result)
