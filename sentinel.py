#!/usr/bin/env python3
# SENTINEL PRO v2.2 FINAL - THE COMPLETE EDITION
# -----------------------------------------------------------------------------
# 最終確認事項:
# 1. 銘柄リスト: 125銘柄以上を完全網羅（漏れなし）
# 2. RS計算: v2.0仕様（係数100）に復元済み
# 3. 表示: 0件防止のため、基準クリア銘柄はステータス問わず表示
# 4. 機能: VCP分析 + バックテスト(PF) + ボラティリティ管理 + 3段階出口戦略
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
    'MAX_POSITIONS': 4,              # 最大ポジション数
    'DISPLAY_LIMIT': 15,             # レポート表示上限数
    'ACCOUNT_RISK_PCT': 0.015,       # 1トレード許容リスク (資金の1.5%)
    
    # フィルタリング基準
    'MIN_RS_RATING': 70,             # RSスコア下限 (強さ)
    'MIN_VCP_SCORE': 50,             # VCPスコア下限 (形)
    'MIN_PROFIT_FACTOR': 1.2,        # PF下限 (実績)
    
    # リスク管理
    'STOP_LOSS_ATR': 2.0,            # ストップ幅 (ATR倍率)
    'MAX_TIGHTNESS': 2.5,            # ボラティリティ許容上限
    
    # 出口戦略 (Reward/Risk倍率)
    'TARGET_CONSERVATIVE': 1.5,      # 利確目標1
    'TARGET_MODERATE': 2.5,          # 利確目標2 (メイン)
    'TARGET_AGGRESSIVE': 4.0,        # 利確目標3
}

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("SENTINEL_PRO")

CACHE_DIR = Path("./cache_pro_final")
CACHE_DIR.mkdir(exist_ok=True)

# ==========================================
# TICKER UNIVERSE (FULL LIST - NO OMISSIONS)
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

    # === Biotech / Healthcare (復元) ===
    'LLY', 'ABBV', 'REGN', 'VRTX', 'NVO', 'BSX', 'HOLX', 'OMER', 'DVAX',
    'RARE', 'RIGL', 'KOD', 'TARS', 'ORKA', 'DSGN',

    # === Fintech / Crypto ===
    'MA', 'V', 'COIN', 'MSTR', 'HOOD', 'PAY', 'MDLN',

    # === New Discoveries / Volume Trend (復元) ===
    'COHR', 'ACN', 'ETN', 'SPOT', 'RDDT', 'RBLX', 'CEVA', 'FFIV',
    'DAKT', 'ITRN', 'TBLA', 'CHA', 'EPAC', 'DJT', 'TV', 'SEM',
    'SCVL', 'INBX', 'CCOI', 'NMAX', 'HY', 'AVR', 'PRSU', 'WBTN',
    'ASTE', 'FULC',

    # === Priority List (復元) ===
    'SNDK', 'WDC', 'STX', 'GEV', 'APH', 'TXN', 'PG', 'UBER',
    'BE', 'LITE', 'IBM', 'CLS', 'CSCO', 'APLD', 'ANET', 'NET',
    'GLW', 'PANW', 'CRWD', 'NBIS', 'RCL', 'ONDS', 'IONQ', 'ROP',
    'PM', 'PEP', 'KO',

    # === ETFs (Market Check) ===
    'SPY', 'QQQ', 'IWM', 'IEMG', 'FXI', 'EWY', 'AGG', 'IJH'
]
# 重複排除とソート
TICKERS = sorted(list(set(TICKERS)))

# ==========================================
# DATA ENGINE
# ==========================================
class DataEngine:
    @staticmethod
    def get_data(ticker, period="700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"
        
        # キャッシュ有効期限 (12時間)
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                with open(cache_file, 'rb') as f: return pickle.load(f)
        
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 200: return None
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            
            required = ['Close', 'High', 'Low', 'Volume']
            if not all(c in df.columns for c in required): return None
            
            with open(cache_file, 'wb') as f: pickle.dump(df, f)
            return df
        except: return None

# ==========================================
# VCP ANALYZER
# ==========================================
class VCPAnalyzer:
    @staticmethod
    def calculate_vcp_score(df):
        try:
            close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']
            
            tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=7).mean().iloc[-1]
            if pd.isna(atr) or atr <= 0: return {'score': 0, 'atr': 0, 'signals': []}
            
            recent_high = high.iloc[-10:].max()
            recent_low = low.iloc[-10:].min()
            tightness = (recent_high - recent_low) / atr
            
            # Tightnessが基準外でも、計算は続行（スコアで弾くため）
            
            score = 0
            signals = []
            
            # 1. Tightness
            if tightness < 0.8: score += 40; signals.append("極度収縮")
            elif tightness < 1.2: score += 30; signals.append("強収縮")
            elif tightness < 1.8: score += 20; signals.append("収縮中")
            elif tightness > 3.0: signals.append("ルーズ") # 情報として記録
            
            # 2. Volume Dry Up
            vol_ma = volume.rolling(50, min_periods=10).mean().iloc[-1]
            if volume.iloc[-1] < vol_ma * 0.8: score += 20; signals.append("Vol枯渇")
            
            # 3. MA Alignment
            curr = close.iloc[-1]
            ma50 = close.rolling(50, min_periods=10).mean().iloc[-1]
            ma200 = close.rolling(200, min_periods=50).mean().iloc[-1]
            if curr > ma50 > ma200: score += 20; signals.append("MA整列")
            elif curr > ma50: score += 10
            
            # 4. Momentum
            mom5 = close.rolling(5, min_periods=3).mean().iloc[-1]
            mom20 = close.rolling(20, min_periods=10).mean().iloc[-1]
            if (mom5 / mom20) > 1.02: score += 20; signals.append("モメンタム+")
            
            if score >= 85: stage = "🔥爆発直前"
            elif score >= 70: stage = "⚡初動圏"
            elif score >= 50: stage = "👁形成中"
            else: stage = "準備段階"
            
            return {'score': score, 'tightness': tightness, 'stage': stage, 'signals': signals, 'atr': atr}
        except: return {'score': 0, 'atr': 0, 'signals': []}

# ==========================================
# RS ANALYZER (RESTORED v2.0 LOGIC)
# ==========================================
class RSAnalyzer:
    @staticmethod
    def calculate_rs_rating(ticker_df, benchmark_df):
        try:
            if benchmark_df is None: return 50
            common = ticker_df.index.intersection(benchmark_df.index)
            if len(common) < 100: return 50
            
            t_c = ticker_df.loc[common, 'Close']
            s_c = benchmark_df.loc[common, 'Close']
            
            periods = {'3mo': 63, '6mo': 126, '9mo': 189, '12mo': 252}
            weights = {'3mo': 0.4, '6mo': 0.2, '9mo': 0.2, '12mo': 0.2}
            raw_score = 0
            
            for p, d in periods.items():
                if len(t_c) > d:
                    t_r = (t_c.iloc[-1] - t_c.iloc[-d]) / t_c.iloc[-d]
                    s_r = (s_c.iloc[-1] - s_c.iloc[-d]) / s_c.iloc[-d]
                    raw_score += (t_r - s_r) * weights[p]
            
            # v2.0仕様: 係数100 (スコアが出やすい設定)
            normalized = min(99, max(1, int(50 + (raw_score * 100))))
            return normalized
        except: return 50

# ==========================================
# BACKTEST ENGINE (v28 LOGIC)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_backtest(df):
        if len(df) < 200: return {'pf': 0, 'winrate': 0}
        close = df['Close']; high = df['High']; low = df['Low']
        
        tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        reward_mult = 2.5
        
        trades = []
        wins = 0; losses = 0
        
        for i in range(200, len(df) - 30):
            try:
                pivot = high.iloc[i-10:i].max() * 1.002
                if high.iloc[i] < pivot: continue
                
                ma50 = close.rolling(50).mean().iloc[i]
                if close.iloc[i] < ma50 * 0.95: continue
                
                entry = pivot
                stop = entry - (atr.iloc[i] * CONFIG['STOP_LOSS_ATR'])
                target = entry + (atr.iloc[i] * CONFIG['STOP_LOSS_ATR'] * reward_mult)
                
                for j in range(i+1, min(i+31, len(df))):
                    if high.iloc[j] >= target: trades.append(reward_mult); wins += 1; break
                    if low.iloc[j] <= stop: trades.append(-1.0); losses += 1; break
                    if j == min(i+30, len(df)-1):
                        pnl = (close.iloc[j] - entry) / (entry - stop)
                        trades.append(pnl)
                        if pnl > 0: wins += 1
                        else: losses += 1
            except: continue
            
        if not trades: return {'pf': 0, 'winrate': 0}
        
        total_wins = sum([t for t in trades if t > 0])
        total_losses = abs(sum([t for t in trades if t < 0]))
        pf = (total_wins / total_losses) if total_losses > 0 else 10.0
        
        return {'pf': pf, 'winrate': (wins / len(trades)) * 100}

# ==========================================
# MAIN EXECUTION
# ==========================================
def analyze_full_universe():
    print(f"🚀 SENTINEL PRO v2.2 FINAL - Scanning {len(TICKERS)} tickers...")
    
    spy_df = DataEngine.get_data('SPY', period="400d")
    if spy_df is None: return "❌ Market Data Error"
    
    curr = spy_df['Close'].iloc[-1]
    ma200 = spy_df['Close'].rolling(200).mean().iloc[-1]
    is_bull = curr > ma200
    
    candidates = []
    stats = {'Scanned': 0, 'Pass': 0}
    
    for ticker in TICKERS:
        if ticker in ['SPY', 'QQQ', 'IWM', 'AGG', 'IEF', 'IEMG', 'FXI', 'EWY', 'IJH']: continue
        stats['Scanned'] += 1
        
        df = DataEngine.get_data(ticker)
        if df is None: continue
        
        # 1. VCP Check
        vcp = VCPAnalyzer.calculate_vcp_score(df)
        if vcp['score'] < CONFIG['MIN_VCP_SCORE']: continue
            
        # 2. RS Check (v2.0 Logic)
        rs = RSAnalyzer.calculate_rs_rating(df, spy_df)
        if rs < CONFIG['MIN_RS_RATING']: continue
            
        # 3. Backtest Check
        bt = BacktestEngine.run_backtest(df)
        if bt['pf'] < CONFIG['MIN_PROFIT_FACTOR']: continue
            
        stats['Pass'] += 1
        
        # Setup
        curr_price = df['Close'].iloc[-1]
        pivot = df['High'].iloc[-10:].max() * 1.002
        stop = pivot - (vcp['atr'] * CONFIG['STOP_LOSS_ATR'])
        
        # Targets
        risk = pivot - stop
        targets = {
            'T1': pivot + (risk * CONFIG['TARGET_CONSERVATIVE']),
            'T2': pivot + (risk * CONFIG['TARGET_MODERATE']),
            'T3': pivot + (risk * CONFIG['TARGET_AGGRESSIVE'])
        }
        
        # Status Determination (Broad ranges to ensure display)
        dist_pct = ((curr_price - pivot) / pivot) * 100
        
        if -2 <= dist_pct < 3: status = "🔥 ACTION"
        elif -6 < dist_pct < -2: status = "👀 WATCH"
        elif dist_pct >= 3: status = "🚀 EXTENDED"
        else: status = "⏳ WAIT"
        
        # Position Sizing
        risk_usd = (CONFIG['CAPITAL_JPY'] * CONFIG['ACCOUNT_RISK_PCT']) / 150
        shares = int(risk_usd / risk) if risk > 0 else 0
        
        candidates.append({
            'ticker': ticker,
            'status': status,
            'vcp': vcp,
            'rs': rs,
            'pf': bt['pf'],
            'winrate': bt['winrate'],
            'current': curr_price,
            'entry': pivot,
            'stop': stop,
            'targets': targets,
            'shares': shares
        })
    
    # Sort: ACTION > WATCH > Score
    # ステータスの優先順位付け
    status_rank = {"🔥 ACTION": 4, "👀 WATCH": 3, "🚀 EXTENDED": 2, "⏳ WAIT": 1}
    candidates.sort(key=lambda x: (status_rank.get(x['status'], 0), x['vcp']['score'], x['pf']), reverse=True)
    
    # Generate Report
    report = []
    report.append("=" * 45)
    report.append("🛡 SENTINEL PRO v2.2 FINAL")
    report.append("=" * 45)
    report.append(f"Market: {'🟢 Bull' if is_bull else '🔴 Bear'}")
    report.append(f"Scan: {stats['Scanned']} | Qualified: {stats['Pass']}")
    report.append("-" * 45)
    
    count = 0
    if not candidates:
        report.append("⚠️ 基準を満たす銘柄なし")
    else:
        for p in candidates:
            if count >= CONFIG['DISPLAY_LIMIT']: break
            
            # アイコン付与
            icon = "💎" if p['pf'] > 1.5 and p['rs'] > 80 else "🔸"
            if p['status'] == "🚀 EXTENDED": icon = "👋"
            
            dist_txt = f"{((p['current']-p['entry'])/p['entry'])*100:+.1f}%"
            
            report.append(f"\n{icon} {p['ticker']} [{p['status']}]")
            report.append(f"   VCP:{p['vcp']['score']} | RS:{p['rs']} | PF:{p['pf']:.2f}")
            report.append(f"   Now:${p['current']:.2f} (Pivot {dist_txt})")
            
            # 詳細表示 (ACTION/WATCH)
            if "ACTION" in p['status'] or "WATCH" in p['status']:
                risk_pct = ((p['entry'] - p['stop']) / p['entry']) * 100
                t2_pct = ((p['targets']['T2'] - p['entry']) / p['entry']) * 100
                jpy_val = p['shares'] * p['entry'] * 150
                
                report.append(f"   📍 Entry: ${p['entry']:.2f}")
                report.append(f"   🛑 Stop : ${p['stop']:.2f} (-{risk_pct:.1f}%)")
                report.append(f"   🎯 Target: ${p['targets']['T2']:.2f} (+{t2_pct:.1f}%)")
                report.append(f"   📦 推奨: {p['shares']}株 (約{jpy_val/10000:.1f}万)")
                report.append(f"   💡 {','.join(p['vcp']['signals'])}")
            
            count += 1

    return "\n".join(report)

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID:
        print(msg)
        return
    MAX_LEN = 4000
    if len(msg) <= MAX_LEN: messages = [msg]
    else: messages = [msg[i:i+MAX_LEN] for i in range(0, len(msg), MAX_LEN)]
    
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"}
    for m in messages:
        try: requests.post(url, headers=headers, json={"to": USER_ID, "messages":[{"type":"text", "text":m}]})
        except: pass

if __name__ == "__main__":
    result = analyze_full_universe()
    send_line(result)
    print(result)

