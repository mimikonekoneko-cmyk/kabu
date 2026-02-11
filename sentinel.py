#!/usr/bin/env python3

# ==========================================================
# 🛡 SENTINEL PRO v3.9 FINAL MASTER (TRUE UNIFICATION)
# ----------------------------------------------------------
# 統合内容:
# 1. 復刻: v3.3.1のコア判定エンジン(20日Pivot/PF判定)を完全移植。ADI等を確実に捕捉。
# 2. 進化: v3.8のプラチナインフラ(JSON保存/VIX市場判定/LINE詳細表示)を維持。
# 3. 救済: 安定成長株がPF不足で落ちないよう、判定ロジックをv3.3.1基準に再調整。
# 4. 網羅: 450銘柄のマンモスリスト。エラー銘柄(FI等)はスキャン時に自動スキップ。
# 5. 可読性: スマホで一目で全てがわかるPLATINUM通知。
# ==========================================================

import os
import time
import logging
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION (v3.3.1の黄金比 + アルファ)
# ==========================================================

CONFIG = {
    "CAPITAL_JPY": 350_000,
    "MAX_POSITIONS": 4,           
    "ACCOUNT_RISK_PCT": 0.015,    
    "MAX_SAME_SECTOR": 2,
    "CORRELATION_LIMIT": 0.80,    

    "MIN_RS_RATING": 70,          # v3.3.1 基準
    "MIN_VCP_SCORE": 50,          # v3.3.1 基準
    "MIN_PROFIT_FACTOR": 1.1,     # 1.2から微調整し、ADIなどの安定株を拾う
    "MAX_TIGHTNESS_PCT": 0.18,    # 0.15から微増し、ボラティリティを許容

    "STOP_LOSS_ATR": 2.0,
    "TARGET_R_MULTIPLE": 2.5,     # v3.3.1 の利確倍率
    "CHEAT_MA50_RANGE": 0.10,     
}

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("SENTINEL_PRO")

CACHE_DIR = Path("./cache_v39")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# TICKER UNIVERSE (450+ 銘柄完全版)
# ==========================================================

ORIGINAL_LIST = [
    'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'MU', 'QCOM', 'MRVL', 'LRCX', 'AMAT',
    'KLAC', 'ADI', 'ON', 'SMCI', 'ARM', 'MPWR', 'TER', 'RKLB', 'ASTS', 'PLTR', 
    'AERO', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AAPL', 'AMZN', 'NFLX', 'CRM', 'NOW',
    'SNOW', 'ADBE', 'INTU', 'ORCL', 'SAP', 'COST', 'WMT', 'TSLA', 'SBUX', 'NKE', 
    'MELI', 'BABA', 'CVNA', 'MTN', 'LLY', 'ABBV', 'REGN', 'VRTX', 'NVO', 'BSX', 
    'HOLX', 'OMER', 'DVAX', 'RARE', 'RIGL', 'KOD', 'TARS', 'ORKA', 'DSGN', 'MA', 
    'V', 'COIN', 'MSTR', 'HOOD', 'PAY', 'MDLN', 'COHR', 'ACN', 'ETN', 'SPOT', 
    'RDDT', 'RBLX', 'CEVA', 'FFIV', 'DAKT', 'ITRN', 'TBLA', 'CHA', 'EPAC', 'DJT', 
    'TV', 'SEM', 'SCVL', 'INBX', 'CCOI', 'NMAX', 'HY', 'AVR', 'PRSU', 'WBTN', 
    'ASTE', 'FULC', 'SNDK', 'WDC', 'STX', 'GEV', 'APH', 'TXN', 'PG', 'UBER', 
    'BE', 'LITE', 'IBM', 'CLS', 'CSCO', 'APLD', 'ANET', 'NET', 'GLW', 'PANW', 
    'CRWD', 'NBIS', 'RCL', 'ONDS', 'IONQ', 'ROP', 'PM', 'PEP', 'KO', 'SPY', 'QQQ', 'IWM'
]

EXPANSION_LIST = [
    'BRK-B','JPM','UNH','XOM','HD','MRK','CVX','BAC','LIN','DIS','TMO','MCD','ABT','WFC',
    'CMCSA','VZ','PFE','CAT','ISRG','GE','SPGI','HON','UNP','RTX','LOW','GS','BKNG','ELV',
    'AXP','COP','MDT','SYK','BLK','NEE','BA','TJX','PGR','ETN','LMT','C','CB','ADP','MMC',
    'PLD','CI','MDLZ','AMT','BX','TMUS','SCHW', 'MO','EOG','DE','SO','DUK','SLB','CME','SHW',
    'CSX','PYPL','CL','EQIX','ICE','FCX', 'MCK','TGT','USB','PH','GD','BDX','ITW','ABNB',
    'HCA','NXPI','PSX','MAR','NSC','EMR', 'AON','PNC','CEG','CDNS','SNPS','MCO','PCAR','COF',
    'FDX','ORLY','ADSK','VLO','OXY','TRV', 'AIG','HLT','WELL','CARR','AZO','PAYX','MSI','TEL',
    'PEG','AJG','ROST','KMB','APD', 'URI','DHI','OKE','WMB','TRGP','SRE','CTAS','AFL','GWW',
    'LHX','MET','PCG','CMI','F','GM','STZ', 'PSA','O','DLR','CCI','KMI','ED','XEL','EIX',
    'WEC','D','AWK','ES','AEP','EXC', 'STM','GFS', 'DDOG','MDB','HUBS','TTD','APP','PATH',
    'MNDY','GTLB', 'IOT', 'DUOL', 'CFLT', 'AI', 'SOUN', 'CLSK', 'MARA', 'RIOT', 'BITF', 'HUT',
    'IREN', 'WULF', 'CORZ', 'CIFR', 'AFRM', 'UPST', 'SOFI', 'DKNG', 'MRNA', 'BNTX', 'UTHR', 'SMMT',
    'VKTX', 'ALT', 'CRSP', 'NTLA', 'BEAM', 'LUNR', 'HII', 'AXON', 'TDG', 'CCJ', 'URA', 'UUUU', 'DNN',
    'NXE', 'UEC', 'SCCO', 'AA', 'NUE', 'STLD', 'TTE', 'CART', 'CAVA', 'BIRK', 'KVUE', 'LULU', 'ONON',
    'DECK', 'CROX', 'WING', 'CMG', 'DPZ', 'YUM', 'CELH', 'MNST', 'GME', 'AMC', 'U', 'OPEN', 'Z',
    'SMH', 'XLF', 'XLV', 'XLE', 'XLI', 'XLK', 'XLC', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE'
]

TICKERS = sorted(list(set(ORIGINAL_LIST + EXPANSION_LIST)))

# ==========================================
# ENGINES
# ==========================================

class CurrencyEngine:
    @staticmethod
    def get_usd_jpy():
        try:
            ticker = yf.Ticker("JPY=X")
            df = ticker.history(period="1d")
            if df.empty: return 152.0
            rate = df['Close'].iloc[-1]
            return round(float(rate), 2) if 130 < rate < 195 else 152.0
        except: return 152.0

class DataEngine:
    @staticmethod
    def get_data(ticker, period="2y"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                try:
                    with open(cache_file, "rb") as f: return pickle.load(f)
                except: pass
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 100: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            with open(cache_file, "wb") as f: pickle.dump(df, f)
            return df
        except: return None

    @staticmethod
    def get_sector(ticker):
        sector_cache_file = CACHE_DIR / "sectors.json"
        sector_map = {}
        if sector_cache_file.exists():
            try:
                with open(sector_cache_file, 'r') as f: sector_map = json.load(f)
            except: pass
        if ticker in sector_map: return sector_map[ticker]
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            sector_map[ticker] = sector
            with open(sector_cache_file, 'w') as f: json.dump(sector_map, f)
            return sector
        except: return "Unknown"

# ==========================================
# CORE ANALYSIS (v3.3.1 ロジック復刻)
# ==========================================

class MarketRegime:
    @staticmethod
    def analyze():
        df = DataEngine.get_data('SPY')
        if df is None: return "UNKNOWN", 0.0, 20.0
        close = df['Close']
        ma200 = close.rolling(200).mean().iloc[-1]; ma50 = close.rolling(50).mean().iloc[-1]; curr = close.iloc[-1]
        try:
            vix_df = yf.download("^VIX", period="1mo", progress=False)
            vix = float(vix_df['Close'].iloc[-1]) if not vix_df.empty else 20.0
        except: vix = 20.0
        if curr < ma200: return "🔴 BEAR MARKET", 0.0, vix
        elif vix > 25: return f"⚠️ HIGH VOLATILITY (VIX:{vix:.1f})", 0.5, vix
        elif curr > ma50: return "🟢 BULL MARKET", 1.0, vix
        else: return "🟡 NEUTRAL", 0.7, vix

class VCPAnalyzer:
    @staticmethod
    def calculate(df):
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
            tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            
            # v3.3.1 の収縮スコアリング
            h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min(); range_pct = (h10 - l10) / (h10 if h10 > 0 else 1)
            
            if range_pct > CONFIG['MAX_TIGHTNESS_PCT']:
                return {"score": 0, "atr": atr, "signals": [f"ルーズ({range_pct*100:.1f}%)"], "is_dryup": False}
            
            # スコア計算 (v3.3.1 のマッピング)
            tight_score = max(0, int(40 * (1 - (range_pct - 0.05) / 0.10))) if range_pct > 0.05 else 40
            
            vol_ma = volume.rolling(50).mean().iloc[-1]; vol_curr = volume.iloc[-1]; vol_ratio = vol_curr / vol_ma if vol_ma > 0 else 1.0
            is_dryup = vol_ratio < 0.7
            vol_score = 30 if is_dryup else (15 if vol_ratio < 1.2 else 0)
            
            ma50 = close.rolling(50).mean().iloc[-1]; ma200 = close.rolling(200).mean().iloc[-1]
            trend_score = (10 if close.iloc[-1] > ma50 else 0) + (10 if ma50 > ma200 else 0) + (10 if close.iloc[-1] > ma200 else 0)
            
            signals = []
            if range_pct < 0.06: signals.append("収縮")
            if is_dryup: signals.append("枯渇")
            if trend_score == 30: signals.append("整列")
            return {"score": tight_score + vol_score + trend_score, "atr": atr, "signals": signals, "is_dryup": is_dryup}
        except: return {"score": 0, "atr": 0, "signals": [], "is_dryup": False}

class RSAnalyzer:
    @staticmethod
    def calculate(ticker_df, benchmark_df):
        try:
            common = ticker_df.index.intersection(benchmark_df.index)
            t = ticker_df.loc[common, "Close"]; s = benchmark_df.loc[common, "Close"]
            periods = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}; weights = {"3m": 0.4, "6m": 0.2, "9m": 0.2, "12m": 0.2}
            raw = sum([((t.iloc[-1]/t.iloc[-d]-1) - (s.iloc[-1]/s.iloc[-d]-1)) * w for p, d, w in [(p, periods[p], weights[p]) for p in periods if len(t) > d]])
            return max(1, min(99, int(50 + raw * 100)))
        except: return 50

class BacktestEngine:
    @staticmethod
    def run_validation(df):
        """v3.3.1 の StrategyValidator ロジックを完全復刻"""
        try:
            if len(df) < 150: return 1.0
            close = df['Close']; high = df['High']; low = df['Low']
            tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
            
            trades = []; in_pos = False; entry_p = 0; stop_p = 0; start_idx = max(50, len(df)-250)
            
            for i in range(start_idx, len(df)):
                if in_pos:
                    if low.iloc[i] <= stop_p: trades.append(-1.0); in_pos = False
                    elif high.iloc[i] >= entry_p + (entry_p - stop_p) * 2.5: trades.append(2.5); in_pos = False
                    elif i == len(df) - 1:
                        pnl = (close.iloc[i] - entry_p) / (entry_p - stop_p) if (entry_p - stop_p) > 0 else 0
                        trades.append(pnl); in_pos = False
                else:
                    if i >= len(df) - 5: continue
                    pivot = high.iloc[i-20:i].max() # v3.3.1 は20日
                    if close.iloc[i] > pivot and close.iloc[i] > close.rolling(50).mean().iloc[i]:
                        in_pos = True; entry_p = close.iloc[i]; stop_p = entry_p - (atr.iloc[i] * 2.0)
            
            if not trades: return 1.0
            pos = sum([t for t in trades if t > 0]); neg = abs(sum([t for t in trades if t < 0]))
            return min(10.0, round(pos / neg, 2) if neg > 0 else 5.0)
        except: return 1.0

# ==========================================
# PORTFOLIO
# ==========================================

def calculate_position(entry, stop, usd_jpy, exposure):
    total_usd = (CONFIG["CAPITAL_JPY"] / usd_jpy) * exposure
    risk_usd = total_usd * CONFIG["ACCOUNT_RISK_PCT"]
    diff = abs(entry - stop)
    if diff <= 0: return 0
    s_risk = int(risk_usd / diff); s_cap = int((total_usd * 0.4) / entry)
    return max(0, min(s_risk, s_cap)) or (1 if s_cap > 0 else 0)

def filter_portfolio(candidates, return_map):
    selected = []; sector_counts = {}
    for c in candidates:
        ticker = c['ticker']; sector = DataEngine.get_sector(ticker); c['sector'] = sector
        if sector_counts.get(sector, 0) >= CONFIG['MAX_SAME_SECTOR'] and sector != "Unknown": continue
        if any([return_map[ticker].corr(return_map[s['ticker']]) > CONFIG['CORRELATION_LIMIT'] for s in selected]): continue
        selected.append(c); sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= CONFIG['MAX_POSITIONS']: break
    return selected

def run():
    st = time.time()
    print("=" * 50); print("🛡 SENTINEL PRO v3.9 FINAL MASTER"); 
    usd_jpy = CurrencyEngine.get_usd_jpy(); m_msg, exposure, vix = MarketRegime.analyze()
    print(f"Rate: {usd_jpy} | Market: {m_msg}\n" + "=" * 50)
    
    if exposure == 0.0:
        send_line(f"🛡 SENTINEL PRO\n{m_msg}\nキャッシュ維持。"); return
    
    benchmark = DataEngine.get_data("^GSPC"); qualified = []; return_map = {}
    print(f"Scanning {len(TICKERS)} tickers...")
    
    for ticker in TICKERS:
        df = DataEngine.get_data(ticker)
        if df is None: continue
        vcp = VCPAnalyzer.calculate(df); rs = RSAnalyzer.calculate(df, benchmark); pf = BacktestEngine.run_validation(df)
        
        # v3.3.1 基準の足切り
        if vcp["score"] < CONFIG["MIN_VCP_SCORE"] or rs < CONFIG["MIN_RS_RATING"] or pf < CONFIG["MIN_PROFIT_FACTOR"]: continue
        
        ma50 = df["Close"].rolling(50).mean().iloc[-1]; price = df["Close"].iloc[-1]
        pivot_std = df["High"].iloc[-20:].max(); pivot_cheat = df["High"].iloc[-3:].max()
        
        is_cheat = (rs > 80 and vcp["is_dryup"] and (ma50 * (1-CONFIG["CHEAT_MA50_RANGE"]) <= price <= ma50 * (1+CONFIG["CHEAT_MA50_RANGE"])))
        e_type = "⚡Cheat" if is_cheat else "Standard"
        entry = pivot_cheat if is_cheat else pivot_std
        stop = entry - vcp["atr"] * 2.0; target = entry + (entry - stop) * 2.5; dist = (price - entry) / (entry if entry > 0 else 1)
        
        # v3.3.1 のACTION判定
        status = "ACTION" if -0.01 <= dist < 0.04 else ("WAIT" if price < entry else "EXTENDED")
        
        shares = calculate_position(entry, stop, usd_jpy, exposure); return_map[ticker] = df["Close"].pct_change().dropna()
        qualified.append({"ticker": ticker, "status": status, "price": round(price,2), "entry": round(entry,2), "stop": round(stop,2), "target": round(target,2), "shares": shares, "vcp": vcp, "rs": rs, "pf": pf, "type": e_type, "sector": DataEngine.get_sector(ticker)})
    
    qualified.sort(key=lambda x: ({"ACTION": 3, "WAIT": 2, "EXTENDED": 1}.get(x["status"], 0), x["vcp"]["score"] + x["rs"]), reverse=True)
    selected = filter_portfolio(qualified, return_map)
    
    today = datetime.now().strftime("%Y-%m-%d"); runtime = round(time.time() - st, 2)
    res = {"date": today, "runtime": f"{runtime}s", "usd_jpy": usd_jpy, "market": m_msg, "vix": vix, "exposure": exposure, "selected_count": len(selected), "qualified_count": len(qualified), "selected": selected, "qualified": qualified}
    with open(RESULTS_DIR / f"{today}.json", 'w', encoding='utf-8') as f: json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print("--- START JSON DATA ---"); print(json.dumps(res, ensure_ascii=False)); print("--- END JSON DATA ---")
    
    msg = [f"🛡 SENTINEL PRO v3.9 (Rate:{usd_jpy})\nMarket: {m_msg}\nScan:{len(TICKERS)} | Sel:{len(selected)}\n" + "="*20]
    for s in selected:
        msg.append(f"\n{'💎' if s['status'] == 'ACTION' else '⏳'} {s['ticker']} [{s['status']}] {'🌟' if s['pf'] >= 3.0 else '✅'}PF:{s['pf']:.2f}\nVCP:{s['vcp']['score']} | RS:{s['rs']} | {s['type']}\n📍Entry:${s['entry']:.2f} 🛑Stop:${s['stop']:.2f}\n📦推奨:{s['shares']}株 | 💡{','.join(s['vcp']['signals'])}\n" + "-"*15)
    send_line("\n".join(msg))

def send_line(message):
    if not ACCESS_TOKEN or not USER_ID: return
    try: requests.post("https://api.line.me/v2/bot/message/push", headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}, json={"to": USER_ID, "messages": [{"type": "text", "text": message[:4000]}]}, timeout=10)
    except: pass

if __name__ == "__main__":
    run()

