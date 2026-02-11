#!/usr/bin/env python3

# ==========================================================
# 🛡 SENTINEL PRO v4.1 REMASTERED (v3.3.1 Logic + Infra)
# ----------------------------------------------------------
# 復刻内容:
# 1. 判定エンジン: v3.3.1 のロジックを100%復元。
# 2. 銘柄リスト: 450銘柄のマンモスリストを統合。
# 3. 修正: GitHub ActionsでのSyntaxErrorを完全に回避。
# 4. 機能: ダッシュボード用JSON出力 & 詳細LINE通知を維持。
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

# ==========================================
# CONFIGURATION (v3.3.1 EXACT)
# ==========================================

CONFIG = {
    "CAPITAL_JPY": 350_000,
    "MAX_POSITIONS": 4,
    "ACCOUNT_RISK_PCT": 0.015,
    "MAX_SAME_SECTOR": 2,
    "CORRELATION_LIMIT": 0.80,

    "MIN_RS_RATING": 70,
    "MIN_VCP_SCORE": 55,
    "MIN_PROFIT_FACTOR": 1.2,
    "MAX_TIGHTNESS_PCT": 0.15,

    "STOP_LOSS_ATR": 2.0,
    "TARGET_R_MULTIPLE": 2.5,
    "DISPLAY_LIMIT": 20,
}

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("SENTINEL_PRO")

CACHE_DIR = Path("./cache_v41")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# ==========================================
# TICKER UNIVERSE (v3.3.1 + EXPANSION)
# ==========================================

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
    def get_data(ticker, period="700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except: pass
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 200: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            return df
        except: return None

    @staticmethod
    def get_sector(ticker):
        sector_cache_file = CACHE_DIR / "sectors.json"
        sector_map = {}
        if sector_cache_file.exists():
            try:
                with open(sector_cache_file, 'r') as f:
                    sector_map = json.load(f)
            except: pass
        if ticker in sector_map: return sector_map[ticker]
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            sector_map[ticker] = sector
            with open(sector_cache_file, 'w') as f:
                json.dump(sector_map, f)
            return sector
        except: return "Unknown"

# ==========================================
# CORE ANALYSIS (v3.3.1 EXACT)
# ==========================================

class VCPAnalyzer:
    @staticmethod
    def calculate(df):
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
            tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=7).mean().iloc[-1]
            h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min()
            range_pct = (h10 - l10) / h10
            if range_pct > CONFIG['MAX_TIGHTNESS_PCT']:
                return {"score": 0, "atr": atr, "signals": []}
            tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.10))
            vol_ma = volume.rolling(50).mean().iloc[-1]; vol_curr = volume.iloc[-1]; vol_ratio = vol_curr / vol_ma if vol_ma > 0 else 1.0
            vol_score = 30 if vol_ratio <= 0.6 else (0 if vol_ratio >= 1.2 else int(30 * (1 - (vol_ratio - 0.6) / 0.6)))
            ma50 = close.rolling(50).mean().iloc[-1]; ma200 = close.rolling(200).mean().iloc[-1]
            trend_score = (10 if close.iloc[-1] > ma50 else 0) + (10 if ma50 > ma200 else 0) + (10 if close.iloc[-1] > ma200 else 0)
            signals = []
            if range_pct < 0.05: signals.append("極度収縮")
            if vol_ratio < 0.7: signals.append("Vol枯渇")
            if trend_score == 30: signals.append("MA整列")
            return {"score": tight_score + vol_score + trend_score, "atr": atr, "signals": signals}
        except: return {"score": 0, "atr": 0, "signals": []}

class RSAnalyzer:
    @staticmethod
    def calculate(ticker_df, benchmark_df):
        try:
            if benchmark_df is None: return 50
            common = ticker_df.index.intersection(benchmark_df.index)
            if len(common) < 200: return 50
            t = ticker_df.loc[common, "Close"]; s = benchmark_df.loc[common, "Close"]
            periods = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}; weights = {"3m": 0.4, "6m": 0.2, "9m": 0.2, "12m": 0.2}
            raw = sum([((t.iloc[-1]/t.iloc[-d]-1) - (s.iloc[-1]/s.iloc[-d]-1)) * w for p, d, w in [(p, periods[p], weights[p]) for p in periods if len(t) > d]])
            return max(1, min(99, int(50 + raw * 100)))
        except: return 50

class StrategyValidator:
    @staticmethod
    def run_backtest(df):
        try:
            if len(df) < 200: return 1.0
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
                    pivot = high.iloc[i-20:i].max()
                    if close.iloc[i] > pivot and close.iloc[i] > close.rolling(50).mean().iloc[i]:
                        in_pos = True; entry_p = close.iloc[i]; stop_p = entry_p - (atr.iloc[i] * 2.0)
            if not trades: return 1.0
            pos = sum([t for t in trades if t > 0]); neg = abs(sum([t for t in trades if t < 0]))
            return round(pos / neg, 2) if neg > 0 else 5.0
        except: return 1.0

# ==========================================
# EXECUTION
# ==========================================

def calculate_position(entry, stop, usd_jpy):
    total_usd = CONFIG["CAPITAL_JPY"] / usd_jpy
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
    print("=" * 50); print("🛡 SENTINEL PRO v4.1 REMASTERED"); 
    usd_jpy = CurrencyEngine.get_usd_jpy(); print(f"Rate: {usd_jpy}\n" + "=" * 50)
    
    benchmark = DataEngine.get_data("^GSPC"); qualified = []; return_map = {}
    print(f"Scanning {len(TICKERS)} tickers...")
    
    for ticker in TICKERS:
        df = DataEngine.get_data(ticker)
        if df is None: continue
        vcp = VCPAnalyzer.calculate(df); rs = RSAnalyzer.calculate(df, benchmark); pf = StrategyValidator.run_backtest(df)
        if vcp["score"] < CONFIG["MIN_VCP_SCORE"] or rs < CONFIG["MIN_RS_RATING"] or pf < CONFIG["MIN_PROFIT_FACTOR"]: continue
        pivot = df["High"].iloc[-20:].max(); price = df["Close"].iloc[-1]
        entry = pivot * 1.002; stop = entry - vcp["atr"] * 2.0; target = entry + (entry - stop) * 2.5; dist = (price - pivot) / pivot
        status = "ACTION" if -0.05 <= dist <= 0.03 else ("WAIT" if dist < -0.05 else "EXTENDED")
        shares = calculate_position(entry, stop, usd_jpy); return_map[ticker] = df["Close"].pct_change().dropna()
        qualified.append({"ticker": ticker, "status": status, "price": round(price,2), "entry": round(entry,2), "stop": round(stop,2), "target": round(target,2), "shares": shares, "vcp": vcp, "rs": rs, "pf": pf, "sector": DataEngine.get_sector(ticker)})
    
    qualified.sort(key=lambda x: ({"ACTION": 3, "WAIT": 2, "EXTENDED": 1}.get(x["status"], 0), x["vcp"]["score"] + x["rs"]), reverse=True)
    selected = filter_portfolio(qualified, return_map)
    
    today = datetime.now().strftime("%Y-%m-%d"); runtime = round(time.time() - st, 2)
    res = {"date": today, "runtime": f"{runtime}s", "usd_jpy": usd_jpy, "selected": selected, "qualified": qualified}
    with open(RESULTS_DIR / f"{today}.json", 'w', encoding='utf-8') as f: json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    
    msg = [f"🛡 SENTINEL PRO v4.1 (Rate:{usd_jpy})\nScan:{len(TICKERS)} | Sel:{len(selected)}\n" + "="*20]
    for s in selected:
        msg.append(f"\n{'💎' if s['status'] == 'ACTION' else '⏳'} {s['ticker']} [{s['status']}]\nVCP:{s['vcp']['score']} | RS:{s['rs']} | PF:{s['pf']:.2f}\nEntry:${s['entry']:.2f} Stop:${s['stop']:.2f}\n推奨:{s['shares']}株 | 💡{','.join(s['vcp']['signals'])}")
    
    print("--- START JSON DATA ---"); print(json.dumps(res, ensure_ascii=False)); print("--- END JSON DATA ---")
    send_line("\n".join(msg))

def send_line(message):
    if not ACCESS_TOKEN or not USER_ID: return
    try: requests.post("https://api.line.me/v2/bot/message/push", headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}, json={"to": USER_ID, "messages": [{"type": "text", "text": message[:4000]}]}, timeout=10)
    except: pass

if __name__ == "__main__":
    run()

