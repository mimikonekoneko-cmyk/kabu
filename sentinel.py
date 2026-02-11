#!/usr/bin/env python3

# ==========================================================
# 🛡 SENTINEL PRO v3.3.1 (Syntax Fixed)
# ----------------------------------------------------------
# 修正内容:
# 1. バグ修正: Pythonの文法エラー(try: with 同一行記述)を修正
# 2. 安定化: インデントを標準的な形式に直し、ランタイムエラーを防止
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

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

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
    "TARGET_CONSERVATIVE": 1.5,
    "TARGET_MODERATE": 2.5,
    "TARGET_AGGRESSIVE": 4.0,
    
    "DISPLAY_LIMIT": 20,
}

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("SENTINEL_PRO")

CACHE_DIR = Path("./cache_v33")
CACHE_DIR.mkdir(exist_ok=True)

# ==========================================================
# TICKER UNIVERSE (MERGED: ORIGINAL + EXPANSION)
# ==========================================================

ORIGINAL_LIST = [
    'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'MU', 'QCOM', 'MRVL', 'LRCX', 'AMAT',
    'KLAC', 'ADI', 'ON', 'SMCI', 'ARM', 'MPWR', 'TER',
    'RKLB', 'ASTS', 'PLTR', 'AERO',
    'MSFT', 'GOOGL', 'GOOG', 'META', 'AAPL', 'AMZN', 'NFLX', 'CRM', 'NOW',
    'SNOW', 'ADBE', 'INTU', 'ORCL', 'SAP',
    'COST', 'WMT', 'TSLA', 'SBUX', 'NKE', 'MELI', 'BABA', 'CVNA', 'MTN',
    'LLY', 'ABBV', 'REGN', 'VRTX', 'NVO', 'BSX', 'HOLX', 'OMER', 'DVAX',
    'RARE', 'RIGL', 'KOD', 'TARS', 'ORKA', 'DSGN',
    'MA', 'V', 'COIN', 'MSTR', 'HOOD', 'PAY', 'MDLN',
    'COHR', 'ACN', 'ETN', 'SPOT', 'RDDT', 'RBLX', 'CEVA', 'FFIV',
    'DAKT', 'ITRN', 'TBLA', 'CHA', 'EPAC', 'DJT', 'TV', 'SEM',
    'SCVL', 'INBX', 'CCOI', 'NMAX', 'HY', 'AVR', 'PRSU', 'WBTN',
    'ASTE', 'FULC',
    'SNDK', 'WDC', 'STX', 'GEV', 'APH', 'TXN', 'PG', 'UBER',
    'BE', 'LITE', 'IBM', 'CLS', 'CSCO', 'APLD', 'ANET', 'NET',
    'GLW', 'PANW', 'CRWD', 'NBIS', 'RCL', 'ONDS', 'IONQ', 'ROP',
    'PM', 'PEP', 'KO',
    'SPY', 'QQQ', 'IWM'
]

EXPANSION_LIST = [
    'BRK-B','JPM','UNH','XOM','HD','MRK','CVX','BAC','LIN','DIS','TMO','MCD','ABT','WFC',
    'CMCSA','VZ','PFE','CAT','ISRG','GE','SPGI','HON','UNP','RTX','LOW','GS','BKNG','ELV',
    'AXP','COP','MDT','SYK','BLK','NEE','BA','TJX','PGR','ETN','LMT','C','CB','ADP','MMC',
    'PLD','CI','MDLZ','AMT','FI','BX','TMUS','SCHW',
    'MO','EOG','DE','SO','DUK','SLB','CME','SHW','CSX','PYPL','CL','EQIX','ICE','FCX',
    'MCK','TGT','USB','PH','GD','BDX','ITW','ABNB','HCA','NXPI','PSX','MAR','NSC','EMR',
    'AON','PNC','CEG','CDNS','SNPS','MCO','PCAR','COF','FDX','ORLY','ADSK','VLO','OXY','TRV',
    'AIG','HLT','WELL','CARR','AZO','PAYX','MSI','TEL','PEG','AJG','ROST','KMB','APD',
    'URI','DHI','OKE','WMB','TRGP','SRE','CTAS','AFL','GWW','LHX','MET','PCG','CMI','F','GM','STZ',
    'PSA','O','DLR','CCI','KMI','ED','XEL','EIX','WEC','D','AWK','ES','AEP','EXC',
    'STM','GFS',
    'DDOG','MDB','HUBS','TTD','APP','PATH','MNDY','GTLB','IOT','DUOL','ZI','CFLT','HCP','AI','SOUN',
    'CLSK','MARA','RIOT','BITF','HUT','IREN','WULF','CORZ','CIFR','SQ','AFRM','UPST','SOFI','DKNG',
    'MRNA','BNTX','UTHR','SMMT','VKTX','ALT','IMGN','CRSP','NTLA','BEAM',
    'LUNR','HII','AXON','TDG',
    'CCJ','URA','UUUU','DNN','NXE','UEC','SCCO','AA','X','NUE','STLD',
    'TTE','PXD','MRO',
    'CART','CAVA','BIRK','KVUE','LULU','ONON','DECK','CROX','WING','CMG','DPZ','YUM','CELH','MNST',
    'GME','AMC','U','OPEN','Z','RDFN',
    'SMH','XLF','XLV','XLE','XLI','XLK','XLC','XLY','XLP','XLB','XLU','XLRE'
]

TICKERS = sorted(list(set(ORIGINAL_LIST + EXPANSION_LIST)))

# ==========================================================
# CURRENCY ENGINE
# ==========================================================
class CurrencyEngine:
    @staticmethod
    def get_usd_jpy():
        try:
            ticker = yf.Ticker("JPY=X")
            df = ticker.history(period="1d")
            rate = df['Close'].iloc[-1]
            if 130 < rate < 180: return round(rate, 2)
            return 152.0
        except: return 152.0

# ==========================================================
# DATA ENGINE
# ==========================================================
class DataEngine:
    @staticmethod
    def get_data(ticker, period="700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except:
                    pass

        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 200: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            required = ["Close", "High", "Low", "Volume"]
            if not all(col in df.columns for col in required): return None
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            return df
        except:
            return None

    @staticmethod
    def get_sector(ticker):
        sector_cache_file = CACHE_DIR / "sectors.json"
        sector_map = {}
        if sector_cache_file.exists():
            try:
                with open(sector_cache_file, 'r') as f:
                    sector_map = json.load(f)
            except:
                pass
        if ticker in sector_map: return sector_map[ticker]
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            sector_map[ticker] = sector
            with open(sector_cache_file, 'w') as f:
                json.dump(sector_map, f)
            return sector
        except:
            return "Unknown"

# ==========================================================
# VCP ANALYZER
# ==========================================================
class VCPAnalyzer:
    @staticmethod
    def calculate(df):
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
            tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=7).mean().iloc[-1]

            h10 = high.iloc[-10:].max()
            l10 = low.iloc[-10:].min()
            range_pct = (h10 - l10) / h10
            
            if range_pct > CONFIG['MAX_TIGHTNESS_PCT']:
                return {"score": 0, "atr": atr, "signals": []}
            
            if range_pct <= 0.05: tight_score = 40
            else: tight_score = int(40 * (1 - (range_pct - 0.05) / 0.10))
            tight_score = max(0, tight_score)

            vol_ma = volume.rolling(50).mean().iloc[-1]
            vol_curr = volume.iloc[-1]
            vol_ratio = vol_curr / vol_ma if vol_ma > 0 else 1.0
            
            if vol_ratio <= 0.6: vol_score = 30
            elif vol_ratio >= 1.2: vol_score = 0
            else: vol_score = int(30 * (1 - (vol_ratio - 0.6) / 0.6))

            curr = close.iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]
            trend_score = 0
            if curr > ma50: trend_score += 10
            if ma50 > ma200: trend_score += 10
            if curr > ma200: trend_score += 10

            score = tight_score + vol_score + trend_score
            signals = []
            if range_pct < 0.05: signals.append("極度収縮")
            if vol_ratio < 0.7: signals.append("Vol枯渇")
            if trend_score == 30: signals.append("MA整列")

            return {"score": score, "atr": atr, "signals": signals}
        except: return {"score": 0, "atr": 0, "signals": []}

# ==========================================================
# RS ANALYZER
# ==========================================================
class RSAnalyzer:
    @staticmethod
    def calculate(ticker_df, benchmark_df):
        try:
            if benchmark_df is None: return 50
            common = ticker_df.index.intersection(benchmark_df.index)
            if len(common) < 200: return 50
            
            t = ticker_df.loc[common, "Close"]
            s = benchmark_df.loc[common, "Close"]
            
            periods = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}
            weights = {"3m": 0.4, "6m": 0.2, "9m": 0.2, "12m": 0.2}
            raw = 0
            for p, d in periods.items():
                if len(t) > d:
                    t_r = (t.iloc[-1] - t.iloc[-d]) / t.iloc[-d]
                    s_r = (s.iloc[-1] - s.iloc[-d]) / s.iloc[-d]
                    raw += (t_r - s_r) * weights[p]
            
            rs = int(50 + raw * 100)
            return max(1, min(99, rs))
        except: return 50

# ==========================================================
# STRATEGY VALIDATOR
# ==========================================================
class StrategyValidator:
    @staticmethod
    def run_backtest(df):
        if len(df) < 200: return 1.0
        close = df['Close']; high = df['High']; low = df['Low']
        
        tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        trades = []
        in_pos = False
        entry_price = 0
        stop_price = 0
        reward_mult = 2.5
        start_idx = max(50, len(df)-250)
        
        for i in range(start_idx, len(df)-10):
            if in_pos:
                if low.iloc[i] <= stop_price:
                    trades.append(-1.0)
                    in_pos = False
                elif high.iloc[i] >= entry_price + (entry_price - stop_price) * reward_mult:
                    trades.append(reward_mult)
                    in_pos = False
                elif i - entry_idx > 20 and close.iloc[i] < entry_price:
                    trades.append((close.iloc[i] - entry_price) / (entry_price - stop_price))
                    in_pos = False
            else:
                pivot = high.iloc[i-20:i].max()
                if close.iloc[i] > pivot and close.iloc[i-1] <= pivot:
                    if close.iloc[i] > close.rolling(50).mean().iloc[i]:
                        in_pos = True; entry_price = close.iloc[i]; entry_idx = i
                        stop_price = entry_price - (atr.iloc[i] * CONFIG['STOP_LOSS_ATR'])
        
        if not trades: return 1.0
        wins = sum([t for t in trades if t > 0])
        losses = abs(sum([t for t in trades if t < 0]))
        if losses == 0: return 10.0
        return round(float(wins / losses), 2)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def filter_portfolio(candidates, return_map):
    selected = []
    sector_counts = {}
    
    for c in candidates:
        ticker = c['ticker']
        sector = DataEngine.get_sector(ticker)
        c['sector'] = sector
        
        if sector_counts.get(sector, 0) >= CONFIG['MAX_SAME_SECTOR'] and sector != "Unknown": continue
            
        is_correlated = False
        if selected:
            my_ret = return_map.get(ticker)
            for s in selected:
                s_ret = return_map.get(s['ticker'])
                try:
                    if my_ret.corr(s_ret) > CONFIG['CORRELATION_LIMIT']:
                        is_correlated = True
                        break
                except: pass
        if is_correlated: continue
            
        selected.append(c)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= CONFIG['MAX_POSITIONS']: break
            
    return selected

def calculate_position(entry, stop, usd_jpy):
    if entry <= 0: return 0
    capital_usd = CONFIG["CAPITAL_JPY"] / usd_jpy
    risk_total_usd = capital_usd * CONFIG["ACCOUNT_RISK_PCT"]
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0: return 0

    shares_by_risk = int(risk_total_usd / risk_per_share)
    shares_by_cap = int((capital_usd * 0.4) / entry)
    if shares_by_risk == 0 and shares_by_cap > 0: return 1
    return max(0, min(shares_by_risk, shares_by_cap))

def run():
    print("=" * 50)
    print("🛡 SENTINEL PRO v3.3.1")
    usd_jpy = CurrencyEngine.get_usd_jpy()
    print(f"Rate: 1 USD = {usd_jpy} JPY")
    print(f"Tracking: {len(TICKERS)} tickers")
    print("=" * 50)

    benchmark = DataEngine.get_data("^GSPC")
    qualified = []
    return_map = {}
    
    print("Scanning... (This may take 2-3 mins)")

    for ticker in TICKERS:
        df = DataEngine.get_data(ticker)
        if df is None: continue
        
        vcp = VCPAnalyzer.calculate(df)
        if vcp["score"] < CONFIG["MIN_VCP_SCORE"]: continue

        rs = RSAnalyzer.calculate(df, benchmark)
        if rs < CONFIG["MIN_RS_RATING"]: continue

        pf = StrategyValidator.run_backtest(df)
        if pf < CONFIG["MIN_PROFIT_FACTOR"]: continue

        pivot = df["High"].iloc[-20:].max()
        price = df["Close"].iloc[-1]
        
        entry = pivot * 1.002
        stop = entry - vcp["atr"] * CONFIG["STOP_LOSS_ATR"]
        target = entry + (entry - stop) * CONFIG["TARGET_R_MULTIPLE"]
        
        dist_pct = ((price - pivot) / pivot) * 100
        if dist_pct > 3.0: status = "EXTENDED"
        elif dist_pct < -5.0: status = "WAIT"
        else: status = "ACTION"

        shares = calculate_position(entry, stop, usd_jpy)
        return_map[ticker] = df["Close"].pct_change().dropna()

        qualified.append({
            "ticker": ticker,
            "status": status,
            "price": price,
            "entry": entry,
            "stop": stop,
            "target": target,
            "shares": shares,
            "vcp": vcp,
            "rs": rs,
            "pf": pf
        })

    status_prio = {"ACTION": 3, "WAIT": 2, "EXTENDED": 1}
    qualified.sort(key=lambda x: (status_prio.get(x["status"], 0), x["vcp"]["score"] + x["rs"] + x["pf"]), reverse=True)

    selected = filter_portfolio(qualified, return_map)

    print(f"Qualified: {len(qualified)} | Selected: {len(selected)}")
    print("-" * 50)

    for s in selected:
        icon = "💎" if s['status'] == 'ACTION' else ("⏳" if s['status'] == 'WAIT' else "👋")
        cost_jpy = s['shares'] * s['entry'] * usd_jpy
        
        print(f"{icon} {s['ticker']} [{s['status']}]")
        print(f"  VCP:{s['vcp']['score']} RS:{s['rs']} PF:{s['pf']}")
        print(f"  Now:${s['price']:.2f}")
        print(f"  📍 Entry:${s['entry']:.2f}")
        print(f"  🛑 Stop :${s['stop']:.2f}")
        print(f"  🎯 Target:${s['target']:.2f}")
        print(f"  📦 推奨:{s['shares']}株 (約{int(cost_jpy/10000)}万円)")
        print(f"  💡 {','.join(s['vcp']['signals'])}")
        print()

    msg_lines = [f"🛡 SENTINEL PRO v3.3.1 (Rate:{usd_jpy})"]
    msg_lines.append(f"Scan:{len(TICKERS)} | Sel:{len(selected)}")
    
    if not selected:
        msg_lines.append("\n⚠️ No candidates.")
    else:
        for s in selected:
            icon = "💎" if s['status'] == 'ACTION' else ("⏳" if s['status'] == 'WAIT' else "👋")
            risk_pct = ((s['entry'] - s['stop']) / s['entry']) * 100
            tgt_pct = ((s['target'] - s['entry']) / s['entry']) * 100
            
            msg_lines.append(f"\n{icon} {s['ticker']} [{s['status']}]")
            msg_lines.append(f"   VCP:{s['vcp']['score']} | RS:{s['rs']} | PF:{s['pf']:.2f}")
            msg_lines.append(f"   Now:${s['price']:.2f}")
            msg_lines.append(f"   📍 Entry:${s['entry']:.2f}")
            msg_lines.append(f"   🛑 Stop :${s['stop']:.2f} (-{risk_pct:.1f}%)")
            msg_lines.append(f"   🎯 T2tgt:${s['target']:.2f} (+{tgt_pct:.1f}%)")
            
            shares_msg = f"{s['shares']}株"
            if s['shares'] == 1 and s['entry']*usd_jpy*0.015 < (s['entry']-s['stop'])*usd_jpy:
                 shares_msg += "(Min)"
            
            msg_lines.append(f"   📦 推奨:{shares_msg} ({s['sector'][:7]})")
            msg_lines.append(f"   💡 {','.join(s['vcp']['signals'])}")
    
    send_line("\n".join(msg_lines))

def send_line(message):
    if not ACCESS_TOKEN or not USER_ID: return
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    
    if len(message) > 4000:
        parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for p in parts:
            payload = {"to": USER_ID, "messages": [{"type": "text", "text": p}]}
            try: requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload, timeout=10)
            except: pass
    else:
        payload = {"to": USER_ID, "messages": [{"type": "text", "text": message}]}
        try: requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload, timeout=10)
        except: pass

if __name__ == "__main__":
    run()

