#!/usr/bin/env python3
# SENTINEL INSTITUTIONAL v3.1 - FULL UNIVERSE
# No Ticker Reduction. Institutional Grade Logic.

import os
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================

CONFIG = {
    "CAPITAL_JPY": 350_000,
    "USDJPY": 150,
    "RISK_PER_TRADE": 0.01,
    "MAX_POSITIONS": 4,
    "MIN_RS_PERCENTILE": 75,
    "MIN_EV": 0.2,
    "ATR_MULTIPLIER": 2.0,
    "SLIPPAGE": 0.001,
}

CACHE_DIR = Path("./cache_inst_full")
CACHE_DIR.mkdir(exist_ok=True)

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# ==========================================
# FULL TICKER UNIVERSE (完全維持)
# ==========================================

TICKERS = sorted(list(set([
'NVDA','AMD','AVGO','TSM','ASML','MU','QCOM','MRVL','LRCX','AMAT',
'KLAC','ADI','ON','SMCI','ARM','MPWR','TER',
'RKLB','ASTS','PLTR','AERO',
'MSFT','GOOGL','GOOG','META','AAPL','AMZN','NFLX','CRM','NOW',
'SNOW','ADBE','INTU','ORCL','SAP',
'COST','WMT','TSLA','SBUX','NKE','MELI','BABA','CVNA','MTN',
'LLY','ABBV','REGN','VRTX','NVO','BSX','HOLX','OMER','DVAX',
'RARE','RIGL','KOD','TARS','ORKA','DSGN',
'MA','V','COIN','MSTR','HOOD','PAY','MDLN',
'COHR','ACN','ETN','SPOT','RDDT','RBLX','CEVA','FFIV',
'DAKT','ITRN','TBLA','CHA','EPAC','DJT','TV','SEM',
'SCVL','INBX','CCOI','NMAX','HY','AVR','PRSU','WBTN',
'ASTE','FULC',
'SNDK','WDC','STX','GEV','APH','TXN','PG','UBER',
'BE','LITE','IBM','CLS','CSCO','APLD','ANET','NET',
'GLW','PANW','CRWD','NBIS','RCL','ONDS','IONQ','ROP',
'PM','PEP','KO',
'SPY','QQQ','IWM'
])))

BENCHMARK = "SPY"

# ==========================================
# DATA ENGINE
# ==========================================

class DataEngine:

    @staticmethod
    def get(ticker, period="3y"):
        f = CACHE_DIR / f"{ticker}.pkl"
        if f.exists() and time.time() - f.stat().st_mtime < 86400:
            return pickle.load(open(f, "rb"))

        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 300:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            pickle.dump(df, open(f, "wb"))
            return df
        except:
            return None

# ==========================================
# ATR
# ==========================================

def calculate_atr(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

# ==========================================
# TRUE RS (Percentile)
# ==========================================

def compute_rs_percentiles(data_dict):
    returns = {}
    for t, df in data_dict.items():
        if df is None or t == BENCHMARK:
            continue
        if len(df) < 126:
            continue
        r = (df["Close"].iloc[-1] / df["Close"].iloc[-126]) - 1
        returns[t] = r

    series = pd.Series(returns)
    pct = series.rank(pct=True) * 100
    return pct.to_dict()

# ==========================================
# WALK FORWARD BACKTEST
# ==========================================

def backtest(df):

    close, high, low = df["Close"], df["High"], df["Low"]
    atr = calculate_atr(df)

    trades = []

    for i in range(250, len(df)-30):

        pivot = high.iloc[i-10:i].max()

        if high.iloc[i] <= pivot:
            continue

        entry = df["Open"].iloc[i+1] * (1 + CONFIG["SLIPPAGE"])
        stop = entry - atr.iloc[i] * CONFIG["ATR_MULTIPLIER"]
        risk = entry - stop

        if risk <= 0:
            continue

        for j in range(i+1, i+30):

            if high.iloc[j] >= entry + risk*2:
                trades.append(2)
                break

            if low.iloc[j] <= stop:
                trades.append(-1)
                break

    if not trades:
        return 0,0,0

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]

    winrate = len(wins) / len(trades)
    pf = sum(wins)/abs(sum(losses)) if losses else 10
    ev = (winrate*2) - ((1-winrate)*1)

    return pf, winrate*100, ev

# ==========================================
# MARKET FILTER
# ==========================================

def market_filter(spy_df):

    ma200 = spy_df["Close"].rolling(200).mean().iloc[-1]
    current = spy_df["Close"].iloc[-1]
    vol = spy_df["Close"].pct_change().rolling(20).std().iloc[-1]

    return current > ma200 and vol < 0.025

# ==========================================
# MAIN
# ==========================================

def run():

    data = {}
    for t in TICKERS:
        data[t] = DataEngine.get(t)

    spy_df = data[BENCHMARK]
    if spy_df is None:
        return "SPY Data Error"

    if not market_filter(spy_df):
        return "⚠ Market Regime Not Favorable"

    rs_percentiles = compute_rs_percentiles(data)

    results = []

    for t in TICKERS:
        if t in ["SPY","QQQ","IWM"]:
            continue

        df = data[t]
        if df is None:
            continue

        rs = rs_percentiles.get(t,0)
        if rs < CONFIG["MIN_RS_PERCENTILE"]:
            continue

        pf, winrate, ev = backtest(df)
        if ev < CONFIG["MIN_EV"]:
            continue

        atr = calculate_atr(df).iloc[-1]
        pivot = df["High"].iloc[-10:].max()
        entry = pivot
        stop = entry - atr * CONFIG["ATR_MULTIPLIER"]

        risk = entry - stop
        capital_usd = CONFIG["CAPITAL_JPY"] / CONFIG["USDJPY"]
        shares = int((capital_usd * CONFIG["RISK_PER_TRADE"]) / risk) if risk > 0 else 0

        results.append({
            "ticker": t,
            "RS": round(rs,1),
            "EV": round(ev,2),
            "PF": round(pf,2),
            "Win": round(winrate,1),
            "Entry": round(entry,2),
            "Stop": round(stop,2),
            "Shares": shares
        })

    results = sorted(results, key=lambda x: x["EV"], reverse=True)
    results = results[:CONFIG["MAX_POSITIONS"]]

    report = []
    report.append("🛡 SENTINEL INSTITUTIONAL v3.1")
    report.append("="*45)

    if not results:
        report.append("No Qualified Setups")
    else:
        for r in results:
            report.append(
                f"{r['ticker']} | RS {r['RS']} | EV {r['EV']} | PF {r['PF']} | "
                f"Entry {r['Entry']} | Stop {r['Stop']} | Shares {r['Shares']}"
            )

    return "\n".join(report)

# ==========================================

if __name__ == "__main__":
    print(run())