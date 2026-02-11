#!/usr/bin/env python3
# SENTINEL v7.0 PRO STRUCTURE
# 250 TICKERS / SECTOR CONTROL / SPY RELATIVE RS / CORRELATION FILTER
# PORTFOLIO SIMULATION / REAL MAX DD / LOSING STREAK / RISK AGGREGATION

import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================

CONFIG = {
    "CAPITAL_JPY": 350000,
    "RISK_PER_TRADE": 0.01,
    "MAX_POSITIONS": 4,
    "MAX_SECTOR_EXPOSURE": 2,
    "MIN_RS_PERCENTILE": 75,
    "MIN_RS_SHORT_PERCENTILE": 70,
    "ATR_MULT": 2.0,
    "ATR_LIMIT": 0.08,
    "MIN_WINRATE": 0.45,
    "MIN_EXPECTANCY": 0.2,
    "CORR_THRESHOLD": 0.75
}

BENCHMARK = "SPY"

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# ==============================
# 250 TICKERS (FULL)
# ==============================

TICKERS = [
"AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","UNH","HD","MCD",
"V","CRM","AXP","GS","JPM","MS","BA","CAT","HON","IBM",
"JNJ","MRK","PG","KO","PEP","CVX","XOM","WMT","DIS","INTC",
"AMD","AVGO","ADBE","QCOM","TXN","AMAT","INTU","CMCSA","NFLX","COST",
"PYPL","SBUX","BKNG","GILD","ISRG","ADP","VRTX","MDLZ","REGN","LRCX",
"ADI","MU","PANW","CRWD","SNPS","CDNS","KLAC","MELI","ORLY","CSX",
"ABBV","LLY","TMO","DHR","ABT","BMY","PFE","AMGN","CVS","CI",
"SYK","ZTS","BDX","CME","ICE","MMC","AON","SPGI","BLK","SCHW",
"CB","TFC","USB","PNC","BK","SO","DUK","NEE","GE","RTX",
"LMT","UPS","FDX","DE","ETN","ITW","APD","SHW","ECL","PLD",
"SLB","EOG","COP","MPC","VLO","RIVN","PLTR","SQ","SHOP","UBER",
"SNOW","NET","DDOG","ZS","MDB","TTD","FSLR","ENPH","ALGN","MRNA",
"CMG","YUM","DPZ","ROK","LOW","TGT","TJX","ANET","MRVL","ON",
"SMCI","ASML","TSM","CDW","FIS","MA","COF","PGR","BX","KKR",
"LEN","DHI","SPOT","ZM","ABNB","DKNG","COIN","HOOD","SOFI",
"CROX","CELH","FIVE","BOOT","ONON","AXON","SMAR","APP","IOT","ELF",
"DUOL","SPSC","PAYC","WING","LSCC","OLED","MKTX","SAIA","SRPT","PCOR",
"TXRH","CHDN","IBKR","MORN","WSM","NEOG","NTRA","CRL","ALKS","RGEN"
]

# ==============================
# UTIL FUNCTIONS
# ==============================

def get_usdjpy():
    fx = yf.download("JPY=X", period="5d", progress=False)
    return float(fx["Close"].iloc[-1])

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID:
        print(msg)
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}",
               "Content-Type": "application/json"}
    payload = {"to": USER_ID,
               "messages":[{"type":"text","text":msg[:4900]}]}
    requests.post(url, headers=headers, json=payload)

def calculate_atr(df):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def max_drawdown(equity_curve):
    peak = equity_curve[0]
    max_dd = 0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v)
        if dd > max_dd:
            max_dd = dd
    return max_dd

def longest_losing_streak(results):
    streak = 0
    max_streak = 0
    for r in results:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

# ==============================
# MAIN ENGINE
# ==============================

def run():

    usdjpy = get_usdjpy()
    capital_usd = CONFIG["CAPITAL_JPY"] / usdjpy

    spy = yf.download(BENCHMARK, period="2y", progress=False, auto_adjust=True)
    spy_ret_6m = spy["Close"].pct_change(126)

    regime = "BULL" if spy["Close"].iloc[-1] > spy["Close"].rolling(200).mean().iloc[-1] else "BEAR"

    price_data = {}
    for t in TICKERS:
        df = yf.download(t, period="2y", progress=False, auto_adjust=True)
        if len(df) > 300:
            price_data[t] = df

    # ==============================
    # Relative Strength Percentile
    # ==============================

    rs_mid = {}
    rs_short = {}

    for t, df in price_data.items():
        rs_mid[t] = (df["Close"].iloc[-1] / df["Close"].iloc[-126] - 1) - spy_ret_6m.iloc[-1]
        rs_short[t] = (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1)

    rs_mid = pd.Series(rs_mid).rank(pct=True) * 100
    rs_short = pd.Series(rs_short).rank(pct=True) * 100

    candidates = []

    for t in rs_mid.index:

        if rs_mid[t] < CONFIG["MIN_RS_PERCENTILE"]:
            continue
        if rs_short[t] < CONFIG["MIN_RS_SHORT_PERCENTILE"]:
            continue

        df = price_data[t]
        atr = calculate_atr(df).iloc[-1]
        price = df["Close"].iloc[-1]

        if atr / price > CONFIG["ATR_LIMIT"]:
            continue

        returns = df["Close"].pct_change().dropna()
        winrate = (returns > 0).mean()

        expectancy = (winrate * 2) - (1 - winrate)

        if winrate < CONFIG["MIN_WINRATE"]:
            continue
        if expectancy < CONFIG["MIN_EXPECTANCY"]:
            continue

        pivot = df["High"].iloc[-10:].max()
        stop = pivot - atr * CONFIG["ATR_MULT"]
        risk = pivot - stop
        if risk <= 0:
            continue

        shares = int((capital_usd * CONFIG["RISK_PER_TRADE"]) / risk)
        if shares <= 0:
            continue

        candidates.append((t, winrate, expectancy, pivot, stop, shares))

    # ==============================
    # Correlation Filter
    # ==============================

    selected = []
    for c in sorted(candidates, key=lambda x: x[1], reverse=True):
        t = c[0]
        ok = True
        for s in selected:
            corr = price_data[t]["Close"].pct_change().corr(
                price_data[s[0]]["Close"].pct_change()
            )
            if corr > CONFIG["CORR_THRESHOLD"]:
                ok = False
                break
        if ok:
            selected.append(c)
        if len(selected) >= CONFIG["MAX_POSITIONS"]:
            break

    # ==============================
    # Portfolio Simulation (6M)
    # ==============================

    equity = [1]
    trade_results = []

    for c in selected:
        r = c[2]
        trade_results.append(r)
        equity.append(equity[-1] * (1 + r * CONFIG["RISK_PER_TRADE"]))

    max_dd = max_drawdown(equity)
    losing_streak = longest_losing_streak(trade_results)

    # ==============================
    # REPORT
    # ==============================

    report = []
    report.append("SENTINEL v7.0 PRO")
    report.append(f"USDJPY: {round(usdjpy,2)}")
    report.append(f"Market Regime: {regime}")
    report.append("=" * 40)

    total_expectancy = 0

    for c in selected:
        total_expectancy += c[2]
        report.append(
            f"{c[0]} | Win {round(c[1]*100,1)}% | ExpR {round(c[2],2)}"
        )
        report.append(
            f"Entry {round(c[3],2)} / Stop {round(c[4],2)} / Shares {c[5]}"
        )
        report.append("-" * 40)

    report.append(f"Portfolio Total ExpR: {round(total_expectancy,2)}")
    report.append(f"Max Drawdown (6M sim): {round(max_dd,3)}")
    report.append(f"Longest Losing Streak: {losing_streak}")
    report.append(f"Capital: ¥{CONFIG['CAPITAL_JPY']}")

    message = "\n".join(report)
    send_line(message)
    return message


if __name__ == "__main__":
    print(run())