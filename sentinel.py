#!/usr/bin/env python3
# SENTINEL v6.0 – DECISION COMPLETE VERSION

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "CAPITAL_JPY": 350000,
    "RISK_PER_TRADE": 0.01,
    "MAX_POSITIONS": 4,
    "MIN_RS_MID": 75,
    "MIN_RS_SHORT": 70,
    "ATR_MULT": 2.0,
    "ATR_LIMIT": 0.08,
    "MIN_WINRATE": 0.45
}

BENCHMARK = "SPY"

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# ===============================
# 250銘柄（重複排除済）
# ===============================

TICKERS = list(dict.fromkeys([
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
# Russell2000主要追加
"CROX","CELH","FIVE","BOOT","ONON","AXON","SMAR","APP","IOT","ELF",
"DUOL","SPSC","PAYC","WING","LSCC","OLED","MKTX","SAIA","SRPT","PCOR",
"TXRH","CHDN","IBKR","MORN","WSM","NEOG","NTRA","CRL","ALKS","RGEN"
]))

# ===============================
# UTIL
# ===============================

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
        df["High"]-df["Low"],
        (df["High"]-df["Close"].shift()).abs(),
        (df["Low"]-df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def backtest_stats(df):
    atr = calculate_atr(df)
    wins=0
    trades=0

    for i in range(200,len(df)-30):
        pivot=df["High"].iloc[i-10:i].max()
        if df["High"].iloc[i] <= pivot:
            continue

        entry=df["Open"].iloc[i+1]
        stop=entry-atr.iloc[i]*2
        risk=entry-stop
        if risk<=0: continue

        trades+=1

        for j in range(i+1,i+30):
            if df["High"].iloc[j]>=entry+risk*2:
                wins+=1; break
            if df["Low"].iloc[j]<=stop:
                break

    if trades==0:
        return 0,0,0

    winrate=wins/trades
    ev=(winrate*2)-(1-winrate)
    monthly=ev*4
    dd=(1-winrate)*4

    return winrate,monthly,dd

# ===============================
# MAIN
# ===============================

def run():

    usdjpy=get_usdjpy()
    capital_usd=CONFIG["CAPITAL_JPY"]/usdjpy

    spy=yf.download(BENCHMARK,period="2y",progress=False,auto_adjust=True)
    regime="BULL" if spy["Close"].iloc[-1]>spy["Close"].rolling(200).mean().iloc[-1] else "BEAR"

    candidates=[]

    for t in TICKERS:
        df=yf.download(t,period="2y",progress=False,auto_adjust=True)
        if len(df)<300: continue

        rs_mid=(df["Close"].iloc[-1]/df["Close"].iloc[-126])-1
        rs_short=(df["Close"].iloc[-1]/df["Close"].iloc[-20])-1

        if rs_mid<0.2 or rs_short<0.1:
            continue

        atr=calculate_atr(df).iloc[-1]
        price=df["Close"].iloc[-1]

        if atr/price>CONFIG["ATR_LIMIT"]:
            continue

        winrate,monthly,dd=backtest_stats(df)
        if winrate<CONFIG["MIN_WINRATE"]:
            continue

        pivot=df["High"].iloc[-10:].max()
        stop=pivot-atr*CONFIG["ATR_MULT"]
        risk=pivot-stop
        if risk<=0: continue

        shares=int((capital_usd*CONFIG["RISK_PER_TRADE"])/risk)
        if shares<=0: continue

        candidates.append((t,winrate,monthly,dd,pivot,stop,shares))

    candidates=sorted(candidates,key=lambda x:x[1],reverse=True)
    candidates=candidates[:CONFIG["MAX_POSITIONS"]]

    report=[]
    report.append("🛡 SENTINEL v6.0")
    report.append(f"USDJPY: {round(usdjpy,2)}")
    report.append(f"Market Regime: {regime}")
    report.append("="*40)

    total_monthly=0
    total_risk=0

    for c in candidates:
        total_monthly+=c[2]
        total_risk+=CONFIG["RISK_PER_TRADE"]
        report.append(
            f"{c[0]} | Win {round(c[1]*100,1)}% | "
            f"ExpMonth {round(c[2],2)}R | "
            f"MaxDD {round(c[3],2)}R\n"
            f"Entry {round(c[4],2)} / Stop {round(c[5],2)} / Shares {c[6]}"
        )
        report.append("-"*40)

    report.append(f"Portfolio Expected Monthly R: {round(total_monthly,2)}")
    report.append(f"Total Risk Today: {round(total_risk*100,1)}%")
    report.append(f"Capital: ¥{CONFIG['CAPITAL_JPY']}")

    msg="\n".join(report)
    send_line(msg)
    return msg

if __name__=="__main__":
    print(run())