#!/usr/bin/env python3
# ==============================================================
# SENTINEL PRO v3.0 - PRO SPEC EDITION
# RS: Percentile Rank
# VCP: Continuous Scoring
# Backtest: Realistic (min trades, slippage, fee, 14d exit)
# Filters: Liquidity, Regime, ATR spike
# ==============================================================

import os
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================
# CONFIG
# ==============================================================

CONFIG = {
    "CAPITAL_JPY": 350_000,
    "ACCOUNT_RISK_PCT": 0.015,
    "DISPLAY_LIMIT": 15,

    # Filters
    "MIN_VCP_SCORE": 55,
    "MIN_RS_PERCENTILE": 70,
    "MIN_PROFIT_FACTOR": 1.2,
    "MIN_TRADES": 25,

    # Risk
    "STOP_LOSS_ATR": 2.0,
    "TARGET_RR": 2.5,

    # Realism
    "SLIPPAGE": 0.001,
    "FEE_PER_SIDE": 0.0025,
    "MAX_HOLD_DAYS": 14,

    # Liquidity
    "MIN_VOLUME": 500_000,

    # Volatility spike filter
    "ATR_SPIKE_LIMIT": 1.5,
}

CACHE_DIR = Path("./cache_v3")
CACHE_DIR.mkdir(exist_ok=True)

# ==============================================================
# TICKERS (same universe)
# ==============================================================

TICKERS = sorted(list(set([
    'NVDA','AMD','AVGO','TSM','ASML','MU','QCOM','MRVL','LRCX','AMAT',
    'KLAC','ADI','ON','SMCI','ARM','MPWR','TER',
    'RKLB','ASTS','PLTR','AERO',
    'MSFT','GOOGL','META','AAPL','AMZN','NFLX','CRM','NOW',
    'SNOW','ADBE','INTU','ORCL',
    'COST','WMT','TSLA','SBUX','NKE',
    'LLY','ABBV','REGN','VRTX','NVO',
    'MA','V','COIN','MSTR','HOOD',
    'SNDK','WDC','STX','GEV','APH','TXN',
    'GLW','PANW','CRWD','IONQ',
    'SPY','QQQ','IWM'
])))

# ==============================================================
# DATA ENGINE
# ==============================================================

class DataEngine:

    @staticmethod
    def get_data(ticker, period="700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"

        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 250:
                return None

            df = df[['Close','High','Low','Volume']].dropna()

            with open(cache_file, "wb") as f:
                pickle.dump(df, f)

            return df

        except:
            return None


# ==============================================================
# RS ANALYZER (Percentile Rank)
# ==============================================================

class RSAnalyzer:

    @staticmethod
    def calculate_raw_rs(df, spy_df):
        common = df.index.intersection(spy_df.index)
        if len(common) < 252:
            return 0

        t = df.loc[common, 'Close']
        s = spy_df.loc[common, 'Close']

        periods = {'3mo':63,'6mo':126,'9mo':189,'12mo':252}
        weights = {'3mo':0.4,'6mo':0.2,'9mo':0.2,'12mo':0.2}

        score = 0
        for p, d in periods.items():
            if len(t) > d:
                tr = (t.iloc[-1] - t.iloc[-d]) / t.iloc[-d]
                sr = (s.iloc[-1] - s.iloc[-d]) / s.iloc[-d]
                score += (tr - sr) * weights[p]

        return score


    @staticmethod
    def percentile_rank(value, distribution):
        count = sum(v <= value for v in distribution)
        return int((count / len(distribution)) * 100)


# ==============================================================
# VCP ANALYZER (Continuous)
# ==============================================================

class VCPAnalyzer:

    @staticmethod
    def calculate_vcp_score(df):

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        atr_latest = atr.iloc[-1]

        if pd.isna(atr_latest) or atr_latest <= 0:
            return {'score':0,'atr':0}

        # Tightness continuous
        recent_high = high.iloc[-10:].max()
        recent_low = low.iloc[-10:].min()
        tightness = (recent_high - recent_low) / atr_latest
        tight_score = max(0, 30 - tightness * 10)
        tight_score = min(30, tight_score)

        # Volume dry up
        vol_ma = volume.rolling(50).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
        vol_score = max(0, 20 * (1 - vol_ratio))
        vol_score = min(20, vol_score)

        # MA alignment
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        ma_score = 15 if close.iloc[-1] > ma50 > ma200 else 0

        # Momentum
        mom = (close.iloc[-1] / close.iloc[-20]) - 1
        mom_score = min(15, max(0, mom * 100))

        total = tight_score + vol_score + ma_score + mom_score

        return {
            'score': round(total,1),
            'atr': atr_latest
        }


# ==============================================================
# BACKTEST ENGINE (Realistic)
# ==============================================================

class BacktestEngine:

    @staticmethod
    def run(df):

        if len(df) < 250:
            return {'pf':0,'winrate':0}

        close = df['Close']
        high = df['High']
        low = df['Low']

        tr = pd.concat([
            high-low,
            (high-close.shift()).abs(),
            (low-close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()

        trades = []

        for i in range(200, len(df)-CONFIG['MAX_HOLD_DAYS']):

            pivot = high.iloc[i-10:i].max() * 1.002
            if high.iloc[i] < pivot:
                continue

            entry = pivot * (1 + CONFIG['SLIPPAGE'])
            stop = entry - (atr.iloc[i] * CONFIG['STOP_LOSS_ATR'])
            target = entry + (atr.iloc[i] * CONFIG['STOP_LOSS_ATR'] * CONFIG['TARGET_RR'])

            for j in range(i+1, i+CONFIG['MAX_HOLD_DAYS']):

                if high.iloc[j] >= target:
                    trades.append(CONFIG['TARGET_RR'] - CONFIG['FEE_PER_SIDE']*2)
                    break

                if low.iloc[j] <= stop:
                    trades.append(-1 - CONFIG['FEE_PER_SIDE']*2)
                    break

                if j == i + CONFIG['MAX_HOLD_DAYS'] - 1:
                    pnl = (close.iloc[j] - entry) / (entry - stop)
                    pnl -= CONFIG['FEE_PER_SIDE']*2
                    trades.append(pnl)

        if len(trades) < CONFIG['MIN_TRADES']:
            return {'pf':0,'winrate':0}

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]

        total_win = sum(wins)
        total_loss = abs(sum(losses)) if losses else 1

        pf = total_win / total_loss
        winrate = len(wins) / len(trades) * 100

        return {'pf':round(pf,2),'winrate':round(winrate,1)}


# ==============================================================
# MAIN
# ==============================================================

def analyze():

    print("🚀 SENTINEL PRO v3.0 PRO SPEC")

    spy_df = DataEngine.get_data("SPY", period="400d")
    if spy_df is None:
        return "Market data error"

    # Market Regime
    spy_ma200 = spy_df['Close'].rolling(200).mean().iloc[-1]
    bull = spy_df['Close'].iloc[-1] > spy_ma200

    if not bull:
        return "🔴 Bear Market - Strategy Disabled"

    # RS raw calculation
    raw_rs = {}
    data_cache = {}

    for t in TICKERS:
        if t in ['SPY','QQQ','IWM']:
            continue

        df = DataEngine.get_data(t)
        if df is None:
            continue

        data_cache[t] = df
        raw_rs[t] = RSAnalyzer.calculate_raw_rs(df, spy_df)

    rs_distribution = list(raw_rs.values())

    results = []

    for t, df in data_cache.items():

        # Liquidity
        if df['Volume'].iloc[-1] < CONFIG['MIN_VOLUME']:
            continue

        # ATR spike filter
        tr = pd.concat([
            df['High']-df['Low'],
            (df['High']-df['Close'].shift()).abs(),
            (df['Low']-df['Close'].shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        if atr.iloc[-1] / atr.iloc[-20:].mean() > CONFIG['ATR_SPIKE_LIMIT']:
            continue

        # VCP
        vcp = VCPAnalyzer.calculate_vcp_score(df)
        if vcp['score'] < CONFIG['MIN_VCP_SCORE']:
            continue

        # RS percentile
        rs_pct = RSAnalyzer.percentile_rank(raw_rs[t], rs_distribution)
        if rs_pct < CONFIG['MIN_RS_PERCENTILE']:
            continue

        # Backtest
        bt = BacktestEngine.run(df)
        if bt['pf'] < CONFIG['MIN_PROFIT_FACTOR']:
            continue

        results.append({
            'ticker': t,
            'vcp': vcp['score'],
            'rs': rs_pct,
            'pf': bt['pf'],
            'winrate': bt['winrate']
        })

    results.sort(key=lambda x: (x['rs'], x['pf'], x['vcp']), reverse=True)

    report = []
    report.append("="*50)
    report.append("🛡 SENTINEL PRO v3.0 REPORT")
    report.append("="*50)
    report.append(f"Market: 🟢 Bull")
    report.append(f"Qualified: {len(results)}")
    report.append("-"*50)

    for r in results[:CONFIG['DISPLAY_LIMIT']]:
        report.append(
            f"{r['ticker']} | RS:{r['rs']} | PF:{r['pf']} | VCP:{r['vcp']} | WR:{r['winrate']}%"
        )

    if not results:
        report.append("⚠️ No candidates")

    return "\n".join(report)


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":
    print(analyze())