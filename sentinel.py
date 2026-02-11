#!/usr/bin/env python3

# ==========================================================
# 🛡 SENTINEL PRO v2.3 ULTIMATE
# 258銘柄完全内包 / VCP改良 / 相関制御 / DD推定 / LINE通知
# ==========================================================

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

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================

CONFIG = {
    "CAPITAL_JPY": 350_000,
    "MAX_POSITIONS": 4,
    "DISPLAY_LIMIT": 20,
    "ACCOUNT_RISK_PCT": 0.015,

    "MIN_RS_RATING": 70,
    "MIN_VCP_SCORE": 50,
    "MIN_PROFIT_FACTOR": 1.2,

    "STOP_LOSS_ATR": 2.0,

    "TARGET_CONSERVATIVE": 1.5,
    "TARGET_MODERATE": 2.5,
    "TARGET_AGGRESSIVE": 4.0,

    "CORRELATION_LIMIT": 0.75,
    "MAX_SAME_SECTOR": 2,
}

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("SENTINEL_PRO")

CACHE_DIR = Path("./cache_pro_v23")
CACHE_DIR.mkdir(exist_ok=True)

# ==========================================================
# 🔥 258銘柄 完全リスト（実体）
# ==========================================================

TICKERS = sorted(list(set([

"AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","AVGO",
"JPM","V","UNH","XOM","MA","PG","JNJ","HD","MRK","COST",
"ABBV","ADBE","CRM","CVX","PEP","KO","BAC","WMT","ACN","TMO",
"MCD","LIN","NFLX","AMD","ORCL","CSCO","INTC","QCOM","IBM","TXN",
"AMAT","INTU","NOW","ISRG","GE","CAT","BA","LMT","RTX","NOC",
"DE","GS","MS","BLK","SPGI","AXP","C","PYPL","SCHW","BK",
"USB","T","VZ","CMCSA","DIS","PFE","ABT","BMY","AMGN","GILD",
"VRTX","REGN","ZTS","MDT","SYK","CI","HUM","CME","ICE","ADP",
"MMM","HON","UPS","FDX","UNP","NSC","CSX","DAL","UAL","LUV",
"F","GM","RIVN","NIO","PLTR","SNOW","SHOP","SQ","COIN","UBER",
"LYFT","PANW","CRWD","ZS","NET","OKTA","DDOG","MDB","TEAM","WDAY",
"ANET","MRVL","MU","KLAC","LRCX","ADI","NXPI","MCHP","ON","TSM",
"ASML","SHOP","ROKU","ETSY","FIVE","TJX","LOW","SBUX","NKE","ADSK",
"FTNT","TTD","ROST","EBAY","KDP","MNST","SIRI","EA","ATVI","TTWO",
"VRTX","BIIB","ILMN","DXCM","EW","BDX","DHR","IDXX","HCA","ELV",
"MO","PM","BTI","SHEL","BP","CVS","WBA","KR","TGT","DG",
"DLTR","KMB","CL","GIS","KHC","HSY","CPB","ADM","CAG","WDC",
"STX","HPQ","DELL","HPE","ORLY","AZO","GPC","OXY","SLB","HAL",
"FCX","NEM","RIO","BHP","AA","VALE","DOW","DD","APD","ECL",
"SHW","LIN","SPOT","PINS","DOCU","ZM","BABA","JD","BIDU","TCEHY",
"SONY","NTES","SE","MELI","SAP","INTU","NOW","CRM","UBS","DB",
"RY","TD","BNS","ENB","SU","TRP","SHOP","FIS","FISV","GPN",
"SQ","PYPL","COIN","MA","V","AXP","NDAQ","CME","ICE","SPGI",
"CB","AIG","MET","PRU","ALL","TRV","AON","MMC","HIG","CINF",
"KMI","ET","WMB","OKE","PXD","EOG","DVN","MPC","PSX","VLO",
"MAR","HLT","ABNB","BKNG","EXPE","RCL","CCL","NCLH","CHTR","TMUS",
"CMG","YUM","DPZ","DASH","WING","CVNA","CAR","TSCO","BBY","ULTA",
"KR","ROST","TJX","M","KSS","JWN","GPS","ANF","LEVI","CPRI"

])))
# ==========================================================
# DATA ENGINE
# ==========================================================

class DataEngine:

    @staticmethod
    def get_data(ticker, period="700d"):

        cache_file = CACHE_DIR / f"{ticker}.pkl"

        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        try:
            df = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True
            )

            if df.empty or len(df) < 250:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = ["Close", "High", "Low", "Volume"]
            if not all(c in df.columns for c in required):
                return None

            with open(cache_file, "wb") as f:
                pickle.dump(df, f)

            return df

        except:
            return None


# ==========================================================
# VCP ANALYZER（60固定排除・連続スコア）
# ==========================================================

class VCPAnalyzer:

    @staticmethod
    def calculate_vcp_score(df):

        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            volume = df["Volume"]

            tr = pd.concat([
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

            atr = tr.rolling(14, min_periods=7).mean().iloc[-1]

            if pd.isna(atr) or atr <= 0:
                return {"score": 0, "atr": 0, "signals": []}

            # --- Tightness ---
            recent_high = high.iloc[-10:].max()
            recent_low = low.iloc[-10:].min()
            tightness = (recent_high - recent_low) / atr

            tight_score = max(0, min(40, int((2.5 - tightness) * 20)))

            # --- Volume Dry Up ---
            vol_ma = volume.rolling(50, min_periods=20).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
            vol_score = max(0, min(20, int((1.2 - vol_ratio) * 50)))

            # --- MA Alignment ---
            curr = close.iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]

            ma_score = 0
            if curr > ma50 > ma200:
                ma_score = 20
            elif curr > ma50:
                ma_score = 10

            # --- Momentum ---
            mom5 = close.rolling(5).mean().iloc[-1]
            mom20 = close.rolling(20).mean().iloc[-1]

            mom_ratio = mom5 / mom20 if mom20 > 0 else 1
            mom_score = max(0, min(20, int((mom_ratio - 1.0) * 200)))

            score = min(100, tight_score + vol_score + ma_score + mom_score)

            signals = []
            if tightness < 1.0:
                signals.append("極度収縮")
            elif tightness < 1.8:
                signals.append("収縮中")
            else:
                signals.append("ルーズ")

            if vol_ratio < 0.8:
                signals.append("Vol枯渇")

            if ma_score == 20:
                signals.append("MA整列")

            if mom_ratio > 1.02:
                signals.append("モメンタム+")

            return {
                "score": score,
                "atr": atr,
                "tightness": tightness,
                "signals": signals
            }

        except:
            return {"score": 0, "atr": 0, "signals": []}


# ==========================================================
# RS ANALYZER
# ==========================================================

class RSAnalyzer:

    @staticmethod
    def calculate_rs_rating(ticker_df, benchmark_df):

        try:
            if benchmark_df is None:
                return 50

            common = ticker_df.index.intersection(benchmark_df.index)
            if len(common) < 200:
                return 50

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
            return min(99, max(1, rs))

        except:
            return 50


# ==========================================================
# BACKTEST ENGINE（簡易PF算出）
# ==========================================================

class BacktestEngine:

    @staticmethod
    def calculate_profit_factor(df):

        try:
            returns = df["Close"].pct_change().dropna()

            wins = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())

            if losses == 0:
                return 1.0

            pf = wins / losses
            return round(float(pf), 2)

        except:
            return 1.0


# ==========================================================
# POSITION SIZING
# ==========================================================

def calculate_position_size(entry, stop):

    risk_per_trade = CONFIG["CAPITAL_JPY"] * CONFIG["ACCOUNT_RISK_PCT"]

    risk_per_share = abs(entry - stop)

    if risk_per_share <= 0:
        return 0

    shares = int(risk_per_trade / risk_per_share)

    max_affordable = int(CONFIG["CAPITAL_JPY"] / entry)

    return max(0, min(shares, max_affordable))


# ==========================================================
# CORRELATION FILTER
# ==========================================================

def filter_by_correlation(selected, candidates):

    final = []

    for c in candidates:

        if not final:
            final.append(c)
            continue

        correlated = False

        for f in final:
            if abs(c["corr"].get(f["ticker"], 0)) > CONFIG["CORRELATION_LIMIT"]:
                correlated = True
                break

        if not correlated:
            final

# ==========================================================
# DRAWDOWN ESTIMATION
# ==========================================================

def estimate_portfolio_dd(selected):

    if not selected:
        return 0

    total_risk = 0

    for s in selected:
        entry = s["entry"]
        stop = s["stop"]
        shares = s["shares"]

        total_risk += abs(entry - stop) * shares

    dd_pct = (total_risk / CONFIG["CAPITAL_JPY"]) * 100
    return round(dd_pct, 1)


# ==========================================================
# SIMPLE SECTOR CONTROL（yfinance sector取得）
# ==========================================================

def sector_filter(candidates):

    sector_count = {}
    filtered = []

    for c in candidates:
        sector = c.get("sector", "Unknown")

        if sector not in sector_count:
            sector_count[sector] = 0

        if sector_count[sector] < CONFIG["MAX_SAME_SECTOR"]:
            filtered.append(c)
            sector_count[sector] += 1

        if len(filtered) >= CONFIG["MAX_POSITIONS"]:
            break

    return filtered


# ==========================================================
# LINE NOTIFIER
# ==========================================================

def send_line(message):

    if not ACCESS_TOKEN or not USER_ID:
        return

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "to": USER_ID,
        "messages": [{"type": "text", "text": message}]
    }

    requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers=headers,
        json=data
    )


# ==========================================================
# MAIN ENGINE
# ==========================================================

def run():

    print("=" * 50)
    print("🛡 SENTINEL PRO v2.3 ULTIMATE")
    print("=" * 50)

    benchmark = DataEngine.get_data("^GSPC")

    results = []
    qualified = []

    price_map = {}
    return_map = {}

    # --- 解析 ---
    for ticker in TICKERS:

        df = DataEngine.get_data(ticker)

        if df is None:
            continue

        vcp = VCPAnalyzer.calculate_vcp_score(df)
        rs = RSAnalyzer.calculate_rs_rating(df, benchmark)
        pf = BacktestEngine.calculate_profit_factor(df)

        if vcp["score"] < CONFIG["MIN_VCP_SCORE"]:
            continue

        if rs < CONFIG["MIN_RS_RATING"]:
            continue

        if pf < CONFIG["MIN_PROFIT_FACTOR"]:
            continue

        price = df["Close"].iloc[-1]

        entry = price * 1.01
        stop = price - vcp["atr"] * CONFIG["STOP_LOSS_ATR"]
        target = price + (price - stop) * CONFIG["TARGET_MODERATE"]

        shares = calculate_position_size(entry, stop)

        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
        except:
            sector = "Unknown"

        returns = df["Close"].pct_change().dropna()
        return_map[ticker] = returns

        qualified.append({
            "ticker": ticker,
            "vcp": vcp,
            "rs": rs,
            "pf": pf,
            "price": price,
            "entry": entry,
            "stop": stop,
            "target": target,
            "shares": shares,
            "sector": sector,
        })

    # --- 相関計算 ---
    for q in qualified:
        corr_map = {}
        for other in qualified:
            if q["ticker"] == other["ticker"]:
                continue
            try:
                corr = return_map[q["ticker"]].corr(return_map[other["ticker"]])
                corr_map[other["ticker"]] = corr
            except:
                corr_map[other["ticker"]] = 0
        q["corr"] = corr_map

    # --- スコア順 ---
    qualified.sort(
        key=lambda x: (x["vcp"]["score"] + x["rs"] + x["pf"]),
        reverse=True
    )

    # --- セクター制御 ---
    sector_filtered = sector_filter(qualified)

    # --- 相関制御 ---
    selected = filter_by_correlation([], sector_filtered)

    # --- DD推定 ---
    est_dd = estimate_portfolio_dd(selected)

    # ======================================================
    # 出力
    # ======================================================

    print(f"Scan: {len(TICKERS)} | Qualified: {len(qualified)}")
    print(f"Selected: {len(selected)} | Est.MaxDD: {est_dd}%")
    print("-" * 50)

    # --- Selected表示 ---
    for s in selected:
        print(f"{s['ticker']} [ACTION]")
        print(f"  VCP:{s['vcp']['score']} RS:{s['rs']} PF:{s['pf']}")
        print(f"  Now:${round(s['price'],2)}")
        print(f"  Entry:${round(s['entry'],2)}")
        print(f"  Stop:${round(s['stop'],2)}")
        print(f"  Target:${round(s['target'],2)}")
        print(f"  推奨:{s['shares']}株")
        print(f"  {','.join(s['vcp']['signals'])}")
        print()

    # --- Qualified簡易表示 ---
    print("---- Qualified (Top 10) ----")
    for q in qualified[:10]:
        print(
            f"{q['ticker']} "
            f"V:{q['vcp']['score']} "
            f"RS:{q['rs']} "
            f"PF:{q['pf']}"
        )

    # --- LINE送信 ---
    message = f"""
🛡 SENTINEL PRO v2.3

Scan:{len(TICKERS)}
Qualified:{len(qualified)}
Selected:{len(selected)}
EstDD:{est_dd}%
"""

    send_line(message)


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":
    run()
