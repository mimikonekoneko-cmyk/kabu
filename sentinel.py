#!/usr/bin/env python3
"""
==============================================================================
🛡️ SENTINEL PRO v5.0 — ローカル統一版
==============================================================================
改善点:
  - VCPAnalyzer を app.py と同じ高精度版に統一（内訳付き）
  - その他クラスは既存のまま維持
==============================================================================
"""

import os, time, json, pickle, warnings, requests
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import feedparser

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

warnings.filterwarnings("ignore")

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================

def _ei(k, d): 
    v = os.getenv(k, "").strip()
    return int(v) if v else int(d)

def _ef(k, d):
    v = os.getenv(k, "").strip()
    return float(v) if v else float(d)

CONFIG = {
    "CAPITAL_JPY":       _ei("CAPITAL_JPY", 350_000),
    "MAX_POSITIONS":     _ei("MAX_POSITIONS", 20),
    "ACCOUNT_RISK_PCT":  _ef("ACCOUNT_RISK_PCT", 0.015),
    "MAX_SAME_SECTOR":   _ei("MAX_SAME_SECTOR", 2),
    "MIN_RS_RATING":     _ei("MIN_RS_RATING", 70),
    "MIN_VCP_SCORE":     _ei("MIN_VCP_SCORE", 55),
    "MIN_PROFIT_FACTOR": _ef("MIN_PROFIT_FACTOR", 1.1),
    "STOP_LOSS_ATR":     _ef("STOP_LOSS_ATR", 2.0),
    "TARGET_R_MULTIPLE": _ef("TARGET_R_MULTIPLE", 2.5),
    "CACHE_EXPIRY":      12 * 3600,
    "NEWS_FETCH_TIMEOUT": 6,
    "NEWS_MAX_CHARS":     400,
}

CACHE_DIR   = Path("./cache_v45"); CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results");   RESULTS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# 📋 TICKER UNIVERSE (450+)
# ==============================================================================

ORIGINAL_LIST = [
    'NVDA','AMD','AVGO','TSM','ASML','MU','QCOM','MRVL','LRCX','AMAT',
    'KLAC','ADI','ON','SMCI','ARM','MPWR','TER','RKLB','ASTS','PLTR',
    'MSFT','GOOGL','GOOG','META','AAPL','AMZN','NFLX','CRM','NOW',
    'SNOW','ADBE','INTU','ORCL','SAP','COST','WMT','TSLA','SBUX','NKE',
    'MELI','BABA','CVNA','LLY','ABBV','REGN','VRTX','NVO','BSX',
    'HOLX','OMER','DVAX','RARE','RIGL','KOD','TARS','MA',
    'V','COIN','MSTR','HOOD','PAY','COHR','ACN','ETN','SPOT',
    'RDDT','RBLX','CEVA','FFIV','DAKT','EPAC',
    'ASTE','SNDK','WDC','STX','GEV','APH','TXN','PG','UBER',
    'BE','LITE','IBM','CLS','CSCO','APLD','ANET','NET','GLW','PANW',
    'CRWD','NBIS','RCL','IONQ','ROP','PM','PEP','KO','SPY','QQQ','IWM',
    'AERO','INBX','CCOI','ONDS',
]

EXPANSION_LIST = [
    'BRK-B','JPM','UNH','XOM','HD','MRK','CVX','BAC','LIN','DIS','TMO','MCD','ABT','WFC',
    'CMCSA','VZ','PFE','CAT','ISRG','GE','SPGI','HON','UNP','RTX','LOW','GS','BKNG','ELV',
    'AXP','COP','MDT','SYK','BLK','NEE','BA','TJX','PGR','ETN','LMT','C','CB','ADP','MMC',
    'PLD','CI','MDLZ','AMT','BX','TMUS','SCHW','MO','EOG','DE','SO','DUK','SLB','CME','SHW',
    'CSX','PYPL','CL','EQIX','ICE','FCX','MCK','TGT','USB','PH','GD','BDX','ITW','ABNB',
    'HCA','NXPI','PSX','MAR','NSC','EMR','AON','PNC','CEG','CDNS','SNPS','MCO','PCAR','COF',
    'FDX','ORLY','ADSK','VLO','OXY','TRV','AIG','HLT','WELL','CARR','AZO','PAYX','MSI','TEL',
    'PEG','AJG','ROST','KMB','APD','URI','DHI','OKE','WMB','TRGP','SRE','CTAS','AFL','GWW',
    'LHX','MET','PCG','CMI','F','GM','STZ','PSA','O','DLR','CCI','KMI','ED','XEL','EIX',
    'WEC','D','AWK','ES','AEP','EXC','STM','GFS',
    'DDOG','MDB','HUBS','TTD','APP','PATH','MNDY','GTLB','IOT','DUOL','CFLT','AI',
    'SOUN','CLSK','MARA','RIOT','BITF','HUT','IREN','WULF','CORZ','CIFR',
    'AFRM','UPST','SOFI','DKNG',
    'MRNA','BNTX','UTHR','SMMT','VKTX','ALT','CRSP','NTLA','BEAM',
    'LUNR','HII','AXON','TDG','CCJ','URA','UUUU','DNN','NXE','UEC',
    'SCCO','AA','NUE','STLD','TTE',
    'CART','CAVA','LULU','ONON','DECK','CROX','WING','CMG','DPZ','YUM','CELH','MNST',
    'GME','AMC','U','OPEN','Z',
    'SMH','XLF','XLV','XLE','XLI','XLK','XLC','XLY','XLP','XLB','XLU','XLRE',
    'VRT','ALAB','OKLO','ASTS',
]

TICKERS = sorted(list(set(ORIGINAL_LIST + EXPANSION_LIST)))

# ==============================================================================
# 💱 CURRENCY ENGINE
# ==============================================================================

class CurrencyEngine:
    @staticmethod
    def get_usd_jpy() -> float:
        try:
            df = yf.Ticker("JPY=X").history(period="1d")
            return round(float(df["Close"].iloc[-1]), 2) if not df.empty else 150.0
        except:
            return 150.0

# ==============================================================================
# 💾 DATA ENGINE
# ==============================================================================

class DataEngine:
    @staticmethod
    def get_data(ticker: str, period: str = "700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < CONFIG["CACHE_EXPIRY"]:
                try:
                    with open(cache_file, "rb") as f: return pickle.load(f)
                except: pass
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 150: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            with open(cache_file, "wb") as f: pickle.dump(df, f)
            return df
        except: return None

    @staticmethod
    def get_sector(ticker: str) -> str:
        cf = CACHE_DIR / "sectors.json"
        sm = {}
        if cf.exists():
            try:
                with open(cf) as f: sm = json.load(f)
            except: pass
        if ticker in sm: return sm[ticker]
        try:
            s = yf.Ticker(ticker).info.get("sector", "Unknown")
            sm[ticker] = s
            with open(cf, "w") as f: json.dump(sm, f)
            return s
        except: return "Unknown"

# ==============================================================================
# 📊 FUNDAMENTAL ENGINE
# ==============================================================================

class FundamentalEngine:
    CACHE_TTL = 24 * 3600

    @staticmethod
    def get(ticker: str) -> dict:
        cf = CACHE_DIR / f"fund_{ticker}.json"
        if cf.exists():
            if time.time() - cf.stat().st_mtime < FundamentalEngine.CACHE_TTL:
                try:
                    with open(cf) as f: return json.load(f)
                except: pass
        try:
            info = yf.Ticker(ticker).info
            data = {
                "analyst_target":      info.get("targetMeanPrice"),
                "analyst_target_high": info.get("targetHighPrice"),
                "analyst_target_low":  info.get("targetLowPrice"),
                "analyst_count":       info.get("numberOfAnalystOpinions"),
                "recommendation":      info.get("recommendationKey"),
                "short_ratio":         info.get("shortRatio"),
                "short_pct_float":     info.get("shortPercentOfFloat"),
                "insider_pct":         info.get("heldPercentInsiders"),
                "institution_pct":     info.get("heldPercentInstitutions"),
                "pe_forward":          info.get("forwardPE"),
                "peg_ratio":           info.get("pegRatio"),
                "revenue_growth":      info.get("revenueGrowth"),
                "earnings_growth":     info.get("earningsGrowth"),
                "earnings_date":       str(info.get("earningsTimestamp", "")),
                "eps_forward":         info.get("forwardEps"),
            }
            with open(cf, "w") as f: json.dump(data, f, default=str)
            return data
        except:
            return {}

# ==============================================================================
# 🏛️ INSIDER ENGINE
# ==============================================================================

class InsiderEngine:
    CACHE_TTL = 6 * 3600

    @staticmethod
    def get_recent(ticker: str) -> dict:
        cf = CACHE_DIR / f"insider_{ticker}.json"
        if cf.exists():
            if time.time() - cf.stat().st_mtime < InsiderEngine.CACHE_TTL:
                try:
                    with open(cf) as f: return json.load(f)
                except: pass

        result = {"buy_count": 0, "sell_count": 0, "net_shares": 0, "recent": []}
        try:
            url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
                f"&dateRange=custom&startdt={(datetime.now()-timedelta(days=30)).strftime('%Y-%m-%d')}"
                f"&forms=4"
            )
            headers = {"User-Agent": "sentinel-pro research@example.com"}
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                hits = r.json().get("hits", {}).get("hits", [])
                for hit in hits[:10]:
                    src = hit.get("_source", {})
                    result["recent"].append({
                        "date":   src.get("period_of_report", ""),
                        "name":   src.get("display_names", ""),
                        "filed":  src.get("file_date", ""),
                    })
        except: pass

        # フォールバック
        try:
            t = yf.Ticker(ticker)
            it = t.insider_transactions
            if it is not None and not it.empty:
                cutoff = datetime.now() - timedelta(days=60)
                recent = it[it.index >= cutoff] if hasattr(it.index, 'tz') else it.head(10)
                for _, row in recent.iterrows():
                    txn = str(row.get("Transaction", "")).lower()
                    shares = int(row.get("Shares", 0) or 0)
                    if "sell" in txn or "sale" in txn:
                        result["sell_count"] += 1
                        result["net_shares"]  -= shares
                    elif "buy" in txn or "purchase" in txn:
                        result["buy_count"]  += 1
                        result["net_shares"] += shares
                    result["recent"].append({
                        "date":  str(row.get("Start Date", "")),
                        "name":  str(row.get("Insider", "")),
                        "trans": str(row.get("Transaction", "")),
                        "shares": shares,
                    })
        except: pass

        with open(cf, "w") as f: json.dump(result, f, default=str)
        return result

# ==============================================================================
# 📰 NEWS ENGINE
# ==============================================================================

class NewsEngine:
    CACHE_TTL = 3600

    @staticmethod
    def get(ticker: str) -> dict:
        cf = CACHE_DIR / f"news_{ticker}.json"
        if cf.exists():
            if time.time() - cf.stat().st_mtime < NewsEngine.CACHE_TTL:
                try:
                    with open(cf) as f: return json.load(f)
                except: pass

        articles = []
        seen = set()

        try:
            for n in (yf.Ticker(ticker).news or [])[:5]:
                title = n.get("title", n.get("headline", ""))
                url   = n.get("link",  n.get("url", ""))
                if title and title not in seen:
                    seen.add(title)
                    articles.append({"title": title, "url": url, "body": ""})
        except: pass

        try:
            feed = feedparser.parse(
                f"https://news.google.com/rss/search?q={ticker}+stock+when:3d&hl=en-US&gl=US&ceid=US:en"
            )
            for e in feed.entries[:5]:
                if e.title not in seen:
                    seen.add(e.title)
                    articles.append({"title": e.title, "url": getattr(e, "link", ""), "body": ""})
        except: pass

        if BS4_OK:
            for art in articles[:3]:
                if not art["url"]: continue
                try:
                    r = requests.get(
                        art["url"],
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=CONFIG["NEWS_FETCH_TIMEOUT"],
                    )
                    soup = BeautifulSoup(r.text, "html.parser")
                    paras = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 50]
                    body  = " ".join(paras)[:CONFIG["NEWS_MAX_CHARS"]]
                    art["body"] = body
                except: pass

        result = {"articles": articles[:8], "fetched_at": datetime.now().isoformat()}
        with open(cf, "w") as f: json.dump(result, f, ensure_ascii=False)
        return result

    @staticmethod
    def format_for_prompt(news: dict) -> str:
        lines = []
        for a in news.get("articles", []):
            lines.append(f"• {a['title']}")
            if a.get("body"):
                lines.append(f"  本文抜粋: {a['body'][:200]}")
        return "\n".join(lines) if lines else "ニュースなし"

# ==============================================================================
# 🎯 VCPAnalyzer (app.py と同じ統一版)
# ==============================================================================

class VCPAnalyzer:
    """
    Mark Minervini VCP 分析エンジン（統一版）
    - Tightness (40pt), Volume (30pt), MA (30pt), Pivot (5pt) = 105pt Max
    """
    @staticmethod
    def calculate(df: pd.DataFrame) -> dict:
        try:
            if df is None or len(df) < 130:
                return VCPAnalyzer._empty_result()

            close_s = df["Close"]
            high_s  = df["High"]
            low_s   = df["Low"]
            vol_s   = df["Volume"]

            # ATR(14)
            tr1 = high_s - low_s
            tr2 = (high_s - close_s.shift(1)).abs()
            tr3 = (low_s - close_s.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_val = float(tr.rolling(14).mean().iloc[-1])
            
            if pd.isna(atr_val) or atr_val <= 0:
                return VCPAnalyzer._empty_result()

            # 1. Tightness
            periods = [20, 30, 40, 60]
            vol_ranges = []
            for p in periods:
                p_high = float(high_s.iloc[-p:].max())
                p_low  = float(low_s.iloc[-p:].min())
                if p_high > 0:
                    vol_ranges.append((p_high - p_low) / p_high)
                else:
                    vol_ranges.append(1.0)
            
            curr_range = vol_ranges[0]
            avg_range = float(np.mean(vol_ranges[:3]))
            is_contracting = vol_ranges[0] < vol_ranges[1] < vol_ranges[2]

            if avg_range < 0.10:   tight_score = 40
            elif avg_range < 0.15: tight_score = 30
            elif avg_range < 0.20: tight_score = 20
            elif avg_range < 0.28: tight_score = 10
            else:                  tight_score = 0
            
            if is_contracting: tight_score += 5
            tight_score = min(40, tight_score)

            # 2. Volume
            v20_avg = float(vol_s.iloc[-20:].mean())
            v60_avg = float(vol_s.iloc[-60:-40].mean())
            
            if pd.isna(v20_avg) or pd.isna(v60_avg):
                return VCPAnalyzer._empty_result()
            
            v_ratio = v20_avg / v60_avg if v60_avg > 0 else 1.0

            if v_ratio < 0.45:   vol_score = 30
            elif v_ratio < 0.60: vol_score = 25
            elif v_ratio < 0.75: vol_score = 15
            else:                vol_score = 0
            
            is_dryup = v_ratio < 0.75

            # 3. MA Alignment
            ma50_v  = float(close_s.rolling(50).mean().iloc[-1])
            ma150_v = float(close_s.rolling(150).mean().iloc[-1])
            ma200_v = float(close_s.rolling(200).mean().iloc[-1])
            price_v = float(close_s.iloc[-1])
            
            m_score = 0
            if price_v > ma50_v:   m_score += 10
            if ma50_v > ma150_v:   m_score += 10
            if ma150_v > ma200_v:  m_score += 10

            # 4. Pivot Bonus
            pivot_v = float(high_s.iloc[-50:].max())
            dist_v = (pivot_v - price_v) / pivot_v
            
            p_bonus = 0
            if 0 <= dist_v <= 0.04:
                p_bonus = 5
            elif 0.04 < dist_v <= 0.08:
                p_bonus = 3

            signals = []
            if tight_score >= 35: signals.append("Tight Base (VCP)")
            if is_contracting:    signals.append("V-Contraction Detected")
            if is_dryup:          signals.append("Volume Dry-up Detected")
            if m_score >= 20:     signals.append("Trend Alignment OK")
            if p_bonus > 0:       signals.append("Near Pivot Point")

            return {
                "score": int(min(105, tight_score + vol_score + m_score + p_bonus)),
                "atr": atr_val,
                "signals": signals,
                "is_dryup": is_dryup,
                "range_pct": round(curr_range, 4),
                "vol_ratio": round(v_ratio, 2),
                "breakdown": {
                    "tight": tight_score,
                    "vol": vol_score,
                    "ma": m_score,
                    "pivot": p_bonus
                }
            }
        except Exception:
            return VCPAnalyzer._empty_result()

    @staticmethod
    def _empty_result():
        return {
            "score": 0, "atr": 0.0, "signals": [], 
            "is_dryup": False, "range_pct": 0.0, "vol_ratio": 1.0,
            "breakdown": {"tight": 0, "vol": 0, "ma": 0, "pivot": 0}
        }

# ==============================================================================
# 📈 RS ANALYZER
# ==============================================================================

class RSAnalyzer:
    @staticmethod
    def get_raw_score(df: pd.DataFrame) -> float:
        try:
            c = df["Close"]
            r12 = (c.iloc[-1]/c.iloc[-252]-1) if len(c)>=252 else (c.iloc[-1]/c.iloc[0]-1)
            r6  = (c.iloc[-1]/c.iloc[-126]-1) if len(c)>=126 else (c.iloc[-1]/c.iloc[0]-1)
            r3  = (c.iloc[-1]/c.iloc[-63] -1) if len(c)>=63  else (c.iloc[-1]/c.iloc[0]-1)
            r1  = (c.iloc[-1]/c.iloc[-21] -1) if len(c)>=21  else (c.iloc[-1]/c.iloc[0]-1)
            return (r12*0.4)+(r6*0.2)+(r3*0.2)+(r1*0.2)
        except: return -999.0

# ==============================================================================
# 🔬 STRATEGY VALIDATOR
# ==============================================================================

class StrategyValidator:
    @staticmethod
    def run_backtest(df: pd.DataFrame) -> float:
        try:
            if len(df) < 200: return 1.0
            close = df["Close"]; high = df["High"]; low = df["Low"]
            tr  = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            trades = []; in_pos = False; entry_p = 0.0; stop_p = 0.0
            for i in range(max(50, len(df)-250), len(df)):
                if in_pos:
                    if low.iloc[i] <= stop_p:
                        trades.append(-1.0); in_pos = False
                    elif high.iloc[i] >= entry_p + (entry_p-stop_p)*CONFIG["TARGET_R_MULTIPLE"]:
                        trades.append(CONFIG["TARGET_R_MULTIPLE"]); in_pos = False
                    elif i == len(df)-1:
                        risk = entry_p - stop_p
                        if risk > 0: trades.append(float((close.iloc[i]-entry_p)/risk))
                        in_pos = False
                else:
                    pivot = high.iloc[i-20:i].max()
                    if close.iloc[i] > pivot and close.iloc[i] > close.rolling(50).mean().iloc[i]:
                        in_pos = True
                        entry_p = float(close.iloc[i])
                        stop_p  = entry_p - float(atr.iloc[i])*CONFIG["STOP_LOSS_ATR"]

            if not trades: return 1.0
            pos = sum(t for t in trades if t>0)
            neg = abs(sum(t for t in trades if t<0))
            return round(float(min(10.0, pos/neg if neg>0 else (5.0 if pos>0 else 1.0))), 2)
        except: return 1.0

# ==============================================================================
# 📐 POSITION SIZING
# ==============================================================================

def calculate_position(entry: float, stop: float, usd_jpy: float) -> int:
    try:
        total_usd   = CONFIG["CAPITAL_JPY"] / usd_jpy
        risk_usd    = total_usd * CONFIG["ACCOUNT_RISK_PCT"]
        diff        = abs(entry - stop)
        if diff <= 0: return 0
        shares_risk = int(risk_usd / diff)
        shares_cap  = int((total_usd * 0.4) / entry)
        return max(0, min(shares_risk, shares_cap))
    except: return 0

# ==============================================================================
# 📲 LINE NOTIFICATION
# ==============================================================================

def send_line(message: str):
    token   = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.getenv("LINE_USER_ID")
    if not token or not user_id: return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    for part in [message[i:i+4000] for i in range(0,len(message),4000)]:
        try:
            requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers=headers,
                json={"to": user_id, "messages": [{"type":"text","text":part}]},
                timeout=15,
            )
        except: pass

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

def run():
    start = time.time()
    print("="*60)
    print("🛡️ SENTINEL PRO v5.0 — ローカル統一版")
    print(f"   Capital: ¥{CONFIG['CAPITAL_JPY']:,}  Universe: {len(TICKERS)} tickers")
    print("="*60)

    usd_jpy = CurrencyEngine.get_usd_jpy()
    print(f"USD/JPY: {usd_jpy}")

    # ── Phase 1: RS raw scores ───────────────────────────────────────
    print(f"\nPhase 1: Scanning {len(TICKERS)} tickers...")
    raw_list = []
    for ticker in TICKERS:
        df = DataEngine.get_data(ticker)
        if df is None: continue
        rs_raw = RSAnalyzer.get_raw_score(df)
        if rs_raw == -999.0: continue
        raw_list.append({"ticker": ticker, "df": df, "raw_rs": rs_raw})

    # ── Phase 2: RS percentile ───────────────────────────────────────
    raw_list.sort(key=lambda x: x["raw_rs"])
    total = len(raw_list)
    for i, item in enumerate(raw_list):
        item["rs_rating"] = int(((i+1)/total)*99)

    # ── Phase 3: Technical + Fundamental filter ──────────────────────
    print(f"Phase 2: Technical + Fundamental validation ({total} candidates)...")
    qualified = []

    for item in raw_list:
        ticker = item["ticker"]
        df     = item["df"]
        rs     = item["rs_rating"]

        vcp = VCPAnalyzer.calculate(df)
        pf  = StrategyValidator.run_backtest(df)

        if rs  < CONFIG["MIN_RS_RATING"] \
        or vcp["score"] < CONFIG["MIN_VCP_SCORE"] \
        or pf  < CONFIG["MIN_PROFIT_FACTOR"]:
            continue

        price  = float(df["Close"].iloc[-1])
        pivot  = float(df["High"].iloc[-20:].max())
        entry  = pivot * 1.002
        stop   = entry - vcp["atr"] * CONFIG["STOP_LOSS_ATR"]
        target = entry + (entry-stop) * CONFIG["TARGET_R_MULTIPLE"]
        shares = calculate_position(entry, stop, usd_jpy)
        if shares <= 0: continue

        fund    = FundamentalEngine.get(ticker)
        insider = InsiderEngine.get_recent(ticker)

        analyst_upside = None
        if fund.get("analyst_target") and price > 0:
            analyst_upside = round((fund["analyst_target"] / price - 1) * 100, 1)

        insider_alert = (
            insider["sell_count"] >= 2 and
            insider["sell_count"] > insider["buy_count"] * 2
        )

        dist_pct = (price - pivot) / pivot
        if   -0.05 <= dist_pct <= 0.03: status = "ACTION"
        elif dist_pct < -0.05:          status = "WAIT"
        else:                           status = "EXTENDED"

        qualified.append({
            "ticker":         ticker,
            "status":         status,
            "price":          round(price, 2),
            "entry":          round(entry, 2),
            "stop":           round(stop, 2),
            "target":         round(target, 2),
            "shares":         int(shares),
            "vcp":            vcp,
            "rs":             int(rs),
            "pf":             float(pf),
            "sector":         DataEngine.get_sector(ticker),
            "analyst_target": fund.get("analyst_target"),
            "analyst_upside": analyst_upside,
            "analyst_count":  fund.get("analyst_count"),
            "recommendation": fund.get("recommendation"),
            "short_ratio":    fund.get("short_ratio"),
            "short_pct":      fund.get("short_pct_float"),
            "insider_pct":    fund.get("insider_pct"),
            "institution_pct":fund.get("institution_pct"),
            "pe_forward":     fund.get("pe_forward"),
            "revenue_growth": fund.get("revenue_growth"),
            "insider_alert":  insider_alert,
            "insider_detail": insider,
        })

    # ── Phase 4: Sort & sector diversification ───────────────────────
    status_rank = {"ACTION":3,"WAIT":2,"EXTENDED":1}
    qualified.sort(
        key=lambda x: (status_rank.get(x["status"],0), x["rs"]+x["vcp"]["score"]+x["pf"]*10),
        reverse=True,
    )

    selected = []; sector_counts = {}
    for q in qualified:
        if q["status"] != "ACTION": continue
        sec = q["sector"]
        if sector_counts.get(sec,0) >= CONFIG["MAX_SAME_SECTOR"] and sec != "Unknown": continue
        selected.append(q)
        sector_counts[sec] = sector_counts.get(sec,0)+1
        if len(selected) >= CONFIG["MAX_POSITIONS"]: break

    # ── Phase 5: News fetch for top picks ────────────────────────────
    print("Phase 3: Fetching news for top picks...")
    for s in (selected + [q for q in qualified if q["status"]=="WAIT"][:5]):
        s["news"] = NewsEngine.get(s["ticker"])

    # ── Save ─────────────────────────────────────────────────────────
    today = datetime.now().strftime("%Y-%m-%d")
    run_info = {
        "date":            today,
        "runtime":         f"{round(time.time()-start,2)}s",
        "usd_jpy":         usd_jpy,
        "scan_count":      len(TICKERS),
        "qualified_count": len(qualified),
        "selected_count":  len(selected),
        "selected":        selected,
        "watchlist_wait":  [q for q in qualified if q["status"]=="WAIT"][:8],
        "qualified_full":  qualified,
    }
    out = RESULTS_DIR / f"{today}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ Saved → {out}")
    print(f"   Qualified: {len(qualified)}  |  Action: {len(selected)}")

    # ── LINE message ─────────────────────────────────────────────────
    lines = [
        f"🛡️ SENTINEL PRO v5.0  {today}",
        f"¥{usd_jpy}  |  Scan:{len(TICKERS)}  |  Action:{len(selected)}",
        "─"*20,
    ]
    if not selected:
        lines.append("⚠️ No actionable setups today.")
    else:
        for s in selected:
            sigs = ", ".join(s["vcp"]["signals"]) or "—"
            upside_str = f"  Analyst: {s['analyst_upside']:+.1f}%" if s.get("analyst_upside") else ""
            alert_str  = "  ⚠️ INSIDER SELL" if s.get("insider_alert") else ""
            lines += [
                f"\n💎 {s['ticker']}  [RS{s['rs']} VCP{s['vcp']['score']} PF{s['pf']:.1f}]",
                f"   {s['shares']}株  Entry${s['entry']}  Stop${s['stop']}  Target${s['target']}",
                f"   {sigs}{upside_str}{alert_str}",
                "─"*15,
            ]
    waits = run_info["watchlist_wait"]
    if waits:
        lines.append("\n📋 Watchlist (WAIT)")
        for w in waits:
            lines.append(f"  • {w['ticker']}  RS{w['rs']} VCP{w['vcp']['score']}")

    msg = "\n".join(lines)
    print("\n"+msg)
    send_line(msg)


if __name__ == "__main__":
    run()
