"""
==============================================================================
🛡️ SENTINEL PRO — app.py (ALL-IN-ONE)
スマホ完結版 | 市場スキャン + リアルタイム診断 + ポートフォリオ管理
==============================================================================
"""

import os, re, time, json, pickle, warnings, datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import feedparser
from openai import OpenAI
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ==============================================================================
# ⚙️ システム設定
# ==============================================================================

NOW       = datetime.datetime.now()
TODAY_STR = NOW.strftime("%Y-%m-%d")

CONFIG = {
    "CAPITAL_JPY":        350_000,
    "MAX_POSITIONS":      20,
    "ACCOUNT_RISK_PCT":   0.015,
    "MAX_SAME_SECTOR":    2,
    "MIN_RS_RATING":      70,
    "MIN_VCP_SCORE":      55,
    "MIN_PROFIT_FACTOR":  1.1,
    "STOP_LOSS_ATR":      2.0,
    "TARGET_R_MULTIPLE":  2.5,
    "CACHE_EXPIRY":       12 * 3600,
}

EXIT_CFG = {
    "STOP_LOSS_ATR_MULT": 2.0,
    "TARGET_R_MULT":      2.5,
    "TRAIL_START_R":      1.5,
    "TRAIL_ATR_MULT":     1.5,
    "SCALE_OUT_R":        1.5,
}

CACHE_DIR   = Path("./cache_v45"); CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results");   RESULTS_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE  = Path("watchlist.json")
PORTFOLIO_FILE  = Path("portfolio.json")

# ==============================================================================
# 🎨 スマホ最適化CSS
# ==============================================================================

st.set_page_config(
    page_title=f"SENTINEL PRO",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",   # スマホではデフォルト閉じる
)

st.markdown("""
<style>
  /* === スマホ基本 === */
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

  /* メトリクスカードを大きく */
  [data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 12px 10px;
  }
  [data-testid="metric-container"] label { font-size: 0.72rem !important; color: #6b7280; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.15rem !important; font-weight: 700; }

  /* ボタンをタップしやすく */
  .stButton > button {
    min-height: 48px;
    font-size: 1rem !important;
    font-weight: 600;
    border-radius: 8px;
  }

  /* タブを大きく */
  .stTabs [data-baseweb="tab"] {
    font-size: 0.9rem;
    padding: 10px 8px;
    font-weight: 600;
  }

  /* AIレポートボックス */
  .ai-box {
    background: #0d1117;
    border-left: 4px solid #00ff7f;
    padding: 18px 16px;
    border-radius: 8px;
    line-height: 1.85;
    font-size: 0.95rem;
  }

  /* ポジションカード */
  .pos-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
  }
  .pos-card.urgent  { border-color: #ef4444; }
  .pos-card.caution { border-color: #f59e0b; }
  .pos-card.profit  { border-color: #00ff7f; }

  .pnl-pos { color: #00ff7f; font-weight: 700; font-size: 1.2rem; }
  .pnl-neg { color: #ef4444; font-weight: 700; font-size: 1.2rem; }
  .pnl-neu { color: #9ca3af; font-weight: 700; font-size: 1.2rem; }

  .exit-info { font-size: 0.8rem; color: #9ca3af; line-height: 1.8; font-family: 'Share Tech Mono', monospace; }

  /* セクション見出し */
  .section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #00ff7f;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 6px;
    margin: 14px 0 10px;
    font-family: 'Share Tech Mono', monospace;
  }

  /* テーブルのスクロール対応 */
  [data-testid="stDataFrame"] { overflow-x: auto; }

  /* サイドバーのティッカーボタン */
  .sidebar-btn { font-size: 0.85rem; }

  /* 余白削減（スマホ） */
  .block-container { padding-top: 0.8rem !important; padding-bottom: 1rem !important; }
  @media (max-width: 768px) {
    .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
  }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 📋 セッション状態
# ==============================================================================

_defaults = {
    "mode": "📊 スキャン",
    "target_ticker": "",
    "trigger_analysis": False,
    "usd_jpy": 152.0,
    "portfolio_dirty": True,
    "portfolio_summary": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# 💱 為替エンジン
# ==============================================================================

@st.cache_data(ttl=600)
def get_usd_jpy() -> float:
    try:
        df = yf.Ticker("JPY=X").history(period="1d")
        return round(float(df["Close"].iloc[-1]), 2) if not df.empty else 152.0
    except:
        return 152.0

# ==============================================================================
# 💾 データエンジン
# ==============================================================================

@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{ticker}.pkl"
    if cache_file.exists():
        if time.time() - cache_file.stat().st_mtime < CONFIG["CACHE_EXPIRY"]:
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except:
                pass
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
        return df
    except:
        return None

@st.cache_data(ttl=300)
def get_current_price(ticker: str) -> Optional[float]:
    try:
        df = yf.Ticker(ticker).history(period="2d", auto_adjust=True)
        return round(float(df["Close"].iloc[-1]), 4) if not df.empty else None
    except:
        return None

@st.cache_data(ttl=600)
def get_sector(ticker: str) -> str:
    sf = CACHE_DIR / "sectors.json"
    sm = {}
    if sf.exists():
        try:
            with open(sf) as f: sm = json.load(f)
        except: pass
    if ticker in sm:
        return sm[ticker]
    try:
        s = yf.Ticker(ticker).info.get("sector", "Unknown")
        sm[ticker] = s
        with open(sf, "w") as f: json.dump(sm, f)
        return s
    except:
        return "Unknown"

@st.cache_data(ttl=300)
def get_atr(ticker: str, period: int = 14) -> Optional[float]:
    try:
        df = yf.Ticker(ticker).history(period="60d", auto_adjust=True)
        if df is None or len(df) < period + 1: return None
        tr = pd.concat([
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"]  - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        v = float(tr.rolling(period).mean().iloc[-1])
        return round(v, 4) if not np.isnan(v) else None
    except:
        return None

@st.cache_data(ttl=600)
def load_historical_json() -> pd.DataFrame:
    all_data = []
    if RESULTS_DIR.exists():
        for file in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    daily = json.load(f)
                date = daily.get("date", file.stem)
                for k in ["selected", "watchlist_wait", "qualified_full"]:
                    for item in daily.get(k, []):
                        item["date"] = date
                        item["vcp_score"] = item.get("vcp", {}).get("score", 0)
                        all_data.append(item)
            except: pass
    return pd.DataFrame(all_data)

# ==============================================================================
# 📰 ニュース取得
# ==============================================================================

@st.cache_data(ttl=600)
def fetch_news(ticker: str) -> str:
    headlines = []
    try:
        for n in (yf.Ticker(ticker).news or [])[:5]:
            headlines.append(f"• {n.get('headline', n.get('title', ''))}")
    except: pass
    try:
        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={ticker}+stock+when:24h&hl=en-US&gl=US&ceid=US:en"
        )
        for e in feed.entries[:5]:
            headlines.append(f"• {e.title}")
    except: pass
    unique = list(dict.fromkeys(headlines))
    return "\n".join(unique[:8]) if unique else "本日、新規材料は未検出。"

# ==============================================================================
# 🧠 VCP分析
# ==============================================================================

def calc_vcp(df: pd.DataFrame) -> dict:
    try:
        close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if np.isnan(atr) or atr <= 0:
            return {"score": 0, "atr": 0, "signals": [], "is_dryup": False}

        h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min()
        range_pct = float((h10 - l10) / h10)
        tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.10))
        tight_score = max(0, min(40, tight_score))

        vol_ma    = volume.rolling(50).mean().iloc[-1]
        vol_ratio = float(volume.iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
        is_dryup  = bool(vol_ratio < 0.7)
        vol_score = 30 if is_dryup else (15 if vol_ratio < 1.1 else 0)

        ma50 = close.rolling(50).mean().iloc[-1]; ma200 = close.rolling(200).mean().iloc[-1]
        trend_score = (
            (10 if close.iloc[-1] > ma50  else 0) +
            (10 if ma50 > ma200            else 0) +
            (10 if close.iloc[-1] > ma200  else 0)
        )
        signals = []
        if range_pct < 0.06: signals.append("極度収縮")
        if is_dryup:         signals.append("Vol枯渇")
        if trend_score == 30: signals.append("MA整列")

        return {
            "score": int(max(0, tight_score + vol_score + trend_score)),
            "atr": atr, "signals": signals, "is_dryup": is_dryup,
        }
    except:
        return {"score": 0, "atr": 0, "signals": [], "is_dryup": False}

# ==============================================================================
# 🤖 Gemini呼び出し
# ==============================================================================

def call_gemini(prompt: str) -> str:
    """DeepSeek-Reasoner を呼び出す（関数名は互換性のため維持）。"""
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "⚠️ DEEPSEEK_API_KEY が未設定です。Streamlit secrets に追加してください。"
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
        )
        # reasoner は reasoning_content（思考過程）と content（最終回答）を返す
        # UIには最終回答のみ表示
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"DeepSeek Error: {e}"

# ==============================================================================
# 📋 Watchlist I/O
# ==============================================================================

def load_watchlist() -> list:
    if WATCHLIST_FILE.exists():
        try:
            with open(WATCHLIST_FILE) as f: return json.load(f)
        except: pass
    return []

def _write_watchlist(data: list):
    tmp = Path("watchlist.tmp")
    with open(tmp, "w") as f: json.dump(data, f)
    tmp.replace(WATCHLIST_FILE)

def add_watchlist(ticker: str) -> bool:
    wl = load_watchlist()
    if ticker not in wl:
        wl.append(ticker); _write_watchlist(wl); return True
    return False

def remove_watchlist(ticker: str) -> bool:
    wl = load_watchlist()
    if ticker in wl:
        wl.remove(ticker); _write_watchlist(wl); return True
    return False

# ==============================================================================
# 💼 Portfolio I/O
# ==============================================================================

def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, encoding="utf-8") as f: return json.load(f)
        except: pass
    return {"positions": {}, "closed": [], "meta": {"created": NOW.isoformat()}}

def _write_portfolio(data: dict):
    tmp = Path("portfolio.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(PORTFOLIO_FILE)

def upsert_position(ticker: str, shares: int, avg_cost: float,
                    memo: str = "", target: float = 0.0, stop: float = 0.0) -> dict:
    ticker = re.sub(r'[^A-Z0-9.\-]', '', ticker.upper())[:10]
    data = load_portfolio(); pos = data["positions"]
    if ticker in pos:
        old = pos[ticker]
        tot = old["shares"] + shares
        pos[ticker].update({
            "shares":     tot,
            "avg_cost":   round((old["shares"]*old["avg_cost"] + shares*avg_cost) / tot, 4),
            "memo":       memo or old.get("memo", ""),
            "target":     target or old.get("target", 0.0),
            "stop":       stop   or old.get("stop",   0.0),
            "updated_at": NOW.isoformat(),
        })
    else:
        pos[ticker] = {
            "ticker": ticker, "shares": shares, "avg_cost": round(avg_cost, 4),
            "memo": memo, "target": round(target, 4), "stop": round(stop, 4),
            "added_at": NOW.isoformat(), "updated_at": NOW.isoformat(),
        }
    _write_portfolio(data)
    return pos[ticker]

def close_position(ticker: str, shares_sold: Optional[int] = None,
                   sell_price: Optional[float] = None) -> bool:
    data = load_portfolio(); pos = data["positions"]
    if ticker not in pos: return False
    p = pos[ticker]
    actual_shares = shares_sold if shares_sold and shares_sold < p["shares"] else p["shares"]
    if sell_price:
        pnl = (sell_price - p["avg_cost"]) * actual_shares
        data["closed"].append({
            "ticker": ticker, "shares": actual_shares,
            "avg_cost": p["avg_cost"], "sell_price": sell_price,
            "pnl_usd": round(pnl, 2),
            "pnl_pct": round((sell_price / p["avg_cost"] - 1) * 100, 2),
            "closed_at": NOW.isoformat(), "memo": p.get("memo", ""),
        })
    if shares_sold and shares_sold < p["shares"]:
        pos[ticker]["shares"] -= shares_sold
    else:
        del pos[ticker]
    _write_portfolio(data)
    return True

# ==============================================================================
# 📊 ポートフォリオ損益計算
# ==============================================================================

def calc_pos_stats(pos: dict, usd_jpy: float) -> dict:
    cp  = get_current_price(pos["ticker"])
    atr = get_atr(pos["ticker"])
    if cp is None:
        return {**pos, "error": True, "current_price": None}

    shares    = pos["shares"]
    avg_cost  = pos["avg_cost"]
    pnl_usd   = (cp - avg_cost) * shares
    pnl_pct   = (cp / avg_cost - 1) * 100
    mv_usd    = cp * shares
    cb_usd    = avg_cost * shares

    ex = {}
    if atr:
        risk   = atr * EXIT_CFG["STOP_LOSS_ATR_MULT"]
        reward = risk * EXIT_CFG["TARGET_R_MULT"]
        dyn_stop = round(cp - risk, 4)
        reg_stop = pos.get("stop", 0.0)
        eff_stop = max(dyn_stop, reg_stop) if reg_stop > 0 else dyn_stop
        cur_r    = (cp - avg_cost) / risk if risk > 0 else 0.0
        reg_tgt  = pos.get("target", 0.0)
        eff_tgt  = reg_tgt if reg_tgt > 0 else round(avg_cost + reward, 4)
        trail    = round(cp - atr * EXIT_CFG["TRAIL_ATR_MULT"], 4) if cur_r >= EXIT_CFG["TRAIL_START_R"] else None
        scale    = round(avg_cost + risk * EXIT_CFG["SCALE_OUT_R"], 4)
        ex = {
            "atr": atr, "risk": round(risk, 4),
            "dyn_stop": dyn_stop, "eff_stop": eff_stop, "eff_tgt": eff_tgt,
            "scale_out": scale, "cur_r": round(cur_r, 2), "trail": trail,
        }

    # ステータス
    cur_r = ex.get("cur_r", 0)
    if pnl_pct <= -8:   status = "🚨"
    elif pnl_pct <= -4: status = "⚠️"
    elif cur_r >= EXIT_CFG["TARGET_R_MULT"]: status = "🎯"
    elif cur_r >= EXIT_CFG["TRAIL_START_R"]: status = "📈"
    elif cur_r >= EXIT_CFG["SCALE_OUT_R"]:  status = "💰"
    elif pnl_pct > 0:   status = "✅"
    else:               status = "🔵"

    return {
        **pos,
        "current_price": round(cp, 4),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_jpy": round(pnl_usd * usd_jpy, 0),
        "mv_usd":  round(mv_usd, 2),
        "cb_usd":  round(cb_usd, 2),
        "exit":    ex,
        "status":  status,
    }

def get_portfolio_summary(usd_jpy: float) -> dict:
    data  = load_portfolio()
    pos_d = data["positions"]
    if not pos_d:
        return {"positions": [], "total": {}, "closed": data.get("closed", [])}

    stats = [calc_pos_stats(p, usd_jpy) for p in pos_d.values()]
    valid = [s for s in stats if not s.get("error")]

    total_mv  = sum(s["mv_usd"]  for s in valid)
    total_cb  = sum(s["cb_usd"]  for s in valid)
    total_pnl = sum(s["pnl_usd"] for s in valid)
    cap_usd   = CONFIG["CAPITAL_JPY"] / usd_jpy

    for s in valid:
        s["pw"] = round(s["mv_usd"] / total_mv * 100, 1) if total_mv > 0 else 0.0

    closed  = data.get("closed", [])
    win_cnt = len([c for c in closed if c.get("pnl_usd", 0) > 0])

    return {
        "positions": stats,
        "total": {
            "count":     len(valid),
            "mv_usd":    round(total_mv, 2),
            "mv_jpy":    round(total_mv * usd_jpy, 0),
            "pnl_usd":   round(total_pnl, 2),
            "pnl_jpy":   round(total_pnl * usd_jpy, 0),
            "pnl_pct":   round(total_pnl / total_cb * 100 if total_cb else 0, 2),
            "exposure":  round(total_mv / cap_usd * 100 if cap_usd else 0, 1),
            "cash_jpy":  round((cap_usd - total_mv) * usd_jpy, 0),
        },
        "closed_stats": {
            "count":     len(closed),
            "pnl_usd":   round(sum(c.get("pnl_usd", 0) for c in closed), 2),
            "pnl_jpy":   round(sum(c.get("pnl_usd", 0) for c in closed) * usd_jpy, 0),
            "win_rate":  round(win_cnt / len(closed) * 100, 1) if closed else 0.0,
        },
        "closed": closed,
    }

# ==============================================================================
# 🖥️ サイドバー（Watchlist）
# ==============================================================================

with st.sidebar:
    st.markdown("### 🛡️ SENTINEL PRO")
    st.caption(TODAY_STR)

    # --- Watchlist ---
    st.markdown("#### ⭐ Watchlist")
    wl = load_watchlist()
    if not wl:
        st.caption("なし")
    else:
        for t in wl:
            c1, c2 = st.columns([4, 1])
            if c1.button(f"🔍 {t}", key=f"sb_{t}", use_container_width=True):
                st.session_state["target_ticker"]   = t
                st.session_state["mode"]            = "🔍 リアルタイム"
                st.session_state["trigger_analysis"] = True
                st.rerun()
            if c2.button("✕", key=f"rm_{t}"):
                remove_watchlist(t); st.rerun()

    st.divider()
    st.caption(f"💱 USD/JPY: {st.session_state['usd_jpy']:.1f}")

# ==============================================================================
# 🔝 ナビゲーションバー（スマホ向け上部タブ）
# =================================================================