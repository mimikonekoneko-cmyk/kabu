import json
import os
import re
import time
import warnings
import datetime
import textwrap
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from openai import OpenAI

# ==============================================================================
# ⚙️ 外部エンジン構成（ローカル単体動作用のスタブ含む）
# ==============================================================================
try:
    # ローカル環境でファイルが揃っている場合
    from config import CONFIG
    from engines.data import CurrencyEngine, DataEngine
    from engines.fundamental import FundamentalEngine, InsiderEngine
    from engines.news import NewsEngine
except ImportError:
    # ⚠️ ファイルが不足している場合の緊急用スタブクラス
    class CurrencyEngine:
        @staticmethod
        def get_usd_jpy(): return 152.65
    class DataEngine:
        @staticmethod
        def get_data(ticker, period): 
            # yfinanceで実際にデータを取得
            return yf.download(ticker, period=period, progress=False)
        @staticmethod
        def get_current_price(ticker):
            try: 
                t = yf.Ticker(ticker)
                return t.fast_info['lastPrice']
            except: return 0.0
        @staticmethod
        def get_atr(ticker): return 1.5
    class FundamentalEngine:
        @staticmethod
        def get(ticker): return {"info": "Data Unavailable (Stub)"}
    class InsiderEngine:
        @staticmethod
        def get(ticker): return {"trades": []}
    class NewsEngine:
        @staticmethod
        def get(ticker): return []

warnings.filterwarnings("ignore")

# ==============================================================================
# 💎 言語設定
# ==============================================================================

LANG = {
    "ja": {
        "title": "🛡️ SENTINEL PRO",
        "tab_scan": "📊 マーケットスキャン",
        "tab_diag": "🔍 AI診断",
        "tab_port": "💼 ポートフォリオ",
        "scan_date": "📅 スキャン日",
        "usd_jpy": "💱 USD/JPY",
        "action_list": "アクション銘柄",
        "wait_list": "ウォッチ銘柄",
        "sector_map": "🗺️ セクター別RSマップ",
        "realtime_scan": "🔍 リアルタイム定量スキャン",
        "ticker_input": "ティッカーシンボル（例：NVDA）",
        "run_quant": "🚀 定量スキャン実行",
        "add_watchlist": "⭐ ウォッチリストに追加",
        "quant_dashboard": "📊 SENTINEL定量ダッシュボード",
        "current_price": "💰 現在値",
        "vcp_score": "🎯 VCPスコア",
        "profit_factor": "📈 プロフィットファクター",
        "rs_momentum": "📏 RSモメンタム",
        "strategic_levels": "🛡️ ATR基準の戦略水準",
        "stop_loss": "ストップロス (2.0R)",
        "target1": "目標① (1.0R)",
        "target2": "目標② (2.5R)",
        "risk_unit": "リスク単価 ($)",
        "vcp_breakdown": "📐 VCPスコア内訳",
        "tightness": "収縮スコア",
        "volume": "出来高スコア",
        "ma_trend": "移動平均トレンド",
        "pivot_bonus": "ピボットボーナス",
        "ai_reasoning": "🤖 SENTINEL AI診断（テクニカル＆ファンダ）",
        "generate_ai": "🚀 AI診断を実行 (DeepSeek-R1)",
        "ai_key_missing": "DEEPSEEK_API_KEY が設定されていません。数値スキャンは完了しましたが、AI分析は実行できません。",
        "portfolio_risk": "💼 ポートフォリオリスク管理",
        "portfolio_empty": "ポートフォリオは空です。",
        "unrealized_jpy": "💰 含み損益 (円)",
        "assets": "📊 保有銘柄数",
        "exposure": "🛡️ エクスポージャー",
        "performance": "📈 平均パフォーマンス",
        "active_positions": "📋 保有中のポジション",
        "close_position": "決済",
        "register_new": "➕ 新規ポジション登録",
        "ticker_symbol": "ティッカーシンボル",
        "shares": "株数",
        "avg_cost": "平均取得単価",
        "add_to_portfolio": "ポートフォリオに追加",
    },
    "en": {
        "title": "🛡️ SENTINEL PRO",
        "tab_scan": "📊 MARKET SCAN",
        "tab_diag": "🔍 AI DIAGNOSIS",
        "tab_port": "💼 PORTFOLIO",
        "scan_date": "📅 Scan Date",
        "usd_jpy": "💱 USD/JPY",
        "action_list": "Action List",
        "wait_list": "Watch List",
        "sector_map": "🗺️ Sector RS Map",
        "realtime_scan": "🔍 REAL-TIME QUANTITATIVE SCAN",
        "ticker_input": "Ticker Symbol (e.g. NVDA)",
        "run_quant": "🚀 RUN QUANTITATIVE SCAN",
        "add_watchlist": "⭐ ADD TO WATCHLIST",
        "quant_dashboard": "📊 SENTINEL QUANTITATIVE DASHBOARD",
        "current_price": "💰 Current Price",
        "vcp_score": "🎯 VCP Score",
        "profit_factor": "📈 Profit Factor",
        "rs_momentum": "📏 RS Momentum",
        "strategic_levels": "🛡️ STRATEGIC LEVELS (ATR-Based)",
        "stop_loss": "Stop Loss (2.0R)",
        "target1": "Target 1 (1.0R)",
        "target2": "Target 2 (2.5R)",
        "risk_unit": "Risk Unit ($)",
        "vcp_breakdown": "📐 VCP SCORE BREAKDOWN",
        "tightness": "Tightness Score",
        "volume": "Volume Dry-up",
        "ma_trend": "MA Trend Score",
        "pivot_bonus": "Pivot Bonus",
        "ai_reasoning": "🤖 SENTINEL AI CONTEXTUAL REASONING",
        "generate_ai": "🚀 GENERATE AI DIAGNOSIS (NEWS & FUNDAMENTALS)",
        "ai_key_missing": "DEEPSEEK_API_KEY is not configured in Secrets. Numerical scan is complete, but AI analysis cannot be performed.",
        "portfolio_risk": "💼 PORTFOLIO RISK MANAGEMENT",
        "portfolio_empty": "Portfolio is currently empty.",
        "unrealized_jpy": "💰 Unrealized JPY",
        "assets": "📊 Assets",
        "exposure": "🛡️ Exposure",
        "performance": "📈 Performance",
        "active_positions": "📋 ACTIVE POSITIONS",
        "close_position": "Close",
        "register_new": "➕ REGISTER NEW POSITION",
        "ticker_symbol": "Ticker Symbol",
        "shares": "Shares",
        "avg_cost": "Avg Cost",
        "add_to_portfolio": "ADD TO PORTFOLIO",
    }
}

# ==============================================================================
# 💎 セッションステートの強制初期化
# ==============================================================================

def initialize_sentinel_state():
    if "target_ticker" not in st.session_state:
        st.session_state.target_ticker = ""
    if "trigger_analysis" not in st.session_state:
        st.session_state.trigger_analysis = False
    if "portfolio_dirty" not in st.session_state:
        st.session_state.portfolio_dirty = True
    if "quant_results_stored" not in st.session_state:
        st.session_state.quant_results_stored = None
    if "ai_analysis_text" not in st.session_state:
        st.session_state.ai_analysis_text = ""
    if "language" not in st.session_state:
        st.session_state.language = "ja"

initialize_sentinel_state()

# ==============================================================================
# 🔧 定数 & 出口戦略構成
# ==============================================================================

NOW         = datetime.datetime.now()
TODAY_STR   = NOW.strftime("%Y-%m-%d")
CACHE_DIR   = Path("./cache_v45"); CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results");   RESULTS_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE = Path("watchlist.json")
PORTFOLIO_FILE = Path("portfolio.json")

EXIT_CFG = {
    "STOP_LOSS_ATR_MULT": 2.0,
    "TARGET_R_MULT":      2.5,
    "TRAIL_START_R":      1.5,
    "TRAIL_ATR_MULT":     1.5,
    "SCALE_OUT_R":        1.5,
}

# ==============================================================================
# 🎨 UI スタイル定義
# ==============================================================================

GLOBAL_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] { 
font-family: 'Rajdhani', sans-serif; 
background-color: #0d1117; 
color: #f0f6fc;
}
.block-container { 
padding-top: 0rem !important; 
padding-bottom: 2rem !important; 
max-width: 95% !important;
}

.ui-push-buffer {
height: 65px;
width: 100%;
background: transparent;
}

.stTabs [data-baseweb="tab-list"] {
display: flex !important;
width: 100% !important;
flex-wrap: nowrap !important;
overflow-x: auto !important;
overflow-y: hidden !important;
background-color: #161b22 !important;
padding: 12px 12px 0 12px !important;
border-radius: 12px 12px 0 0 !important;
gap: 12px !important;
border-bottom: 2px solid #30363d !important;
scrollbar-width: none !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }

.stTabs [data-baseweb="tab"] {
min-width: 185px !important; 
flex-shrink: 0 !important;
font-size: 1.05rem !important;
font-weight: 700 !important;
color: #8b949e !important;
padding: 22px 32px !important;
background-color: transparent !important;
border: none !important;
white-space: nowrap !important;
text-align: center !important;
}

.stTabs [aria-selected="true"] {
color: #ffffff !important;
background-color: #238636 !important;
border-radius: 12px 12px 0 0 !important;
}

.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

.sentinel-grid {
display: grid;
grid-template-columns: repeat(2, 1fr);
gap: 16px;
margin: 20px 0 30px 0;
}
@media (min-width: 992px) {
.sentinel-grid { grid-template-columns: repeat(4, 1fr); }
}
.sentinel-card {
background: #161b22;
border: 1px solid #30363d;
border-radius: 12px;
padding: 24px;
box-shadow: 0 4px 25px rgba(0,0,0,0.7);
}
.sentinel-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.25em; margin-bottom: 12px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.sentinel-value { font-size: 1.45rem; font-weight: 700; color: #f0f6fc; line-height: 1.1; }
.sentinel-delta { font-size: 0.95rem; font-weight: 600; margin-top: 12px; }

.diagnostic-panel {
background: #0d1117;
border: 1px solid #30363d;
border-radius: 12px;
padding: 28px;
margin-bottom: 26px;
}
.diag-row {
display: flex;
justify-content: space-between;
padding: 16px 0;
border-bottom: 1px solid #21262d;
}
.diag-row:last-child { border-bottom: none; }
.diag-key { color: #8b949e; font-size: 1.0rem; font-weight: 600; }
.diag-val { color: #f0f6fc; font-weight: 700; font-family: 'Share Tech Mono', monospace; font-size: 1.15rem; }

.section-header { 
font-size: 1.2rem; font-weight: 700; color: #58a6ff; 
border-bottom: 1px solid #30363d; padding-bottom: 16px; 
margin: 45px 0 28px; text-transform: uppercase; letter-spacing: 4px;
display: flex; align-items: center; gap: 14px;
}

.pos-card { 
background: #0d1117; border: 1px solid #30363d; border-radius: 18px; 
padding: 30px; margin-bottom: 24px; border-left: 12px solid #30363d; 
}
.pos-card.urgent { border-left-color: #f85149; }
.pos-card.caution { border-left-color: #d29922; }
.pos-card.profit { border-left-color: #3fb950; }
.pnl-pos { color: #3fb950; font-weight: 700; font-size: 1.3rem; }
.pnl-neg { color: #f85149; font-weight: 700; font-size: 1.3rem; }
.exit-info { font-size: 0.95rem; color: #8b949e; font-family: 'Share Tech Mono', monospace; margin-top: 18px; border-top: 1px solid #21262d; padding-top: 18px; line-height: 1.8; }

.stButton > button { min-height: 60px; border-radius: 14px; font-weight: 700; font-size: 1.1rem; }
[data-testid="stMetric"] { display: none !important; }
</style>
"""

# ==============================================================================
# 🎯 VCPAnalyzer (統一版)
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

            if avg_range < 0.10:    tight_score = 40
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

            if v_ratio < 0.45:    vol_score = 30
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
            if price_v > ma50_v:    m_score += 10
            if ma50_v > ma150_v:    m_score += 10
            if ma150_v > ma200_v:   m_score += 10

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
# 📈 RSAnalyzer
# ==============================================================================

class RSAnalyzer:
    @staticmethod
    def get_raw_score(df: pd.DataFrame) -> float:
        try:
            c = df["Close"]
            if len(c) < 252:
                return -999.0
            
            r12m = (c.iloc[-1] / c.iloc[-252]) - 1
            r6m  = (c.iloc[-1] / c.iloc[-126]) - 1
            r3m  = (c.iloc[-1] / c.iloc[-63])  - 1
            r1m  = (c.iloc[-1] / c.iloc[-21])  - 1
            
            return (r12m * 0.4) + (r6m * 0.2) + (r3m * 0.2) + (r1m * 0.2)
        except Exception:
            return -999.0

# ==============================================================================
# 🔬 StrategyValidator
# ==============================================================================

class StrategyValidator:
    @staticmethod
    def run(df: pd.DataFrame) -> float:
        try:
            if len(df) < 252:
                return 1.0
            
            c_data = df["Close"]
            h_data = df["High"]
            l_data = df["Low"]
            
            tr_calc = pd.concat([
                h_data - l_data,
                (h_data - c_data.shift(1)).abs(),
                (l_data - c_data.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_s = tr_calc.rolling(14).mean()
            
            trade_results = []
            is_in_pos = False
            entry_p = 0.0
            stop_p  = 0.0
            
            t_mult = EXIT_CFG["TARGET_R_MULT"]
            s_mult = EXIT_CFG["STOP_LOSS_ATR_MULT"]
            
            idx_start = max(65, len(df) - 252)
            for i in range(idx_start, len(df)):
                if is_in_pos:
                    if float(l_data.iloc[i]) <= stop_p:
                        trade_results.append(-1.0)
                        is_in_pos = False
                    elif float(h_data.iloc[i]) >= entry_p + (entry_p - stop_p) * t_mult:
                        trade_results.append(t_mult)
                        is_in_pos = False
                    elif i == len(df) - 1:
                        risk_unit = entry_p - stop_p
                        if risk_unit > 0:
                            pnl_r = (float(c_data.iloc[i]) - entry_p) / risk_unit
                            trade_results.append(pnl_r)
                        is_in_pos = False
                else:
                    if i < 20: continue
                    local_high_20 = float(h_data.iloc[i-20:i].max())
                    ma50_c = float(c_data.rolling(50).mean().iloc[i])
                    
                    if float(c_data.iloc[i]) > local_high_20 and float(c_data.iloc[i]) > ma50_c:
                        is_in_pos = True
                        entry_p = float(c_data.iloc[i])
                        atr_now = float(atr_s.iloc[i])
                        stop_p = entry_p - (atr_now * s_mult)
            
            if not trade_results:
                return 1.0
            
            gp = sum(res for res in trade_results if res > 0)
            gl = abs(sum(res for res in trade_results if res < 0))
            
            if gl == 0:
                return round(min(10.0, gp if gp > 0 else 1.0), 2)
            
            return round(min(10.0, float(gp / gl)), 2)
            
        except Exception:
            return 1.0

# ==============================================================================
# 📋 UI ヘルパー
# ==============================================================================

def draw_sentinel_grid_ui(metrics: List[Dict[str, Any]]):
    html_out = '<div class="sentinel-grid">'
    for m in metrics:
        delta_s = ""
        if "delta" in m and m["delta"]:
            is_pos = "+" in str(m["delta"]) or (isinstance(m["delta"], (int, float)) and m["delta"] > 0)
            c_code = "#3fb950" if is_pos else "#f85149"
            delta_s = f'<div class="sentinel-delta" style="color:{c_code}">{m["delta"]}</div>'
        item = (
            '<div class="sentinel-card">'
            f'<div class="sentinel-label">{m["label"]}</div>'
            f'<div class="sentinel-value">{m["value"]}</div>'
            f'{delta_s}'
            '</div>'
        )
        html_out += item
    html_out += '</div>'
    st.markdown(html_out.strip(), unsafe_allow_html=True)

def load_portfolio_json() -> dict:
    if not PORTFOLIO_FILE.exists():
        return {"positions": {}, "closed": [], "meta": {"last_update": ""}}
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"positions": {}, "closed": []}

def save_portfolio_json(data: dict):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_watchlist_data() -> list:
    if not WATCHLIST_FILE.exists():
        return []
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_watchlist_data(data: list):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f)

# ==============================================================================
# 🧭 メイン UI
# ==============================================================================

st.set_page_config(
    page_title="SENTINEL PRO", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown('<div class="ui-push-buffer"></div>', unsafe_allow_html=True)
st.markdown(GLOBAL_STYLE, unsafe_allow_html=True)

# --- サイドバー（言語選択＋ウォッチリスト）---
with st.sidebar:
    st.markdown("### 🌐 Language")
    lang = st.selectbox("", ["日本語", "English"], index=0 if st.session_state.language == "ja" else 1)
    st.session_state.language = "ja" if lang == "日本語" else "en"
    txt = LANG[st.session_state.language]

    st.markdown(f"### {txt['title']} ウォッチリスト")
    wl_t = load_watchlist_data()
    for t_n in wl_t:
        col_n, col_d = st.columns([4, 1])
        if col_n.button(t_n, key=f"side_{t_n}", use_container_width=True):
            st.session_state.target_ticker = t_n
            st.session_state.trigger_analysis = True
            st.rerun()
        if col_d.button("×", key=f"rm_{t_n}"):
            wl_t.remove(t_n)
            save_watchlist_data(wl_t)
            st.rerun()
    st.divider()
    st.caption(f"🛡️ SENTINEL V4.5 | {NOW.strftime('%H:%M:%S')}")

fx_rate = CurrencyEngine.get_usd_jpy()

# メインタブ
tab_scan, tab_diag, tab_port = st.tabs([txt["tab_scan"], txt["tab_diag"], txt["tab_port"]])

# ------------------------------------------------------------------------------
# タブ1: マーケットスキャン
# ------------------------------------------------------------------------------
with tab_scan:
    st.markdown(f'<div class="section-header">{txt["tab_scan"]}</div>', unsafe_allow_html=True)
    if RESULTS_DIR.exists():
        f_list = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
        if f_list:
            try:
                with open(f_list[0], "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                s_df = pd.DataFrame(s_data.get("qualified_full", []))
                draw_sentinel_grid_ui([
                    {"label": txt["scan_date"], "value": s_data.get("date", TODAY_STR)},
                    {"label": txt["usd_jpy"], "value": f"¥{fx_rate:.2f}"},
                    {"label": txt["action_list"], "value": len(s_df[s_df["status"]=="ACTION"]) if not s_df.empty else 0},
                    {"label": txt["wait_list"], "value": len(s_df[s_df["status"]=="WAIT"]) if not s_df.empty else 0}
                ])
                if not s_df.empty:
                    st.markdown(f'<div class="section-header">{txt["sector_map"]}</div>', unsafe_allow_html=True)
                    s_df["vcp_score"] = s_df["vcp"].apply(lambda x: x.get("score", 0))
                    m_fig = px.treemap(
                        s_df,
                        path=["sector", "ticker"],
                        values="vcp_score",
                        color="rs",
                        color_continuous_scale="RdYlGn",
                        range_color=[70, 100]
                    )
                    m_fig.update_layout(template="plotly_dark", height=600, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(m_fig, use_container_width=True)
                    st.dataframe(
                        s_df[["ticker", "status", "vcp_score", "rs", "sector"]].sort_values("vcp_score", ascending=False),
                        use_container_width=True,
                        height=500
                    )
            except Exception as e:
                st.error(f"Error loading scan results: {e}")

# ------------------------------------------------------------------------------
# タブ2: AI診断（リアルタイム定量スキャン）
# ------------------------------------------------------------------------------
with tab_diag:
    st.markdown(f'<div class="section-header">{txt["realtime_scan"]}</div>', unsafe_allow_html=True)

    t_input = st.text_input(txt["ticker_input"], value=st.session_state.target_ticker).upper().strip()

    col_q, col_w = st.columns(2)
    start_quant = col_q.button(txt["run_quant"], type="primary", use_container_width=True)
    add_watchlist = col_w.button(txt["add_watchlist"], use_container_width=True)

    if add_watchlist and t_input:
        wl = load_watchlist_data()
        if t_input not in wl:
            wl.append(t_input)
            save_watchlist_data(wl)
            st.success(f"Added {t_input}")

    if (start_quant or st.session_state.pop("trigger_analysis", False)) and t_input:
        with st.spinner(f"SENTINEL ENGINE: Scanning {t_input}..."):
            df_raw = DataEngine.get_data(t_input, "2y")
            if df_raw is not None and not df_raw.empty:
                vcp_res = VCPAnalyzer.calculate(df_raw)
                rs_val = RSAnalyzer.get_raw_score(df_raw)
                pf_val = StrategyValidator.run(df_raw)
                p_curr = DataEngine.get_current_price(t_input) or df_raw["Close"].iloc[-1]

                st.session_state.quant_results_stored = {
                    "vcp": vcp_res, "rs": rs_val, "pf": pf_val, "price": p_curr, "ticker": t_input
                }
                st.session_state.ai_analysis_text = ""
            else:
                st.error(f"Failed to fetch data for {t_input}.")

if st.session_state.quant_results_stored and st.session_state.quant_results_stored["ticker"] == t_input:
    q = st.session_state.quant_results_stored
    vcp_res, rs_val, pf_val, p_curr = q["vcp"], q["rs"], q["pf"], q["price"]

    # ---- 数値の安全な変換 ----
    try:
        rs_val = float(rs_val) if rs_val is not None else 0.0
    except (TypeError, ValueError):
        rs_val = 0.0

    try:
        pf_val = float(pf_val) if pf_val is not None else 0.0
    except (TypeError, ValueError):
        pf_val = 0.0

    try:
        p_curr = float(p_curr) if p_curr is not None else 0.0
    except (TypeError, ValueError):
        p_curr = 0.0
    # ---- ここまで ----

    st.markdown(f'<div class="section-header">{txt["quant_dashboard"]}</div>', unsafe_allow_html=True)
    draw_sentinel_grid_ui([
        {"label": txt["current_price"], "value": f"${p_curr:.2f}"},
        {"label": txt["vcp_score"], "value": f"{vcp_res['score']}/105"},
        {"label": txt["profit_factor"], "value": f"x{pf_val:.2f}"},
        {"label": txt["rs_momentum"], "value": f"{rs_val*100:+.1f}%"}
    ])

    d1, d2 = st.columns(2)
    with d1:
        risk = vcp_res['atr'] * EXIT_CFG["STOP_LOSS_ATR_MULT"]
        panel_html1 = f'''
<div class="diagnostic-panel">
<b>{txt["strategic_levels"]}</b>
<div class="diag-row"><span class="diag-key">{txt["stop_loss"]}</span><span class="diag-val">${p_curr - risk:.2f}</span></div>
<div class="diag-row"><span class="diag-key">{txt["target1"]}</span><span class="diag-val">${p_curr + risk:.2f}</span></div>
<div class="diag-row"><span class="diag-key">{txt["target2"]}</span><span class="diag-val">${p_curr + risk*2.5:.2f}</span></div>
<div class="diag-row"><span class="diag-key">{txt["risk_unit"]}</span><span class="diag-val">${risk:.2f}</span></div>
</div>'''
        st.markdown(panel_html1.strip(), unsafe_allow_html=True)
    with d2:
        bd = vcp_res['breakdown']
        panel_html2 = f'''
<div class="diagnostic-panel">
<b>{txt["vcp_breakdown"]}</b>
<div class="diag-row"><span class="diag-key">{txt["tightness"]}</span><span class="diag-val">{bd.get("tight", 0)}/45</span></div>
<div class="diag-row"><span class="diag-key">{txt["volume"]}</span><span class="diag-val">{bd.get("vol", 0)}/30</span></div>
<div class="diag-row"><span class="diag-key">{txt["ma_trend"]}</span><span class="diag-val">{bd.get("ma", 0)}/30</span></div>
<div class="diag-row"><span class="diag-key">{txt["pivot_bonus"]}</span><span class="diag-val">+{bd.get("pivot", 0)}pt</span></div>
</div>'''
        st.markdown(panel_html2.strip(), unsafe_allow_html=True)

        # --------------------------------------------------------------
        # 📈 チャート描画エリア (画面全体に拡張)
        # --------------------------------------------------------------
        df_raw = DataEngine.get_data(t_input, "2y")
        if df_raw is not None and not df_raw.empty:
            df_t = df_raw.tail(150) # 表示期間を少し長めに
            c_fig = go.Figure(data=[go.Candlestick(
                x=df_t.index, 
                open=df_t['Open'], 
                high=df_t['High'], 
                low=df_t['Low'], 
                close=df_t['Close'],
                name=t_input
            )])
            
            # MAの追加
            ma50 = df_raw['Close'].rolling(50).mean().tail(150)
            ma200 = df_raw['Close'].rolling(200).mean().tail(150)
            c_fig.add_trace(go.Scatter(x=df_t.index, y=ma50, line=dict(color='orange', width=1), name='MA50'))
            c_fig.add_trace(go.Scatter(x=df_t.index, y=ma200, line=dict(color='blue', width=1), name='MA200'))

            c_fig.update_layout(
                template="plotly_dark", 
                height=750,  # ★高さを大幅アップ
                title=f"{t_input} DAILY CHART",
                margin=dict(t=40, b=0, l=0, r=0), 
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            # カラム分けせず、フル幅で表示
            st.plotly_chart(c_fig, use_container_width=True)

        # --------------------------------------------------------------
        # 🤖 AI診断セクション (チャート認識機能付き)
        # --------------------------------------------------------------
        st.markdown(f'<div class="section-header">{txt["ai_reasoning"]}</div>', unsafe_allow_html=True)
        if st.button(txt["generate_ai"], use_container_width=True):
            key = st.secrets.get("DEEPSEEK_API_KEY")
            if not key:
                st.error(txt["ai_key_missing"])
            else:
                with st.spinner(f"AI Reasoning for {t_input}..."):
                    # ニュースとファンダメンタル
                    news = NewsEngine.get(t_input); fund = FundamentalEngine.get(t_input)

                    # --- ★チャート情報の言語化処理 ---
                    if df_raw is not None and not df_raw.empty:
                        hist_close = df_raw['Close']
                        ma50_val = hist_close.rolling(50).mean().iloc[-1]
                        ma200_val = hist_close.rolling(200).mean().iloc[-1]
                        
                        # トレンド判定
                        if p_curr > ma50_val and ma50_val > ma200_val:
                            trend_desc = "強力な上昇トレンド (パーフェクトオーダー)"
                        elif p_curr < ma50_val and ma50_val < ma200_val:
                            trend_desc = "下降トレンド"
                        elif p_curr > ma50_val:
                            trend_desc = "回復局面 (MA50ブレイク済み)"
                        else:
                            trend_desc = "調整局面 / 方向感なし"
                        
                        # 直近5日の値動き
                        recent_moves = hist_close.tail(5).tolist()
                        recent_str = " -> ".join([f"${p:.2f}" for p in recent_moves])
                    else:
                        trend_desc = "データ不足"
                        ma50_val = 0
                        ma200_val = 0
                        recent_str = "不明"
                    # ------------------------------------

                    prompt = (
                        f"あなたは伝説的投資家 Mark Minervini の理論を極めた AI ファンドマネージャー「SENTINEL」です。\n"
                        f"銘柄 {t_input} の診断結果に基づき、プロの投資判断を下してください。\n\n"
                        f"━━━ 定量的データ (SENTINEL ENGINE) ━━━\n"
                        f"現在値: ${p_curr:.2f} | VCPスコア: {vcp_res['score']}/105 | PF: {pf_val:.2f} | RS: {rs_val*100:+.2f}%\n"
                        f"━━━ チャート形状情報 (テクニカル視覚代替) ━━━\n"
                        f"トレンド判定: {trend_desc}\n"
                        f"主要MA: 50日線=${ma50_val:.2f}, 200日線=${ma200_val:.2f}\n"
                        f"直近5日の推移: {recent_str}\n\n"
                        f"━━━ 外部情報 ━━━\n"
                        f"ファンダメンタル要約: {str(fund)[:1000]}\n"
                        f"ニュース: {str(news)[:1500]}\n\n"
                        f"指示: PF数値、RS値、そして上記の『チャート形状』を統合し、テクニカル分析の観点から投資妙味を論評せよ。1,500文字以上で記述せよ。"
                    )
                    
                    cl = OpenAI(api_key=key, base_url="https://api.deepseek.com")
                    try:
                        res_ai = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
                        st.session_state.ai_analysis_text = res_ai.choices[0].message.content.replace("$", r"\$")
                    except Exception as ai_e:
                        st.error(f"AI Error: {ai_e}")

        if st.session_state.ai_analysis_text:
            st.markdown("---")
            st.markdown(st.session_state.ai_analysis_text)

# ------------------------------------------------------------------------------
# タブ3: ポートフォリオ
# ------------------------------------------------------------------------------
with tab_port:
    st.markdown(f'<div class="section-header">{txt["portfolio_risk"]}</div>', unsafe_allow_html=True)
    p_j = load_portfolio_json()
    pos_m = p_j.get("positions", {})
    if not pos_m:
        st.info(txt["portfolio_empty"])
    else:
        stats_list = []
        for s_k, s_d in pos_m.items():
            l_p = DataEngine.get_current_price(s_k)
            if l_p:
                pnl_u = (l_p - s_d["avg_cost"]) * s_d["shares"]
                pnl_p = (l_p / s_d["avg_cost"] - 1) * 100
                atr_l = DataEngine.get_atr(s_k) or 0.0
                risk_l = atr_l * EXIT_CFG["STOP_LOSS_ATR_MULT"]
                stop_l = max(l_p - risk_l, s_d.get("stop", 0)) if risk_l else s_d.get("stop", 0)
                stats_list.append({
                    "ticker": s_k, "shares": s_d["shares"], "avg": s_d["avg_cost"],
                    "cp": l_p, "pnl_usd": pnl_u, "pnl_pct": pnl_p,
                    "cl": "profit" if pnl_p > 0 else "urgent", "stop": stop_l
                })
        total_pnl_j = sum(s["pnl_usd"] for s in stats_list) * fx_rate
        draw_sentinel_grid_ui([
            {"label": txt["unrealized_jpy"], "value": f"¥{total_pnl_j:,.0f}"},
            {"label": txt["assets"], "value": len(stats_list)},
            {"label": txt["exposure"], "value": f"${sum(s['shares']*s['avg'] for s in stats_list):,.0f}"},
            {"label": txt["performance"], "value": f"{np.mean([s['pnl_pct'] for s in stats_list]):.2f}%" if stats_list else "0%"}
        ])

        st.markdown(f'<div class="section-header">{txt["active_positions"]}</div>', unsafe_allow_html=True)
        for s in stats_list:
            pnl_c = "pnl-pos" if s["pnl_pct"] > 0 else "pnl-neg"
            st.markdown(f'''
<div class="pos-card {s['cl']}">
<div style="display: flex; justify-content: space-between; align-items: center;">
<b>{s['ticker']}</b>
<span class="{pnl_c}">{s['pnl_pct']:+.2f}% (¥{s['pnl_usd']*fx_rate:+,.0f})</span>
</div>
<div style="font-size: 0.95rem; color: #f0f6fc; margin-top: 10px;">
{s['shares']} shares @ ${s['avg']:.2f} (Live: ${s['cp']:.2f})
</div>
<div class="exit-info">🛡️ DYNAMIC STOP: ${s['stop']:.2f}</div>
</div>''', unsafe_allow_html=True)

            if st.button(f"{txt['close_position']} {s['ticker']}", key=f"cl_{s['ticker']}"):
                del pos_m[s['ticker']]
                save_portfolio_json(p_j)
                st.rerun()

    # 新規ポジション登録
    st.markdown(f'<div class="section-header">{txt["register_new"]}</div>', unsafe_allow_html=True)
    with st.form("add_port"):
        c1, c2, c3 = st.columns(3)
        f_ticker = c1.text_input(txt["ticker_symbol"]).upper().strip()
        f_shares = c2.number_input(txt["shares"], min_value=1, value=10)
        f_cost   = c3.number_input(txt["avg_cost"], min_value=0.01, value=100.0)
        if st.form_submit_button(txt["add_to_portfolio"], use_container_width=True):
            if f_ticker:
                p = load_portfolio_json()
                p["positions"][f_ticker] = {"ticker": f_ticker, "shares": f_shares, "avg_cost": f_cost, "added_at": TODAY_STR}
                save_portfolio_json(p)
                st.success(f"Added {f_ticker}")
                st.rerun()

st.divider()
st.caption(f"🛡️ SENTINEL PRO SYSTEM | CORE ENGINE: UNIFIED | UI: MULTILINGUAL")

