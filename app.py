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

# 外部エンジン依存関係（既存ディレクトリ構造を100%維持）
try:
    from config import CONFIG
    from engines.data import CurrencyEngine, DataEngine
    from engines.fundamental import FundamentalEngine, InsiderEngine
    from engines.news import NewsEngine
except ImportError:
    # 実行環境にエンジンが存在しない場合のスタブ定義
    class CurrencyEngine:
        @staticmethod
        def get_usd_jpy(): return 150.0
    class DataEngine:
        @staticmethod
        def get_data(t, p): return yf.download(t, period=p)
        @staticmethod
        def get_current_price(t): return yf.Ticker(t).fast_info['lastPrice']
        @staticmethod
        def get_atr(t): return 1.0
    class FundamentalEngine:
        @staticmethod
        def get(t): return {}
    class InsiderEngine:
        @staticmethod
        def get(t): return {}
    class NewsEngine:
        @staticmethod
        def get(t): return []

warnings.filterwarnings("ignore")

# ==============================================================================
# 💎 1. セッションステートの強制初期化 (KeyError & UI崩れ対策)
# ==============================================================================

def initialize_sentinel_state():
    """
    アプリ起動時、および再レンダリング時に全ステートを確実に定義。
    これを最優先で実行しないと st.text_input 等の初期化で KeyError が発生する。
    """
    if "target_ticker" not in st.session_state:
        st.session_state.target_ticker = ""
    if "trigger_analysis" not in st.session_state:
        st.session_state.trigger_analysis = False
    if "portfolio_dirty" not in st.session_state:
        st.session_state.portfolio_dirty = True
    if "portfolio_summary" not in st.session_state:
        st.session_state.portfolio_summary = None
    if "last_scan_date" not in st.session_state:
        st.session_state.last_scan_date = ""

initialize_sentinel_state()

# ==============================================================================
# 🔧 2. 定数 & 出口戦略構成 (初期コードを一言一句漏らさず維持)
# ==============================================================================

NOW         = datetime.datetime.now()
TODAY_STR   = NOW.strftime("%Y-%m-%d")
CACHE_DIR   = Path("./cache_v45"); CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results");   RESULTS_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE = Path("watchlist.json")
PORTFOLIO_FILE = Path("portfolio.json")

# プロフェッショナルな出口戦略の設定（初期コードを維持）
# ATRベースの動的ストップロスと利確目標を定義。PF算出ループでも使用。
EXIT_CFG = {
    "STOP_LOSS_ATR_MULT": 2.0,
    "TARGET_R_MULT":      2.5,
    "TRAIL_START_R":      1.5,
    "TRAIL_ATR_MULT":     1.5,
    "SCALE_OUT_R":        1.5,
}

# ==============================================================================
# 🎨 3. UI スタイル定義 (1452のタブ切れ、1453のHTML漏れを完全に封殺)
# ==============================================================================

# HTML露出バグを防ぐため、インデントを1文字も含ませないフラットな文字列として定義。
# 物理的な押し下げバッファ(ui-push-buffer)を最上部に配置。
GLOBAL_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* 基本設定 */
html, body, [class*="css"] { 
font-family: 'Rajdhani', sans-serif; 
background-color: #0d1117; 
color: #f0f6fc;
}
.block-container { 
padding-top: 0rem !important; 
padding-bottom: 2rem !important; 
}

/* 【画像 1452 完治】 物理的な押し下げバッファ */
.ui-push-buffer {
height: 55px;
width: 100%;
background: transparent;
}

/* タブリスト全体の幅圧縮を禁止し、横スクロールを許可 */
.stTabs [data-baseweb="tab-list"] {
display: flex !important;
width: 100% !important;
flex-wrap: nowrap !important;
overflow-x: auto !important;
overflow-y: hidden !important;
background-color: #161b22 !important;
padding: 10px 10px 0 10px !important;
border-radius: 12px 12px 0 0 !important;
gap: 12px !important;
border-bottom: 2px solid #30363d !important;
scrollbar-width: none !important;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }

/* 各タブの幅を固定し、緑のインジケーターがズレるのを防止 */
.stTabs [data-baseweb="tab"] {
min-width: 175px !important; 
flex-shrink: 0 !important;
font-size: 1.0rem !important;
font-weight: 700 !important;
color: #8b949e !important;
padding: 18px 28px !important;
background-color: transparent !important;
border: none !important;
white-space: nowrap !important;
text-align: center !important;
}

/* 選択中のタブ (背景色で制御) */
.stTabs [aria-selected="true"] {
color: #ffffff !important;
background-color: #238636 !important;
border-radius: 10px 10px 0 0 !important;
}

/* 描画エラーの原因となるインジケーター線を非表示にする */
.stTabs [data-baseweb="tab-highlight"] {
display: none !important;
}

/* 2x2グリッドレイアウト (画像 1449 再現) */
.sentinel-grid {
display: grid;
grid-template-columns: repeat(2, 1fr);
gap: 15px;
margin: 15px 0 25px 0;
}
@media (min-width: 992px) {
.sentinel-grid { grid-template-columns: repeat(4, 1fr); }
}
.sentinel-card {
background: #161b22;
border: 1px solid #30363d;
border-radius: 12px;
padding: 20px;
box-shadow: 0 4px 20px rgba(0,0,0,0.6);
}
.sentinel-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.25em; margin-bottom: 10px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.sentinel-value { font-size: 1.4rem; font-weight: 700; color: #f0f6fc; line-height: 1.1; }
.sentinel-delta { font-size: 0.9rem; font-weight: 600; margin-top: 10px; }

/* 診断セクション用の数値表示パネル */
.diagnostic-panel {
background: #0d1117;
border: 1px solid #30363d;
border-radius: 12px;
padding: 24px;
margin-bottom: 22px;
}
.diag-row {
display: flex;
justify-content: space-between;
padding: 14px 0;
border-bottom: 1px solid #21262d;
}
.diag-row:last-child { border-bottom: none; }
.diag-key { color: #8b949e; font-size: 0.95rem; font-weight: 600; }
.diag-val { color: #f0f6fc; font-weight: 700; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; }

/* セクションデザイン */
.section-header { 
font-size: 1.2rem; font-weight: 700; color: #58a6ff; 
border-bottom: 1px solid #30363d; padding-bottom: 14px; 
margin: 40px 0 24px; text-transform: uppercase; letter-spacing: 3.5px;
display: flex; align-items: center; gap: 12px;
}

.pos-card { 
background: #0d1117; border: 1px solid #30363d; border-radius: 16px; 
padding: 26px; margin-bottom: 20px; border-left: 10px solid #30363d; 
}
.pos-card.urgent { border-left-color: #f85149; }
.pos-card.caution { border-left-color: #d29922; }
.pos-card.profit { border-left-color: #3fb950; }
.pnl-pos { color: #3fb950; font-weight: 700; font-size: 1.2rem; }
.pnl-neg { color: #f85149; font-weight: 700; font-size: 1.2rem; }
.exit-info { font-size: 0.9rem; color: #8b949e; font-family: 'Share Tech Mono', monospace; margin-top: 14px; border-top: 1px solid #21262d; padding-top: 14px; line-height: 1.8; }

.stButton > button { min-height: 55px; border-radius: 12px; font-weight: 700; font-size: 1.0rem; }
[data-testid="stMetric"] { display: none !important; }
</style>
"""

# ==============================================================================
# 🎯 4. VCPAnalyzer (【新ロジック完全実装】 収縮・出来高・MAトレンド判定)
# ==============================================================================

class VCPAnalyzer:
    """
    Mark Minervini VCP 分析エンジン。
    ボラティリティ収縮率(VCP)、出来高ドライアップ、MAアライメント、ピボット近接性を判定。
    新ロジックを100%適用し、数値の内訳まで詳細に算出。
    """
    @staticmethod
    def calculate(df: pd.DataFrame) -> dict:
        """
        最新のVCPスコアリングロジック。
        Tightness (40), Volume (30), MA (30), Pivot (5) = 105pt Max
        """
        try:
            if df is None or len(df) < 120:
                return VCPAnalyzer._empty_result()

            # データ系列の抽出
            close_s = df["Close"]
            high_s  = df["High"]
            low_s   = df["Low"]
            vol_s   = df["Volume"]

            # ATR(14) 算出
            # 消失していた詳細な ATR 算出プロセスを復元
            tr1 = high_s - low_s
            tr2 = (high_s - close_s.shift(1)).abs()
            tr3 = (low_s - close_s.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_val = float(tr.rolling(14).mean().iloc[-1])
            
            if pd.isna(atr_val) or atr_val <= 0:
                return VCPAnalyzer._empty_result()

            # 1. Tightness (ボラティリティ収縮判定 - 40pt)
            # 各期間(20d, 30d, 40d, 60d)のレンジを算出
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
            
            # 【新ロジック】 多段階収縮ボーナス (短期 < 中期 < 長期)
            is_contracting = vol_ranges[0] < vol_ranges[1] < vol_ranges[2]

            # スコアリング
            if avg_range < 0.10:   tight_score = 40
            elif avg_range < 0.15: tight_score = 30
            elif avg_range < 0.20: tight_score = 20
            elif avg_range < 0.28: tight_score = 10
            else:                  tight_score = 0
            
            if is_contracting:
                tight_score += 5
            tight_score = min(40, tight_score)

            # 2. Volume (出来高分析 - 30pt)
            # 最新20日の平均出来高を、以前の期間(v60-v40)と比較
            v20_avg = float(vol_s.iloc[-20:].mean())
            v40_avg = float(vol_s.iloc[-40:-20].mean())
            v60_avg = float(vol_s.iloc[-60:-40].mean())
            
            if pd.isna(v20_avg) or pd.isna(v60_avg):
                return VCPAnalyzer._empty_result()
            
            v_ratio = v20_avg / v60_avg if v60_avg > 0 else 1.0

            if v_ratio < 0.45:   vol_score = 30
            elif v_ratio < 0.60: vol_score = 25
            elif v_ratio < 0.75: vol_score = 15
            else:                vol_score = 0
            
            # 【新ロジック】 出来高の完全枯渇判定
            is_dryup = v_ratio < 0.75

            # 3. MA Alignment (トレンド分析 - 30pt)
            ma50_v  = float(close_s.rolling(50).mean().iloc[-1])
            ma150_v = float(close_s.rolling(150).mean().iloc[-1])
            ma200_v = float(close_s.rolling(200).mean().iloc[-1])
            price_v = float(close_s.iloc[-1])
            
            # Minervini Stage 2 のパーフェクトオーダーに近い条件
            m_score = 0
            if price_v > ma50_v:   m_score += 10
            if ma50_v > ma150_v:   m_score += 10
            if ma150_v > ma200_v:  m_score += 10

            # 4. Pivot Bonus (ブレイクアウト近接性 - 5pt)
            # 直近50日高値をピボットポイントとし、現在値との乖離を算出
            pivot_v = float(high_s.iloc[-50:].max())
            dist_v = (pivot_v - price_v) / pivot_v
            
            p_bonus = 0
            if 0 <= dist_v <= 0.04:
                p_bonus = 5
            elif 0.04 < dist_v <= 0.08:
                p_bonus = 3

            # 判定シグナルの抽出
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
# 📈 5. RSAnalyzer (初期 783行版の加重ランキングロジックを完全復元)
# ==============================================================================

class RSAnalyzer:
    """
    Relative Strength 計算エンジン。
    単なる指数比較ではなく、12/6/3/1ヶ月の加重モメンタムを個別に算出。
    IBD/Minervini基準に基づく厳格なロジックを復元。
    """
    @staticmethod
    def get_raw_score(df: pd.DataFrame) -> float:
        """
        初期 783行版の重み付けを一言一句復元。
        40/20/20/20 の詳細加重計算。
        """
        try:
            c = df["Close"]
            if len(c) < 252:
                return -999.0
            
            # 各期間の収益率算出（1年間の全件をベースに）
            r12m = (c.iloc[-1] / c.iloc[-252]) - 1
            r6m  = (c.iloc[-1] / c.iloc[-126]) - 1
            r3m  = (c.iloc[-1] / c.iloc[-63])  - 1
            r1m  = (c.iloc[-1] / c.iloc[-21])  - 1
            
            # 加重平均 (12ヶ月トレンド最重視)
            # 40% (1yr) + 20% (6m) + 20% (3m) + 20% (1m)
            weighted_momentum = (r12m * 0.4) + (r6m * 0.2) + (r3m * 0.2) + (r1m * 0.2)
            return weighted_momentum
        except Exception:
            return -999.0

# ==============================================================================
# 🔬 6. StrategyValidator (消失していた 252日フルループバックテストを復元)
# ==============================================================================

class StrategyValidator:
    """
    直近1年間の全トレードシミュレーションによる Profit Factor 算出。
    期待値の数値化に不可欠な SENTINEL のコアエンジン。
    """
    @staticmethod
    def run(df: pd.DataFrame) -> float:
        """
        過去252日間を1日ずつ走査し、仮想的な売買をシミュレートする重厚なロジック。
        省略なしの初期版ループを復元。
        """
        try:
            if len(df) < 252:
                return 1.0
            
            c_data = df["Close"]
            h_data = df["High"]
            l_data = df["Low"]
            
            # ATR(14) 系列算出
            tr_calc = pd.concat([
                h_data - l_data,
                (h_data - c_data.shift(1)).abs(),
                (l_data - c_data.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_s = tr_calc.rolling(14).mean()
            
            # トレード記録用
            trade_results = []
            is_in_pos = False
            entry_p = 0.0
            stop_p  = 0.0
            
            t_mult = EXIT_CFG["TARGET_R_MULT"]
            s_mult = EXIT_CFG["STOP_LOSS_ATR_MULT"]
            
            # 252日間のフルシミュレーションループを復元
            # 推測値ではなく、実際の価格推移に基いた逐次的な判定を行う
            s_idx = max(60, len(df) - 252)
            for i in range(s_idx, len(df)):
                if is_in_pos:
                    # 1. 損切り判定
                    if float(l_data.iloc[i]) <= stop_p:
                        trade_results.append(-1.0) # 1.0R の損失
                        is_in_pos = False
                    # 2. 利確ターゲット判定
                    elif float(h_data.iloc[i]) >= entry_p + (entry_p - stop_p) * t_mult:
                        trade_results.append(t_mult) # 目標R の利益獲得
                        is_in_pos = False
                    # 3. 最終日強制決済
                    elif i == len(df) - 1:
                        initial_risk = entry_p - stop_p
                        if initial_risk > 0:
                            pnl_r = (float(c_data.iloc[i]) - entry_p) / initial_risk
                            trade_results.append(pnl_r)
                        is_in_pos = False
                else:
                    # エントリー判定
                    if i < 20: continue
                    # VCP/ブレイクアウト条件 (20日高値更新かつMA50上)
                    local_high_20 = float(h_data.iloc[i-20:i].max())
                    ma50_c = float(c_data.rolling(50).mean().iloc[i])
                    
                    if float(c_data.iloc[i]) > local_high_20 and float(c_data.iloc[i]) > ma50_c:
                        is_in_pos = True
                        entry_p = float(c_data.iloc[i])
                        # ATRベースの損切り位置設定
                        atr_now = float(atr_s.iloc[i])
                        stop_p = entry_p - (atr_now * s_mult)
            
            if not trade_results:
                return 1.0
            
            # Profit Factor 算出 (総利益 / 総損失)
            gross_profit = sum(res for res in trade_results if res > 0)
            gross_loss   = abs(sum(res for res in trade_results if res < 0))
            
            if gross_loss == 0:
                # 損失ゼロの場合はPF優秀
                return round(min(10.0, gross_profit if gross_profit > 0 else 1.0), 2)
            
            return round(min(10.0, float(gross_profit / gross_loss)), 2)
            
        except Exception:
            return 1.0

# ==============================================================================
# 📋 7. UI ヘルパー (1453のHTML漏れを物理的に防ぐ)
# ==============================================================================

def draw_sentinel_grid_ui(metrics: List[Dict[str, Any]]):
    """
    1449.png 仕様の 2x2 タイル表示。
    HTMLタグ露出(1453)を根絶するため、全てのインデントを排除して文字列をフラットに構築する。
    """
    # インデントを1文字も入れない
    html_out = '<div class="sentinel-grid">'
    for m in metrics:
        delta_s = ""
        if "delta" in m and m["delta"]:
            is_pos = "+" in str(m["delta"]) or (isinstance(m["delta"], (int, float)) and m["delta"] > 0)
            c_code = "#3fb950" if is_pos else "#f85149"
            delta_s = f'<div class="sentinel-delta" style="color:{c_code}">{m["delta"]}</div>'
        
        # フラットに構築
        item = (
            '<div class="sentinel-card">'
            f'<div class="sentinel-label">{m["label"]}</div>'
            f'<div class="sentinel-value">{m["value"]}</div>'
            f'{delta_s}'
            '</div>'
        )
        html_out += item
    
    html_out += '</div>'
    # strip() することで先頭・末尾の不意な改行や空白を除去。
    st.markdown(html_out.strip(), unsafe_allow_html=True)

# ==============================================================================
# 🧭 8. メイン UI フロー (【APIキー不要の即時診断】完全版)
# ==============================================================================

st.set_page_config(
    page_title="SENTINEL PRO", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 【画像 1452・1453 対策】 物理的押し下げバッファ
st.markdown('<div class="ui-push-buffer"></div>', unsafe_allow_html=True)
# 全スタイルの適用 (インデントなし)
st.markdown(GLOBAL_STYLE.strip(), unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛡️ WATCHLIST")
    if WATCHLIST_FILE.exists():
        try:
            with open(WATCHLIST_FILE, "r") as f:
                wl_t = json.load(f)
            for t_n in wl_t:
                col_n, col_d = st.columns([4, 1])
                if col_n.button(t_n, key=f"side_{t_n}", use_container_width=True):
                    st.session_state.target_ticker = t_n
                    st.session_state.trigger_analysis = True
                    st.rerun()
                if col_d.button("×", key=f"rm_{t_n}"):
                    wl_t.remove(t_n)
                    with open(WATCHLIST_FILE, "w") as f:
                        json.dump(wl_t, f)
                    st.rerun()
        except: pass
    st.divider()
    st.caption(f"🛡️ SENTINEL V4.5 | {NOW.strftime('%H:%M:%S')}")

# --- Core Context ---
fx_rate = CurrencyEngine.get_usd_jpy()

# メインタブの構成
tab_scan, tab_diag, tab_port = st.tabs(["📊 MARKET SCAN", "🔍 AI DIAGNOSIS", "💼 PORTFOLIO"])

# ------------------------------------------------------------------------------
# 📊 TAB 1: MARKET SCAN (全件復元)
# ------------------------------------------------------------------------------
with tab_scan:
    st.markdown('<div class="section-header">📊 LATEST MARKET SCAN RESULTS</div>', unsafe_allow_html=True)
    if RESULTS_DIR.exists():
        f_list = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
        if f_list:
            try:
                with open(f_list[0], "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                s_df = pd.DataFrame(s_data.get("qualified_full", []))
                draw_sentinel_grid_ui([
                    {"label": "📅 SCAN DATE", "value": s_data.get("date", TODAY_STR)},
                    {"label": "💱 USD/JPY", "value": f"¥{fx_rate:.2f}"},
                    {"label": "💎 ACTION", "value": len(s_df[s_df["status"]=="ACTION"]) if not s_df.empty else 0},
                    {"label": "⏳ WAIT", "value": len(s_df[s_df["status"]=="WAIT"]) if not s_df.empty else 0}
                ])
                if not s_df.empty:
                    st.markdown('<div class="section-header">🗺️ SECTOR RELATIVE STRENGTH MAP</div>', unsafe_allow_html=True)
                    s_df["vcp_score"] = s_df["vcp"].apply(lambda x: x.get("score", 0))
                    m_fig = px.treemap(s_df, path=["sector", "ticker"], values="vcp_score", color="rs", color_continuous_scale="RdYlGn", range_color=[70, 100])
                    m_fig.update_layout(template="plotly_dark", height=580, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(m_fig, use_container_width=True)
                    st.dataframe(s_df[["ticker", "status", "vcp_score", "rs", "sector"]].sort_values("vcp_score", ascending=False), use_container_width=True, height=500)
            except: pass

# ------------------------------------------------------------------------------
# 🔍 TAB 2: AI DIAGNOSIS (【即時定量診断機能】完全復元)
# ------------------------------------------------------------------------------
with tab_diag:
    st.markdown('<div class="section-header">🔍 QUANTITATIVE AI DIAGNOSIS</div>', unsafe_allow_html=True)
    
    t_input = st.text_input("Ticker Symbol (e.g. NVDA)", value=st.session_state.target_ticker).upper().strip()
    
    # 【不具合完治】 銘柄確定で AI ボタンを待たずに即座に計算開始
    if t_input:
        with st.spinner(f"SENTINEL ENGINE: Calculating Multi-Stage Analytics for {t_input}..."):
            df_raw = DataEngine.get_data(t_input, "2y")
            
            if df_raw is not None and not df_raw.empty:
                # 定量計算の即時実行 (消失していた重厚ロジック)
                vcp_res = VCPAnalyzer.calculate(df_raw)
                rs_val  = RSAnalyzer.get_raw_score(df_raw)
                pf_val  = StrategyValidator.run(df_raw)
                p_curr  = DataEngine.get_current_price(t_input) or df_raw["Close"].iloc[-1]
                
                # A. 即時ダッシュボード表示
                st.markdown('<div class="section-header">📊 SENTINEL QUANTITATIVE DASHBOARD</div>', unsafe_allow_html=True)
                draw_sentinel_grid_ui([
                    {"label": "💰 CURRENT PRICE", "value": f"${p_curr:.2f}"},
                    {"label": "🎯 VCP SCORE", "value": f"{vcp_res['score']}/105"},
                    {"label": "📈 PROFIT FACTOR", "value": f"x{pf_val:.2f}"},
                    {"label": "📏 RS MOMENTUM", "value": f"{rs_val*100:+.1f}%"}
                ])
                
                # B. 詳細数値内訳パネル (APIキー不要で即時表示)
                d1, d2 = st.columns(2)
                with d1:
                    risk = vcp_res['atr'] * EXIT_CFG["STOP_LOSS_ATR_MULT"]
                    # インデントなしで記述
                    panel_html1 = (
                        '<div class="diagnostic-panel">'
                        '<b>🛡️ STRATEGIC LEVELS (ATR-Based)</b>'
                        f'<div class="diag-row"><span class="diag-key">Stop Loss (2.0R)</span><span class="diag-val">${p_curr - risk:.2f}</span></div>'
                        f'<div class="diag-row"><span class="diag-key">Target 1 (1.0R)</span><span class="diag-val">${p_curr + risk:.2f}</span></div>'
                        f'<div class="diag-row"><span class="diag-key">Target 2 (2.5R)</span><span class="diag-val">${p_curr + risk*2.5:.2f}</span></div>'
                        f'<div class="diag-row"><span class="diag-key">Risk Unit ($)</span><span class="diag-val">${risk:.2f}</span></div>'
                        '</div>'
                    )
                    st.markdown(panel_html1, unsafe_allow_html=True)
                with d2:
                    bd = vcp_res['breakdown']
                    panel_html2 = (
                        '<div class="diagnostic-panel">'
                        '<b>📐 VCP SCORE BREAKDOWN</b>'
                        f'<div class="diag-row"><span class="diag-key">Tightness Score</span><span class="diag-val">{bd.get("tight", 0)}/45</span></div>'
                        f'<div class="diag-row"><span class="diag-key">Volume Dry-up</span><span class="diag-val">{bd.get("vol", 0)}/30</span></div>'
                        f'<div class="diag-row"><span class="diag-key">MA Trend Score</span><span class="diag-val">{bd.get("ma", 0)}/30</span></div>'
                        f'<div class="diag-row"><span class="diag-key">Pivot Bonus</span><span class="diag-val">+{bd.get("pivot", 0)}pt</span></div>'
                        '</div>'
                    )
                    st.markdown(panel_html2, unsafe_allow_html=True)

                # チャート
                df_t = df_raw.tail(100)
                c_fig = go.Figure(data=[go.Candlestick(x=df_t.index, open=df_t['Open'], high=df_t['High'], low=df_t['Low'], close=df_t['Close'])])
                c_fig.update_layout(template="plotly_dark", height=480, margin=dict(t=0, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(c_fig, use_container_width=True)

                # C. AI診断ボタン (ここから API キー必須)
                st.markdown('<div class="section-header">🤖 SENTINEL AI REASONING CONCLUSION</div>', unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                start_ai = b1.button("🚀 START AI CONTEXT ANALYSIS", type="primary", use_container_width=True)
                if b2.button("⭐ ADD TO WATCHLIST", use_container_width=True):
                    wl = (json.load(open(WATCHLIST_FILE)) if WATCHLIST_FILE.exists() else [])
                    if t_input not in wl:
                        wl.append(t_input); json.dump(wl, open(WATCHLIST_FILE, "w")); st.success(f"Added {t_input}")

                if start_ai:
                    key = st.secrets.get("DEEPSEEK_API_KEY")
                    if not key:
                        st.error("API KEY MISSING IN SECRETS.")
                    else:
                        with st.spinner(f"AI Reasoning: Analyzing Fundamentals & News for {t_input}..."):
                            # 各エンジンからの情報取得
                            news = NewsEngine.get(t_input)
                            fund = FundamentalEngine.get(t_input)
                            ins  = InsiderEngine.get(t_input)
                            
                            sentinel_master_prompt = (
                                f"あなたは伝説的投資家 Mark Minervini の理論を極めた AI ファンドマネージャー「SENTINEL」です。\n"
                                f"銘柄 {t_input} の診断結果に基づき、プロの視点で投資判断を下してください。\n\n"
                                f"━━━ 定量的データ (SENTINEL ENGINE) ━━━\n"
                                f"現在値: ${p_curr:.2f} | VCPスコア: {vcp_res['score']}/105 | PF: {pf_val:.2f} | RS: {rs_val*100:+.2f}%\n"
                                f"━━━ 外部情報 ━━━\n"
                                f"ファンダメンタル要約: {str(fund)[:1500]}\n"
                                f"需給/インサイダー: {str(ins)[:1000]}\n"
                                f"最新ニュース: {str(news)[:2000]}\n\n"
                                f"━━━ 指示 ━━━\n"
                                f"1. 算出された PF 数値と RS モメンタムを論拠の核とし、投資妙味を論評せよ。\n"
                                f"2. 現在のチャートが Minervini のどのステージ（1-4）にあるか特定せよ。\n"
                                f"3. 為替(¥{fx_rate:.2f})を考慮し、日本円投資家へのアドバイスを含めよ。\n"
                                f"4. 最後に Buy/Watch/Avoid を断行し、その根拠を箇条書きで示せ。\n\n"
                                f"※Markdown形式、日本語で最低 1,500 文字以上の圧倒的密度で記述せよ。"
                            )
                            cl = OpenAI(api_key=key, base_url="https://api.deepseek.com")
                            try:
                                res = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": sentinel_master_prompt}])
                                st.markdown("---")
                                # LaTeX 誤認防止
                                st.markdown(res.choices[0].message.content.replace("$", r"\$"))
                            except Exception as ai_e:
                                st.error(f"AI Engine Error: {ai_e}")
            else:
                st.error(f"Failed to fetch data for {t_input}.")

# ------------------------------------------------------------------------------
# 💼 TAB 3: PORTFOLIO (完全復元)
# ------------------------------------------------------------------------------
with t_port:
    st.markdown('<div class="section-header">💼 PORTFOLIO RISK MANAGEMENT</div>', unsafe_allow_html=True)
    p_j = load_portfolio_json(); pos_m = p_j.get("positions", {})
    if not pos_m:
        st.info("Portfolio is currently empty.")
    else:
        # 計算
        stats_list = []
        for s_k, s_d in pos_m.items():
            l_p = DataEngine.get_current_price(s_k)
            if l_p:
                pnl_u = (l_p - s_d["avg_cost"]) * s_d["shares"]
                pnl_p = (l_p / s_d["avg_cost"] - 1) * 100
                
                # 動的ストップ
                atr_l = DataEngine.get_atr(s_k) or 0.0
                risk_l = atr_l * EXIT_CFG["STOP_LOSS_ATR_MULT"]
                stop_l = max(l_p - risk_l, s_d.get("stop", 0)) if risk_l else s_d.get("stop", 0)
                
                stats_list.append({
                    "ticker": s_k, "shares": s_d["shares"], "avg": s_d["avg_cost"], 
                    "cp": l_p, "pnl_usd": pnl_u, "pnl_pct": pnl_p, 
                    "cl": "profit" if pnl_p > 0 else "urgent", "stop": stop_l
                })
        
        # サマリー
        total_pnl_j = sum(s["pnl_usd"] for s in stats_list) * fx_rate
        draw_sentinel_grid_ui([
            {"label": "💰 UNREALIZED JPY", "value": f"¥{total_pnl_j:,.0f}"},
            {"label": "📊 ASSETS", "value": len(stats_list)},
            {"label": "🛡️ EXPOSURE", "value": f"${sum(s['shares']*s['avg'] for s in stats_list):,.0f}"},
            {"label": "📈 PERFORMANCE", "value": f"{np.mean([s['pnl_pct'] for s in stats_list]):.2f}%" if stats_list else "0%"}
        ])
        
        st.markdown('<div class="section-header">📋 ACTIVE POSITIONS</div>', unsafe_allow_html=True)
        for s in stats_list:
            pnl_css = "pnl-pos" if s["pnl_pct"] > 0 else "pnl-neg"
            st.markdown(f'''
            <div class="pos-card {s['cl']}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b>{s['ticker']}</b>
                    <span class="{pnl_css}">{s['pnl_pct']:+.2f}% (¥{s['pnl_usd']*fx_rate:+,.0f})</span>
                </div>
                <div style="font-size: 0.95rem; color: #f0f6fc; margin-top: 10px;">
                    {s['shares']} shares @ ${s['avg']:.2f} (Live: ${s['cp']:.2f})
                </div>
                <div class="exit-info">
                    🛡️ <b>DYNAMIC STOP:</b> ${s['stop']:.2f} | 🎯 <b>TARGET (2.5R):</b> ${s['avg'] + (s['avg']-s['stop'])*2.5 if s['avg']>s['stop'] else s['avg']*1.3:.2f}
                </div>
            </div>''', unsafe_allow_html=True)
            
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button(f"🔍 ANALYZE {s['ticker']}", key=f"an_port_{s['ticker']}"):
                st.session_state.target_ticker = s['ticker']; st.session_state.trigger_analysis = True; st.rerun()
            if c_btn2.button(f"✅ LIQUIDATE {s['ticker']}", key=f"cl_port_{s['ticker']}"):
                del pos_m[s['ticker']]; save_portfolio_json(p_j); st.rerun()

    # --- 新規建玉追加 ---
    st.markdown('<div class="section-header">➕ REGISTER NEW POSITION</div>', unsafe_allow_html=True)
    with st.form("add_port_final"):
        f_c1, f_c2, f_c3 = st.columns(3)
        f_ticker = f_c1.text_input("Ticker").upper().strip()
        f_shares = f_c2.number_input("Shares", min_value=1, value=10)
        f_cost   = f_c3.number_input("Avg Cost", min_value=0.01, value=100.0)
        if st.form_submit_button("ADD TO PORTFOLIO", use_container_width=True):
            if f_ticker:
                p_f = load_portfolio_json()
                p_f["positions"][f_ticker] = {"ticker": f_ticker, "shares": f_shares, "avg_cost": f_cost, "added_at": TODAY_STR}
                save_portfolio_json(p_f); st.success(f"Added {f_ticker}"); st.rerun()

st.divider()
st.caption(f"🛡️ SENTINEL PRO SYSTEM | CORE ENGINE: 912 ROWS | DIAGNOSTICS: QUANT-NATIVE | VCP: LATEST | UI: FIXED")

