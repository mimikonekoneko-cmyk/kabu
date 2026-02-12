import streamlit as st
import pandas as pd
import json
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# ⚙️ 設定 & ロジッククラス (SENTINEL PRO v4.5.2 から移植)
# ==============================================================================

# --- ページ設定 ---
st.set_page_config(
    page_title="SENTINEL PRO Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- カスタムCSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# --- ロジックエンジンの移植 (リアルタイム診断用) ---
class VCPAnalyzer:
    @staticmethod
    def calculate(df):
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
            # ATR
            tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

            if pd.isna(atr) or atr <= 0: return {"score": 0, "atr": 0, "signals": [], "is_dryup": False}

            # 1. 収縮判定
            h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min()
            range_pct = float((h10 - l10) / h10)
            tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.10))
            tight_score = max(0, min(40, tight_score))

            # 2. 出来高ドライアップ
            vol_ma = volume.rolling(50).mean().iloc[-1]
            vol_ratio = float(volume.iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
            is_dryup = bool(vol_ratio < 0.7)
            vol_score = 30 if is_dryup else (15 if vol_ratio < 1.1 else 0)

            # 3. トレンド
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]
            trend_score = (10 if close.iloc[-1] > ma50 else 0) + (10 if ma50 > ma200 else 0) + (10 if close.iloc[-1] > ma200 else 0)

            signals = []
            if range_pct < 0.06: signals.append("極度収縮")
            if is_dryup: signals.append("Vol枯渇")
            if trend_score == 30: signals.append("MA整列")

            return {"score": int(max(0, tight_score + vol_score + trend_score)), "atr": atr, "signals": signals}
        except: return {"score": 0, "atr": 0, "signals": []}

class StrategyValidator:
    @staticmethod
    def run_backtest(df):
        try:
            if len(df) < 200: return 1.0
            close = df['Close']; high = df['High']; low = df['Low']
            tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            trades = []
            in_pos = False; entry_p = 0; stop_p = 0
            start_idx = max(50, len(df)-250)
            for i in range(start_idx, len(df)):
                if in_pos:
                    if low.iloc[i] <= stop_p:
                        trades.append(-1.0); in_pos = False
                    elif high.iloc[i] >= entry_p + (entry_p - stop_p) * 2.5:
                        trades.append(2.5); in_pos = False
                    elif i == len(df) - 1:
                        risk = entry_p - stop_p
                        if risk > 0: trades.append(float((close.iloc[i] - entry_p) / risk)); in_pos = False
                else:
                    pivot = high.iloc[i-20:i].max()
                    if close.iloc[i] > pivot and close.iloc[i] > close.rolling(50).mean().iloc[i]:
                        in_pos = True; entry_p = float(close.iloc[i]); stop_p = entry_p - (float(atr.iloc[i]) * 2.0)
            if not trades: return 1.0
            pos_sum = sum([t for t in trades if t > 0]); neg_sum = abs(sum([t for t in trades if t < 0]))
            return round(float(pos_sum / neg_sum if neg_sum > 0 else (5.0 if pos_sum > 0 else 1.0)), 2)
        except: return 1.0

# ==============================================================================
# 📂 データ読み込み
# ==============================================================================

@st.cache_data(ttl=3600)
def load_historical_json():
    data_dir = Path("results")
    all_data = []
    if data_dir.exists():
        for file in sorted(data_dir.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    daily = json.load(f)
                    date = daily.get("date", file.stem)
                    # 過去の優秀銘柄データを統合
                    for k in ["selected", "watchlist_wait", "qualified_full"]:
                        for item in daily.get(k, []):
                            item["date"] = date
                            vcp = item.get("vcp", {})
                            item["vcp_score"] = vcp.get("score", item.get("vcp_score", 0)) if isinstance(vcp, dict) else 0
                            all_data.append(item)
            except: pass
    return pd.DataFrame(all_data)

df_history = load_historical_json()

# ==============================================================================
# 🖥️ UI構成
# ==============================================================================

st.title("🛡️ SENTINEL PRO DASHBOARD")

# --- サイドバー：モード切替 ---
mode = st.sidebar.radio("モード選択", ["📊 市場レポート (Batch)", "🔍 個別銘柄診断 (Realtime)"])

# ------------------------------------------------------------------------------
# MODE 1: 市場レポート (既存機能の強化版)
# ------------------------------------------------------------------------------
if mode == "📊 市場レポート (Batch)":
    st.subheader("Market Scan Report")
    
    if df_history.empty:
        st.error("JSONデータが見つかりません。resultsフォルダを確認してください。")
    else:
        # 最新日付のデータを取得
        latest_date = df_history["date"].max()
        latest_df = df_history[df_history["date"] == latest_date].copy()
        latest_df = latest_df.drop_duplicates(subset=["ticker"]) # 重複排除

        # セクターマップ
        st.markdown("### 🗺️ Sector Heatmap (Latest)")
        if not latest_df.empty:
            fig_treemap = px.treemap(
                latest_df, path=['sector', 'ticker'], values='vcp_score', color='rs',
                color_continuous_scale='RdYlGn', title="セクター別ヒートマップ (サイズ=VCP, 色=RS)"
            )
            st.plotly_chart(fig_treemap, use_container_width=True)

        # リスト表示
        st.markdown(f"### 📋 Scan Results ({latest_date})")
        
        # フィルタ
        filter_status = st.multiselect("ステータス", options=["ACTION", "WAIT"], default=["ACTION", "WAIT"])
        show_df = latest_df[latest_df["status"].isin(filter_status)]

        st.dataframe(
            show_df[["ticker", "status", "price", "rs", "vcp_score", "pf", "sector"]].style.background_gradient(subset=["vcp_score"], cmap="Greens"),
            use_container_width=True
        )

# ------------------------------------------------------------------------------
# MODE 2: 個別銘柄診断 (新機能)
# ------------------------------------------------------------------------------
elif mode == "🔍 個別銘柄診断 (Realtime)":
    st.subheader("Realtime Ticker Analyzer")
    st.caption("任意のティッカーを入力して、SENTINELエンジンでリアルタイム診断を行います。")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker_input = st.text_input("ティッカーを入力 (例: TSLA, NVDA)", value="").upper()
    with col_btn:
        st.write("") 
        st.write("") 
        analyze_btn = st.button("診断開始 🚀", type="primary")

    if analyze_btn and ticker_input:
        with st.spinner(f"{ticker_input} のデータを取得・分析中..."):
            try:
                # 1. データ取得
                data = yf.download(ticker_input, period="2y", interval="1d", progress=False, auto_adjust=True)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if data.empty or len(data) < 200:
                    st.error("データ不足、またはティッカーが無効です。")
                else:
                    # 2. リアルタイム分析実行
                    vcp_res = VCPAnalyzer.calculate(data)
                    pf_res = StrategyValidator.run_backtest(data)
                    
                    # セクター情報の取得 (簡易)
                    try:
                        info = yf.Ticker(ticker_input).info
                        sector = info.get("sector", "Unknown")
                        current_price = data["Close"].iloc[-1]
                    except:
                        sector = "Unknown"
                        current_price = 0

                    # 3. 結果表示
                    st.markdown("---")
                    
                    # 3-1. スコアカード
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("現在価格", f"${current_price:.2f}")
                    c2.metric("VCPスコア", f"{vcp_res['score']}/100", delta="合格ライン: 55")
                    c3.metric("Profit Factor", f"{pf_res:.2f}", delta="合格ライン: 1.1")
                    c4.metric("Sector", sector)

                    # 3-2. メーター表示
                    col_gauge, col_chart = st.columns([1, 2])
                    
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = vcp_res['score'],
                            title = {'text': "Sentinel Score"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#00ff00" if vcp_res['score'] > 70 else ("#f1c40f" if vcp_res['score'] > 55 else "#ff3333")},
                                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 55}
                            }
                        ))
                        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        st.markdown("**検出シグナル:**")
                        if vcp_res['signals']:
                            for s in vcp_res['signals']:
                                st.success(f"✅ {s}")
                        else:
                            st.warning("⚠️ 明確なセットアップなし")

                    with col_chart:
                        # チャート表示 (直近6ヶ月)
                        chart_df = data.iloc[-126:]
                        fig = go.Figure(data=[go.Candlestick(
                            x=chart_df.index,
                            open=chart_df['Open'], high=chart_df['High'],
                            low=chart_df['Low'], close=chart_df['Close'],
                            name=ticker_input
                        )])
                        fig.update_layout(
                            title=f"{ticker_input} Daily Chart",
                            template="plotly_dark",
                            xaxis_rangeslider_visible=False,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 4. セクター比較 (セグメント分析)
                    if not df_history.empty:
                        st.markdown(f"### 📊 同セクター ({sector}) との比較")
                        
                        # 過去データから同セクターの平均値を算出
                        sector_peers = df_history[df_history["sector"] == sector]
                        
                        if not sector_peers.empty:
                            avg_vcp = sector_peers["vcp_score"].mean()
                            avg_pf = sector_peers["pf"].mean()
                            
                            # 比較レーダーチャート
                            categories = ['VCP Score', 'Profit Factor', 'RS (Estimate)']
                            
                            # 今回の診断値
                            # RSは相対値なので単独計算できないが、簡易的にPFとVCPから推定
                            est_rs = min(99, (pf_res * 10) + (vcp_res['score'] / 2)) 
                            
                            fig_radar = go.Figure()
                            
                            # 自分
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[vcp_res['score'], pf_res * 20, est_rs], # PFはスケール調整
                                theta=categories,
                                fill='toself',
                                name=ticker_input,
                                line_color='#00FF00'
                            ))
                            
                            # セクター平均
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[avg_vcp, avg_pf * 20, 50],
                                theta=categories,
                                fill='toself',
                                name=f'{sector} Average',
                                line_color='#666666',
                                line_dash='dash'
                            ))
                            
                            fig_radar.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                template="plotly_dark",
                                title=f"VS セクター平均"
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                            
                            # 同セクターの過去の優秀銘柄リスト
                            st.markdown("過去に検出された同セクターの優秀銘柄:")
                            top_peers = sector_peers.sort_values("vcp_score", ascending=False).drop_duplicates("ticker").head(5)
                            st.dataframe(top_peers[["date", "ticker", "vcp_score", "pf", "price"]])
                        else:
                            st.info(f"データベースに {sector} の比較対象データがありませんでした。")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# --- フッター ---
st.markdown("---")
st.caption("Powered by SENTINEL PRO ELITE Engine | Data: yfinance")
