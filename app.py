import streamlit as st
import pandas as pd
import json
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# --- ページ設定 ---
st.set_page_config(
    page_title="SENTINEL PRO Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" # スマホで見やすくするため最初は閉じる
)

# --- カスタムCSS（ダークモード最適化 & スマホ調整） ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #00FF00;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SENTINEL PRO ELITE")
st.caption("AI-Powered US Stock Screening System")

# --- データ読み込み (キャッシュ) ---
@st.cache_data(ttl=3600)
def load_data():
    data_dir = Path("results")
    all_data = []
    if data_dir.exists():
        for file in sorted(data_dir.glob("*.json"), reverse=True): # 最新順
            try:
                with open(file, "r", encoding="utf-8") as f:
                    daily = json.load(f)
                    date = daily.get("date", file.stem)
                    # ACTION
                    for item in daily.get("selected", []):
                        item["status"] = "ACTION"
                        item["date"] = date
                        # VCPの階層をフラット化
                        vcp = item.pop("vcp", {})
                        item["vcp_score"] = vcp.get("score", 0)
                        item["signals"] = vcp.get("signals", [])
                        all_data.append(item)
                    # WAIT
                    for item in daily.get("watchlist_wait", []):
                        item["status"] = "WAIT"
                        item["date"] = date
                        vcp = item.pop("vcp", {})
                        item["vcp_score"] = vcp.get("score", 0)
                        item["signals"] = vcp.get("signals", [])
                        all_data.append(item)
            except: pass
            
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()
if df.empty:
    st.error("データがありません。")
    st.stop()

# --- 最新データのみ抽出 ---
latest_date = df["date"].max()
latest_df = df[df["date"] == latest_date].copy()

# --- 1. トップ指標（KPI）エリア ---
st.markdown("### 📊 Market Pulse")
col1, col2, col3, col4 = st.columns(4)
with col1:
    action_count = len(latest_df[latest_df['status']=='ACTION'])
    st.metric("ACTION Signals", f"{action_count} 銘柄", delta="即エントリー可", delta_color="normal")
with col2:
    wait_count = len(latest_df[latest_df['status']=='WAIT'])
    st.metric("WAIT List", f"{wait_count} 銘柄", delta="監視候補", delta_color="off")
with col3:
    avg_rs = latest_df[latest_df['status']=='ACTION']['rs'].mean()
    st.metric("Avg RS Rating", f"{avg_rs:.1f}", delta="市場強度")
with col4:
    avg_vcp = latest_df[latest_df['status']=='ACTION']['vcp_score'].mean()
    st.metric("Avg VCP Score", f"{avg_vcp:.1f}", delta="チャート品質")

st.markdown("---")

# --- 2. セクターヒートマップ（Plotly） ---
st.markdown("### 🗺️ Sector Heatmap")
if not latest_df.empty:
    # セクターごとの銘柄数をカウント
    sector_df = latest_df.groupby('sector').size().reset_index(name='count')
    # 平均RSも計算して色に使う
    sector_rs = latest_df.groupby('sector')['rs'].mean().reset_index(name='avg_rs')
    sector_data = pd.merge(sector_df, sector_rs, on='sector')
    
    fig_treemap = px.treemap(
        latest_df, 
        path=['sector', 'ticker'], 
        values='rs',
        color='rs',
        color_continuous_scale='RdYlGn', # 赤→黄→緑
        title="セクター別・銘柄強度マップ (サイズ=RS, 色=RS)"
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

# --- 3. メインリスト & 詳細 ---
st.markdown("### 💎 Focus List")

# タブで表示切り替え
tab1, tab2 = st.tabs(["📋 リスト表示", "📈 詳細チャート"])

with tab1:
    # データフレームに装飾をつける
    def highlight_status(val):
        color = '#06982d' if val == 'ACTION' else '#b38600'
        return f'background-color: {color}'

    display_df = latest_df[["ticker", "status", "price", "entry", "target", "stop", "rs", "vcp_score", "pf", "shares", "sector"]]
    st.dataframe(
        display_df.style.applymap(highlight_status, subset=['status'])
        .format({"price": "{:.2f}", "entry": "{:.2f}", "target": "{:.2f}", "stop": "{:.2f}", "pf": "{:.2f}"}),
        use_container_width=True,
        height=400
    )

with tab2:
    tickers = latest_df["ticker"].unique()
    selected_ticker = st.selectbox("分析する銘柄を選択", tickers)
    
    if selected_ticker:
        row = latest_df[latest_df["ticker"] == selected_ticker].iloc[0]
        
        # 3カラムレイアウト
        c1, c2, c3 = st.columns([1, 2, 1])
        
        with c1:
            st.markdown(f"## {row['ticker']}")
            st.caption(f"{row['sector']}")
            st.metric("現在値", f"${row['price']}", delta=f"Entryまで {row['entry'] - row['price']:.2f}")
            
            # リスクリワード計算
            risk = row['entry'] - row['stop']
            reward = row['target'] - row['entry']
            rr_ratio = reward / risk if risk > 0 else 0
            st.markdown(f"**リスクリワード比:** 1 : {rr_ratio:.2f}")
            
            st.info(f"推奨株数: **{row['shares']}株**")
            st.success(f"利確目標: **${row['target']}**")
            st.error(f"損切ライン: **${row['stop']}**")

        with c2:
            # yfinanceでデータ取得 & Plotly CandleStick
            with st.spinner("Loading Chart..."):
                stock = yf.download(selected_ticker, period="6mo", interval="1d", progress=False)
                if isinstance(stock.columns, pd.MultiIndex):
                    stock.columns = stock.columns.get_level_values(0)
                
                # Plotlyチャート（TradingView風）
                fig = go.Figure(data=[go.Candlestick(
                    x=stock.index,
                    open=stock['Open'],
                    high=stock['High'],
                    low=stock['Low'],
                    close=stock['Close'],
                    name=selected_ticker
                )])
                
                # エントリー、利確、損切ラインを描画
                fig.add_hline(y=row['entry'], line_dash="dash", line_color="yellow", annotation_text="ENTRY")
                fig.add_hline(y=row['target'], line_dash="dash", line_color="green", annotation_text="TARGET")
                fig.add_hline(y=row['stop'], line_dash="dash", line_color="red", annotation_text="STOP")

                fig.update_layout(
                    title=f"{selected_ticker} Technical Chart",
                    yaxis_title="Price (USD)",
                    template="plotly_dark", # ダークモード
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

        with c3:
            st.markdown("### 🤖 Signals")
            # VCPスコアをゲージで表示
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = row['vcp_score'],
                title = {'text': "VCP Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00ff00" if row['vcp_score'] > 70 else "#f1c40f"},
                    'steps': [{'range': [0, 50], 'color': "gray"}]
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=0, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("**検出シグナル:**")
            if row['signals']:
                for sig in row['signals']:
                    st.markdown(f"- ✅ {sig}")
            else:
                st.markdown("- 特になし")
            
            st.markdown("---")
            st.markdown(f"**RS Rating:** {row['rs']}/99")
            st.progress(row['rs'] / 100)
            
            st.markdown(f"**Profit Factor:** {row['pf']}")

st.markdown("---")
st.caption("Generated by SENTINEL PRO ELITE Engine")
