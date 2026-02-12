import streamlit as st
import pandas as pd
import json
from pathlib import Path
import os
import yfinance as yf
import altair as alt

# ページ設定（スマホでも見やすいようにwide）
st.set_page_config(
    page_title="SENTINEL PRO 分析ダッシュボード",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡 SENTINEL PRO 分析ダッシュボード")
st.markdown("毎日蓄積されたACTION / WAITデータを分析します。株価推移もyfinanceでリアルタイム取得。")

# データ読み込み関数
@st.cache_data(ttl=3600)  # 1時間キャッシュ
def load_all_data():
    data_dir = Path("results")
    all_data = []
    
    if data_dir.exists():
        for file in sorted(data_dir.glob("*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    daily = json.load(f)
                    date = daily.get("date", file.stem)
                    
                    # selected (ACTION)
                    for item in daily.get("selected", []):
                        row = item.copy()
                        row["date"] = date
                        row["status"] = "ACTION"
                        vcp = row.pop("vcp", {})
                        row["vcp_score"] = vcp.get("score")
                        row["vcp_signals"] = ", ".join(vcp.get("signals", []))
                        all_data.append(row)
                    
                    # watchlist_wait (WAIT)
                    for item in daily.get("watchlist_wait", []):
                        row = item.copy()
                        row["date"] = date
                        row["status"] = "WAIT"
                        vcp = row.pop("vcp", {})
                        row["vcp_score"] = vcp.get("score")
                        row["vcp_signals"] = ", ".join(vcp.get("signals", []))
                        all_data.append(row)
            except Exception as e:
                st.warning(f"ファイル読み込みエラー: {file} → {e}")
    
    if not all_data:
        st.info("resultsフォルダにJSONデータがありません。GitHub Actionsの実行をお待ちください。")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False)

# データ読み込み
df = load_all_data()

if df.empty:
    st.stop()

# サイドバー：フィルタ
st.sidebar.header("フィルタ")
status_filter = st.sidebar.multiselect(
    "ステータス",
    options=["ACTION", "WAIT"],
    default=["ACTION"]
)

min_rs = st.sidebar.slider("最低RS", 50, 99, 70)
min_vcp = st.sidebar.slider("最低VCPスコア", 0, 100, 50)

df_filtered = df[
    (df["status"].isin(status_filter)) &
    (df["rs"] >= min_rs) &
    (df["vcp_score"] >= min_vcp)
]

# 概要メトリクス
st.subheader("概要")
col1, col2, col3, col4 = st.columns(4)
col1.metric("総エントリ数", len(df_filtered))
col2.metric("ユニーク銘柄数", df_filtered["ticker"].nunique())
col3.metric("平均RS", round(df_filtered["rs"].mean(), 1))
col4.metric("平均VCPスコア", round(df_filtered["vcp_score"].mean(), 1))

# 時系列トレンド
st.subheader("RS / VCPスコア推移（日次平均）")
if not df_filtered.empty:
    trend = df_filtered.groupby("date")[["rs", "vcp_score"]].mean().reset_index()
    st.line_chart(trend.set_index("date"))

# セクター分布
st.subheader("セクター分布")
sector_counts = df_filtered["sector"].value_counts()
st.bar_chart(sector_counts)

# 全データテーブル
st.subheader("全データテーブル")
display_cols = [
    "date", "ticker", "status", "rs", "vcp_score", "vcp_signals",
    "pf", "sector", "price", "entry", "target", "shares"
]
st.dataframe(df_filtered[display_cols])

# 銘柄別詳細 + 株価チャート
st.subheader("銘柄詳細 & 株価推移")
available_tickers = sorted(df["ticker"].unique())
ticker = st.selectbox("銘柄を選択", options=available_tickers)

if ticker:
    ticker_df = df[df["ticker"] == ticker].sort_values("date")
    
    st.markdown(f"**{ticker} の履歴**")
    st.dataframe(ticker_df[["date", "status", "rs", "vcp_score", "pf", "price", "entry", "target"]])
    
    # RS / VCP 推移チャート
    st.markdown("**RS / VCPスコア推移**")
    st.line_chart(ticker_df.set_index("date")[["rs", "vcp_score"]])
    
    # 株価チャート（yfinance）
    st.markdown("**株価推移（始値・終値・ローソク足）**")
    with st.spinner(f"{ticker} の株価データを取得中..."):
        try:
            period = st.selectbox("期間", ["1mo", "3mo", "6mo", "1y"], index=0)
            stock_data = yf.download(ticker, period=period, progress=False)
            
            if not stock_data.empty:
                # テーブル
                st.dataframe(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
                
                # 線チャート（Open/Close）
                chart_data = stock_data[['Open', 'Close']].reset_index()
                st.line_chart(chart_data.set_index('Date'))
                
                # ローソク足（Altair）
                c = alt.Chart(stock_data.reset_index()).mark_candlestick(
                    open='Open', high='High', low='Low', close='Close'
                ).encode(
                    x='Date:T',
                    y='Close:Q',
                    color=alt.condition(
                        alt.datum.Close >= alt.datum.Open,
                        alt.value("#00cc00"),  # 上昇：緑
                        alt.value("#ff3333")   # 下降：赤
                    )
                ).interactive()
                st.altair_chart(c, use_container_width=True)
            else:
                st.warning(f"{ticker} のデータが取得できませんでした。")
        except Exception as e:
            st.error(f"株価取得エラー: {e}")

st.markdown("---")
st.caption("データはGitHub Actionsで毎日更新 | 株価はyfinanceリアルタイム取得 | 最終更新: " + 
           (df["date"].max().strftime("%Y-%m-%d") if not df.empty else "データなし"))