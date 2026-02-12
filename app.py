import streamlit as st
import pandas as pd
import json
from pathlib import Path
import os

st.set_page_config(page_title="Sentinel PRO Dashboard", layout="wide")

st.title("SENTINEL PRO 分析ダッシュボード")
st.markdown("毎日蓄積されたACTION/WAITデータを時系列で分析します。")

# resultsフォルダから全JSON読み込み
data_dir = Path("results")
all_data = []

if data_dir.exists():
    for file in sorted(data_dir.glob("*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                daily = json.load(f)
                date = daily.get("date", file.stem)
                
                # selected (ACTION) をフラット化
                for item in daily.get("selected", []):
                    row = item.copy()
                    row["date"] = date
                    row["status"] = "ACTION"
                    vcp = row.pop("vcp", {})
                    row["vcp_score"] = vcp.get("score")
                    row["vcp_signals"] = ", ".join(vcp.get("signals", []))
                    all_data.append(row)
                
                # watchlist_wait (WAIT) も追加
                for item in daily.get("watchlist_wait", []):
                    row = item.copy()
                    row["date"] = date
                    row["status"] = "WAIT"
                    vcp = row.pop("vcp", {})
                    row["vcp_score"] = vcp.get("score")
                    row["vcp_signals"] = ", ".join(vcp.get("signals", []))
                    all_data.append(row)
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")

if all_data:
    df = pd.DataFrame(all_data)
    
    # フィルタ
    status_filter = st.multiselect("Status", options=["ACTION", "WAIT"], default=["ACTION"])
    df_filtered = df[df["status"].isin(status_filter)]
    
    # 基本統計
    st.subheader("概要")
    col1, col2, col3 = st.columns(3)
    col1.metric("総エントリ数", len(df))
    col2.metric("ユニーク銘柄数", df["ticker"].nunique())
    col3.metric("平均RS", round(df["rs"].mean(), 1))
    
    # 時系列チャート
    st.subheader("RS / VCPスコア推移（平均）")
    trend = df_filtered.groupby("date")[["rs", "vcp_score"]].mean().reset_index()
    st.line_chart(trend.set_index("date"))
    
    # セクター分布
    st.subheader("セクター分布")
    sector_counts = df_filtered["sector"].value_counts()
    st.bar_chart(sector_counts)
    
    # テーブル表示（最新日付順）
    st.subheader("全データテーブル")
    st.dataframe(df_filtered.sort_values("date", ascending=False)[
        ["date", "ticker", "status", "rs", "vcp_score", "pf", "sector", "price", "entry", "target"]
    ])
    
    # 銘柄別詳細
    ticker = st.selectbox("銘柄を選択", options=sorted(df["ticker"].unique()))
    if ticker:
        st.subheader(f"{ticker} の推移")
        ticker_df = df[df["ticker"] == ticker].sort_values("date")
        st.dataframe(ticker_df[["date", "status", "rs", "vcp_score", "pf", "price", "entry"]])
        st.line_chart(ticker_df.set_index("date")[["rs", "vcp_score"]])
else:
    st.info("resultsフォルダにJSONがありません。Actionsが実行されるのを待ってください。")