import streamlit as st
import pandas as pd
import json
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import warnings
import datetime
import feedparser

warnings.filterwarnings("ignore")

# ==============================================================================
# ⚙️ システム設定 & 時間管理
# ==============================================================================

NOW = datetime.datetime.now()
TODAY_STR = NOW.strftime("%Y-%m-%d")

st.set_page_config(
    page_title=f"SENTINEL PRO - {TODAY_STR}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .ai-report { background-color: #0E1117; border-left: 5px solid #00FF00; padding: 25px; border-radius: 5px; line-height: 1.8; }
    .ai-individual { background-color: #1c2333; border: 1px solid #00FF00; padding: 30px; border-radius: 12px; line-height: 1.9; }
    .watchlist-card { background-color: #111; border: 1px solid #333; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 📂 ウォッチリスト・データ管理
# ==============================================================================

WATCHLIST_FILE = Path("watchlist.json")

def load_watchlist():
    if WATCHLIST_FILE.exists():
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    return []

def save_watchlist(ticker):
    watchlist = load_watchlist()
    if ticker not in watchlist:
        watchlist.append(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f)
        return True
    return False

def remove_from_watchlist(ticker):
    watchlist = load_watchlist()
    if ticker in watchlist:
        watchlist.remove(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f)
        return True
    return False

@st.cache_data(ttl=600)
def load_historical_json():
    data_dir = Path("results")
    all_data = []
    if data_dir.exists():
        for file in sorted(data_dir.glob("*.json"), reverse=True):
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
# 🧠 分析エンジン
# ==============================================================================

class VCPAnalyzer:
    @staticmethod
    def calculate(df):
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
            h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min()
            range_pct = float((h10 - l10) / h10)
            tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.15))
            tight_score = max(0, min(40, tight_score))

            vol_ma = volume.rolling(50).mean().iloc[-1]
            vol_ratio = float(volume.iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
            vol_score = 30 if vol_ratio < 0.7 else (15 if vol_ratio < 1.1 else 0)

            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]
            trend_score = (10 if close.iloc[-1] > ma50 else 0) + (10 if ma50 > ma200 else 0) + (10 if close.iloc[-1] > ma200 else 0)

            signals = []
            if range_pct < 0.06: signals.append("極度収縮")
            if vol_ratio < 0.7: signals.append("Vol枯渇")
            if trend_score == 30: signals.append("MA整列")

            return {
                "score": int(max(0, tight_score + vol_score + trend_score)),
                "signals": signals,
                "raw": {"range": range_pct, "vol": vol_ratio}
            }
        except: return {"score": 0, "signals": [], "raw": {"range": 0, "vol": 0}}

# ==============================================================================
# 🛰️ ニュース & AIエンジン
# ==============================================================================

def fetch_fresh_news(ticker):
    headlines = []
    try:
        yf_news = yf.Ticker(ticker).news
        for n in (yf_news or [])[:5]:
            headlines.append(f"- {n.get('headline', n.get('title', 'No Title'))}")
    except: pass
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:24h&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]: headlines.append(f"- {entry.title}")
    except: pass
    return "\n".join(list(set(headlines))) if headlines else "本日、特筆すべき新規材料は未検出。"

def call_gemini(prompt):
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key: return "⚠️ APIキー未設定"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, safety_settings={cat: HarmBlockThreshold.BLOCK_NONE for cat in [HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT]})
        return response.text
    except Exception as e: return f"Gemini Error: {str(e)}"

# ==============================================================================
# 🖥️ メインUI構成
# ==============================================================================

st.title(f"🛡️ SENTINEL PRO")

# --- サイドバー：ウォッチリスト ---
st.sidebar.header("⭐ Watchlist")
watchlist = load_watchlist()
if not watchlist:
    st.sidebar.info("お気に入りがありません。リアルタイム診断から追加してください。")
else:
    for t in watchlist:
        col_t, col_r = st.sidebar.columns([3, 1])
        if col_t.button(f"🔍 {t}", key=f"btn_{t}", use_container_width=True):
            st.session_state.ticker_from_list = t
        if col_r.button("❌", key=f"rm_{t}"):
            remove_from_watchlist(t)
            st.rerun()

mode = st.sidebar.radio("分析モード", ["📊 市場スキャン (Batch)", "🔍 リアルタイム診断 (Realtime)"])

df_all = load_historical_json()

if mode == "📊 市場スキャン (Batch)":
    if df_all.empty: st.error("データ未検出")
    else:
        latest_date = df_all["date"].max()
        latest_df = df_all[df_all["date"] == latest_date].copy().drop_duplicates(subset=["ticker"])
        
        st.markdown(f"### 🤖 SENTINEL Briefing: {latest_date}")
        report_key = f"report_{latest_date}"
        if report_key not in st.session_state:
            with st.spinner("市況解析中..."):
                spy_news = fetch_fresh_news("SPY")
                action_list = latest_df[latest_df['status']=='ACTION']['ticker'].tolist()
                prompt = f"伝説の投資家AI「SENTINEL」として、{latest_date}の市場を800文字以上で分析せよ。\nニュース:\n{spy_news}\n注目銘柄: {action_list[:5]}"
                st.session_state[report_key] = call_gemini(prompt)

        st.markdown(f"""<div class="ai-report">{st.session_state[report_key]}</div>""", unsafe_allow_html=True)
        
        # セクターマップ
        st.plotly_chart(px.treemap(latest_df, path=['sector', 'ticker'], values='vcp_score', color='rs', color_continuous_scale='RdYlGn'), use_container_width=True)
        st.dataframe(latest_df[["ticker", "status", "price", "vcp_score", "sector"]].style.background_gradient(subset=["vcp_score"], cmap="Greens"), use_container_width=True)

        # --- 銘柄詳細チャート（復活） ---
        st.divider()
        st.subheader("🔍 Selected Ticker Deep Drill")
        drill_ticker = st.selectbox("詳細チャートを表示する銘柄を選択", options=latest_df['ticker'].unique())
        if drill_ticker:
            with st.spinner(f"{drill_ticker} のチャートを生成中..."):
                t_data = yf.Ticker(drill_ticker).history(period="1y", auto_adjust=True)
                if not t_data.empty:
                    fig = go.Figure(data=[go.Candlestick(x=t_data.index[-120:], open=t_data['Open'][-120:], high=t_data['High'][-120:], low=t_data['Low'][-120:], close=t_data['Close'][-120:])])
                    fig.update_layout(title=f"{drill_ticker} - Daily Chart (6 months)", template="plotly_dark", xaxis_rangeslider_visible=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)

elif mode == "🔍 リアルタイム診断 (Realtime)":
    st.subheader("Realtime Ticker Analyzer 🤖")
    
    # サイドバーのウォッチリストからの入力を反映
    default_ticker = st.session_state.get("ticker_from_list", "")
    ticker_input = st.text_input("ティッカーを入力", value=default_ticker).upper()
    
    col_run, col_fav = st.columns([1, 1])
    with col_run: run_btn = st.button("診断開始 🚀", type="primary", use_container_width=True)
    with col_fav: 
        if st.button("⭐ Watchlistに追加", use_container_width=True) and ticker_input:
            if save_watchlist(ticker_input): st.success(f"{ticker_input} を追加しました！")
            else: st.warning("既に追加されています。")

    if run_btn and ticker_input:
        with st.spinner(f"{ticker_input} を深層解析中..."):
            try:
                t_obj = yf.Ticker(ticker_input)
                data = t_obj.history(period="2y", auto_adjust=True)
                news = fetch_fresh_news(ticker_input)
                if data.empty: st.error("データ取得不可")
                else:
                    vcp = VCPAnalyzer.calculate(data)
                    prompt = f"ウォール街のプロAIとして{ticker_input}を診断せよ。今日:{TODAY_STR}\nニュース:\n{news}\nスコア:{vcp['score']}/100\n結論(BUY/WAIT/PASS)を800文字以上で語れ。"
                    report = call_gemini(prompt)
                    st.markdown(f"""<div class="ai-individual"><h5>🤖 SENTINEL Deep Diagnosis</h5>{report}</div>""", unsafe_allow_html=True)
                    
                    st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index[-60:], open=data['Open'][-60:], high=data['High'][-60:], low=data['Low'][-60:], close=data['Close'][-60:])]).update_layout(template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")

st.markdown("---")
st.caption(f"System Time: {TODAY_STR} | Powered by SENTINEL PRO ELITE")
