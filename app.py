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
# ⚙️ 設定 & スタイル
# ==============================================================================

st.set_page_config(
    page_title="SENTINEL PRO Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E; border: 1px solid #333; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;
    }
    .ai-report {
        background-color: #0E1117; 
        border-left: 5px solid #00FF00; 
        padding: 25px; 
        margin-bottom: 20px; 
        border-radius: 5px;
        font-family: 'Helvetica Neue', sans-serif;
        line-height: 1.8;
        font-size: 1.1em;
    }
    .ai-individual {
        background-color: #1c2333; 
        border: 1px solid #00FF00; 
        padding: 30px; 
        border-radius: 12px; 
        margin-top: 10px;
        line-height: 1.9;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 分析エンジン
# ==============================================================================

class VCPAnalyzer:
    @staticmethod
    def calculate(df):
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
            tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if pd.isna(atr) or atr <= 0: return {"score": 0, "atr": 0, "signals": []}

            h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min()
            range_pct = float((h10 - l10) / h10)
            tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.15))
            tight_score = max(0, min(40, tight_score))

            vol_ma = volume.rolling(50).mean().iloc[-1]
            vol_ratio = float(volume.iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
            is_dryup = bool(vol_ratio < 0.7)
            vol_score = 30 if is_dryup else (15 if vol_ratio < 1.1 else 0)

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
                    if low.iloc[i] <= stop_p: trades.append(-1.0); in_pos = False
                    elif high.iloc[i] >= entry_p + (entry_p - stop_p) * 2.5: trades.append(2.5); in_pos = False
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
# 📂 データ読み込み関数
# ==============================================================================

@st.cache_data(ttl=3600)
def load_historical_json():
    data_dir = Path("results")
    all_data = []
    meta_data = {}
    if data_dir.exists():
        for file in sorted(data_dir.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    daily = json.load(f)
                    date = daily.get("date", file.stem)
                    meta_data[date] = {"scan_count": daily.get("scan_count", 450), "qualified_count": daily.get("qualified_count", 0)}
                    for k in ["selected", "watchlist_wait", "qualified_full"]:
                        for item in daily.get(k, []):
                            item["date"] = date
                            vcp = item.get("vcp", {})
                            item["vcp_score"] = vcp.get("score", item.get("vcp_score", 0)) if isinstance(vcp, dict) else 0
                            all_data.append(item)
            except: pass
    return pd.DataFrame(all_data), meta_data

# ==============================================================================
# 🛰️ ニュース収集エンジン
# ==============================================================================

def fetch_safe_news(ticker):
    headlines = []
    try:
        yf_news = yf.Ticker(ticker).news
        for n in (yf_news or [])[:5]:
            headlines.append(f"- {n.get('headline', n.get('title', 'No Title'))}")
    except: pass
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            title = f"- {entry.title}"
            if title not in headlines: headlines.append(title)
    except: pass
    
    context = "\n".join(headlines)
    if not context or "No Headline" in context:
        return "※現在、最新ニュースを外部確認中...（取得制限により一時的に表示されませんが、材料がないことを意味しません）"
    return context

# ==============================================================================
# 🤖 AIエンジン (Gemini 2.0 Flash)
# ==============================================================================

def call_gemini_pure(prompt):
    api_key = None
    try: api_key = st.secrets["GEMINI_API_KEY"]
    except: api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "⚠️ APIキー未設定"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        safety_settings = {cat: HarmBlockThreshold.BLOCK_NONE for cat in [
            HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
        ]}
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e: return f"Gemini Error: {str(e)}"

# ==============================================================================
# 🖥️ メインUI構成
# ==============================================================================

st.title("🛡️ SENTINEL PRO DASHBOARD")

# 先に関数を定義してから呼び出す
df_history, meta_history = load_historical_json()

mode = st.sidebar.radio("モード選択", ["📊 市場レポート (Batch)", "🔍 個別銘柄診断 (Realtime)"])

if mode == "📊 市場レポート (Batch)":
    if df_history.empty:
        st.error("データが見つかりません。")
    else:
        latest_date = df_history["date"].max()
        latest_df = df_history[df_history["date"] == latest_date].copy().drop_duplicates(subset=["ticker"])
        
        st.markdown(f"### 🤖 SENTINEL AI Briefing")
        
        if "market_ai_pure" not in st.session_state:
            with st.spinner("AIが市況を深く精査中..."):
                spy_news = fetch_safe_news("SPY")
                action_list = latest_df[latest_df['status']=='ACTION']['ticker'].tolist()
                top_sector = latest_df['sector'].value_counts().idxmax() if not latest_df.empty else "None"
                
                prompt = f"""
                あなたは伝説の投資家AI「SENTINEL」です。
                【最新ニュース(SPY)】\n{spy_news}
                【内部データ】\n- ACTION: {len(action_list)}銘柄 ({', '.join(action_list[:5])})\n- 主導セクター: {top_sector}\n- VCP平均: {latest_df['vcp_score'].mean():.1f}
                【指示】
                市場環境を読み解き、今日の戦い方を800文字程度で論理的に解説してください。
                1. 市況判断 2. セクター動向 3. 今日の具体的戦略。スコアが高いほど強気であることを忘れずに。
                """
                st.session_state.market_ai_pure = call_gemini_pure(prompt)
        
        st.markdown(f"""<div class="ai-report">{st.session_state.market_ai_pure}</div>""", unsafe_allow_html=True)

        if not latest_df.empty:
            st.plotly_chart(px.treemap(latest_df, path=['sector', 'ticker'], values='vcp_score', color='rs', color_continuous_scale='RdYlGn'), use_container_width=True)
        st.dataframe(latest_df[["ticker", "status", "price", "rs", "vcp_score", "pf", "sector"]].style.background_gradient(subset=["vcp_score"], cmap="Greens"), use_container_width=True)

elif mode == "🔍 個別銘柄診断 (Realtime)":
    st.subheader("Realtime Ticker Analyzer 🤖")
    ticker_input = st.text_input("ティッカー (例: WDC)", value="").upper()
    if st.button("診断開始 🚀", type="primary") and ticker_input:
        with st.spinner(f"{ticker_input} を深層分析中..."):
            try:
                ticker_obj = yf.Ticker(ticker_input)
                data = ticker_obj.history(period="2y", auto_adjust=True)
                news_context = fetch_safe_news(ticker_input)
                
                if data.empty: st.error("データ取得失敗")
                else:
                    vcp = VCPAnalyzer().calculate(data)
                    pf_res = StrategyValidator().run_backtest(data)
                    price = data["Close"].iloc[-1]
                    try: sector = ticker_obj.info.get("sector", "Unknown")
                    except: sector = "Unknown"
                    
                    prompt = f"""
                    あなたはウォール街の冷徹なプロ投資家AIです。【{ticker_input}】を技術的・ファンダメンタルズ両面から診断します。
                    【最新ニュース】\n{news_context}
                    【テクニカル】\n- VCPスコア: {vcp['score']} / 100\n- PF: {pf_res:.2f}\n- シグナル: {vcp['signals']}
                    【最重要ルール】
                    1. スコアが高いほど「買い推奨」です。低いスコアを無理に褒めないでください。
                    2. 直近で大きな材料（自社株買い等）があった場合、一時的にボラティリティが拡大しスコアが下がることがありますが、これは「ふるい落とし（Shakeout）」である可能性を考慮してください。
                    【指示】
                    現在の状況を800文字程度で論理的に解説し、最後に「BUY」「WAIT」「PASS」を断言してください。
                    """
                    ai_report = call_gemini_pure(prompt)
                    
                    st.markdown("---")
                    st.markdown(f"""<div class="ai-individual"><h5>🤖 SENTINEL Deep Diagnosis</h5>{ai_report}</div>""", unsafe_allow_html=True)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"${price:.2f}")
                    c2.metric("VCP Score", f"{vcp['score']}")
                    c3.metric("Profit Factor", f"{pf_res:.2f}")
                    c4.metric("Sector", sector)

                    # レーダーチャート
                    categories = ['VCP Score', 'Profit Factor', 'RS Rating']
                    h_max = data["High"].max(); l_min = data["Low"].min()
                    est_rs = ((price - l_min) / (h_max - l_min)) * 100 if h_max > l_min else 50
                    hist_data = df_history[df_history["ticker"] == ticker_input]
                    my_rs = hist_data.iloc[0]["rs"] if not hist_data.empty else est_rs

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(r=[vcp['score'], min(100, pf_res*20), my_rs], theta=categories, fill='toself', name=ticker_input, line_color='#00FF00'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark", height=300)
                    st.plotly_chart(fig_radar, use_container_width=True)

                    st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index[-126:], open=data['Open'][-126:], high=data['High'][-126:], low=data['Low'][-126:], close=data['Close'][-126:])]).update_layout(template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")

st.markdown("---")
st.caption("Powered by SENTINEL PRO ELITE & Google Gemini 2.0 Flash")
