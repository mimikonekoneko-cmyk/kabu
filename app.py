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
# ⚙️ 時間管理ロジック
# ==============================================================================

# 「システム上の今日」を定義
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
            # 1. 収縮判定 (10日間の変動幅)
            h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min()
            range_pct = float((h10 - l10) / h10)
            tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.15))
            tight_score = max(0, min(40, tight_score))

            # 2. 出来高枯渇判定 (50日平均比)
            vol_ma = volume.rolling(50).mean().iloc[-1]
            vol_ratio = float(volume.iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
            is_dryup = bool(vol_ratio < 0.7)
            vol_score = 30 if is_dryup else (15 if vol_ratio < 1.1 else 0)

            # 3. トレンド判定 (MA整列)
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]
            trend_score = (10 if close.iloc[-1] > ma50 else 0) + (10 if ma50 > ma200 else 0) + (10 if close.iloc[-1] > ma200 else 0)

            signals = []
            if range_pct < 0.06: signals.append("極度収縮")
            if is_dryup: signals.append("Vol枯渇")
            if trend_score == 30: signals.append("MA整列")

            return {
                "score": int(max(0, tight_score + vol_score + trend_score)),
                "signals": signals,
                "raw": {"range": range_pct, "vol": vol_ratio}
            }
        except: return {"score": 0, "signals": [], "raw": {"range": 0, "vol": 0}}

# ==============================================================================
# 🛰️ ニュースエンジン (24時間限定)
# ==============================================================================

def fetch_fresh_news(ticker):
    """過去24時間の最新ニュースのみに絞り込む"""
    headlines = []
    # yfinance
    try:
        yf_news = yf.Ticker(ticker).news
        for n in (yf_news or [])[:5]:
            headlines.append(f"- {n.get('headline', n.get('title', 'No Title'))}")
    except: pass
    # Google RSS (when:24h に変更してノイズを排除)
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:24h&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            headlines.append(f"- {entry.title}")
    except: pass
    
    return "\n".join(list(set(headlines))) if headlines else "本日、特筆すべき新規材料は検出されませんでした。"

# ==============================================================================
# 📂 データロード
# ==============================================================================

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
# 🤖 AIエンジン (Gemini 2.0 Flash)
# ==============================================================================

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
# 🖥️ UI構成
# ==============================================================================

st.title(f"🛡️ SENTINEL PRO DASHBOARD")
df_all = load_historical_json()

mode = st.sidebar.radio("分析モード", ["📊 市場スキャン (Batch)", "🔍 リアルタイム診断 (Realtime)"])

if mode == "📊 市場スキャン (Batch)":
    if df_all.empty: st.error("データ未検出")
    else:
        latest_date = df_all["date"].max()
        latest_df = df_all[df_all["date"] == latest_date].copy().drop_duplicates(subset=["ticker"])
        
        st.markdown(f"### 🤖 SENTINEL Briefing: {latest_date}")
        
        # キャッシュキーに日付を含めて毎日更新させる
        report_key = f"report_{latest_date}"
        if report_key not in st.session_state:
            with st.spinner(f"{latest_date} のデータを解析中..."):
                spy_news = fetch_fresh_news("SPY")
                action_list = latest_df[latest_df['status']=='ACTION']['ticker'].tolist()
                
                prompt = f"""
                あなたはプロ投資戦略AI「SENTINEL」です。
                【現在時刻】{TODAY_STR}
                【解析対象日】{latest_date}
                
                【対象日の市場ニュース(SPY)】\n{spy_news}
                【スキャン結果】\n- ACTION銘柄: {', '.join(action_list[:5])}\n- VCP平均: {latest_df['vcp_score'].mean():.1f}
                
                【指示】
                解析対象日({latest_date})の市場を800文字以上で論理的に解説してください。
                1. 市況判断 2. セクター動向 3. 具体的戦略。
                24時間以内に発生したニュースを最優先し、古い雇用統計などのノイズは完全に無視してください。
                """
                st.session_state[report_key] = call_gemini(prompt)

        st.markdown(f"""<div class="ai-report">{st.session_state[report_key]}</div>""", unsafe_allow_html=True)
        st.dataframe(latest_df[["ticker", "status", "price", "vcp_score", "sector"]].style.background_gradient(subset=["vcp_score"], cmap="Greens"), use_container_width=True)

elif mode == "🔍 リアルタイム診断 (Realtime)":
    st.subheader("Realtime Ticker Analyzer 🤖")
    ticker_input = st.text_input("ティッカーを入力 (例: WDC, ITRN)", key="realtime_ticker").upper()
    
    if st.button("診断開始 🚀", type="primary") and ticker_input:
        with st.spinner(f"{ticker_input} の「今」を解析中..."):
            try:
                t_obj = yf.Ticker(ticker_input)
                data = t_obj.history(period="1y", auto_adjust=True)
                news = fetch_fresh_news(ticker_input)
                
                if data.empty: st.error("データ取得不可")
                else:
                    vcp = VCPAnalyzer.calculate(data)
                    price = data["Close"].iloc[-1]
                    
                    prompt = f"""
                    あなたはウォール街のプロAI「SENTINEL」です。【{ticker_input}】をリアルタイム診断します。
                    【現在時刻】{TODAY_STR}
                    
                    【直近24時間のニュース】\n{news}
                    【テクニカル】\n- VCPスコア: {vcp['score']}/100\n- 変動幅: {vcp['raw']['range']:.2%}\n- 出来高比: {vcp['raw']['vol']:.2f}\n- シグナル: {vcp['signals']}
                    
                    【指示】
                    現在の状況を800文字以上で論理的に分析し、「BUY」「WAIT」「PASS」を断言してください。
                    VCPスコアが高いほど買いです。直近の急騰による一時的なスコア低下は「ふるい落とし」として解釈してください。
                    """
                    report = call_gemini(prompt)
                    st.markdown(f"""<div class="ai-individual"><h5>🤖 SENTINEL Deep Diagnosis</h5>{report}</div>""", unsafe_allow_html=True)
                    
                    # メトリクス表示
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Price", f"${price:.2f}")
                    c2.metric("VCP Score", f"{vcp['score']}")
                    c3.metric("Range (10d)", f"{vcp['raw']['range']:.1%}")
                    c4.metric("Signals", ", ".join(vcp['signals']) if vcp['signals'] else "None")
                    
                    st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index[-60:], open=data['Open'][-60:], high=data['High'][-60:], low=data['Low'][-60:], close=data['Close'][-60:])]).update_layout(template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")

st.markdown("---")
st.caption(f"System Time: {TODAY_STR} | Powered by SENTINEL PRO ELITE")
