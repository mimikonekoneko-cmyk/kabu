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
import feedparser # RSSパース用 (pip install feedparser)

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
        padding: 20px; 
        margin-bottom: 20px; 
        border-radius: 5px;
        font-family: 'Helvetica Neue', sans-serif;
        line-height: 1.8;
    }
    .ai-individual {
        background-color: #1c2333; 
        border: 1px solid #00FF00; 
        padding: 25px; 
        border-radius: 12px; 
        margin-top: 10px;
        line-height: 1.8;
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
            # 収縮判定: 数値が高いほど優秀
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

# ==============================================================================
# 🛰️ 合法的・非侵襲的ニュース収集エンジン
# ==============================================================================

def fetch_safe_news(ticker):
    """yfinanceとGoogle RSSを組み合わせた安全なニュース収集"""
    headlines = []
    
    # 1. yfinanceからの取得 (公式API準拠)
    try:
        yf_news = yf.Ticker(ticker).news
        for n in (yf_news or [])[:5]:
            headlines.append(f"- {n.get('headline', n.get('title', 'No Title'))}")
    except: pass
    
    # 2. Google News RSSからの取得 (配信規格準拠)
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:7d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            if f"- {entry.title}" not in headlines:
                headlines.append(f"- {entry.title}")
    except: pass
    
    return "\n".join(headlines) if headlines else "※現在、最新ニュースを外部確認中...（自動取得制限あり）"

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
        safety_settings = {category: HarmBlockThreshold.BLOCK_NONE for category in [
            HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
        ]}
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e: return f"Gemini Error: {str(e)}"

# ==============================================================================
# 🖥️ UI構成
# ==============================================================================

df_history, meta_history = load_historical_json() # 既存のロード関数

mode = st.sidebar.radio("モード選択", ["📊 市場レポート (Batch)", "🔍 個別銘柄診断 (Realtime)"])

if mode == "📊 市場レポート (Batch)":
    # 既存の市場レポートロジック (Geminiプロンプトのみ500文字指定に強化)
    pass 

elif mode == "🔍 個別銘柄診断 (Realtime)":
    st.subheader("Realtime Ticker Analyzer 🤖")
    ticker_input = st.text_input("ティッカー (例: WDC)", value="").upper()
    if st.button("診断開始 🚀", type="primary") and ticker_input:
        with st.spinner(f"{ticker_input} を深層分析中..."):
            try:
                # 1. データとニュースの取得
                data = yf.Ticker(ticker_input).history(period="2y", auto_adjust=True)
                news_context = fetch_safe_news(ticker_input)
                
                if data.empty: st.error("銘柄が見つかりません。")
                else:
                    vcp = VCPAnalyzer().calculate(data)
                    price = data["Close"].iloc[-1]
                    
                    # 2. AIプロンプトの構築 (WDCの自社株買い等の文脈を意識)
                    prompt = f"""
                    あなたはウォール街の伝説的投資家AI「SENTINEL」です。
                    
                    【銘柄情報】 {ticker_input} (${price:.2f})
                    【最新ニュース】
                    {news_context}
                    
                    【テクニカルデータ】
                    - VCPスコア: {vcp['score']} / 100
                    - 特徴: {', '.join(vcp['signals']) if vcp['signals'] else '収縮待ち'}
                    
                    【最重要ルール】
                    1. スコアが高いほど「買い推奨」です。低いスコアを無理に褒めないでください。
                    2. 直近で大きな材料（自社株買い等）があった場合、一時的にチャートが荒れてVCPスコアが下がることがあります。これは「エネルギーの再充電」や「ふるい落とし」の過程であることを考慮してください。
                    3. ニュースに「Buyback(自社株買い)」等があれば、そのインパクトを重視してください。
                    
                    【指示】
                    現在の「仕上がり具合」をプロの視点で800文字程度で論理的に解説してください。
                    結論は「BUY」「WAIT」「PASS」を太字で示してください。
                    """
                    ai_report = call_gemini_pure(prompt)
                    
                    # 3. 表示
                    st.markdown(f"""<div class="ai-individual"><h5>🤖 SENTINEL Deep Diagnosis</h5>{ai_report}</div>""", unsafe_allow_html=True)
                    
                    # レーダーチャート (3角形)
                    # (以下、以前のコードと同じ描画ロジック)
            except Exception as e: st.error(f"Error: {e}")
