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
        border-left: 5px solid #4285F4; 
        padding: 20px; 
        margin-bottom: 20px; 
        border-radius: 5px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .ai-individual {
        background-color: #1c2333; 
        border: 1px solid #4285F4; 
        padding: 15px; 
        border-radius: 8px; 
        margin-top: 10px;
    }
    .stProgress > div > div > div > div { background-color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 分析ロジック
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
            tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.10))
            
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

df_history, meta_history = load_historical_json()

# ==============================================================================
# 🤖 AIエンジン (Gemini 2.0 Flash - 修正版)
# ==============================================================================

def call_gemini_ai(prompt, use_search=False):
    """Google Gemini API呼び出し (Grounding修正版)"""
    api_key = None
    try: api_key = st.secrets["GEMINI_API_KEY"]
    except: api_key = os.getenv("GEMINI_API_KEY")

    if not api_key: return "⚠️ APIキー未設定 (Secrets: GEMINI_API_KEY)"

    try:
        genai.configure(api_key=api_key)
        
        # ツール設定: 正しいフォーマット [{"google_search": {}}] を使用
        tools = []
        if use_search:
            tools = [{"google_search": {}}]
        
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            tools=tools
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e: return f"Gemini Error: {str(e)}"

# ==============================================================================
# 🖥️ UI構成
# ==============================================================================

st.title("🛡️ SENTINEL PRO DASHBOARD")
st.caption("Powered by Google Gemini 2.0 Flash")

mode = st.sidebar.radio("モード選択", ["📊 市場レポート (Batch)", "🔍 個別銘柄診断 (Realtime)"])

if mode == "📊 市場レポート (Batch)":
    if df_history.empty:
        st.error("データが見つかりません。")
    else:
        latest_date = df_history["date"].max()
        latest_df = df_history[df_history["date"] == latest_date].copy()
        latest_df = latest_df.drop_duplicates(subset=["ticker"])
        
        st.markdown(f"### 🤖 SENTINEL AI Briefing")
        
        if "market_ai_gemini" not in st.session_state:
            with st.spinner("GeminiがGoogle検索と内部データを統合分析中...🌍"):
                action_list = latest_df[latest_df['status']=='ACTION']['ticker'].tolist()
                wait_list = latest_df[latest_df['status']=='WAIT']['ticker'].tolist()
                top_sector = latest_df['sector'].value_counts().idxmax() if not latest_df.empty else "None"
                avg_vcp = latest_df['vcp_score'].mean() if not latest_df.empty else 0
                today_str = datetime.date.today().strftime("%Y-%m-%d")

                prompt = f"""
                あなたはウォール街のトップストラテジストAI「SENTINEL」です。
                【内部スキャンデータ ({latest_date})】とGoogle検索による【最新市況】を統合し、
                日本の個人投資家向けに具体的かつ洞察に満ちた市場レポートを作成してください。

                【内部データ】
                - ACTION (即エントリー): {len(action_list)}銘柄 ({', '.join(action_list[:5]) if action_list else 'なし'})
                - WAIT (監視): {len(wait_list)}銘柄
                - 主導セクター: {top_sector}
                - 市場VCP平均: {avg_vcp:.1f} (基準55)

                【要件】
                今日({today_str})の米国市場重要ニュース(FOMC, 経済指標等)を検索・加味し、
                内部データの数値と関連付けて分析してください。

                【フォーマット】
                マークダウン形式、**400〜600文字**。
                ### 1. Market Overview (市況概況)
                ### 2. Sector Rotation (資金の流れ)
                ### 3. Sentinel Strategy (投資戦略)
                """
                st.session_state.market_ai_gemini = call_gemini_ai(prompt, use_search=True)
        
        st.markdown(f"""<div class="ai-report">{st.session_state.market_ai_gemini}</div>""", unsafe_allow_html=True)

        st.markdown("### 🗺️ Sector Heatmap")
        if not latest_df.empty:
            fig_treemap = px.treemap(latest_df, path=['sector', 'ticker'], values='vcp_score', color='rs', color_continuous_scale='RdYlGn', title="セクター別ヒートマップ")
            st.plotly_chart(fig_treemap, use_container_width=True)

        st.markdown("### 📋 Scan Results")
        filter_status = st.multiselect("ステータス", options=["ACTION", "WAIT"], default=["ACTION", "WAIT"])
        show_df = latest_df[latest_df["status"].isin(filter_status)]
        st.dataframe(show_df[["ticker", "status", "price", "rs", "vcp_score", "pf", "sector"]].style.background_gradient(subset=["vcp_score"], cmap="Greens"), use_container_width=True)

elif mode == "🔍 個別銘柄診断 (Realtime)":
    st.subheader("Realtime Ticker Analyzer 🤖")
    col_input, col_btn = st.columns([3, 1])
    with col_input: ticker_input = st.text_input("ティッカー (例: TSLA)", value="").upper()
    with col_btn: 
        st.write(""); st.write("")
        analyze_btn = st.button("診断開始 🚀", type="primary")

    if analyze_btn and ticker_input:
        with st.spinner(f"{ticker_input} をGeminiが徹底分析中..."):
            try:
                data = yf.download(ticker_input, period="2y", interval="1d", progress=False, auto_adjust=True)
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                if data.empty or len(data) < 200: st.error("データ不足です。")
                else:
                    vcp_res = VCPAnalyzer.calculate(data)
                    pf_res = StrategyValidator.run_backtest(data)
                    try: info = yf.Ticker(ticker_input).info; sector = info.get("sector", "Unknown"); price = data["Close"].iloc[-1]
                    except: sector="Unknown"; price=0

                    prompt_ind = f"""
                    あなたはプロトレーダーAIです。以下の銘柄を診断し、投資判断を下してください。
                    Google検索で「{ticker_input} stock news」を検索し、直近の材料も加味してください。

                    【銘柄】{ticker_input} (${price:.2f}) - {sector}
                    【テクニカル】VCP:{vcp_res['score']}/100, PF:{pf_res:.2f}, Signal:{vcp_res['signals']}

                    【出力要件】
                    - 結論: 「BUY」「WAIT」「PASS」のいずれかを太字で。
                    - 分析: テクニカルとニュースの両面から、なぜその結論なのかを300文字程度で辛口解説。
                    """
                    ai_comment = call_gemini_ai(prompt_ind, use_search=True)

                    st.markdown("---")
                    st.markdown(f"""<div class="ai-individual"><h5>🤖 Gemini 2.0 Diagnosis</h5><div style="font-size: 1.05em;">{ai_comment}</div></div>""", unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"${price:.2f}")
                    c2.metric("VCP Score", f"{vcp_res['score']}/100")
                    c3.metric("Profit Factor", f"{pf_res:.2f}")
                    c4.metric("Sector", sector)

                    col_g, col_c = st.columns([1, 2])
                    with col_g:
                        categories = ['VCP Score', 'Profit Factor', 'RS (Strength)']
                        if not df_history.empty:
                            peers = df_history[df_history["sector"]==sector]
                            avg_vcp = peers["vcp_score"].mean() if not peers.empty else 50
                            avg_pf = peers["pf"].mean() if not peers.empty else 1.0
                            avg_rs = peers["rs"].mean() if not peers.empty else 50
                        else: avg_vcp, avg_pf, avg_rs = 50, 1.0, 50
                        
                        hist_data = df_history[df_history["ticker"] == ticker_input]
                        if not hist_data.empty: my_rs = hist_data.iloc[0]["rs"]
                        else:
                            try:
                                y_low = data["Low"].min(); y_high = data["High"].max()
                                my_rs = ((price - y_low)/(y_high - y_low))*100
                            except: my_rs = 50

                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(r=[vcp_res['score'], min(100, pf_res*20), my_rs], theta=categories, fill='toself', name=ticker_input, line_color='#00FF00'))
                        fig_radar.add_trace(go.Scatterpolar(r=[avg_vcp, min(100, avg_pf*20), avg_rs], theta=categories, fill='toself', name='Sector Avg', line_color='#666666', line_dash='dash'))
                        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark", height=300, margin=dict(l=30,r=30,t=30,b=30))
                        st.plotly_chart(fig_radar, use_container_width=True)

                    with col_c:
                        chart_df = data.iloc[-126:]
                        fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name=ticker_input)])
                        fig.update_layout(title=f"{ticker_input} Daily Chart", template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e: st.error(f"Error: {e}")

st.markdown("---")
st.caption("Powered by SENTINEL PRO ELITE & Google Gemini")
