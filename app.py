import streamlit as st
import pandas as pd
import json
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
import os
import warnings

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

# カスタムCSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E; border: 1px solid #333; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;
    }
    .ai-report {
        background-color: #0E1117; border-left: 5px solid #4B91F1; padding: 20px; margin-bottom: 20px; border-radius: 5px;
    }
    .ai-individual {
        background-color: #1c2333; border: 1px solid #4B91F1; padding: 15px; border-radius: 8px; margin-top: 10px;
    }
    .stProgress > div > div > div > div { background-color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 分析ロジック (VCP & Backtest)
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

# ==============================================================================
# 📂 データ読み込み
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

df_history, meta_history = load_historical_json()

# ==============================================================================
# 🐳 AIエンジン (DeepSeek-R1 Reasoner)
# ==============================================================================

def call_deepseek_r1(prompt):
    """DeepSeek API呼び出し共通関数"""
    api_key = None
    try: api_key = st.secrets["DEEPSEEK_API_KEY"]
    except: api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key: return "⚠️ APIキー未設定 (Secrets: DEEPSEEK_API_KEY)"

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner", # DeepSeek-R1
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e: return f"DeepSeek Error: {str(e)}"

# ==============================================================================
# 🖥️ UI構成
# ==============================================================================

st.title("🛡️ SENTINEL PRO DASHBOARD")
mode = st.sidebar.radio("モード選択", ["📊 市場レポート (Batch)", "🔍 個別銘柄診断 (Realtime)"])

# ------------------------------------------------------------------------------
# MODE 1: 市場レポート
# ------------------------------------------------------------------------------
if mode == "📊 市場レポート (Batch)":
    if df_history.empty:
        st.error("データが見つかりません。")
    else:
        latest_date = df_history["date"].max()
        latest_df = df_history[df_history["date"] == latest_date].copy()
        latest_df = latest_df.drop_duplicates(subset=["ticker"])
        
        # --- 🐳 Market Report ---
        st.markdown(f"### 🐳 SENTINEL AI Briefing (DeepSeek-R1)")
        
        if "market_ai" not in st.session_state:
            with st.spinner("DeepSeek-R1が市場を推論中...🧠"):
                action_stocks = latest_df[latest_df['status']=='ACTION']['ticker'].tolist()
                wait_stocks = latest_df[latest_df['status']=='WAIT']['ticker'].tolist()
                top_sector = latest_df['sector'].value_counts().idxmax() if not latest_df.empty else "None"
                avg_vcp = latest_df['vcp_score'].mean() if not latest_df.empty else 0
                
                prompt = f"""
                あなたはヘッジファンドAI「SENTINEL」です。以下の市場データを分析し、相場環境と戦略を解説してください。
                【市場データ】
                - 監視: 450銘柄
                - ACTION: {len(action_stocks)}銘柄 ({', '.join(action_stocks[:5]) if action_stocks else 'なし'})
                - WAIT: {len(wait_stocks)}銘柄
                - 主導セクター: {top_sector}
                - 平均VCP: {avg_vcp:.1f} (基準55)
                【出力】
                200文字以内の日本語で、相場フェーズ（強気/調整）と投資家の取るべき行動を断言してください。
                """
                st.session_state.market_ai = call_deepseek_r1(prompt)
        
        st.markdown(f"""<div class="ai-report"><h4>🎙️ Market Analyst</h4><p style="font-size: 1.1em;">{st.session_state.market_ai}</p></div>""", unsafe_allow_html=True)

        # セクターマップ
        st.markdown("### 🗺️ Sector Heatmap")
        if not latest_df.empty:
            fig_treemap = px.treemap(latest_df, path=['sector', 'ticker'], values='vcp_score', color='rs', color_continuous_scale='RdYlGn', title="セクター別ヒートマップ")
            st.plotly_chart(fig_treemap, use_container_width=True)

        # リスト
        st.markdown("### 📋 Scan Results")
        filter_status = st.multiselect("ステータス", options=["ACTION", "WAIT"], default=["ACTION", "WAIT"])
        show_df = latest_df[latest_df["status"].isin(filter_status)]
        st.dataframe(show_df[["ticker", "status", "price", "rs", "vcp_score", "pf", "sector"]].style.background_gradient(subset=["vcp_score"], cmap="Greens"), use_container_width=True)

# ------------------------------------------------------------------------------
# MODE 2: 個別銘柄診断 (AI搭載)
# ------------------------------------------------------------------------------
elif mode == "🔍 個別銘柄診断 (Realtime)":
    st.subheader("Realtime Ticker Analyzer 🤖")
    col_input, col_btn = st.columns([3, 1])
    with col_input: ticker_input = st.text_input("ティッカー (例: TSLA)", value="").upper()
    with col_btn: 
        st.write(""); st.write("")
        analyze_btn = st.button("診断開始 🚀", type="primary")

    if analyze_btn and ticker_input:
        with st.spinner(f"{ticker_input} を詳細分析中..."):
            try:
                data = yf.download(ticker_input, period="2y", interval="1d", progress=False, auto_adjust=True)
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                if data.empty or len(data) < 200: st.error("データ不足です。")
                else:
                    vcp_res = VCPAnalyzer.calculate(data)
                    pf_res = StrategyValidator.run_backtest(data)
                    try: info = yf.Ticker(ticker_input).info; sector = info.get("sector", "Unknown"); price = data["Close"].iloc[-1]
                    except: sector="Unknown"; price=0

                    # --- 🐳 Individual AI Diagnosis ---
                    prompt_ind = f"""
                    あなたはプロのトレーダーAIです。以下の銘柄を技術的に診断し、「買い」「様子見」「パス」のいずれかを判定してください。
                    【銘柄】{ticker_input} (${price:.2f}) - {sector}
                    【テクニカル指標】
                    - VCPスコア: {vcp_res['score']}/100 (高いほど良い。55以上が基準)
                    - シグナル: {', '.join(vcp_res['signals']) if vcp_res['signals'] else '特になし'}
                    - 直近トレンドの強さ(PF): {pf_res:.2f} (1.1以上で合格)
                    【出力】
                    150文字以内の日本語で、テクニカル根拠に基づいたアドバイスをしてください。甘口ではなく辛口でお願いします。
                    """
                    ai_comment = call_deepseek_r1(prompt_ind)

                    # 結果表示
                    st.markdown("---")
                    
                    # AIコメント表示エリア
                    st.markdown(f"""
                    <div class="ai-individual">
                        <h5>🤖 DeepSeek-R1 Diagnosis</h5>
                        <p style="font-size: 1.1em; font-weight: bold;">{ai_comment}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"${price:.2f}")
                    c2.metric("VCP Score", f"{vcp_res['score']}/100")
                    c3.metric("Profit Factor", f"{pf_res:.2f}")
                    c4.metric("Sector", sector)

                    col_g, col_c = st.columns([1, 2])
                    with col_g:
                        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=vcp_res['score'], title={'text': "Sentinel Score"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00ff00" if vcp_res['score']>70 else "#f1c40f"}, 'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 55}}))
                        fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col_c:
                        chart_df = data.iloc[-126:]
                        fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name=ticker_input)])
                        fig.update_layout(title=f"{ticker_input} Daily Chart", template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                    if not df_history.empty:
                        sector_peers = df_history[df_history["sector"] == sector]
                        if not sector_peers.empty:
                            avg_vcp = sector_peers["vcp_score"].mean()
                            fig_radar = go.Figure()
                            fig_radar.add_trace(go.Scatterpolar(r=[vcp_res['score'], pf_res*20], theta=['VCP', 'PF'], fill='toself', name=ticker_input, line_color='#00FF00'))
                            fig_radar.add_trace(go.Scatterpolar(r=[avg_vcp, sector_peers["pf"].mean()*20], theta=['VCP', 'PF'], fill='toself', name='Sector Avg', line_color='#666666', line_dash='dash'))
                            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark", title="VS Sector Avg")
                            st.plotly_chart(fig_radar, use_container_width=True)

            except Exception as e: st.error(f"Error: {e}")

st.markdown("---")
st.caption("Powered by SENTINEL PRO ELITE & DeepSeek-R1")
