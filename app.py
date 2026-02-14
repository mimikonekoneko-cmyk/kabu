import json
import os
import time
import warnings
import datetime
import pickle
import requests
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import feedparser
from openai import OpenAI

# スクレイピング用ライブラリ
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

# ==============================================================================
# 設定・定数
# ==============================================================================

NOW = datetime.datetime.now()
TODAY_STR = NOW.strftime("%Y-%m-%d")
CACHE_DIR = Path("./cache_v45"); CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results"); RESULTS_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE = Path("watchlist.json")
PORTFOLIO_FILE = Path("portfolio.json")

# ニュース設定
NEWS_CONFIG = {"FETCH_TIMEOUT": 6, "MAX_CHARS": 400, "CACHE_TTL": 3600}

# エグジット設定
EXIT_CFG = {
    "STOP_LOSS_ATR_MULT": 2.0,
    "TARGET_R_MULT":      2.5,
    "TRAIL_START_R":      1.5,
    "TRAIL_ATR_MULT":     1.5,
    "SCALE_OUT_R":        1.5,
}

warnings.filterwarnings("ignore")

# ==============================================================================
# エンジン群
# ==============================================================================

class CurrencyEngine:
    @staticmethod
    def get_usd_jpy():
        """リアルタイムに近い為替レートを取得"""
        try:
            # 1分足等の短い足ではなく、直近の気配値を取得
            ticker = yf.Ticker("JPY=X")
            # fast_infoから取得を試みる
            price = ticker.fast_info.get('lastPrice')
            if price is None:
                # 取得できなければ履歴から
                df = ticker.history(period="1d")
                price = float(df["Close"].iloc[-1]) if not df.empty else 150.0
            return round(price, 2)
        except:
            return 150.00

class DataEngine:
    @staticmethod
    def get_data(ticker, period):
        """チャート表示用にデータを整形して取得"""
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period)
            if df is None or df.empty:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return None

            # MultiIndex解消とカラム名統一
            if isinstance(df.columns, pd.MultiIndex):
                # Closeが含まれるレベルを探す
                target_level = None
                for i in range(df.columns.nlevels):
                    if 'Close' in df.columns.get_level_values(i):
                        df.columns = df.columns.get_level_values(i)
                        target_level = i
                        break
                if target_level is None:
                    df.columns = df.columns.get_level_values(0)

            # カラム名のクリーニング
            new_cols = []
            for c in df.columns:
                s_c = str(c)
                # タプルの場合
                if isinstance(c, tuple):
                    s_c = str(c[0])
                
                if s_c.lower() in ['open', 'high', 'low', 'close', 'volume']:
                    new_cols.append(s_c.capitalize())
                elif s_c.lower() in ['adj close', 'adjclose']:
                    new_cols.append('Close')
                else:
                    new_cols.append(s_c)
            df.columns = new_cols

            # インデックスのタイムゾーン削除
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            else:
                df.index = pd.to_datetime(df.index)

            # 必須カラムチェックと補完
            if 'Close' in df.columns:
                if 'Open' not in df.columns: df['Open'] = df['Close']
                if 'High' not in df.columns: df['High'] = df['Close']
                if 'Low' not in df.columns: df['Low'] = df['Close']
                if 'Volume' not in df.columns: df['Volume'] = 0
            else:
                return None

            # 数値型変換
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df.dropna(subset=['Close'], inplace=True)
            return df
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return None

    @staticmethod
    def get_current_price(ticker):
        try:
            t = yf.Ticker(ticker)
            p = t.fast_info.get('lastPrice')
            if p is None:
                h = t.history(period="1d")
                if not h.empty: p = h["Close"].iloc[-1]
            return float(p) if p else 0.0
        except: return 0.0
    
    @staticmethod
    def get_atr(ticker): return 1.5

    @staticmethod
    def get_market_overview():
        try:
            spy = yf.Ticker("SPY").history(period="5d")
            vix = yf.Ticker("^VIX").history(period="1d")
            spy_p = spy["Close"].iloc[-1] if not spy.empty else 0
            spy_chg = (spy_p / spy["Close"].iloc[-2] - 1) * 100 if len(spy) >= 2 else 0
            vix_p = vix["Close"].iloc[-1] if not vix.empty else 0
            return {"spy": spy_p, "spy_change": spy_chg, "vix": vix_p}
        except: return {"spy": 0, "spy_change": 0, "vix": 0}

class FundamentalEngine:
    CACHE_TTL = 24 * 3600
    @staticmethod
    def get(ticker: str) -> dict:
        cf = CACHE_DIR / f"fund_{ticker}.json"
        if cf.exists() and (time.time() - cf.stat().st_mtime < FundamentalEngine.CACHE_TTL):
            try:
                with open(cf) as f: return json.load(f)
            except: pass
        try:
            i = yf.Ticker(ticker).info
            d = {
                "analyst_target": i.get("targetMeanPrice"),
                "sector": i.get("sector", "Unknown"),
                "industry": i.get("industry", "Unknown"),
                "market_cap": i.get("marketCap"),
                "pe_forward": i.get("forwardPE")
            }
            with open(cf, "w") as f: json.dump(d, f, default=str)
            return d
        except: return {}

class NewsEngine:
    @staticmethod
    def get(ticker: str) -> dict:
        cf = CACHE_DIR / f"news_{ticker}.json"
        if cf.exists() and (time.time() - cf.stat().st_mtime < NEWS_CONFIG["CACHE_TTL"]):
            try:
                with open(cf) as f: return json.load(f)
            except: pass
        
        articles = []
        seen = set()
        # 1. YFinance
        try:
            for n in (yf.Ticker(ticker).news or [])[:3]:
                t = n.get("title", "")
                if t and t not in seen:
                    seen.add(t)
                    articles.append({"title": t, "url": n.get("link", ""), "body": ""})
        except: pass
        # 2. Google RSS
        try:
            f = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}+stock+when:3d&hl=en-US&gl=US&ceid=US:en")
            for e in f.entries[:3]:
                if e.title not in seen:
                    seen.add(e.title)
                    articles.append({"title": e.title, "url": getattr(e, "link", ""), "body": ""})
        except: pass
        # 3. Scraping
        if BS4_OK:
            for a in articles[:3]:
                if not a["url"]: continue
                try:
                    r = requests.get(a["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=NEWS_CONFIG["FETCH_TIMEOUT"])
                    s = BeautifulSoup(r.text, "html.parser")
                    # 本文抽出ロジック（簡易）
                    ps = [p.get_text().strip() for p in s.find_all("p") if len(p.get_text().strip()) > 50]
                    a["body"] = " ".join(ps)[:NEWS_CONFIG["MAX_CHARS"]]
                except: pass
        
        res = {"articles": articles[:5]}
        with open(cf, "w") as f: json.dump(res, f, ensure_ascii=False)
        return res

    @staticmethod
    def get_general_market() -> dict:
        cf = CACHE_DIR / "news_market_general.json"
        if cf.exists() and (time.time() - cf.stat().st_mtime < NEWS_CONFIG["CACHE_TTL"]):
            try:
                with open(cf) as f: return json.load(f)
            except: pass
        
        articles = []
        try:
            f = feedparser.parse("https://news.google.com/rss/search?q=stock+market+news+when:1d&hl=en-US&gl=US&ceid=US:en")
            for e in f.entries[:5]:
                articles.append({"title": e.title, "url": getattr(e, "link", "")})
        except: pass
        res = {"articles": articles}
        with open(cf, "w") as f: json.dump(res, f, ensure_ascii=False)
        return res

    @staticmethod
    def format_for_prompt(news: dict) -> str:
        lines = []
        for a in news.get("articles", []):
            lines.append(f"• {a['title']} (URL: {a['url']})")
            if a.get("body"): lines.append(f"  内容: {a['body']}...")
        return "\n".join(lines) if lines else "特になし"

# ==============================================================================
# 分析ロジック
# ==============================================================================

class VCPAnalyzer:
    @staticmethod
    def calculate(df: pd.DataFrame) -> dict:
        # 省略せず実装
        try:
            if df is None or len(df) < 130: return VCPAnalyzer._empty()
            c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]
            tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            
            # Tightness
            periods = [20, 30, 40, 60]
            rngs = []
            for p in periods:
                ph = float(h.iloc[-p:].max()); pl = float(l.iloc[-p:].min())
                rngs.append((ph-pl)/ph if ph>0 else 1.0)
            avg_rng = np.mean(rngs[:3])
            is_contracting = rngs[0] < rngs[1] < rngs[2]
            
            t_sc = 40 if avg_rng < 0.10 else (30 if avg_rng < 0.15 else (20 if avg_rng < 0.20 else (10 if avg_rng < 0.28 else 0)))
            if is_contracting: t_sc += 5
            t_sc = min(40, t_sc)

            # Volume
            v20 = float(v.iloc[-20:].mean()); v60 = float(v.iloc[-60:-40].mean())
            v_rat = v20/v60 if v60>0 else 1.0
            v_sc = 30 if v_rat < 0.45 else (25 if v_rat < 0.60 else (15 if v_rat < 0.75 else 0))
            is_dry = v_rat < 0.75

            # MA
            ma50 = float(c.rolling(50).mean().iloc[-1]); ma150 = float(c.rolling(150).mean().iloc[-1]); ma200 = float(c.rolling(200).mean().iloc[-1])
            price = float(c.iloc[-1])
            m_sc = 0
            if price > ma50: m_sc += 10
            if ma50 > ma150: m_sc += 10
            if ma150 > ma200: m_sc += 10

            # Pivot
            piv = float(h.iloc[-50:].max()); dist = (piv - price)/piv
            p_bon = 5 if 0 <= dist <= 0.04 else (3 if 0.04 < dist <= 0.08 else 0)

            score = min(105, t_sc + v_sc + m_sc + p_bon)
            return {"score": score, "atr": atr, "breakdown": {"tight": t_sc, "vol": v_sc, "ma": m_sc, "pivot": p_bon}}
        except: return VCPAnalyzer._empty()

    @staticmethod
    def _empty(): return {"score": 0, "atr": 0.0, "breakdown": {"tight": 0, "vol": 0, "ma": 0, "pivot": 0}}

class RSAnalyzer:
    @staticmethod
    def get_raw_score(df: pd.DataFrame) -> float:
        try:
            c = df["Close"]
            if len(c) < 252: return 0.0
            r12 = c.iloc[-1]/c.iloc[-252]-1; r6 = c.iloc[-1]/c.iloc[-126]-1
            r3 = c.iloc[-1]/c.iloc[-63]-1; r1 = c.iloc[-1]/c.iloc[-21]-1
            return r12*0.4 + r6*0.2 + r3*0.2 + r1*0.2
        except: return 0.0

class StrategyValidator:
    @staticmethod
    def run(df: pd.DataFrame) -> float:
        try:
            # 簡易バックテスト
            if len(df) < 200: return 1.0
            return 1.5 # スタブ（処理負荷軽減のため固定値を返すか、必要なら実装）
        except: return 1.0

# ==============================================================================
# UI ヘルパー
# ==============================================================================

def draw_sentinel_grid_ui(metrics: List[Dict[str, Any]]):
    html = '<div class="sentinel-grid">'
    for m in metrics:
        delta = ""
        if m.get("delta"):
            col = "#3fb950" if "+" in str(m["delta"]) else "#f85149"
            delta = f'<div class="sentinel-delta" style="color:{col}">{m["delta"]}</div>'
        html += f'<div class="sentinel-card"><div class="sentinel-label">{m["label"]}</div><div class="sentinel-value">{m["value"]}</div>{delta}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def load_portfolio_json() -> dict:
    default = {"positions": {}, "cash": {"jpy": 350000, "usd": 0}} # デフォルト現金
    if not PORTFOLIO_FILE.exists(): return default
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            d = json.load(f)
            if "cash" not in d: d["cash"] = {"jpy": 350000, "usd": 0}
            return d
    except: return default

def save_portfolio_json(data: dict):
    with open(PORTFOLIO_FILE, "w") as f: json.dump(data, f, indent=2)

def load_watchlist():
    if not WATCHLIST_FILE.exists(): return []
    try:
        with open(WATCHLIST_FILE, "r") as f: return json.load(f)
    except: return []

def save_watchlist(data):
    with open(WATCHLIST_FILE, "w") as f: json.dump(data, f)

# ==============================================================================
# UI Styles
# ==============================================================================

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #0d1117; color: #f0f6fc; }
.block-container { padding-top: 0rem !important; }
.ui-push-buffer { height: 60px; }
.stTabs [data-baseweb="tab-list"] { background-color: #161b22; padding: 10px; border-radius: 10px; border-bottom: 2px solid #30363d; gap: 10px; }
.stTabs [data-baseweb="tab"] { color: #8b949e; border: none; font-weight: 700; }
.stTabs [aria-selected="true"] { color: #fff; background-color: #238636; border-radius: 8px; }
.sentinel-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }
@media(min-width: 900px){ .sentinel-grid { grid-template-columns: repeat(4, 1fr); } }
.sentinel-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }
.sentinel-label { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; font-weight: 600; }
.sentinel-value { font-size: 1.5rem; font-weight: 700; color: #f0f6fc; margin-top: 5px; }
.section-header { font-size: 1.25rem; font-weight: 700; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin: 30px 0 20px; }
.pos-card { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 15px; border-left: 8px solid #30363d; }
.pos-card.profit { border-left-color: #3fb950; }
.pos-card.loss { border-left-color: #f85149; }
</style>
"""

# ==============================================================================
# MAIN APP
# ==============================================================================

st.set_page_config(page_title="SENTINEL PRO", layout="wide", initial_sidebar_state="collapsed")
st.markdown(STYLE, unsafe_allow_html=True)

# State Init
if "target_ticker" not in st.session_state: st.session_state.target_ticker = ""
if "ai_text" not in st.session_state: st.session_state.ai_text = ""

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ SENTINEL V6.0")
    wl = load_watchlist()
    for t in wl:
        c1, c2 = st.columns([4,1])
        if c1.button(t, key=f"side_{t}"):
            st.session_state.target_ticker = t
        if c2.button("×", key=f"del_{t}"):
            wl.remove(t)
            save_watchlist(wl)
            st.rerun()

# Tabs
tabs = st.tabs(["📊 MARKET", "🔍 AI DIAGNOSIS", "💼 PORTFOLIO"])
usd_jpy = CurrencyEngine.get_usd_jpy()

# --- TAB 1: MARKET ---
with tabs[0]:
    st.markdown(f'<div class="section-header">MARKET OVERVIEW (USD/JPY: ¥{usd_jpy:.2f})</div>', unsafe_allow_html=True)
    m_ctx = DataEngine.get_market_overview()
    
    # AI Market Analysis Button
    if st.button("🤖 ANALYZE MARKET CONDITIONS", use_container_width=True, type="primary"):
        k = st.secrets.get("DEEPSEEK_API_KEY")
        if not k: st.error("No API Key")
        else:
            with st.spinner("Analyzing..."):
                news = NewsEngine.get_general_market()
                n_txt = NewsEngine.format_for_prompt(news)
                p = f"""あなたは「AI投資家SENTINEL」。
現在日時: {TODAY_STR}
SPY: ${m_ctx['spy']:.2f}, VIX: {m_ctx['vix']:.2f}
ニュース:
{n_txt}
指示:
1. 現在の市場環境（強気/弱気/調整）を定義せよ。
2. ニュースから重要材料を抽出せよ。未来の日付は無視。
3. 推奨ポジション比率を提示せよ。
4. 600字以内。文末に「最終判断: [BULL/BEAR/NEUTRAL]」を記述。
"""
                try:
                    cl = OpenAI(api_key=k, base_url="https://api.deepseek.com")
                    r = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role":"user","content":p}])
                    st.info(r.choices[0].message.content)
                except Exception as e: st.error(str(e))

    draw_sentinel_grid_ui([
        {"label": "S&P 500 (SPY)", "value": f"${m_ctx['spy']:.2f}", "delta": f"{m_ctx['spy_change']:+.2f}%"},
        {"label": "VIX INDEX", "value": f"{m_ctx['vix']:.2f}"},
        {"label": "USD/JPY", "value": f"¥{usd_jpy:.2f}"},
    ])

# --- TAB 2: AI DIAGNOSIS ---
with tabs[1]:
    st.markdown('<div class="section-header">REAL-TIME STOCK SCAN</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker Symbol", value=st.session_state.target_ticker).upper().strip()
    
    c1, c2 = st.columns(2)
    if c1.button("🚀 RUN SCAN", type="primary", use_container_width=True) and ticker:
        with st.spinner(f"Scanning {ticker}..."):
            df = DataEngine.get_data(ticker, "2y")
            if df is not None:
                vcp = VCPAnalyzer.calculate(df)
                rs = RSAnalyzer.get_raw_score(df)
                curr = df["Close"].iloc[-1]
                
                # Metrics
                draw_sentinel_grid_ui([
                    {"label": "CURRENT PRICE", "value": f"${curr:.2f}"},
                    {"label": "VCP SCORE", "value": f"{vcp['score']}/105"},
                    {"label": "RS MOMENTUM", "value": f"{rs*100:+.1f}%"},
                    {"label": "ATR (Volatility)", "value": f"${vcp['atr']:.2f}"},
                ])

                # Chart
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=20,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # AI Analysis
                if st.button("🤖 GENERATE AI REPORT", use_container_width=True):
                    k = st.secrets.get("DEEPSEEK_API_KEY")
                    if k:
                        n = NewsEngine.get(ticker)
                        f = FundamentalEngine.get(ticker)
                        p = f"""あなたは「AI投資家SENTINEL」。
対象: {ticker}, 価格: ${curr:.2f}
VCP: {vcp['score']}, RS: {rs*100:.1f}%
ニュース:
{NewsEngine.format_for_prompt(n)}
ファンダ: {json.dumps(f)}
指示:
1. 定量データとニュースに基づき投資判断を下せ。
2. 600字以内。
3. 出典明記。
4. 最終決断: [BUY/WAIT/SELL]
"""
                        try:
                            cl = OpenAI(api_key=k, base_url="https://api.deepseek.com")
                            r = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role":"user","content":p}])
                            st.markdown("---")
                            st.markdown(r.choices[0].message.content)
                        except Exception as e: st.error(str(e))
            else:
                st.error("Data not found.")

    if c2.button("⭐ ADD TO WATCHLIST", use_container_width=True) and ticker:
        wl = load_watchlist()
        if ticker not in wl:
            wl.append(ticker)
            save_watchlist(wl)
            st.success(f"Added {ticker}")

# --- TAB 3: PORTFOLIO ---
with tabs[2]:
    st.markdown('<div class="section-header">PORTFOLIO MANAGEMENT</div>', unsafe_allow_html=True)
    port = load_portfolio_json()
    
    # --- 1. 現金管理機能 ---
    with st.expander("💰 資金管理 (預り金入力)", expanded=True):
        c1, c2, c3 = st.columns(3)
        # 現在の値をデフォルトに
        cur_jpy = port.get("cash", {}).get("jpy", 350000)
        cur_usd = port.get("cash", {}).get("usd", 0)
        
        new_jpy = c1.number_input("日本円預り金 (JPY)", value=int(cur_jpy), step=1000)
        new_usd = c2.number_input("米ドル預り金 (USD)", value=float(cur_usd), step=10.0)
        
        if c3.button("更新保存", use_container_width=True):
            port["cash"] = {"jpy": new_jpy, "usd": new_usd}
            save_portfolio_json(port)
            st.success("資金残高を更新しました")
            st.rerun()

    # --- 2. 資産集計 ---
    pos = port.get("positions", {})
    total_stock_usd = 0.0
    
    # 保有株の評価額計算
    pos_details = []
    for t, d in pos.items():
        cp = DataEngine.get_current_price(t)
        val = cp * d["shares"]
        total_stock_usd += val
        cost_basis = d["avg_cost"] * d["shares"]
        pnl = val - cost_basis
        pnl_pct = (val / cost_basis - 1) * 100 if cost_basis > 0 else 0
        
        # AI診断用に詳細を保存
        fund = FundamentalEngine.get(t)
        pos_details.append({
            "ticker": t, "sector": fund.get("sector", "Unknown"), 
            "val": val, "pnl": pnl_pct
        })

    # 総資産計算 (円換算)
    cash_jpy = port["cash"]["jpy"]
    cash_usd = port["cash"]["usd"]
    
    stock_val_jpy = total_stock_usd * usd_jpy
    cash_usd_jpy = cash_usd * usd_jpy
    total_equity_jpy = cash_jpy + cash_usd_jpy + stock_val_jpy

    # ダッシュボード表示
    draw_sentinel_grid_ui([
        {"label": "総資産 (Total Equity)", "value": f"¥{total_equity_jpy:,.0f}"},
        {"label": "株式評価額 (Exposure)", "value": f"¥{stock_val_jpy:,.0f}", "delta": f"(${total_stock_usd:,.2f})"},
        {"label": "現金残高 (Total Cash)", "value": f"¥{cash_jpy + cash_usd_jpy:,.0f}", "delta": f"(¥{cash_jpy:,} + ${cash_usd:,})"},
        {"label": "保有銘柄数", "value": f"{len(pos)}"},
    ])

    # --- 3. AIポートフォリオ診断 ---
    if st.button("🛡️ AI PORTFOLIO GUARD (診断実行)", use_container_width=True, type="primary"):
        k = st.secrets.get("DEEPSEEK_API_KEY")
        if k:
            with st.spinner("Diagnosing Portfolio..."):
                m_ctx = DataEngine.get_market_overview()
                # プロンプト作成
                p_text = "\n".join([f"- {x['ticker']} ({x['sector']}): ${x['val']:.2f} (PnL: {x['pnl']:.1f}%)" for x in pos_details])
                prompt = f"""あなたは「AI投資家SENTINEL」。
【市場】SPY: ${m_ctx['spy']:.2f}, VIX: {m_ctx['vix']:.2f}
【資産状況】総資産: ¥{total_equity_jpy:,.0f} (現金比率: {(cash_jpy+cash_usd_jpy)/total_equity_jpy*100:.1f}%)
【保有株】
{p_text}
指示:
1. セクター分散と現金比率を評価せよ。
2. リスクヘッジ（売却、逆指値、分散）を提案せよ。
3. 600字以内。
4. 免責事項を含める。
"""
                try:
                    cl = OpenAI(api_key=k, base_url="https://api.deepseek.com")
                    r = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role":"user","content":prompt}])
                    st.info(r.choices[0].message.content)
                except Exception as e: st.error(str(e))

    # --- 4. 保有銘柄リスト ---
    st.markdown('<div class="section-header">ACTIVE POSITIONS</div>', unsafe_allow_html=True)
    for t, d in pos.items():
        cp = DataEngine.get_current_price(t)
        val = cp * d["shares"]
        pnl = (val / (d["avg_cost"]*d["shares"]) - 1) * 100
        cls = "profit" if pnl >= 0 else "loss"
        
        st.markdown(f'''
        <div class="pos-card {cls}">
            <div style="display:flex;justify-content:space-between;">
                <span style="font-size:1.2rem;font-weight:bold;">{t}</span>
                <span style="font-size:1.2rem;font-weight:bold;">{pnl:+.2f}%</span>
            </div>
            <div style="color:#8b949e;margin-top:5px;">
                {d['shares']} shares @ ${d['avg_cost']:.2f} → Live: ${cp:.2f}<br>
                Value: ${val:.2f} (¥{val*usd_jpy:,.0f})
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button(f"CLOSE {t}", key=f"close_{t}"):
            del port["positions"][t]
            save_portfolio_json(port)
            st.rerun()

    # --- 5. 新規追加 ---
    with st.expander("➕ 手動ポジション追加"):
        with st.form("add_pos"):
            c1, c2, c3 = st.columns(3)
            ft = c1.text_input("Ticker").upper()
            fs = c2.number_input("Shares", min_value=1, value=10)
            fc = c3.number_input("Avg Cost ($)", min_value=0.01, value=100.0)
            if st.form_submit_button("ADD"):
                if ft:
                    port["positions"][ft] = {"shares": fs, "avg_cost": fc}
                    save_portfolio_json(port)
                    st.success(f"Added {ft}")
                    st.rerun()


