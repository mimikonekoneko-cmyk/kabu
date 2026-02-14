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

# スクレイピング用ライブラリのインポート試行
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

# ==============================================================================
# 設定・定数
# ==============================================================================

NOW = datetime.datetime.now()
TODAY_STR = NOW.strftime("%Y-%m-%d") # 現在の日付文字列
CACHE_DIR = Path("./cache_v45"); CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results"); RESULTS_DIR.mkdir(exist_ok=True)
WATCHLIST_FILE = Path("watchlist.json")
PORTFOLIO_FILE = Path("portfolio.json")

# AI分析用のニュース設定
NEWS_CONFIG = {
    "FETCH_TIMEOUT": 6,
    "MAX_CHARS": 400,
    "CACHE_TTL": 3600
}

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
        try:
            # リアルタイム性を高めるためfast_info優先
            t = yf.Ticker("JPY=X")
            price = t.fast_info.get('lastPrice')
            if price is None:
                df = t.history(period="1d")
                price = float(df["Close"].iloc[-1]) if not df.empty else 152.65
            return round(price, 2)
        except:
            return 152.65

class DataEngine:
    @staticmethod
    def get_data(ticker, period):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period)
            if df is None or df.empty:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return None

            # MultiIndex解消
            if isinstance(df.columns, pd.MultiIndex):
                target_level = None
                for i in range(df.columns.nlevels):
                    if 'Close' in df.columns.get_level_values(i):
                        df.columns = df.columns.get_level_values(i)
                        target_level = i
                        break
                if target_level is None:
                    df.columns = df.columns.get_level_values(0)

            # カラム名クリーニング
            new_cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    found = False
                    for part in c:
                        s_part = str(part)
                        if s_part in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                            new_cols.append(s_part)
                            found = True
                            break
                    if not found:
                        new_cols.append(str(c[0]))
                else:
                    new_cols.append(str(c))
            df.columns = new_cols
            df.columns = [c.strip().capitalize() for c in df.columns]
            
            rename_map = {'Adj close': 'Close', 'Adj Close': 'Close', 'Last': 'Close'}
            df.rename(columns=rename_map, inplace=True)

            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            else:
                df.index = pd.to_datetime(df.index)

            required = {'Open', 'High', 'Low', 'Close'}
            if 'Volume' not in df.columns: df['Volume'] = 0
            
            if not required.issubset(df.columns):
                if 'Close' in df.columns:
                    if 'Open' not in df.columns: df['Open'] = df['Close']
                    if 'High' not in df.columns: df['High'] = df['Close']
                    if 'Low' not in df.columns:  df['Low'] = df['Close']
                else:
                    return None
            
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
            for c in cols_to_numeric:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df.dropna(subset=['Close'], inplace=True)
            return df
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return None

    @staticmethod
    def get_current_price(ticker):
        try:
            ticker_dat = yf.Ticker(ticker)
            price = ticker_dat.fast_info.get('lastPrice')
            if price is None:
                hist = ticker_dat.history(period="1d")
                if not hist.empty:
                    price = hist.iloc[-1, 3] # Close
            return float(price) if price else 0.0
        except:
            return 0.0
    
    @staticmethod
    def get_atr(ticker): return 1.5

    @staticmethod
    def get_market_overview():
        """S&P500(SPY)とVIXの直近データを取得"""
        try:
            spy = yf.Ticker("SPY").history(period="5d")
            vix = yf.Ticker("^VIX").history(period="1d")
            
            spy_price = spy["Close"].iloc[-1] if not spy.empty else 0
            spy_change = (spy_price / spy["Close"].iloc[-2] - 1) * 100 if len(spy) >= 2 else 0
            vix_price = vix["Close"].iloc[-1] if not vix.empty else 0
            
            return {"spy": spy_price, "spy_change": spy_change, "vix": vix_price}
        except:
            return {"spy": 0, "spy_change": 0, "vix": 0}

class FundamentalEngine:
    CACHE_TTL = 24 * 3600

    @staticmethod
    def get(ticker: str) -> dict:
        cf = CACHE_DIR / f"fund_{ticker}.json"
        if cf.exists():
            if time.time() - cf.stat().st_mtime < FundamentalEngine.CACHE_TTL:
                try:
                    with open(cf) as f: return json.load(f)
                except: pass
        try:
            info = yf.Ticker(ticker).info
            data = {
                "analyst_target":      info.get("targetMeanPrice"),
                "analyst_count":       info.get("numberOfAnalystOpinions"),
                "recommendation":      info.get("recommendationKey"),
                "short_ratio":         info.get("shortRatio"),
                "short_pct_float":     info.get("shortPercentOfFloat"),
                "insider_pct":         info.get("heldPercentInsiders"),
                "institution_pct":     info.get("heldPercentInstitutions"),
                "pe_forward":          info.get("forwardPE"),
                "revenue_growth":      info.get("revenueGrowth"),
                "sector":              info.get("sector", "Unknown"),
                "industry":            info.get("industry", "Unknown"),
                "market_cap":          info.get("marketCap")
            }
            with open(cf, "w") as f: json.dump(data, f, default=str)
            return data
        except:
            return {}

class NewsEngine:
    @staticmethod
    def get(ticker: str) -> dict:
        cf = CACHE_DIR / f"news_{ticker}.json"
        if cf.exists():
            if time.time() - cf.stat().st_mtime < NEWS_CONFIG["CACHE_TTL"]:
                try:
                    with open(cf) as f: return json.load(f)
                except: pass

        articles = []
        seen = set()

        # 1. yfinance news
        try:
            for n in (yf.Ticker(ticker).news or [])[:3]:
                title = n.get("title", n.get("headline", ""))
                url   = n.get("link",  n.get("url", ""))
                if title and title not in seen:
                    seen.add(title)
                    articles.append({"title": title, "url": url, "body": ""})
        except: pass

        # 2. Google News RSS
        try:
            feed = feedparser.parse(
                f"https://news.google.com/rss/search?q={ticker}+stock+when:3d&hl=en-US&gl=US&ceid=US:en"
            )
            for e in feed.entries[:3]:
                if e.title not in seen:
                    seen.add(e.title)
                    articles.append({"title": e.title, "url": getattr(e, "link", ""), "body": ""})
        except: pass

        # 3. Scraping Body
        if BS4_OK:
            for art in articles[:3]: # 上位3件のみ本文取得
                if not art["url"]: continue
                try:
                    r = requests.get(
                        art["url"],
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=NEWS_CONFIG["FETCH_TIMEOUT"],
                    )
                    soup = BeautifulSoup(r.text, "html.parser")
                    paras = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 50]
                    body  = " ".join(paras)[:NEWS_CONFIG["MAX_CHARS"]]
                    art["body"] = body
                except: pass

        result = {"articles": articles[:5], "fetched_at": datetime.datetime.now().isoformat()}
        with open(cf, "w") as f: json.dump(result, f, ensure_ascii=False)
        return result

    @staticmethod
    def get_general_market() -> dict:
        """市場全体のニュースを取得"""
        cf = CACHE_DIR / "news_market_general.json"
        if cf.exists():
            if time.time() - cf.stat().st_mtime < NEWS_CONFIG["CACHE_TTL"]:
                try:
                    with open(cf) as f: return json.load(f)
                except: pass
        
        articles = []
        seen = set()
        try:
            # Google News RSS (Market)
            feed = feedparser.parse("https://news.google.com/rss/search?q=stock+market+news+when:1d&hl=en-US&gl=US&ceid=US:en")
            for e in feed.entries[:5]:
                if e.title not in seen:
                    seen.add(e.title)
                    articles.append({"title": e.title, "url": getattr(e, "link", ""), "body": ""})
        except: pass
        
        # スクレイピングは省略（地合い判断用なのでタイトルだけで十分）

        result = {"articles": articles, "fetched_at": datetime.datetime.now().isoformat()}
        with open(cf, "w") as f: json.dump(result, f, ensure_ascii=False)
        return result

    @staticmethod
    def format_for_prompt(news: dict) -> str:
        lines = []
        for a in news.get("articles", []):
            lines.append(f"• タイトル: {a['title']}")
            lines.append(f"  URL: {a['url']}")
            if a.get("body"):
                lines.append(f"  (内容: {a['body']})")
            lines.append("---")
        return "\n".join(lines) if lines else "特になし"

# ==============================================================================
# 分析ロジック
# ==============================================================================

class VCPAnalyzer:
    @staticmethod
    def calculate(df: pd.DataFrame) -> dict:
        try:
            if df is None or len(df) < 130:
                return VCPAnalyzer._empty_result()

            close_s = df["Close"]
            high_s  = df["High"]
            low_s   = df["Low"]
            vol_s   = df["Volume"]

            tr1 = high_s - low_s
            tr2 = (high_s - close_s.shift(1)).abs()
            tr3 = (low_s - close_s.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_val = float(tr.rolling(14).mean().iloc[-1])

            if pd.isna(atr_val) or atr_val <= 0:
                return VCPAnalyzer._empty_result()

            periods = [20, 30, 40, 60]
            vol_ranges = []
            for p in periods:
                p_high = float(high_s.iloc[-p:].max())
                p_low  = float(low_s.iloc[-p:].min())
                if p_high > 0:
                    vol_ranges.append((p_high - p_low) / p_high)
                else:
                    vol_ranges.append(1.0)

            curr_range = vol_ranges[0]
            avg_range = float(np.mean(vol_ranges[:3]))
            is_contracting = vol_ranges[0] < vol_ranges[1] < vol_ranges[2]

            if avg_range < 0.10:   tight_score = 40
            elif avg_range < 0.15: tight_score = 30
            elif avg_range < 0.20: tight_score = 20
            elif avg_range < 0.28: tight_score = 10
            else:                  tight_score = 0

            if is_contracting: tight_score += 5
            tight_score = min(40, tight_score)

            v20_avg = float(vol_s.iloc[-20:].mean())
            v60_avg = float(vol_s.iloc[-60:-40].mean())
            if pd.isna(v20_avg) or pd.isna(v60_avg): return VCPAnalyzer._empty_result()
            v_ratio = v20_avg / v60_avg if v60_avg > 0 else 1.0

            if v_ratio < 0.45:   vol_score = 30
            elif v_ratio < 0.60: vol_score = 25
            elif v_ratio < 0.75: vol_score = 15
            else:                vol_score = 0
            is_dryup = v_ratio < 0.75

            ma50_v  = float(close_s.rolling(50).mean().iloc[-1])
            ma150_v = float(close_s.rolling(150).mean().iloc[-1])
            ma200_v = float(close_s.rolling(200).mean().iloc[-1])
            price_v = float(close_s.iloc[-1])

            m_score = 0
            if price_v > ma50_v:   m_score += 10
            if ma50_v > ma150_v:   m_score += 10
            if ma150_v > ma200_v:  m_score += 10

            pivot_v = float(high_s.iloc[-50:].max())
            dist_v = (pivot_v - price_v) / pivot_v
            p_bonus = 0
            if 0 <= dist_v <= 0.04: p_bonus = 5
            elif 0.04 < dist_v <= 0.08: p_bonus = 3

            signals = []
            if tight_score >= 35: signals.append("Tight Base (VCP)")
            if is_contracting:    signals.append("V-Contraction Detected")
            if is_dryup:          signals.append("Volume Dry-up Detected")
            if m_score >= 20:     signals.append("Trend Alignment OK")
            if p_bonus > 0:       signals.append("Near Pivot Point")

            return {
                "score": int(min(105, tight_score + vol_score + m_score + p_bonus)),
                "atr": atr_val,
                "signals": signals,
                "is_dryup": is_dryup,
                "range_pct": round(curr_range, 4),
                "vol_ratio": round(v_ratio, 2),
                "breakdown": {"tight": tight_score, "vol": vol_score, "ma": m_score, "pivot": p_bonus}
            }
        except Exception:
            return VCPAnalyzer._empty_result()

    @staticmethod
    def _empty_result():
        return {
            "score": 0, "atr": 0.0, "signals": [], 
            "is_dryup": False, "range_pct": 0.0, "vol_ratio": 1.0,
            "breakdown": {"tight": 0, "vol": 0, "ma": 0, "pivot": 0}
        }

class RSAnalyzer:
    @staticmethod
    def get_raw_score(df: pd.DataFrame) -> float:
        try:
            c = df["Close"]
            if len(c) < 252: return -999.0
            r12m = (c.iloc[-1] / c.iloc[-252]) - 1
            r6m  = (c.iloc[-1] / c.iloc[-126]) - 1
            r3m  = (c.iloc[-1] / c.iloc[-63])  - 1
            r1m  = (c.iloc[-1] / c.iloc[-21])  - 1
            return (r12m * 0.4) + (r6m * 0.2) + (r3m * 0.2) + (r1m * 0.2)
        except Exception:
            return -999.0

class StrategyValidator:
    @staticmethod
    def run(df: pd.DataFrame) -> float:
        try:
            if len(df) < 252: return 1.0
            c_data = df["Close"]; h_data = df["High"]; l_data = df["Low"]
            tr_calc = pd.concat([h_data - l_data, (h_data - c_data.shift(1)).abs(), (l_data - c_data.shift(1)).abs()], axis=1).max(axis=1)
            atr_s = tr_calc.rolling(14).mean()
            trade_results = []
            is_in_pos = False; entry_p = 0.0; stop_p = 0.0
            t_mult = EXIT_CFG["TARGET_R_MULT"]; s_mult = EXIT_CFG["STOP_LOSS_ATR_MULT"]

            idx_start = max(65, len(df) - 252)
            for i in range(idx_start, len(df)):
                if is_in_pos:
                    if float(l_data.iloc[i]) <= stop_p:
                        trade_results.append(-1.0); is_in_pos = False
                    elif float(h_data.iloc[i]) >= entry_p + (entry_p - stop_p) * t_mult:
                        trade_results.append(t_mult); is_in_pos = False
                    elif i == len(df) - 1:
                        risk_unit = entry_p - stop_p
                        if risk_unit > 0:
                            pnl_r = (float(c_data.iloc[i]) - entry_p) / risk_unit
                            trade_results.append(pnl_r)
                        is_in_pos = False
                else:
                    if i < 20: continue
                    local_high_20 = float(h_data.iloc[i-20:i].max())
                    ma50_c = float(c_data.rolling(50).mean().iloc[i])
                    if float(c_data.iloc[i]) > local_high_20 and float(c_data.iloc[i]) > ma50_c:
                        is_in_pos = True
                        entry_p = float(c_data.iloc[i])
                        atr_now = float(atr_s.iloc[i])
                        stop_p = entry_p - (atr_now * s_mult)

            if not trade_results: return 1.0
            gp = sum(res for res in trade_results if res > 0)
            gl = abs(sum(res for res in trade_results if res < 0))
            if gl == 0: return round(min(10.0, gp if gp > 0 else 1.0), 2)
            return round(min(10.0, float(gp / gl)), 2)
        except Exception:
            return 1.0

# ==============================================================================
# UI ヘルパー
# ==============================================================================

def draw_sentinel_grid_ui(metrics: List[Dict[str, Any]]):
    html_out = '<div class="sentinel-grid">'
    for m in metrics:
        delta_s = ""
        if "delta" in m and m["delta"]:
            is_pos = "+" in str(m["delta"]) or (isinstance(m["delta"], (int, float)) and m["delta"] > 0)
            c_code = "#3fb950" if is_pos else "#f85149"
            delta_s = f'<div class="sentinel-delta" style="color:{c_code}">{m["delta"]}</div>'
        item = (
            '<div class="sentinel-card">'
            f'<div class="sentinel-label">{m["label"]}</div>'
            f'<div class="sentinel-value">{m["value"]}</div>'
            f'{delta_s}'
            '</div>'
        )
        html_out += item
    html_out += '</div>'
    st.markdown(html_out.strip(), unsafe_allow_html=True)

def load_portfolio_json() -> dict:
    default = {"positions": {}, "cash_jpy": 350000, "cash_usd": 0} # デフォルトで35万円
    if not PORTFOLIO_FILE.exists():
        return default
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
            # 既存データにキーがない場合の互換性維持
            if "cash_jpy" not in d: d["cash_jpy"] = 350000
            if "cash_usd" not in d: d["cash_usd"] = 0
            return d
    except:
        return default

def save_portfolio_json(data: dict):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_watchlist_data() -> list:
    if not WATCHLIST_FILE.exists(): return []
    try:
        with open(WATCHLIST_FILE, "r") as f: return json.load(f)
    except: return []

def save_watchlist_data(data: list):
    with open(WATCHLIST_FILE, "w") as f: json.dump(data, f)

# ==============================================================================
# UIスタイル
# ==============================================================================

GLOBAL_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #0d1117; color: #f0f6fc; }
.block-container { padding-top: 0rem !important; padding-bottom: 2rem !important; }
.ui-push-buffer { height: 65px; width: 100%; background: transparent; }
.stTabs [data-baseweb="tab-list"] { display: flex !important; width: 100% !important; flex-wrap: nowrap !important; overflow-x: auto !important; background-color: #161b22 !important; padding: 12px 12px 0 12px !important; border-radius: 12px 12px 0 0 !important; gap: 12px !important; border-bottom: 2px solid #30363d !important; }
.stTabs [data-baseweb="tab"] { min-width: 185px !important; flex-shrink: 0 !important; font-size: 1.05rem !important; font-weight: 700 !important; color: #8b949e !important; padding: 22px 32px !important; background-color: transparent !important; border: none !important; }
.stTabs [aria-selected="true"] { color: #ffffff !important; background-color: #238636 !important; border-radius: 12px 12px 0 0 !important; }
.sentinel-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 20px 0 30px 0; }
@media (min-width: 992px) { .sentinel-grid { grid-template-columns: repeat(4, 1fr); } }
.sentinel-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; box-shadow: 0 4px 25px rgba(0,0,0,0.7); }
.sentinel-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.25em; margin-bottom: 12px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.sentinel-value { font-size: 1.45rem; font-weight: 700; color: #f0f6fc; line-height: 1.1; }
.sentinel-delta { font-size: 0.95rem; font-weight: 600; margin-top: 12px; }
.diagnostic-panel { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 28px; margin-bottom: 26px; }
.diag-row { display: flex; justify-content: space-between; padding: 16px 0; border-bottom: 1px solid #21262d; }
.diag-row:last-child { border-bottom: none; }
.diag-key { color: #8b949e; font-size: 1.0rem; font-weight: 600; }
.diag-val { color: #f0f6fc; font-weight: 700; font-family: 'Share Tech Mono', monospace; font-size: 1.15rem; }
.section-header { font-size: 1.2rem; font-weight: 700; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 16px; margin: 45px 0 28px; text-transform: uppercase; letter-spacing: 4px; display: flex; align-items: center; gap: 14px; }
.pos-card { background: #0d1117; border: 1px solid #30363d; border-radius: 18px; padding: 30px; margin-bottom: 24px; border-left: 12px solid #30363d; }
.pos-card.urgent { border-left-color: #f85149; }
.pos-card.caution { border-left-color: #d29922; }
.pos-card.profit { border-left-color: #3fb950; }
.pnl-pos { color: #3fb950; font-weight: 700; font-size: 1.3rem; }
.pnl-neg { color: #f85149; font-weight: 700; font-size: 1.3rem; }
.exit-info { font-size: 0.95rem; color: #8b949e; font-family: 'Share Tech Mono', monospace; margin-top: 18px; border-top: 1px solid #21262d; padding-top: 18px; line-height: 1.8; }
.stButton > button { min-height: 60px; border-radius: 14px; font-weight: 700; font-size: 1.1rem; }
[data-testid="stMetric"] { display: none !important; }
.js-plotly-plot, .plotly, .plot-container { width: 100% !important; }
</style>
"""

# ==============================================================================
# 言語定義
# ==============================================================================

LANG = {
    "ja": {
        "title": "🛡️ SENTINEL PRO",
        "tab_scan": "📊 マーケットスキャン",
        "tab_diag": "🔍 AI診断",
        "tab_port": "💼 ポートフォリオ",
        "scan_date": "📅 スキャン日",
        "usd_jpy": "💱 USD/JPY",
        "action_list": "アクション銘柄",
        "wait_list": "ウォッチ銘柄",
        "sector_map": "🗺️ セクター別RSマップ",
        "realtime_scan": "🔍 リアルタイム定量スキャン",
        "ticker_input": "ティッカーシンボル（例：NVDA）",
        "run_quant": "🚀 定量スキャン実行",
        "add_watchlist": "⭐ ウォッチリストに追加",
        "quant_dashboard": "📊 SENTINEL定量ダッシュボード",
        "current_price": "💰 現在値",
        "vcp_score": "🎯 VCPスコア",
        "profit_factor": "📈 プロフィットファクター",
        "rs_momentum": "📏 RSモメンタム",
        "strategic_levels": "🛡️ ATR基準の戦略水準",
        "stop_loss": "ストップロス (2.0R)",
        "target1": "目標① (1.0R)",
        "target2": "目標② (2.5R)",
        "risk_unit": "リスク単価 ($)",
        "vcp_breakdown": "📐 VCPスコア内訳",
        "tightness": "収縮スコア",
        "volume": "出来高スコア",
        "ma_trend": "移動平均トレンド",
        "pivot_bonus": "ピボットボーナス",
        "ai_reasoning": "🤖 SENTINEL AI診断",
        "generate_ai": "🚀 AI診断を生成（ニュース＆ファンダメンタル）",
        "ai_key_missing": "DEEPSEEK_API_KEY が設定されていません。",
        "portfolio_risk": "💼 ポートフォリオリスク管理",
        "portfolio_empty": "ポートフォリオは空です。",
        "unrealized_jpy": "💰 評価額合計 (Total)",
        "assets": "📊 保有銘柄数",
        "exposure": "🛡️ 米国株式 (Stocks)",
        "performance": "📈 平均パフォーマンス",
        "active_positions": "📋 保有中のポジション",
        "close_position": "決済",
        "register_new": "➕ 新規ポジション登録",
        "ticker_symbol": "ティッカーシンボル",
        "shares": "株数",
        "avg_cost": "平均取得単価",
        "add_to_portfolio": "ポートフォリオに追加",
        "port_ai_btn": "🛡️ AIポートフォリオ診断 (SENTINEL PORTFOLIO GUARD)",
        "jpy_cash": "💰 預り金 (JPY)",
        "usd_cash": "💵 USドル (USD)",
        "market_ai_btn": "🤖 AI市場分析 (SENTINEL MARKET EYE)",
    },
    "en": {
        "title": "🛡️ SENTINEL PRO",
        "tab_scan": "📊 MARKET SCAN",
        "tab_diag": "🔍 AI DIAGNOSIS",
        "tab_port": "💼 PORTFOLIO",
        "scan_date": "📅 Scan Date",
        "usd_jpy": "💱 USD/JPY",
        "action_list": "Action List",
        "wait_list": "Watch List",
        "sector_map": "🗺️ Sector RS Map",
        "realtime_scan": "🔍 REAL-TIME QUANTITATIVE SCAN",
        "ticker_input": "Ticker Symbol (e.g. NVDA)",
        "run_quant": "🚀 RUN QUANTITATIVE SCAN",
        "add_watchlist": "⭐ ADD TO WATCHLIST",
        "quant_dashboard": "📊 SENTINEL QUANTITATIVE DASHBOARD",
        "current_price": "💰 Current Price",
        "vcp_score": "🎯 VCP Score",
        "profit_factor": "📈 Profit Factor",
        "rs_momentum": "📏 RS Momentum",
        "strategic_levels": "🛡️ STRATEGIC LEVELS (ATR-Based)",
        "stop_loss": "Stop Loss (2.0R)",
        "target1": "Target 1 (1.0R)",
        "target2": "Target 2 (2.5R)",
        "risk_unit": "Risk Unit ($)",
        "vcp_breakdown": "📐 VCP SCORE BREAKDOWN",
        "tightness": "Tightness Score",
        "volume": "Volume Dry-up",
        "ma_trend": "MA Trend Score",
        "pivot_bonus": "Pivot Bonus",
        "ai_reasoning": "🤖 SENTINEL AI CONTEXTUAL REASONING",
        "generate_ai": "🚀 GENERATE AI DIAGNOSIS (NEWS & FUNDAMENTALS)",
        "ai_key_missing": "DEEPSEEK_API_KEY is not configured.",
        "portfolio_risk": "💼 PORTFOLIO RISK MANAGEMENT",
        "portfolio_empty": "Portfolio is currently empty.",
        "unrealized_jpy": "💰 Total Equity",
        "assets": "📊 Assets",
        "exposure": "🛡️ Stock Value",
        "performance": "📈 Performance",
        "active_positions": "📋 ACTIVE POSITIONS",
        "close_position": "Close",
        "register_new": "➕ REGISTER NEW POSITION",
        "ticker_symbol": "Ticker Symbol",
        "shares": "Shares",
        "avg_cost": "Avg Cost",
        "add_to_portfolio": "ADD TO PORTFOLIO",
        "port_ai_btn": "🛡️ AI PORTFOLIO REVIEW",
        "jpy_cash": "💰 JPY Cash",
        "usd_cash": "💵 USD Cash",
        "market_ai_btn": "🤖 AI MARKET ANALYSIS",
    }
}

# ==============================================================================
# メイン UI
# ==============================================================================

def initialize_sentinel_state():
    if "target_ticker" not in st.session_state: st.session_state.target_ticker = ""
    if "trigger_analysis" not in st.session_state: st.session_state.trigger_analysis = False
    if "quant_results_stored" not in st.session_state: st.session_state.quant_results_stored = None
    if "ai_analysis_text" not in st.session_state: st.session_state.ai_analysis_text = ""
    if "ai_market_text" not in st.session_state: st.session_state.ai_market_text = ""
    if "ai_port_text" not in st.session_state: st.session_state.ai_port_text = ""
    if "language" not in st.session_state: st.session_state.language = "ja"

initialize_sentinel_state()

st.set_page_config(page_title="SENTINEL PRO", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")
st.markdown('<div class="ui-push-buffer"></div>', unsafe_allow_html=True)
st.markdown(GLOBAL_STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🌐 Language")
    lang = st.selectbox("", ["日本語", "English"], index=0 if st.session_state.language == "ja" else 1)
    st.session_state.language = "ja" if lang == "日本語" else "en"
    txt = LANG[st.session_state.language]
    st.markdown(f"### {txt['title']} ウォッチリスト")
    wl_t = load_watchlist_data()
    for t_n in wl_t:
        col_n, col_d = st.columns([4, 1])
        if col_n.button(t_n, key=f"side_{t_n}", use_container_width=True):
            st.session_state.target_ticker = t_n
            st.session_state.trigger_analysis = True
            st.rerun()
        if col_d.button("×", key=f"rm_{t_n}"):
            wl_t.remove(t_n)
            save_watchlist_data(wl_t)
            st.rerun()
    st.divider()
    st.caption(f"🛡️ SENTINEL V5.0 | {NOW.strftime('%H:%M:%S')}")

fx_rate = CurrencyEngine.get_usd_jpy()
tab_scan, tab_diag, tab_port = st.tabs([txt["tab_scan"], txt["tab_diag"], txt["tab_port"]])

# --- Tab 1: スキャン結果 & AI地合い分析 ---
with tab_scan:
    st.markdown(f'<div class="section-header">{txt["tab_scan"]}</div>', unsafe_allow_html=True)
    
    # スキャンデータの読み込み
    s_data = {}
    s_df = pd.DataFrame()
    if RESULTS_DIR.exists():
        f_list = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
        if f_list:
            try:
                with open(f_list[0], "r", encoding="utf-8") as f: s_data = json.load(f)
                s_df = pd.DataFrame(s_data.get("qualified_full", []))
            except: pass

    # AI地合い分析ボタン
    if st.button(txt["market_ai_btn"], use_container_width=True, type="primary"):
        key = st.secrets.get("DEEPSEEK_API_KEY")
        if not key:
            st.error(txt["ai_key_missing"])
        else:
            with st.spinner("Analyzing Market Conditions (SPY, VIX, News, Scan Data)..."):
                # データ収集
                m_ctx = DataEngine.get_market_overview()
                m_news = NewsEngine.get_general_market()
                m_news_txt = NewsEngine.format_for_prompt(m_news)
                
                # スキャン統計
                act_count = len(s_df[s_df["status"]=="ACTION"]) if not s_df.empty else 0
                wait_count = len(s_df[s_df["status"]=="WAIT"]) if not s_df.empty else 0
                sectors = s_df["sector"].value_counts().to_dict() if not s_df.empty else {}
                top_sectors = list(sectors.keys())[:3] if sectors else []

                prompt = (
                    f"あなたは「ウォール街のAI投資家SENTINEL」です。以下のデータに基づき、本日の市場環境（地合い）を分析し、投資家への助言を行ってください。\n\n"
                    f"【現在日時】: {TODAY_STR}\n"
                    f"【市場インジケータ】\n"
                    f"S&P500(SPY): ${m_ctx['spy']:.2f} (5日前比変化率: {m_ctx['spy_change']:.2f}%)\n"
                    f"VIX指数: {m_ctx['vix']:.2f}\n\n"
                    f"【SENTINELスキャン統計】\n"
                    f"買いシグナル(ACTION)数: {act_count}銘柄\n"
                    f"待機シグナル(WAIT)数: {wait_count}銘柄\n"
                    f"主導セクター: {', '.join(top_sectors)}\n\n"
                    f"【市場ニュース（スクレイピング結果）】\n"
                    f"{m_news_txt}\n\n"
                    f"【指示】\n"
                    f"1. 書き出しは「ウォール街のAI投資家SENTINELだ。」\n"
                    f"2. 市場フェーズを定義せよ（例：上昇トレンド、調整局面、下落トレンド）。SPYとVIXの関係、およびスキャン結果（ACTIONが多いなら強気、少ないなら弱気など）を根拠にすること。\n"
                    f"3. ニュースから読み取れる市場の懸念点や好材料を挙げること（ハルシネーション禁止。日付が未来のものは無視）。\n"
                    f"4. 推奨エクスポージャー（積極投資か、キャッシュ比率を高めるべきか）を助言せよ。\n"
                    f"5. 600文字以内でまとめること。\n"
                    f"6. 参照したニュースのソースを明記すること。\n"
                    f"7. 最後に免責事項を含めること。"
                )
                cl = OpenAI(api_key=key, base_url="https://api.deepseek.com")
                try:
                    res = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
                    st.session_state.ai_market_text = res.choices[0].message.content.replace("$", r"\$")
                except Exception as e:
                    st.error(f"AI Error: {e}")

    if st.session_state.ai_market_text:
        st.info(st.session_state.ai_market_text)

    # グリッド表示
    draw_sentinel_grid_ui([
        {"label": txt["scan_date"], "value": s_data.get("date", TODAY_STR)},
        {"label": txt["usd_jpy"], "value": f"¥{fx_rate:.2f}"},
        {"label": txt["action_list"], "value": len(s_df[s_df["status"]=="ACTION"]) if not s_df.empty else 0},
        {"label": txt["wait_list"], "value": len(s_df[s_df["status"]=="WAIT"]) if not s_df.empty else 0}
    ])
    if not s_df.empty:
        st.markdown(f'<div class="section-header">{txt["sector_map"]}</div>', unsafe_allow_html=True)
        s_df["vcp_score"] = s_df["vcp"].apply(lambda x: x.get("score", 0))
        m_fig = px.treemap(s_df, path=["sector", "ticker"], values="vcp_score", color="rs", color_continuous_scale="RdYlGn", range_color=[70, 100])
        m_fig.update_layout(template="plotly_dark", height=600, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(m_fig, use_container_width=True, key="sector_treemap")
        st.dataframe(s_df[["ticker", "status", "vcp_score", "rs", "sector"]].sort_values("vcp_score", ascending=False), use_container_width=True, height=500)

# --- Tab 2: AI診断 (個別) ---
with tab_diag:
    st.markdown(f'<div class="section-header">{txt["realtime_scan"]}</div>', unsafe_allow_html=True)
    t_input = st.text_input(txt["ticker_input"], value=st.session_state.target_ticker).upper().strip()
    col_q, col_w = st.columns(2)
    start_quant = col_q.button(txt["run_quant"], type="primary", use_container_width=True)
    add_watchlist = col_w.button(txt["add_watchlist"], use_container_width=True)

    if add_watchlist and t_input:
        wl = load_watchlist_data()
        if t_input not in wl:
            wl.append(t_input)
            save_watchlist_data(wl)
            st.success(f"Added {t_input}")

    if (start_quant or st.session_state.pop("trigger_analysis", False)) and t_input:
        with st.spinner(f"SENTINEL ENGINE: Scanning {t_input}..."):
            df_raw = DataEngine.get_data(t_input, "2y")
            if df_raw is not None and not df_raw.empty:
                vcp_res = VCPAnalyzer.calculate(df_raw)
                rs_val = RSAnalyzer.get_raw_score(df_raw)
                pf_val = StrategyValidator.run(df_raw)
                p_curr = df_raw["Close"].iloc[-1]
                st.session_state.quant_results_stored = {"vcp": vcp_res, "rs": rs_val, "pf": pf_val, "price": p_curr, "ticker": t_input}
                st.session_state.ai_analysis_text = ""
            else: st.error(f"Failed to fetch data for {t_input}.")

    if st.session_state.quant_results_stored and st.session_state.quant_results_stored["ticker"] == t_input:
        q = st.session_state.quant_results_stored
        vcp_res, rs_val, pf_val, p_curr = q["vcp"], q["rs"], q["pf"], q["price"]
        rs_val = float(rs_val) if rs_val else 0.0
        pf_val = float(pf_val) if pf_val else 0.0
        p_curr = float(p_curr) if p_curr else 0.0

        st.markdown(f'<div class="section-header">{txt["quant_dashboard"]}</div>', unsafe_allow_html=True)
        draw_sentinel_grid_ui([
            {"label": txt["current_price"], "value": f"${p_curr:.2f}"},
            {"label": txt["vcp_score"], "value": f"{vcp_res['score']}/105"},
            {"label": txt["profit_factor"], "value": f"x{pf_val:.2f}"},
            {"label": txt["rs_momentum"], "value": f"{rs_val*100:+.1f}%"}
        ])

        d1, d2 = st.columns(2)
        with d1:
            risk = vcp_res['atr'] * EXIT_CFG["STOP_LOSS_ATR_MULT"]
            st.markdown(f'''<div class="diagnostic-panel"><b>{txt["strategic_levels"]}</b>
<div class="diag-row"><span class="diag-key">{txt["stop_loss"]}</span><span class="diag-val">${p_curr - risk:.2f}</span></div>
<div class="diag-row"><span class="diag-key">{txt["target1"]}</span><span class="diag-val">${p_curr + risk:.2f}</span></div>
<div class="diag-row"><span class="diag-key">{txt["target2"]}</span><span class="diag-val">${p_curr + risk*2.5:.2f}</span></div>
<div class="diag-row"><span class="diag-key">{txt["risk_unit"]}</span><span class="diag-val">${risk:.2f}</span></div>
</div>''', unsafe_allow_html=True)
        with d2:
            bd = vcp_res['breakdown']
            st.markdown(f'''<div class="diagnostic-panel"><b>{txt["vcp_breakdown"]}</b>
<div class="diag-row"><span class="diag-key">{txt["tightness"]}</span><span class="diag-val">{bd.get("tight", 0)}/45</span></div>
<div class="diag-row"><span class="diag-key">{txt["volume"]}</span><span class="diag-val">{bd.get("vol", 0)}/30</span></div>
<div class="diag-row"><span class="diag-key">{txt["ma_trend"]}</span><span class="diag-val">{bd.get("ma", 0)}/30</span></div>
<div class="diag-row"><span class="diag-key">{txt["pivot_bonus"]}</span><span class="diag-val">+{bd.get("pivot", 0)}pt</span></div>
</div>''', unsafe_allow_html=True)

        st.markdown("### 📈 価格チャート")
        with st.spinner("チャートを読み込み中..."):
            df_raw = DataEngine.get_data(t_input, "2y")
            if df_raw is not None and not df_raw.empty:
                df_t = df_raw.tail(120).copy()
                fig = go.Figure(data=[go.Candlestick(x=df_t.index, open=df_t['Open'], high=df_t['High'], low=df_t['Low'], close=df_t['Close'], name=t_input)])
                fig.update_layout(template="plotly_dark", height=500, margin=dict(t=30, b=0, l=0, r=0), xaxis_rangeslider_visible=False, title=dict(text=f"{t_input} Daily Chart", x=0.05))
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{t_input}")

        st.markdown(f'<div class="section-header">{txt["ai_reasoning"]}</div>', unsafe_allow_html=True)
        if st.button(txt["generate_ai"], use_container_width=True):
            key = st.secrets.get("DEEPSEEK_API_KEY")
            if not key:
                st.error(txt["ai_key_missing"])
            else:
                with st.spinner(f"Fetching News & Fundamentals for {t_input}..."):
                    news_data = NewsEngine.get(t_input)
                    news_text = NewsEngine.format_for_prompt(news_data)
                    fund_data = FundamentalEngine.get(t_input)
                    fund_text = json.dumps(fund_data, indent=2, ensure_ascii=False)
                    prompt = (
                        f"あなたは「ウォール街のAI投資家SENTINEL」です。以下のデータのみに基づき、冷徹な相場観で投資判断を下してください。\n\n"
                        f"【現在日時】: {TODAY_STR}\n"
                        f"【定量的データ】\n銘柄: {t_input}\n現在値: ${p_curr:.2f}\nVCPスコア: {vcp_res['score']}/105\nPF: {pf_val:.2f}\nRS: {rs_val*100:+.1f}%\n\n"
                        f"【ニュース】\n{news_text}\n\n【ファンダメンタルズ】\n{fund_text}\n\n"
                        f"【制約】\n1. 書き出しは「ウォール街のAI投資家SENTINELだ。」\n2. ハルシネーション禁止。提供データのみ使用。\n3. 未来日付のニュースは無視。\n4. 600字以内。\n5. 最終投資決断[BUY/WAIT/SELL]を提示。\n6. ニュースソースを明記。\n7. 免責事項を含める。"
                    )
                    cl = OpenAI(api_key=key, base_url="https://api.deepseek.com")
                    try:
                        res_ai = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
                        st.session_state.ai_analysis_text = res_ai.choices[0].message.content.replace("$", r"\$")
                    except Exception as ai_e:
                        st.error(f"AI Error: {ai_e}")

        if st.session_state.ai_analysis_text:
            st.markdown("---")
            st.markdown(st.session_state.ai_analysis_text)

# --- Tab 3: ポートフォリオ & AI診断 ---
with tab_port:
    st.markdown(f'<div class="section-header">{txt["portfolio_risk"]}</div>', unsafe_allow_html=True)
    p_j = load_portfolio_json()
    pos_m = p_j.get("positions", {})

    # --- 1. 資金管理 (預り金設定) ---
    with st.expander("💰 資金管理 (預り金設定)", expanded=True):
        c1, c2, c3 = st.columns(3)
        # JSONに保存された値を読み込み (デフォルト0)
        curr_jpy = p_j.get("cash_jpy", 350000)
        curr_usd = p_j.get("cash_usd", 0)
        
        in_jpy = c1.number_input(txt["jpy_cash"], value=int(curr_jpy), step=1000)
        in_usd = c2.number_input(txt["usd_cash"], value=float(curr_usd), step=100.0)
        
        if c3.button("残高更新", use_container_width=True):
            p_j["cash_jpy"] = in_jpy
            p_j["cash_usd"] = in_usd
            save_portfolio_json(p_j)
            st.success("資金残高を更新しました。")
            st.rerun()

    # --- 2. 資産集計 ---
    total_stock_val_usd = 0.0
    pos_details = []
    
    for t, d in pos_m.items():
        # セクター情報の自動取得
        fund = FundamentalEngine.get(t)
        sec = fund.get("sector", "Unknown")
        ind = fund.get("industry", "Unknown")
        
        # 現在価格と含み損益
        curr_p = DataEngine.get_current_price(t)
        val_usd = curr_p * d['shares']
        total_stock_usd += val_usd
        
        pnl_pct = ((curr_p / d['avg_cost']) - 1) * 100 if d['avg_cost'] > 0 else 0
        
        pos_details.append({
            "ticker": t, "sector": sec, 
            "val": val_usd, "pnl": pnl_pct, 
            "shares": d['shares'], "cost": d['avg_cost'], "curr": curr_p
        })

    # 円換算 (リアルタイム為替)
    stock_val_jpy = total_stock_usd * fx_rate
    usd_cash_jpy = in_usd * fx_rate
    total_equity_jpy = stock_val_jpy + in_jpy + usd_cash_jpy

    # ダッシュボード表示 (スクショの構成に合わせる)
    draw_sentinel_grid_ui([
        {"label": txt["unrealized_jpy"], "value": f"¥{total_equity_jpy:,.0f}"}, # 評価額合計
        {"label": txt["exposure"], "value": f"¥{stock_val_jpy:,.0f}", "delta": f"(${total_stock_usd:,.2f})"}, # 米国株式
        {"label": txt["jpy_cash"], "value": f"¥{in_jpy:,.0f}"}, # 預り金
        {"label": txt["usd_cash"], "value": f"¥{usd_cash_jpy:,.0f}", "delta": f"(${in_usd:,.2f})"}, # USドル
    ])
    
    # AIポートフォリオ診断ボタン
    if st.button(txt["port_ai_btn"], use_container_width=True, type="primary"):
        key = st.secrets.get("DEEPSEEK_API_KEY")
        if not key:
            st.error(txt["ai_key_missing"])
        else:
            with st.spinner("Analyzing Portfolio Risk & Hedging Strategies..."):
                m_ctx = DataEngine.get_market_overview()
                
                # プロンプト用保有株リスト
                p_text = "\n".join([f"- {x['ticker']} [Sector: {x['sector']}]: ${x['val']:.2f} (PnL: {x['pnl']:+.1f}%)" for x in pos_details])
                
                prompt = (
                    f"あなたは「ウォール街のAI投資家SENTINEL」です。以下のポートフォリオと市場環境に基づき、リスク管理とリバランスの提案を行ってください。\n\n"
                    f"【現在日時】: {TODAY_STR}\n"
                    f"【市場環境】\nSPY: ${m_ctx['spy']:.2f}, VIX: {m_ctx['vix']:.2f}\n\n"
                    f"【資産状況】\n総資産: ¥{total_equity_jpy:,.0f}\n現金比率(円+ドル): {(in_jpy+usd_cash_jpy)/total_equity_jpy*100:.1f}%\n"
                    f"【保有ポートフォリオ詳細（セクター含む）】\n" + p_text + "\n\n"
                    f"【指示】\n"
                    f"1. 書き出しは「ウォール街のAI投資家SENTINELだ。」\n"
                    f"2. 取得されたセクター情報に基づき、特定のセクターへの集中リスクや分散状況を具体的に評価せよ。\n"
                    f"3. VIX指数を考慮し、現在の市場でヘッジ（例: キャッシュ化、逆指値の引き上げ）が必要か助言せよ。\n"
                    f"4. 600文字以内でまとめること。\n"
                    f"5. 最後に免責事項を含めること。"
                )
                cl = OpenAI(api_key=key, base_url="https://api.deepseek.com")
                try:
                    res_p = cl.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
                    st.session_state.ai_port_text = res_p.choices[0].message.content.replace("$", r"\$")
                except Exception as e:
                    st.error(f"AI Error: {e}")

    if st.session_state.ai_port_text:
        st.info(st.session_state.ai_port_text)

    if not pos_m:
        st.info(txt["portfolio_empty"])
    else:
        st.markdown(f'<div class="section-header">{txt["active_positions"]}</div>', unsafe_allow_html=True)
        for p in pos_details:
            t = p["ticker"]
            val = p["val"]
            pnl_pct = p["pnl"]
            cost = p["cost"] * p["shares"]
            pnl_val_jpy = (val - cost) * fx_rate
            
            pnl_c = "pnl-pos" if pnl_pct >= 0 else "pnl-neg"
            cls = "profit" if pnl_pct >= 0 else "urgent"
            
            st.markdown(f'''<div class="pos-card {cls}">
<div style="display: flex; justify-content: space-between; align-items: center;"><b>{t}</b><span class="{pnl_c}">{pnl_pct:+.2f}% (¥{pnl_val_jpy:+,.0f})</span></div>
<div style="font-size: 0.95rem; color: #f0f6fc; margin-top: 10px;">{p["shares"]} shares @ ${p["cost"]:.2f} (Live: ${p["curr"]:.2f})</div>
<div class="exit-info">Sector: {p["sector"]} | Value: ${val:.2f} (¥{val*fx_rate:,.0f})</div></div>''', unsafe_allow_html=True)
            if st.button(f"{txt['close_position']} {t}", key=f"cl_{t}"):
                del port["positions"][t]
                save_portfolio_json(port)
                st.rerun()

    st.markdown(f'<div class="section-header">{txt["register_new"]}</div>', unsafe_allow_html=True)
    with st.form("add_port"):
        c1, c2, c3 = st.columns(3)
        f_ticker = c1.text_input(txt["ticker_symbol"]).upper().strip()
        f_shares = c2.number_input(txt["shares"], min_value=1, value=10)
        f_cost   = c3.number_input(txt["avg_cost"], min_value=0.01, value=100.0)
        if st.form_submit_button(txt["add_to_portfolio"], use_container_width=True):
            if f_ticker:
                port["positions"][f_ticker] = {"shares": f_shares, "avg_cost": f_cost}
                save_portfolio_json(port)
                st.success(f"Added {f_ticker}")
                st.rerun()

st.divider()
st.caption(f"🛡️ SENTINEL PRO SYSTEM | FULL AI INTEGRATION | V5.0")


