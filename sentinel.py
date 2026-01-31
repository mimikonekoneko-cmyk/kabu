#!/usr/bin/env python3
# SENTINEL v27.1 PRIORITIZED - ETF/Stock Split Notification
# Multi-dimensional scoring with VCP maturity and institutional intelligence
# Philosophy: "Price and volume are the cause, news is the result"
# Target: 10% annual return by catching institutional accumulation BEFORE news
# 
# Requirements: pandas, numpy, yfinance, requests, beautifulsoup4
# Usage: python sentinel_v27_1_prioritized.py

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("SENTINEL")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler("sentinel_debug.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(fh)

# ---------------------------
# CONFIG
# ---------------------------
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

INITIAL_CAPITAL_JPY = 350_000
TRADING_RATIO = 0.75

ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25
MAX_SECTOR_CONCENTRATION = 0.40

# Minimum position size to ensure high-value stocks are tradeable
MIN_POSITION_USD = 500  # Minimum $500 per position

MAX_TIGHTNESS_BASE = 2.0
MAX_NOTIFICATIONS = 10
MIN_DAILY_VOLUME_USD = 10_000_000

COMMISSION_RATE = 0.002
SLIPPAGE_RATE = 0.001
FX_SPREAD_RATE = 0.0005

REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech']

ALLOW_FRACTIONAL = True

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------
# TICKER UNIVERSE
# ---------------------------
TICKERS = {
    # --- ETF Categories ---
    'SPY':'Index', 'QQQ':'Index', 'IVV':'Index', 'VOO':'Index', 'DIA':'Index',
    'IWM':'Index', 'RSP':'Index', 'VTI':'Index', 'EEM':'Index', 'EFA':'Index',
    'VEA':'Index', 'EWZ':'Index', 'XLK':'Sector', 'XLF':'Sector', 'XLE':'Sector',
    'XLI':'Sector', 'XLV':'Sector', 'SMH':'Sector', 'GDX':'Metal', 'HYG':'Bond',
    'LQD':'Bond', 'TLT':'Bond', 'IAU':'Metal', 'GLDM':'Metal', 'SLV':'Metal',
    'TQQQ':'Leveraged', 'SQQQ':'Leveraged', 'SOXL':'Leveraged',

    # --- Individual Stocks ---
    'NVDA':'AI', 'AMD':'Semi', 'AVGO':'Semi', 'TSM':'Semi', 'ASML':'Semi',
    'MU':'Semi', 'INTC':'Semi', 'LRCX':'Semi', 'AMAT':'Semi', 'KLAC':'Semi',
    'TXN':'Semi', 'QCOM':'Semi', 'MRVL':'Semi', 'ADI':'Semi', 'ON':'Semi',
    'MSFT':'Cloud', 'AAPL':'Device', 'GOOGL':'Ad', 'GOOG':'Ad', 'META':'Ad',
    'AMZN':'Retail', 'NFLX':'Service', 'ORCL':'Soft', 'IBM':'Soft', 'INTU':'Soft',
    'ADBE':'Soft', 'CRM':'Soft', 'NOW':'Soft', 'APP':'Soft', 'SNOW':'Cloud',
    'JPM':'Bank', 'GS':'Bank', 'BAC':'Bank', 'WFC':'Bank', 'COF':'Bank',
    'MA':'Fin', 'V':'Fin', 'BLK':'Fin', 'SCHW':'Fin', 'AXP':'Fin',
    'LLY':'Bio', 'UNH':'Health', 'JNJ':'Health', 'ABBV':'Bio',
    'ABT':'Health', 'TMO':'Health', 'PFE':'Pharma', 'MRK':'Pharma',
    'XOM':'Energy', 'CVX':'Energy', 'FCX':'Material', 'NEM':'Gold',
    'COP':'Energy', 'MPC':'Energy',
    'WMT':'Retail', 'COST':'Retail', 'HD':'Retail', 'SBUX':'Cons', 'PG':'Cons',
    'KO':'Cons', 'PEP':'Cons',
    'GE':'Ind', 'CAT':'Ind', 'BA':'Ind', 'APH':'Ind', 'HON':'Ind',
    'MSTR':'Crypto', 'COIN':'Crypto', 'HOOD':'Fintech',
    'PLTR':'AI', 'RKLB':'Space', 'ASTS':'Space',
}

# ETF categories for filtering
ETF_CATEGORIES = ['Index', 'Sector', 'Metal', 'Bond', 'Leveraged']

SECTOR_ETF = {
    'Index':'SPY',
    'Sector':'SPY',
    'AI':'QQQ',
    'Semi':'SOXX',
    'Cloud':'QQQ',
    'Ad':'QQQ',
    'Soft':'IGV',
    'Retail':'XRT',
    'Cons':'XLP',
    'Bank':'XLF',
    'Fin':'VFH',
    'Health':'XLV',
    'Bio':'IBB',
    'Pharma':'XLV',
    'Energy':'XLE',
    'Ind':'XLI',
    'Material':'XLB',
    'Metal':'GLD',
    'Gold':'GLD',
    'Bond':'LQD',
    'Crypto':'BTC-USD',
    'Cannabis':'MJ',
    'China':'MCHI',
    'Gaming':'BETZ',
    'Biotech':'XBI',
    'Media':'XLC',
    'EV':'DRIV',
    'Tech':'XLK',
    'Renewable':'ICLN',
    'Mining':'XME',
    'Utilities':'XLU',
    'Travel':'JETS',
    'Space':'UFO',
    'RealEstate':'XLRE',
    'Telecom':'VOX',
    'Leveraged':'QQQ',
    'Space':'UFO',
    'Unknown':'SPY'
}

# ---------------------------
# VCP Maturity Analyzer
# ---------------------------
class VCPAnalyzer:
    @staticmethod
    def calculate_vcp_maturity(df, result):
        try:
            maturity = 0
            signals = []

            # 1. Volatility Contraction (40 pts)
            tightness = result.get('tightness', 999)
            if tightness < 1.0:
                maturity += 40
                signals.append("極度収縮")
            elif tightness < 1.5:
                maturity += 30
                signals.append("強収縮")
            elif tightness < 2.0:
                maturity += 20
                signals.append("収縮中")
            elif tightness < 2.5:
                maturity += 10
                signals.append("軽度収縮")

            # 2. Higher Lows (30 pts)
            if 'Close' in df.columns and len(df) >= 20:
                close = df['Close'].astype(float)
                recent_lows = close.iloc[-20:].rolling(5).min()

                if len(recent_lows) >= 10:
                    if recent_lows.iloc[-1] > recent_lows.iloc[-10] > recent_lows.iloc[-20]:
                        maturity += 30
                        signals.append("切上完了")
                    elif recent_lows.iloc[-1] > recent_lows.iloc[-10]:
                        maturity += 20
                        signals.append("切上中")
                    elif recent_lows.iloc[-1] >= recent_lows.iloc[-5]:
                        maturity += 10
                        signals.append("底固め")

            # 3. Volume Drying (20 pts)
            reasons = result.get('reasons', '')
            if 'VolDry' in reasons:
                maturity += 20
                signals.append("出来高縮小")

            # 4. MA Structure (10 pts)
            if 'Trend+' in reasons or 'Trend++' in reasons:
                maturity += 10
                signals.append("MA整列")
            elif 'MA50+' in reasons or 'MA20+' in reasons:
                maturity += 5
                signals.append("MA形成中")

            # Stage determination
            if maturity >= 85:
                stage = "🔥爆発直前"
                stage_en = "BREAKOUT_READY"
            elif maturity >= 70:
                stage = "⚡初動圏"
                stage_en = "EARLY_STAGE"
            elif maturity >= 50:
                stage = "👁形成中"
                stage_en = "FORMING"
            elif maturity >= 30:
                stage = "⏳準備段階"
                stage_en = "PREPARING"
            else:
                stage = "❌未成熟"
                stage_en = "IMMATURE"

            return {
                'maturity': maturity,
                'stage': stage,
                'stage_en': stage_en,
                'signals': signals
            }

        except Exception as e:
            logger.debug("VCP maturity calculation failed: %s", e)
            return {
                'maturity': 0,
                'stage': '❌計算不可',
                'stage_en': 'UNKNOWN',
                'signals': []
            }

# ---------------------------
# Comprehensive Signal Quality Scoring
# ---------------------------
class SignalQuality:
    @staticmethod
    def calculate_comprehensive_score(result, vcp_analysis, inst_analysis):
        # Technical Score (0-40) - Based on VCP maturity
        tech_score = min(vcp_analysis['maturity'] * 0.4, 40)

        # Risk/Reward Score (0-30)
        ev = result['bt'].get('net_expectancy', 0)
        wr = result['bt'].get('winrate', 0) / 100.0

        rr_score = 0
        if ev > 0.6 and wr > 0.5:
            rr_score = 30
        elif ev > 0.4 and wr > 0.45:
            rr_score = 25
        elif ev > 0.3 and wr > 0.42:
            rr_score = 20
        elif ev > 0.2 and wr > 0.40:
            rr_score = 15
        elif ev > 0.1 and wr > 0.35:
            rr_score = 10
        elif ev > 0 and wr > 0.3:
            rr_score = 5

        # Institutional Score (0-30)
        risk_score = inst_analysis.get('risk_score', 0)

        if risk_score < 0:
            inst_score = 30
        elif risk_score < 20:
            inst_score = 25
        elif risk_score < 40:
            inst_score = 20
        elif risk_score < 60:
            inst_score = 15
        else:
            inst_score = max(0, 15 - (risk_score - 60) // 10)

        total = tech_score + rr_score + inst_score

        # Tier Classification
        if total >= 75:
            tier = 'CORE'
            tier_emoji = '🔥'
            priority = 1
        elif total >= 60:
            tier = 'SECONDARY'
            tier_emoji = '⚡'
            priority = 2
        elif total >= 45:
            tier = 'WATCH'
            tier_emoji = '👁'
            priority = 3
        else:
            tier = 'AVOID'
            tier_emoji = '❌'
            priority = 4

        return {
            'total_score': int(total),
            'tech_score': int(tech_score),
            'rr_score': int(rr_score),
            'inst_score': int(inst_score),
            'tier': tier,
            'tier_emoji': tier_emoji,
            'priority': priority
        }

    @staticmethod
    def generate_why_now(result, vcp_analysis, inst_analysis, quality):
        reasons = []

        # VCP Stage
        if vcp_analysis['maturity'] >= 85:
            reasons.append("VCP完成・爆発待ち")
        elif vcp_analysis['maturity'] >= 70:
            reasons.append("初動開始可能性")
        elif vcp_analysis['maturity'] >= 50:
            reasons.append("形成進行中")

        # Institutional Intelligence
        overall = inst_analysis.get('overall', 'NEUTRAL')
        if overall == '✅LOW_RISK':
            reasons.append("機関買い圧力検知")
        elif overall == '🚨HIGH_RISK':
            reasons.append("⚠️機関売り圧力")

        # RR Quality
        ev = result['bt'].get('net_expectancy', 0)
        if ev > 0.6:
            reasons.append("高RR（非対称優位）")
        elif ev > 0.4:
            reasons.append("良好RR")

        # Price Action
        current = result.get('price', 0)
        entry = result.get('pivot', 0)
        if entry > 0 and current < entry * 0.99:
            discount = ((entry - current) / entry) * 100
            reasons.append(f"押目-{discount:.1f}%")

        # Options Signal
        signals = inst_analysis.get('signals', {})
        options_sig = signals.get('options', {}).get('signal', '')
        if options_sig == '🐂BULLISH':
            reasons.append("Opt強気")
        elif options_sig == '🐻BEARISH':
            reasons.append("Opt弱気")

        return " | ".join(reasons) if reasons else "基準達成"

# ---------------------------
# Institutional Modules (simplified for length)
# ---------------------------
class InsiderTracker:
    @staticmethod
    def get_insider_activity(ticker, days=30):
        try:
            cache_file = CACHE_DIR / f"insider_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)

            stock = yf.Ticker(ticker)
            insider_trades = stock.insider_transactions

            if insider_trades is None or insider_trades.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}

            cutoff_date = datetime.now() - timedelta(days=days)
            recent = insider_trades[insider_trades.index >= cutoff_date]

            if recent.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}

            buy_shares = recent[recent['Shares'] > 0]['Shares'].sum()
            sell_shares = abs(recent[recent['Shares'] < 0]['Shares'].sum())
            ratio = sell_shares / max(buy_shares, 1)

            if ratio > 5:
                signal = '🚨SELL'
            elif ratio > 2:
                signal = '⚠️CAUTION'
            elif ratio < 0.5:
                signal = '✅BUY'
            else:
                signal = 'NEUTRAL'

            result = {'buy_shares': int(buy_shares), 'sell_shares': int(sell_shares), 'ratio': float(ratio), 'signal': signal}
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.debug("Insider tracking failed for %s: %s", ticker, e)
            return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}

class ShortInterestTracker:
    @staticmethod
    def get_short_interest(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            short_percent = info.get('shortPercentOfFloat', 0)
            if short_percent > 20:
                signal = '🚨HIGH'
            elif short_percent > 10:
                signal = '⚠️ELEVATED'
            else:
                signal = 'NORMAL'
            return {'short_percent': float(short_percent), 'signal': signal}
        except Exception:
            return {'short_percent': 0, 'signal': 'UNKNOWN'}

class SentimentAnalyzer:
    @staticmethod
    def get_reddit_sentiment(ticker):
        return {'mentions': 0, 'hype_score': 50, 'signal': 'UNKNOWN'}

class InstitutionalOwnership:
    @staticmethod
    def get_institutional_holdings(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            inst_percent = info.get('heldPercentInstitutions', 0) * 100
            if inst_percent > 80:
                signal = '✅STRONG'
            elif inst_percent < 40:
                signal = '⚠️WEAK'
            else:
                signal = 'NORMAL'
            return {'inst_percent': float(inst_percent), 'signal': signal}
        except Exception:
            return {'inst_percent': 0, 'signal': 'UNKNOWN'}

class OptionFlowAnalyzer:
    @staticmethod
    def get_put_call_ratio(ticker):
        try:
            stock = yf.Ticker(ticker)
            exp_dates = stock.options
            if not exp_dates:
                return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}
            opt = stock.option_chain(exp_dates[0])
            calls = opt.calls
            puts = opt.puts
            if calls.empty or puts.empty:
                return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            ratio = put_volume / max(call_volume, 1)
            if ratio > 1.5:
                signal = '🐻BEARISH'
            elif ratio < 0.7:
                signal = '🐂BULLISH'
            else:
                signal = 'NEUTRAL'
            return {'put_call_ratio': float(ratio), 'signal': signal}
        except Exception:
            return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}

class MacroAnalyzer:
    @staticmethod
    def get_macro_environment():
        try:
            cache_file = CACHE_DIR / f"macro_{datetime.now().strftime('%Y%m%d')}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            tnx = yf.download("^TNX", period="5d", progress=False)
            treasury_10y = float(tnx['Close'].iloc[-1]) if not tnx.empty and 'Close' in tnx.columns else 4.5
            vix_data = yf.download("^VIX", period="5d", progress=False)
            vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty and 'Close' in vix_data.columns else 20.0
            rate_env = '⚠️ELEVATED' if treasury_10y > 4.0 else '✅LOW_RATE'
            vol_env = '✅LOW_VOL' if vix < 20 else '⚠️ELEVATED'
            result = {'treasury_10y': treasury_10y, 'vix': vix, 'rate_env': rate_env, 'vol_env': vol_env}
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            return result
        except Exception:
            return {'treasury_10y': 4.5, 'vix': 20.0, 'rate_env': 'UNKNOWN', 'vol_env': 'UNKNOWN'}

class InstitutionalAnalyzer:
    @staticmethod
    def analyze(ticker):
        signals = {}
        alerts = []
        risk_score = 0

        insider = InsiderTracker.get_insider_activity(ticker)
        signals['insider'] = insider
        if insider['signal'] == '🚨SELL':
            alerts.append(f"Insider売{insider['ratio']:.1f}x")
            risk_score += 30
        elif insider['signal'] == '✅BUY':
            risk_score -= 10

        short = ShortInterestTracker.get_short_interest(ticker)
        signals['short'] = short
        if short['signal'] == '🚨HIGH':
            alerts.append(f"空売{short['short_percent']:.0f}%")
            risk_score += 20

        sentiment = SentimentAnalyzer.get_reddit_sentiment(ticker)
        signals['sentiment'] = sentiment

        inst = InstitutionalOwnership.get_institutional_holdings(ticker)
        signals['institutional'] = inst
        if inst['signal'] == '⚠️WEAK':
            alerts.append(f"機関{inst['inst_percent']:.0f}%")
            risk_score += 10

        options = OptionFlowAnalyzer.get_put_call_ratio(ticker)
        signals['options'] = options
        if options['signal'] == '🐻BEARISH':
            alerts.append(f"P/C{options['put_call_ratio']:.2f}")
            risk_score += 15
        elif options['signal'] == '🐂BULLISH':
            risk_score -= 10

        if risk_score > 60:
            overall = '🚨HIGH_RISK'
        elif risk_score > 30:
            overall = '⚠️CAUTION'
        elif risk_score < 0:
            overall = '✅LOW_RISK'
        else:
            overall = 'NEUTRAL'

        return {'signals': signals, 'alerts': alerts, 'risk_score': risk_score, 'overall': overall}

# ---------------------------
# Core modules (abbreviated)
# ---------------------------
def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="5d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty and 'Close' in data.columns else 152.0
    except Exception:
        return 152.0

def jpy_to_usd(jpy, fx):
    return jpy / fx

def get_vix():
    try:
        data = yf.download("^VIX", period="5d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty and 'Close' in data.columns else 20.0
    except Exception:
        return 20.0

def check_market_trend():
    try:
        spy = yf.download("SPY", period="400d", progress=False)
        if spy.empty:
            return True, "Unknown", 0.0
        close = spy['Close'].dropna() if 'Close' in spy.columns else None
        if close is None or len(close) < 210:
            return True, "Unknown", 0.0
        curr = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        dist = ((curr - ma200) / ma200) * 100
        return curr > ma200, f"{'Bull' if curr > ma200 else 'Bear'} ({dist:+.1f}%)", dist
    except Exception:
        return True, "Unknown", 0.0

def safe_download(ticker, period="700d", retry=3):
    for attempt in range(retry):
        try:
            time.sleep(1.5)  # Rate limiting protection
            df = yf.download(ticker, period=period, progress=False)
            return df.to_frame() if isinstance(df, pd.Series) else df
        except Exception as e:
            logger.warning("yf.download attempt %d failed for %s: %s", attempt+1, ticker, e)
            time.sleep(3 + attempt * 2)
    return pd.DataFrame()

def ensure_df(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.copy() if df is not None else pd.DataFrame()

def safe_rolling_last(series, window, min_periods=1, default=np.nan):
    try:
        val = series.rolling(window, min_periods=min_periods).mean().iloc[-1]
        return float(val) if not pd.isna(val) else default
    except Exception:
        try:
            return float(series.iloc[-1])
        except Exception:
            return default

def is_earnings_near(ticker, days_window=2):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None:
            return False
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            date_val = cal.iloc[0, 0]
        elif isinstance(cal, dict):
            date_val = cal.get('Earnings Date', [None])[0]
        else:
            return False
        if date_val is None:
            return False
        ed = pd.to_datetime(date_val).date()
        days_until = (ed - datetime.now().date()).days
        return abs(days_until) <= days_window
    except Exception:
        return False

def sector_is_strong(sector):
    try:
        sector_key = str(sector[0]) if isinstance(sector, (pd.Series, np.ndarray, list, tuple)) and len(sector) > 0 else str(sector)
        etf = SECTOR_ETF.get(sector_key)
        if not etf:
            return True
        etf_sym = str(etf[0]) if isinstance(etf, (pd.Series, np.ndarray, list, tuple)) and len(etf) > 0 else str(etf)
        df = safe_download(etf_sym, period="300d", retry=2)
        if df is None or df.empty:
            return True
        if 'Close' not in df.columns:
            for c in df.columns:
                if 'close' in str(c).lower():
                    df['Close'] = df[c]
                    break
        if 'Close' not in df.columns:
            return True
        close = df['Close'].dropna()
        if len(close) < 220:
            return True
        ma200 = close.rolling(200, min_periods=50).mean().dropna()
        if len(ma200) < 12:
            return True
        last = float(ma200.iloc[-1])
        prev = float(ma200.iloc[-10])
        slope = (last - prev) / prev if prev != 0 else 0.0
        return bool(slope >= 0.0)
    except Exception as e:
        logger.exception("sector_is_strong error for %s: %s", sector, e)
        return True

class TransactionCostModel:
    @staticmethod
    def calculate_total_cost_usd(val_usd):
        return (val_usd * COMMISSION_RATE + val_usd * SLIPPAGE_RATE) * 2

class PositionSizer:
    @staticmethod
    def calculate_position(cap_usd, winrate, rr, atr_pct, vix, sec_exp):
        try:
            if rr <= 0:
                return 0.0, 0.0
            kelly = max(0.0, (winrate - (1 - winrate) / rr))
            kelly = min(kelly * 0.5, MAX_POSITION_SIZE)
            v_f = 0.7 if atr_pct > 0.05 else 0.85 if atr_pct > 0.03 else 1.0
            m_f = 0.7 if vix > 30 else 0.85 if vix > 20 else 1.0
            s_f = 0.7 if sec_exp > MAX_SECTOR_CONCENTRATION else 1.0
            final_frac = min(kelly * v_f * m_f * s_f, MAX_POSITION_SIZE)
            pos_val = cap_usd * final_frac

            # Apply minimum position size
            if pos_val > 0 and pos_val < MIN_POSITION_USD:
                pos_val = MIN_POSITION_USD
                final_frac = pos_val / cap_usd

            return pos_val, final_frac
        except Exception:
            return 0.0, 0.0

def simulate_past_performance_v2(df, sector, lookback_years=3):
    try:
        df = ensure_df(df)
        if 'Close' not in df.columns:
            for c in df.columns:
                if 'close' in str(c).lower():
                    df['Close'] = df[c]; break
        if 'High' not in df.columns:
            for c in df.columns:
                if 'high' in str(c).lower():
                    df['High'] = df[c]; break
        if 'Low' not in df.columns:
            for c in df.columns:
                if 'low' in str(c).lower():
                    df['Low'] = df[c]; break
        close = df['Close'].dropna() if 'Close' in df.columns else pd.Series(dtype=float)
        high = df['High'].dropna() if 'High' in df.columns else pd.Series(dtype=float)
        low = df['Low'].dropna() if 'Low' in df.columns else pd.Series(dtype=float)
        if len(close) < 60 or len(high) < 60 or len(low) < 60:
            return {'winrate':0, 'net_expectancy':0, 'message':'LowData'}
        end_date = close.index[-1]
        start_date = end_date - pd.DateOffset(years=lookback_years)
        mask = close.index >= start_date
        close = close.loc[mask]
        high = high.loc[mask]
        low = low.loc[mask]
        if len(close) < 60:
            return {'winrate':0, 'net_expectancy':0, 'message':'ShortWindow'}
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=7).mean().dropna()
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        wins = 0; losses = 0; total_r = 0.0; samples = 0
        for i in range(50, len(close)-40):
            try:
                window_high = high.iloc[i-5:i].max()
                pivot = window_high * 1.002
                if high.iloc[i] < pivot:
                    continue
                ma50 = close.rolling(50, min_periods=10).mean().iloc[i]
                ma200 = close.rolling(200, min_periods=50).mean().iloc[i] if i >= 200 else None
                if ma200 is not None and not (close.iloc[i] > ma50 or ma50 > ma200):
                    continue
                stop_dist = atr.iloc[i] * ATR_STOP_MULT if i < len(atr) else atr.iloc[-1] * ATR_STOP_MULT
                entry = pivot
                target = entry + stop_dist * reward_mult
                outcome = None
                for j in range(1, 31):
                    if i + j >= len(close):
                        break
                    if high.iloc[i+j] >= target:
                        outcome = 'win'; break
                    if low.iloc[i+j] <= entry - stop_dist:
                        outcome = 'loss'; break
                if outcome is None:
                    last_close = close.iloc[min(i+30, len(close)-1)]
                    pnl = (last_close - entry) / stop_dist if stop_dist != 0 else 0
                    if pnl > 0:
                        wins += 1; total_r += min(pnl, reward_mult)
                    else:
                        losses += 1; total_r -= abs(pnl)
                    samples += 1
                else:
                    samples += 1
                    if outcome == 'win':
                        wins += 1; total_r += reward_mult
                    else:
                        losses += 1; total_r -= 1.0
            except Exception:
                continue
        total = wins + losses
        if total < 8:
            return {'winrate':0, 'net_expectancy':0, 'message':f'LowSample:{total}'}
        wr = (wins / total)
        ev = total_r / total
        return {'winrate':wr*100, 'net_expectancy':ev - 0.05, 'message':f"WR{wr*100:.0f}% EV{ev:.2f}"}
    except Exception as e:
        logger.exception("Backtest error: %s", e)
        return {'winrate':0, 'net_expectancy':0, 'message':'BT Error'}

class StrategicAnalyzerV2:
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_position_value_usd, vix, sec_exposures, cap_usd, market_is_bull):
        try:
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return None, "❌DATA"
            df = ensure_df(df)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = [' '.join(map(str, c)).strip() for c in df.columns.values]
                except Exception:
                    pass
            if 'Close' not in df.columns:
                for c in df.columns:
                    if 'adj close' in str(c).lower() or 'adj_close' in str(c).lower():
                        df['Close'] = df[c]; break
                if 'Close' not in df.columns:
                    for c in df.columns:
                        if 'close' in str(c).lower():
                            df['Close'] = df[c]; break
            if 'High' not in df.columns:
                for c in df.columns:
                    if 'high' in str(c).lower():
                        df['High'] = df[c]; break
            if 'Low' not in df.columns:
                for c in df.columns:
                    if 'low' in str(c).lower():
                        df['Low'] = df[c]; break
            if 'Volume' not in df.columns:
                for c in df.columns:
                    if 'volume' in str(c).lower():
                        df['Volume'] = df[c]; break
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            if 'Close' not in df.columns:
                logger.debug("analyze_ticker: missing Close column after normalization for ticker=%s, cols=%s", ticker, list(df.columns))
                return None, "❌DATA"
            df = df.dropna(subset=['Close'])
            if df.empty:
                return None, "❌DATA"
            df[['High','Low','Close','Volume']] = df[['High','Low','Close','Volume']].ffill().bfill()
            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            vol = df['Volume'].astype(float)
            if len(close) < 60:
                return None, "❌DATA"
            curr = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else 0.0
            if curr <= 0:
                return None, "❌PRICE"
            try:
                max_shares = int(max_position_value_usd // curr)
            except Exception:
                max_shares = 0
            fractional_possible = (max_position_value_usd / curr) if curr > 0 else 0.0
            if ALLOW_FRACTIONAL:
                can_trade = fractional_possible >= 0.01
            else:
                can_trade = max_shares >= 1
            if not can_trade:
                return None, "❌PRICE"
            ma50 = safe_rolling_last(close, 50, min_periods=10, default=curr)
            ma200 = safe_rolling_last(close, 200, min_periods=50, default=None) if len(close) >= 50 else None
            if ma200 is not None:
                if not (curr > ma50 or ma50 > ma200):
                    return None, "❌TREND"
            else:
                if not (curr > ma50):
                    return None, "❌TREND"
            try:
                tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr14 = tr.rolling(14, min_periods=7).mean().iloc[-1]
            except Exception:
                atr14 = np.nan
            if pd.isna(atr14) or atr14 <= 0:
                try:
                    alt = (high - low).rolling(14, min_periods=7).mean().iloc[-1]
                    atr14 = max(alt if not pd.isna(alt) else 0.0, 1e-6)
                except Exception:
                    atr14 = 1e-6
            atr_pct = atr14 / curr if curr > 0 else 0.0
            try:
                tightness = (high.iloc[-5:].max() - low.iloc[-5:].min()) / (atr14 if atr14 > 0 else 1.0)
            except Exception:
                tightness = 999.0
            max_tightness = MAX_TIGHTNESS_BASE
            if market_is_bull and vix < 20:
                max_tightness = MAX_TIGHTNESS_BASE * 1.4
            elif vix > 25:
                max_tightness = MAX_TIGHTNESS_BASE * 0.9
            if tightness > max_tightness:
                return None, "❌LOOSE"
            score = 0; reasons = []
            try:
                if tightness < 0.8:
                    score += 30; reasons.append("VCP+++")
                elif tightness < 1.2:
                    score += 20; reasons.append("VCP+")
                vol50 = safe_rolling_last(vol, 50, min_periods=10, default=np.nan)
                if not pd.isna(vol50) and vol.iloc[-1] < vol50:
                    score += 15; reasons.append("VolDry")
                mom5 = safe_rolling_last(close, 5, min_periods=3, default=np.nan)
                mom20 = safe_rolling_last(close, 20, min_periods=10, default=np.nan)
                if not pd.isna(mom5) and not pd.isna(mom20) and (mom5 / mom20) > 1.02:
                    score += 20; reasons.append("Mom+")
                if ma200 is not None and ((ma50 - ma200) / ma200) > 0.03:
                    score += 20; reasons.append("Trend+")
                elif ma200 is None and (curr > ma50):
                    score += 10; reasons.append("Trend?")
            except Exception:
                pass
            bt = simulate_past_performance_v2(df, sector)
            winrate = bt.get('winrate', 0) / 100.0
            try:
                pos_val_usd, frac = PositionSizer.calculate_position(cap_usd, winrate, 2.0, atr_pct, vix, float(sec_exposures.get(sector, 0.0)))
            except Exception as e:
                logger.exception("PositionSizer error for %s: %s", ticker, e)
                pos_val_usd, frac = 0.0, 0.0
            try:
                if ALLOW_FRACTIONAL:
                    est_shares = pos_val_usd / curr if curr > 0 else 0.0
                else:
                    est_shares = int(pos_val_usd // curr) if curr > 0 else 0
                    if est_shares < 1 and max_shares >= 1:
                        est_shares = 1
                if not ALLOW_FRACTIONAL and est_shares < 1:
                    return None, "❌PRICE"
                if not ALLOW_FRACTIONAL and est_shares > max_shares:
                    est_shares = max_shares
            except Exception:
                return None, "❌PRICE"
            pivot = high.iloc[-5:].max() * 1.002 if len(high) >= 5 else curr * 1.002
            stop = pivot - (atr14 * ATR_STOP_MULT)
            result = {
                'score': int(score),
                'reasons': ' '.join(reasons),
                'pivot': pivot,
                'stop': stop,
                'sector': sector,
                'bt': bt,
                'pos_usd': pos_val_usd,
                'pos_frac': frac,
                'est_shares': est_shares,
                'tightness': tightness,
                'price': curr,
                'atr_pct': atr_pct,
                'vol': int(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0,
                'df': df
            }
            return result, "✅PASS"
        except Exception as e:
            logger.exception("Analyze error for %s: %s", ticker, e)
            return None, "❌ERROR"

def send_line(msg):
    logger.info("LINE message prepared.")
    if not ACCESS_TOKEN or not USER_ID:
        logger.debug("LINE credentials missing; skipping send.")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages":[{"type":"text", "text":msg}]}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("LINE push succeeded.")
        else:
            logger.warning("LINE push failed status=%d body=%s", resp.status_code, resp.text)
    except Exception as e:
        logger.exception("LINE send failed: %s", e)

# ---------------------------
# Helper function to split ETFs and Stocks
# ---------------------------
def split_etf_stock(data_list):
    """Split a list of (ticker, data) tuples into ETF and Stock lists"""
    etfs = []
    stocks = []

    for ticker, data in data_list:
        sector = data.get('sector', '')
        if sector in ETF_CATEGORIES:
            etfs.append((ticker, data))
        else:
            stocks.append((ticker, data))

    return etfs, stocks

# ---------------------------
# Main mission - v27.1 PRIORITIZED with ETF/Stock split
# ---------------------------
def run_mission():
    fx = get_current_fx_rate()
    vix = get_vix()
    is_bull, market_status, _ = check_market_trend()
    logger.info("Market: %s | VIX: %.1f | FX: ¥%.2f", market_status, vix, fx)
    macro = MacroAnalyzer.get_macro_environment()
    initial_cap_usd = jpy_to_usd(INITIAL_CAPITAL_JPY, fx)
    trading_cap_usd = initial_cap_usd * TRADING_RATIO
    results = []
    stats = {"Earnings":0, "Sector":0, "Trend":0, "Price":0, "Loose":0, "Data":0, "Pass":0, "Error":0}
    sec_exposures = {s: 0.0 for s in set(TICKERS.values())}

    for ticker, sector in TICKERS.items():
        try:
            earnings_flag = is_earnings_near(ticker, days_window=2)
            if earnings_flag:
                stats["Earnings"] += 1
            try:
                sector_flag = not bool(sector_is_strong(sector))
            except Exception:
                logger.exception("sector check failed for %s", sector)
                sector_flag = False
            if sector_flag:
                stats["Sector"] += 1
            df_t = safe_download(ticker, period="700d")
            if df_t is None or df_t.empty:
                stats["Data"] += 1
                logger.debug("No data for %s", ticker)
                continue
            max_pos_val_usd = trading_cap_usd * MAX_POSITION_SIZE
            res, reason = StrategicAnalyzerV2.analyze_ticker(
                ticker, df_t, sector, max_pos_val_usd, vix, sec_exposures, trading_cap_usd, is_bull
            )
            if res:
                res['is_earnings'] = earnings_flag
                res['is_sector_weak'] = sector_flag
                vcp_analysis = VCPAnalyzer.calculate_vcp_maturity(res['df'], res)
                res['vcp_analysis'] = vcp_analysis
                inst_analysis = InstitutionalAnalyzer.analyze(ticker)
                res['institutional'] = inst_analysis
                quality = SignalQuality.calculate_comprehensive_score(res, vcp_analysis, inst_analysis)
                res['quality'] = quality
                why_now = SignalQuality.generate_why_now(res, vcp_analysis, inst_analysis, quality)
                res['why_now'] = why_now
                results.append((ticker, res))
                if not earnings_flag and not sector_flag:
                    stats["Pass"] += 1
                    sec_exposures[sector] += res['pos_usd'] / trading_cap_usd
            else:
                if reason is None:
                    stats["Error"] += 1
                elif "TREND" in reason:
                    stats["Trend"] += 1
                elif "PRICE" in reason:
                    stats["Price"] += 1
                elif "LOOSE" in reason:
                    stats["Loose"] += 1
                elif "DATA" in reason:
                    stats["Data"] += 1
                elif "ERROR" in reason:
                    stats["Error"] += 1
                else:
                    stats["Error"] += 1
        except Exception as e:
            logger.exception("Loop error for %s: %s", ticker, e)
            stats["Error"] += 1
            continue

    all_sorted = sorted(results, key=lambda x: x[1]['quality']['total_score'], reverse=True)
    passed_core = [r for r in all_sorted if r[1]['quality']['tier'] == 'CORE' and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]
    passed_secondary = [r for r in all_sorted if r[1]['quality']['tier'] == 'SECONDARY' and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]
    passed_watch = [r for r in all_sorted if r[1]['quality']['tier'] == 'WATCH' and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]

    # Split into ETFs and Stocks
    core_etfs, core_stocks = split_etf_stock(passed_core)
    secondary_etfs, secondary_stocks = split_etf_stock(passed_secondary)
    watch_etfs, watch_stocks = split_etf_stock(passed_watch)
    all_etfs, all_stocks = split_etf_stock(all_sorted)

    report_lines = []
    report_lines.append("="*50)
    report_lines.append("SENTINEL v27.1 PRIORITIZED - ETF/Stock Split")
    report_lines.append("Catch institutional accumulation BEFORE the news")
    report_lines.append("="*50)
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    report_lines.append(f"Market: {market_status} | VIX: {vix:.1f} | FX: ¥{fx:.2f}")
    report_lines.append(f"10Y: {macro['treasury_10y']:.2f}% | {macro['rate_env']} {macro['vol_env']}")
    report_lines.append("")
    report_lines.append("【TARGET】10% Annual / 0.8% Monthly")
    report_lines.append(f"Capital: ¥{INITIAL_CAPITAL_JPY:,} → ${initial_cap_usd:.0f} | Trading: ${trading_cap_usd:.0f}")
    report_lines.append("")
    report_lines.append("【STATISTICS】")
    report_lines.append(f"Analyzed: {len(TICKERS)} | Pass: {len(all_sorted)}")
    report_lines.append(f"Blocked: Earn={stats['Earnings']} Sec={stats['Sector']} Trend={stats['Trend']} Loose={stats['Loose']}")
    report_lines.append(f"Errors: Data={stats['Data']} Internal={stats['Error']}")
    report_lines.append("="*50)

    report_lines.append("\n【PRIORITY SIGNALS】")
    report_lines.append(f"🔥 CORE STOCKS: {len(core_stocks)} | 🏆 CORE ETFs: {len(core_etfs)}")
    report_lines.append(f"⚡ SECONDARY STOCKS: {len(secondary_stocks)} | 🏅 SECONDARY ETFs: {len(secondary_etfs)}")
    report_lines.append(f"👁 WATCH STOCKS: {len(watch_stocks)} | 📊 WATCH ETFs: {len(watch_etfs)}")
    report_lines.append("")

    # TODAY'S TOP PRIORITY (From Stocks only)
    if core_stocks:
        top = core_stocks[0]
        ticker = top[0]
        r = top[1]

        actual_shares = int(r['est_shares'])
        actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

        report_lines.append(f"🎯 TODAY'S TOP PRIORITY (STOCK): {ticker}")
        report_lines.append(f"   Score: {r['quality']['total_score']}/100 (Tech:{r['quality']['tech_score']} RR:{r['quality']['rr_score']} Inst:{r['quality']['inst_score']})")

        if actual_shares > 0:
            report_lines.append(f"   {actual_shares}株 @ ${r['price']:.2f} = ${actual_cost:.0f}")
        else:
            report_lines.append(f"   ⚠️ 1株未満 (${r['price']:.2f})")

        report_lines.append(f"   Why Now: {r['why_now']}")
        report_lines.append("")

    # CORE STOCKS - IMMEDIATE CONSIDERATION
    if core_stocks:
        report_lines.append("🔥 CORE STOCKS - IMMEDIATE CONSIDERATION (Top 5)")
        for i, (ticker, r) in enumerate(core_stocks[:5], 1):
            q = r['quality']
            vcp = r['vcp_analysis']
            inst = r['institutional']

            actual_shares = int(r['est_shares'])
            actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

            report_lines.append(f"\n[{i}] {ticker} {q['total_score']}/100 | VCP:{vcp['maturity']}% {vcp['stage']}")
            report_lines.append(f"    Tech:{q['tech_score']} RR:{q['rr_score']} Inst:{q['inst_score']} | Risk:{inst['risk_score']}")

            if actual_shares > 0:
                report_lines.append(f"    {actual_shares}株 @ ${r['price']:.2f} = ${actual_cost:.0f} | Entry: ${r['pivot']:.2f}")
            else:
                report_lines.append(f"    ⚠️ 1株未満 (${r['price']:.2f}) | Entry: ${r['pivot']:.2f}")

            report_lines.append(f"    BT: {r['bt']['message']} | T:{r['tightness']:.2f}")
            report_lines.append(f"    💡 {r['why_now']}")
            if inst['alerts']:
                report_lines.append(f"    ⚠️  {' | '.join(inst['alerts'][:3])}")

    # CORE ETFs - IMMEDIATE CONSIDERATION
    if core_etfs:
        report_lines.append("\n🏆 CORE ETFs - IMMEDIATE CONSIDERATION (Top 5)")
        for i, (ticker, r) in enumerate(core_etfs[:5], 1):
            q = r['quality']
            vcp = r['vcp_analysis']

            actual_shares = int(r['est_shares'])
            actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

            report_lines.append(f"\n[{i}] {ticker} {q['total_score']}/100 | VCP:{vcp['maturity']}% {vcp['stage']}")

            if actual_shares > 0:
                report_lines.append(f"    {actual_shares}株 @ ${r['price']:.2f} = ${actual_cost:.0f} | Entry: ${r['pivot']:.2f}")
            else:
                report_lines.append(f"    ⚠️ 1株未満 (${r['price']:.2f}) | Entry: ${r['pivot']:.2f}")

            report_lines.append(f"    {r['why_now']}")

    # SECONDARY STOCKS
    if secondary_stocks:
        report_lines.append("\n⚡ SECONDARY STOCKS - CONDITIONAL WATCH (Top 10)")
        for i, (ticker, r) in enumerate(secondary_stocks[:10], 1):
            q = r['quality']
            vcp = r['vcp_analysis']

            actual_shares = int(r['est_shares'])
            actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

            report_lines.append(f"\n[{i}] {ticker} {q['total_score']}/100 | VCP:{vcp['maturity']}% {vcp['stage']}")

            if actual_shares > 0:
                report_lines.append(f"    {actual_shares}株 @ ${r['price']:.2f} = ${actual_cost:.0f} | Entry: ${r['pivot']:.2f}")
            else:
                report_lines.append(f"    ⚠️ 1株未満 (${r['price']:.2f}) | Entry: ${r['pivot']:.2f}")

            report_lines.append(f"    {r['why_now']}")

    # WATCH LIST SUMMARY
    if watch_stocks:
        watch_str = ", ".join([f"{t} {r['quality']['total_score']}" for t, r in watch_stocks[:15]])
        report_lines.append("\n👁 WATCH STOCKS - MONITORING (Top 15)")
        report_lines.append(f"    {watch_str}")

    if watch_etfs:
        etf_watch_str = ", ".join([f"{t} {r['quality']['total_score']}" for t, r in watch_etfs[:5]])
        report_lines.append("\n📊 WATCH ETFs - MONITORING (Top 5)")
        report_lines.append(f"    {etf_watch_str}")

    # TOP 15 INDIVIDUAL STOCKS COMPREHENSIVE ANALYSIS
    report_lines.append("\n" + "="*50)
    report_lines.append("【TOP 15 INDIVIDUAL STOCKS - COMPREHENSIVE ANALYSIS】")
    for i, (ticker, r) in enumerate(all_stocks[:15], 1):
        q = r['quality']
        vcp = r['vcp_analysis']
        tag = "✅OK"
        if r.get('is_earnings'): 
            tag = "❌EARN"
        elif r.get('is_sector_weak'): 
            tag = "❌SEC"
        report_lines.append(f"\n{i:2}. {ticker:5} {q['total_score']:3}/100 {q['tier_emoji']} | {tag}")
        report_lines.append(f"    VCP:{vcp['maturity']:3}% {vcp['stage']} | WR:{r['bt']['winrate']:.0f}% EV:{r['bt']['net_expectancy']:+.2f}")
        report_lines.append(f"    {' '.join(vcp['signals'])}")
        report_lines.append(f"    {r['why_now']}")

    # TOP 5 ETFs COMPREHENSIVE ANALYSIS
    report_lines.append("\n" + "="*50)
    report_lines.append("【TOP 5 ETFs - COMPREHENSIVE ANALYSIS】")
    for i, (ticker, r) in enumerate(all_etfs[:5], 1):
        q = r['quality']
        vcp = r['vcp_analysis']
        tag = "✅OK"
        if r.get('is_earnings'): 
            tag = "❌EARN"
        elif r.get('is_sector_weak'): 
            tag = "❌SEC"
        report_lines.append(f"\n{i:2}. {ticker:5} {q['total_score']:3}/100 {q['tier_emoji']} | {tag}")
        report_lines.append(f"    VCP:{vcp['maturity']:3}% {vcp['stage']} | WR:{r['bt']['winrate']:.0f}% EV:{r['bt']['net_expectancy']:+.2f}")
        report_lines.append(f"    {' '.join(vcp['signals'])}")
        report_lines.append(f"    {r['why_now']}")

    report_lines.append("\n" + "="*50)
    report_lines.append("【PHILOSOPHY】")
    report_lines.append("✓ Price & volume are the CAUSE")
    report_lines.append("✓ News is the RESULT")
    report_lines.append("✓ Catch institutional accumulation BEFORE headlines")
    report_lines.append("="*50)

    final_report = "\n".join(report_lines)
    logger.info("\n%s", final_report)
    send_line(final_report)

if __name__ == "__main__":
    run_mission()