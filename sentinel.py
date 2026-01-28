#!/usr/bin/env python3
# SENTINEL v26.0 INSTITUTIONAL - FULL FREE PACKAGE
# Complete institutional tracking with NO API costs
# Requirements: pandas, numpy, yfinance, requests, beautifulsoup4, praw, pytrends, fredapi
# Usage: python sentinel_v26_institutional.py

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import re
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

MIN_SCORE_STRICT = 65
MIN_SCORE_STANDARD = 55
MIN_SCORE_RELAXED = 45
MIN_SCORE = MIN_SCORE_STANDARD

MAX_TIGHTNESS_BASE = 2.0
MAX_NOTIFICATIONS = 10
MIN_DAILY_VOLUME_USD = 10_000_000

COMMISSION_RATE = 0.002
SLIPPAGE_RATE = 0.001
FX_SPREAD_RATE = 0.0005

REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech']

ALLOW_FRACTIONAL = True

# Cache for institutional data (avoid repeated requests)
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------
# TICKER UNIVERSE
# ---------------------------
TICKERS = {
    'NVDA':'AI', 'AMD':'Semi', 'AVGO':'Semi', 'TSM':'Semi', 'ASML':'Semi', 'MU':'Semi',
    'ARM':'Semi', 'INTC':'Semi', 'QCOM':'Semi', 'ON':'Semi', 'LRCX':'Semi', 'AMAT':'Semi',
    'MSFT':'Cloud', 'GOOGL':'Ad', 'META':'Ad', 'PLTR':'AI', 'NOW':'Soft', 'CRM':'Soft',
    'ADBE':'Soft', 'SNOW':'Cloud', 'DDOG':'Cloud', 'WDAY':'Soft', 'TEAM':'Soft',
    'ANET':'Cloud', 'ZS':'Sec', 'MDB':'Cloud', 'SHOP':'Retail', 'PANW':'Sec',
    'CRWD':'Sec', 'FTNT':'Sec', 'NET':'Sec', 'OKTA':'Sec', 'AAPL':'Device',
    'TSLA':'Auto', 'AMZN':'Retail', 'NFLX':'Service', 'COST':'Retail', 'WMT':'Retail',
    'TJX':'Retail', 'TGT':'Retail', 'NKE':'Cons', 'LULU':'Cons', 'SBUX':'Cons',
    'PEP':'Cons', 'KO':'Cons', 'PG':'Cons', 'ELF':'Cons', 'CELH':'Cons', 'MELI':'Retail',
    'V':'Fin', 'MA':'Fin', 'PYPL':'Fintech', 'SQ':'Fintech', 'JPM':'Bank', 'GS':'Bank',
    'MS':'Bank', 'AXP':'Fin', 'BLK':'Fin', 'COIN':'Crypto', 'SOFI':'Fintech', 'NU':'Fintech',
    'LLY':'Bio', 'UNH':'Health', 'ABBV':'Bio', 'ISRG':'Health', 'VRTX':'Bio', 'MRK':'Bio',
    'PFE':'Bio', 'AMGN':'Bio', 'HCA':'Health', 'TDOC':'Health', 'GE':'Ind', 'CAT':'Ind',
    'DE':'Ind', 'BA':'Ind', 'ETN':'Power', 'VRT':'Power', 'TT':'Ind', 'PH':'Ind',
    'TDG':'Ind', 'XOM':'Energy', 'CVX':'Energy', 'MPC':'Energy', 'UBER':'Platform',
    'BKNG':'Travel', 'ABNB':'Travel', 'MAR':'Travel', 'RCL':'Travel', 'DKNG':'Bet',
    'RBLX':'Service', 'DASH':'Service', 'SMCI':'AI'
}

SECTOR_ETF = {
    'Energy':'XLE', 'Semi':'SOXX', 'Bank':'XLF', 'Retail':'XRT', 'Soft':'IGV', 'AI':'QQQ',
    'Fin':'VFH', 'Device':'QQQ', 'Cloud':'QQQ', 'Ad':'QQQ', 'Service':'QQQ', 'Sec':'HACK',
    'Cons':'XLP', 'Bio':'IBB', 'Health':'XLV', 'Ind':'XLI', 'Auto':'CARZ', 'Crypto':'BTC-USD',
    'Power':'XLI', 'Platform':'QQQ', 'Travel':'XLY', 'Bet':'BETZ', 'Fintech':'ARKF'
}

# ---------------------------
# Institutional Data Modules
# ---------------------------

class InsiderTracker:
    """Track insider trading from SEC EDGAR (FREE)"""
    
    @staticmethod
    def get_insider_activity(ticker, days=30):
        """Get recent insider trades from SEC EDGAR"""
        try:
            cache_file = CACHE_DIR / f"insider_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Check cache (daily)
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Use yfinance insider transactions
            stock = yf.Ticker(ticker)
            insider_trades = stock.insider_transactions
            
            if insider_trades is None or insider_trades.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}
            
            # Filter recent trades
            cutoff_date = datetime.now() - timedelta(days=days)
            recent = insider_trades[insider_trades.index >= cutoff_date]
            
            if recent.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}
            
            # Calculate buy/sell
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
            
            result = {
                'buy_shares': int(buy_shares),
                'sell_shares': int(sell_shares),
                'ratio': float(ratio),
                'signal': signal
            }
            
            # Cache
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
            
        except Exception as e:
            logger.debug("Insider tracking failed for %s: %s", ticker, e)
            return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': 'NEUTRAL'}


class ShortInterestTracker:
    """Track short interest from FINRA (FREE)"""
    
    @staticmethod
    def get_short_interest(ticker):
        """Get short interest ratio"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            short_ratio = info.get('shortRatio', 0)
            short_percent = info.get('shortPercentOfFloat', 0)
            
            if short_percent > 20:
                signal = '🚨HIGH'
            elif short_percent > 10:
                signal = '⚠️ELEVATED'
            elif short_percent < 3:
                signal = '✅LOW'
            else:
                signal = 'NORMAL'
            
            return {
                'short_ratio': float(short_ratio),
                'short_percent': float(short_percent),
                'signal': signal
            }
            
        except Exception as e:
            logger.debug("Short interest failed for %s: %s", ticker, e)
            return {'short_ratio': 0, 'short_percent': 0, 'signal': 'UNKNOWN'}


class SentimentAnalyzer:
    """Analyze social sentiment (FREE - Reddit)"""
    
    @staticmethod
    def get_reddit_sentiment(ticker):
        """Get sentiment from Reddit mentions"""
        try:
            cache_file = CACHE_DIR / f"reddit_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Simple web scraping of r/wallstreetbets
            # Note: This is a simplified version. For production, use PRAW
            url = f"https://www.reddit.com/search.json?q={ticker}&sort=new&t=week"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    posts = data.get('data', {}).get('children', [])
                    mention_count = len(posts)
                    
                    if mention_count > 100:
                        signal = '🔥HYPE'
                        hype_score = 90
                    elif mention_count > 50:
                        signal = '⚠️POPULAR'
                        hype_score = 70
                    elif mention_count > 20:
                        signal = 'NORMAL'
                        hype_score = 50
                    else:
                        signal = '✅QUIET'
                        hype_score = 20
                    
                    result = {
                        'mentions': mention_count,
                        'hype_score': hype_score,
                        'signal': signal
                    }
                    
                    with open(cache_file, 'w') as f:
                        json.dump(result, f)
                    
                    return result
            except Exception:
                pass
            
            return {'mentions': 0, 'hype_score': 50, 'signal': 'UNKNOWN'}
            
        except Exception as e:
            logger.debug("Reddit sentiment failed for %s: %s", ticker, e)
            return {'mentions': 0, 'hype_score': 50, 'signal': 'UNKNOWN'}


class GoogleTrendsAnalyzer:
    """Analyze Google search trends (FREE)"""
    
    @staticmethod
    def get_search_trend(ticker):
        """Get Google Trends interest"""
        try:
            cache_file = CACHE_DIR / f"trends_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Note: pytrends requires installation: pip install pytrends
            try:
                from pytrends.request import TrendReq
                
                pytrend = TrendReq(hl='en-US', tz=360)
                pytrend.build_payload([ticker], timeframe='today 1-m')
                trends = pytrend.interest_over_time()
                
                if not trends.empty:
                    current = int(trends[ticker].iloc[-1])
                    avg = int(trends[ticker].mean())
                    
                    if current > avg * 2:
                        signal = '🔥SPIKE'
                    elif current > avg * 1.5:
                        signal = '⚠️RISING'
                    elif current < avg * 0.5:
                        signal = '✅QUIET'
                    else:
                        signal = 'NORMAL'
                    
                    result = {
                        'current': current,
                        'average': avg,
                        'signal': signal
                    }
                    
                    with open(cache_file, 'w') as f:
                        json.dump(result, f)
                    
                    return result
                    
            except ImportError:
                logger.debug("pytrends not installed - skipping Google Trends")
            except Exception:
                pass
            
            return {'current': 50, 'average': 50, 'signal': 'UNKNOWN'}
            
        except Exception as e:
            logger.debug("Google Trends failed for %s: %s", ticker, e)
            return {'current': 50, 'average': 50, 'signal': 'UNKNOWN'}


class InstitutionalOwnership:
    """Track institutional ownership changes (FREE - Yahoo Finance)"""
    
    @staticmethod
    def get_institutional_holdings(ticker):
        """Get institutional ownership percentage"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            inst_percent = info.get('heldPercentInstitutions', 0) * 100
            
            # Get institutional holders
            holders = stock.institutional_holders
            
            if holders is not None and not holders.empty:
                top_holders = len(holders)
            else:
                top_holders = 0
            
            if inst_percent > 80:
                signal = '✅STRONG'
            elif inst_percent > 60:
                signal = 'NORMAL'
            elif inst_percent < 40:
                signal = '⚠️WEAK'
            else:
                signal = 'NEUTRAL'
            
            return {
                'inst_percent': float(inst_percent),
                'top_holders': top_holders,
                'signal': signal
            }
            
        except Exception as e:
            logger.debug("Institutional holdings failed for %s: %s", ticker, e)
            return {'inst_percent': 0, 'top_holders': 0, 'signal': 'UNKNOWN'}


class OptionFlowAnalyzer:
    """Analyze options Put/Call ratio (FREE - Yahoo Finance)"""
    
    @staticmethod
    def get_put_call_ratio(ticker):
        """Get put/call ratio from options chain"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get options expiration dates
            exp_dates = stock.options
            
            if not exp_dates:
                return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}
            
            # Use nearest expiration
            opt = stock.option_chain(exp_dates[0])
            
            calls = opt.calls
            puts = opt.puts
            
            if calls.empty or puts.empty:
                return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}
            
            # Volume-weighted
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            
            if call_volume == 0:
                ratio = 2.0
            else:
                ratio = put_volume / call_volume
            
            if ratio > 1.5:
                signal = '🐻BEARISH'
            elif ratio > 1.0:
                signal = '⚠️CAUTIOUS'
            elif ratio < 0.7:
                signal = '🐂BULLISH'
            else:
                signal = 'NEUTRAL'
            
            return {
                'put_call_ratio': float(ratio),
                'call_volume': int(call_volume),
                'put_volume': int(put_volume),
                'signal': signal
            }
            
        except Exception as e:
            logger.debug("Options flow failed for %s: %s", ticker, e)
            return {'put_call_ratio': 1.0, 'signal': 'UNKNOWN'}


class MacroAnalyzer:
    """Analyze macro environment (FREE - FRED)"""
    
    @staticmethod
    def get_macro_environment():
        """Get current macro indicators"""
        try:
            cache_file = CACHE_DIR / f"macro_{datetime.now().strftime('%Y%m%d')}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # Use yfinance for treasury yields (free alternative to FRED)
            tnx = yf.download("^TNX", period="5d", progress=False)
            
            if not tnx.empty and 'Close' in tnx.columns:
                treasury_10y = float(tnx['Close'].iloc[-1])
            else:
                treasury_10y = 4.5  # default
            
            # VIX already available
            vix_data = yf.download("^VIX", period="5d", progress=False)
            if not vix_data.empty and 'Close' in vix_data.columns:
                vix = float(vix_data['Close'].iloc[-1])
            else:
                vix = 20.0
            
            # Determine environment
            if treasury_10y > 5.0:
                rate_env = '🔴HIGH_RATE'
            elif treasury_10y > 4.0:
                rate_env = '⚠️ELEVATED'
            elif treasury_10y < 3.0:
                rate_env = '✅LOW_RATE'
            else:
                rate_env = 'NORMAL'
            
            if vix > 30:
                vol_env = '🔴HIGH_VOL'
            elif vix > 20:
                vol_env = '⚠️ELEVATED'
            else:
                vol_env = '✅LOW_VOL'
            
            result = {
                'treasury_10y': treasury_10y,
                'vix': vix,
                'rate_env': rate_env,
                'vol_env': vol_env
            }
            
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
            
        except Exception as e:
            logger.warning("Macro analysis failed: %s", e)
            return {
                'treasury_10y': 4.5,
                'vix': 20.0,
                'rate_env': 'UNKNOWN',
                'vol_env': 'UNKNOWN'
            }


# ---------------------------
# Institutional Analyzer (Master)
# ---------------------------
class InstitutionalAnalyzer:
    """Master analyzer combining all institutional signals"""
    
    @staticmethod
    def analyze(ticker):
        """Run all institutional checks"""
        
        signals = {}
        alerts = []
        risk_score = 0  # 0-100, higher = more risky
        
        # 1. Insider Activity
        insider = InsiderTracker.get_insider_activity(ticker)
        signals['insider'] = insider
        if insider['signal'] == '🚨SELL':
            alerts.append(f"Insider売り優勢 {insider['ratio']:.1f}x")
            risk_score += 30
        elif insider['signal'] == '✅BUY':
            alerts.append(f"Insider買い優勢")
            risk_score -= 10
        
        # 2. Short Interest
        short = ShortInterestTracker.get_short_interest(ticker)
        signals['short'] = short
        if short['signal'] == '🚨HIGH':
            alerts.append(f"空売り{short['short_percent']:.0f}%")
            risk_score += 20
        
        # 3. Social Sentiment  
        sentiment = SentimentAnalyzer.get_reddit_sentiment(ticker)
        signals['sentiment'] = sentiment
        if sentiment['signal'] == '🔥HYPE':
            alerts.append(f"SNS過熱 ({sentiment['mentions']}件)")
            risk_score += 25
        elif sentiment['signal'] == '✅QUIET':
            risk_score -= 5
        
        # 4. Google Trends
        trends = GoogleTrendsAnalyzer.get_search_trend(ticker)
        signals['trends'] = trends
        if trends['signal'] == '🔥SPIKE':
            alerts.append("検索急増")
            risk_score += 15
        
        # 5. Institutional Ownership
        inst = InstitutionalOwnership.get_institutional_holdings(ticker)
        signals['institutional'] = inst
        if inst['signal'] == '⚠️WEAK':
            alerts.append(f"機関保有{inst['inst_percent']:.0f}%")
            risk_score += 10
        
        # 6. Options Flow
        options = OptionFlowAnalyzer.get_put_call_ratio(ticker)
        signals['options'] = options
        if options['signal'] == '🐻BEARISH':
            alerts.append(f"Put/Call {options['put_call_ratio']:.2f}")
            risk_score += 15
        elif options['signal'] == '🐂BULLISH':
            risk_score -= 10
        
        # Overall assessment
        if risk_score > 60:
            overall = '🚨HIGH_RISK'
        elif risk_score > 30:
            overall = '⚠️CAUTION'
        elif risk_score < 0:
            overall = '✅LOW_RISK'
        else:
            overall = 'NEUTRAL'
        
        return {
            'signals': signals,
            'alerts': alerts,
            'risk_score': risk_score,
            'overall': overall
        }


# ---------------------------
# Original modules (from v25.2)
# ---------------------------

def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="5d", progress=False)
        if data is None or data.empty:
            return 152.0
        if 'Close' in data.columns:
            return float(data['Close'].iloc[-1])
        return float(data.iloc[-1, -1])
    except Exception as e:
        logger.warning("FX fetch failed: %s", e)
        return 152.0

def jpy_to_usd(jpy, fx):
    return jpy / fx

def get_vix():
    try:
        data = yf.download("^VIX", period="5d", progress=False)
        if data is None or data.empty:
            return 20.0
        if 'Close' in data.columns:
            return float(data['Close'].iloc[-1])
        return float(data.iloc[-1, -1])
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)
        return 20.0

def check_market_trend():
    try:
        spy = yf.download("SPY", period="400d", progress=False)
        if spy is None or spy.empty:
            return True, "Unknown", 0.0
        close = None
        if 'Close' in spy.columns:
            close = spy['Close'].dropna()
        else:
            for c in spy.columns:
                if 'close' in str(c).lower():
                    close = spy[c].dropna()
                    break
        if close is None or len(close) < 210:
            return True, "Unknown", 0.0
        curr = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        dist = ((curr - ma200) / ma200) * 100
        return curr > ma200, f"{'Bull' if curr > ma200 else 'Bear'} ({dist:+.1f}%)", dist
    except Exception as e:
        logger.warning("Market trend check failed: %s", e)
        return True, "Unknown", 0.0

def safe_download(ticker, period="700d", retry=3):
    for attempt in range(retry):
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df, pd.Series):
                df = df.to_frame()
            return df
        except Exception as e:
            logger.warning("yf.download attempt %d failed for %s: %s", attempt+1, ticker, e)
            time.sleep(1 + attempt)
    return pd.DataFrame()

def ensure_df(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df is None:
        return pd.DataFrame()
    return df.copy()

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
            try:
                date_val = cal.iloc[0, 0]
            except Exception:
                return False
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
        if isinstance(sector, (pd.Series, np.ndarray, list, tuple)):
            if len(sector) == 0:
                return True
            sector_key = str(sector[0])
        else:
            sector_key = str(sector)

        etf = SECTOR_ETF.get(sector_key)
        if not etf:
            return True

        if isinstance(etf, (pd.Series, np.ndarray, list, tuple)):
            if len(etf) == 0:
                return True
            etf_sym = str(etf[0])
        else:
            etf_sym = str(etf)

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
        comm = val_usd * COMMISSION_RATE
        slip = val_usd * SLIPPAGE_RATE
        return (comm + slip) * 2

    @staticmethod
    def calculate_total_cost_jpy(val_usd, fx):
        return TransactionCostModel.calculate_total_cost_usd(val_usd) * fx + (val_usd * FX_SPREAD_RATE * fx) * 2

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
                'vol': int(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0
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
# Main mission - WITH INSTITUTIONAL ANALYSIS
# ---------------------------
def run_mission():
    fx = get_current_fx_rate()
    vix = get_vix()
    is_bull, market_status, _ = check_market_trend()
    logger.info("Market: %s | VIX: %.1f | FX: ¥%.2f", market_status, vix, fx)

    # Get macro environment
    macro = MacroAnalyzer.get_macro_environment()

    initial_cap_usd = jpy_to_usd(INITIAL_CAPITAL_JPY, fx)
    trading_cap_usd = initial_cap_usd * TRADING_RATIO

    results = []
    stats = {"Earnings":0, "Sector":0, "Trend":0, "Price":0, "Loose":0, "Data":0, "Pass":0, "Error":0}
    sec_exposures = {s: 0.0 for s in set(TICKERS.values())}

    # Track institutional data for all tickers
    institutional_data = {}

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
                
                # ADD INSTITUTIONAL ANALYSIS
                inst_analysis = InstitutionalAnalyzer.analyze(ticker)
                res['institutional'] = inst_analysis
                institutional_data[ticker] = inst_analysis
                
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

    all_sorted = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    
    passed_strict = [r for r in all_sorted if r[1]['score'] >= MIN_SCORE_STRICT and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]
    passed_standard = [r for r in all_sorted if r[1]['score'] >= MIN_SCORE_STANDARD and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]
    passed_relaxed = [r for r in all_sorted if r[1]['score'] >= MIN_SCORE_RELAXED and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]

    report_lines = []
    report_lines.append("SENTINEL v26.0 INSTITUTIONAL")
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    report_lines.append(f"Mkt: {market_status}")
    report_lines.append(f"VIX: {vix:.1f} | FX: ¥{fx:.2f}")
    report_lines.append(f"10Y: {macro['treasury_10y']:.2f}% | {macro['rate_env']} {macro['vol_env']}")
    report_lines.append("="*40)
    report_lines.append("【TARGET STATUS】")
    report_lines.append(f"Goal: 10% annual / 0.8% monthly")
    report_lines.append(f"Capital: ¥{INITIAL_CAPITAL_JPY:,} (${initial_cap_usd:.0f})")
    report_lines.append(f"Trading: ${trading_cap_usd:.0f} ({TRADING_RATIO*100:.0f}%)")
    report_lines.append("")
    report_lines.append("【STATISTICS】")
    report_lines.append(f"Analyzed: {len(TICKERS)} tickers")
    report_lines.append(f"Blocked by Earnings: {stats['Earnings']}")
    report_lines.append(f"Blocked by Sector:   {stats['Sector']}")
    report_lines.append(f"Blocked by Trend:    {stats['Trend']}")
    report_lines.append(f"Blocked by Loose:    {stats['Loose']}")
    report_lines.append(f"VCP/Score Pass:      {len(all_sorted)}")
    report_lines.append(f"Data Error:          {stats['Data']} / Internal Error: {stats['Error']}")
    report_lines.append("="*40)
    
    report_lines.append("【SIGNALS BY TIER】")
    report_lines.append(f"STRICT   (≥{MIN_SCORE_STRICT}pt): {len(passed_strict)} signals")
    report_lines.append(f"STANDARD (≥{MIN_SCORE_STANDARD}pt): {len(passed_standard)} signals")
    report_lines.append(f"RELAXED  (≥{MIN_SCORE_RELAXED}pt): {len(passed_relaxed)} signals")
    report_lines.append("")
    
    report_lines.append("【BUY SIGNALS - INSTITUTIONAL FILTERED】")

    passed = passed_standard
    
    if not passed:
        report_lines.append(f"No candidates at STANDARD tier ({MIN_SCORE_STANDARD}+ pts).")
        if len(passed_relaxed) > 0:
            report_lines.append(f"→ RELAXED tier has {len(passed_relaxed)} candidates")
    else:
        for i, (ticker, r) in enumerate(passed[:MAX_NOTIFICATIONS], 1):
            pos_usd = r['pos_usd']
            price = r['price']
            est_shares = r['est_shares']
            roundtrip_cost_usd = TransactionCostModel.calculate_total_cost_usd(pos_usd)
            shares_str = f"{est_shares:.4f}" if ALLOW_FRACTIONAL else f"{int(est_shares)}"
            
            inst = r.get('institutional', {})
            risk_score = inst.get('risk_score', 0)
            overall = inst.get('overall', 'UNKNOWN')
            alerts = inst.get('alerts', [])
            
            risk_icon = ""
            if overall == '🚨HIGH_RISK':
                risk_icon = "🚨"
            elif overall == '⚠️CAUTION':
                risk_icon = "⚠️"
            elif overall == '✅LOW_RISK':
                risk_icon = "✅"
            
            report_lines.append(f"{risk_icon} [{i}] {ticker} {r['score']}pt | Risk:{risk_score} {overall}")
            report_lines.append(f"   Entry: ${r['pivot']:.2f} / Price: ${price:.2f} / Shares: {shares_str}")
            report_lines.append(f"   Pos(USD): ${pos_usd:,.2f} / Cost: ${roundtrip_cost_usd:,.2f}")
            report_lines.append(f"   BT: {r['bt']['message']} T:{r['tightness']:.2f}")
            
            if alerts:
                report_lines.append(f"   ⚠️ {' | '.join(alerts[:3])}")

    report_lines.append("\n【TOP 15 INSTITUTIONAL ANALYSIS】")
    for i, (ticker, r) in enumerate(all_sorted[:15], 1):
        tag = "✅OK"
        if r.get('is_earnings'): 
            tag = "❌EARN"
        elif r.get('is_sector_weak'): 
            tag = "❌SEC"
        elif r['score'] < MIN_SCORE_STANDARD: 
            tag = f"⚠️{r['score']}pt"
        
        inst = r.get('institutional', {})
        risk = inst.get('risk_score', 0)
        overall = inst.get('overall', 'UNK')
        
        shares_str = f"{r.get('est_shares', 0):.2f}" if ALLOW_FRACTIONAL else f"{int(r.get('est_shares', 0))}"
        report_lines.append(f"{i:2}. {ticker:5} {r['score']:3}pt | {tag} | Risk:{risk:2} {overall}")
        report_lines.append(f"    T:{r['tightness']:.2f} WR:{r['bt']['winrate']:.0f}% Sh:{shares_str}")
        
        # Show key institutional signals
        signals = inst.get('signals', {})
        insider_sig = signals.get('insider', {}).get('signal', '')
        options_sig = signals.get('options', {}).get('signal', '')
        sentiment_sig = signals.get('sentiment', {}).get('signal', '')
        
        sig_str = []
        if insider_sig and insider_sig != 'NEUTRAL':
            sig_str.append(f"Ins:{insider_sig}")
        if options_sig and options_sig != 'NEUTRAL':
            sig_str.append(f"Opt:{options_sig}")
        if sentiment_sig and sentiment_sig != 'NORMAL':
            sig_str.append(f"Sent:{sentiment_sig}")
        
        if sig_str:
            report_lines.append(f"    {' | '.join(sig_str)}")

    report_lines.append("\n【MACRO INSIGHTS】")
    report_lines.append(f"✓ 10Y Treasury: {macro['treasury_10y']:.2f}% ({macro['rate_env']})")
    report_lines.append(f"✓ VIX: {vix:.1f} ({macro['vol_env']})")
    if macro['rate_env'] == '🔴HIGH_RATE':
        report_lines.append("⚠️ 高金利環境 - グロース株に逆風")
    elif macro['rate_env'] == '✅LOW_RATE':
        report_lines.append("✅ 低金利環境 - グロース株に追い風")

    final_report = "\n".join(report_lines)
    logger.info("\n%s", final_report)
    send_line(final_report)

if __name__ == "__main__":
    run_mission()
