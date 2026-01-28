#!/usr/bin/env python3
# SENTINEL v26.0 - TARGET-DRIVEN VERSION
# Goal: 10% annual return with 350k JPY initial + 30k JPY quarterly additions
# Requirements: pandas, numpy, yfinance, requests
# Usage: python sentinel_v26_optimized.py

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import requests
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
# CONFIG - TARGET-DRIVEN PARAMETERS
# ---------------------------
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

# Portfolio settings
INITIAL_CAPITAL_JPY = 350_000
QUARTERLY_ADDITION_JPY = 30_000
TRADING_RATIO = 0.75  # 70% → 75% (より積極的に投資)

# Risk management
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25
MAX_SECTOR_CONCENTRATION = 0.40

# Scoring - MULTI-TIER SYSTEM for better opportunity capture
SCORE_THRESHOLDS = {
    'strict': 70,      # Ultra-conservative
    'standard': 60,    # Original baseline
    'relaxed': 50,     # Moderate risk
    'aggressive': 40   # High risk tolerance
}

# Tightness - DYNAMIC based on VIX
TIGHTNESS_DYNAMIC = True  # Enable dynamic adjustment

MAX_NOTIFICATIONS = 10  # Show more candidates

MIN_DAILY_VOLUME_USD = 10_000_000

# Transaction costs
COMMISSION_RATE = 0.002
SLIPPAGE_RATE = 0.001
FX_SPREAD_RATE = 0.0005

# Reward multipliers
REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech']

ALLOW_FRACTIONAL = True  # Enable fractional shares for better capital utilization

# Target tracking
ANNUAL_TARGET_RETURN = 0.10  # 10% annual target
MONTHLY_TARGET_RETURN = (1 + ANNUAL_TARGET_RETURN) ** (1/12) - 1  # ~0.797%

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
# Utilities
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

# ---------------------------
# Data helpers
# ---------------------------
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

# ---------------------------
# Earnings & sector
# ---------------------------
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
    """Robust sector strength check that always returns a Python bool."""
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

# ---------------------------
# Transaction cost model
# ---------------------------
class TransactionCostModel:
    @staticmethod
    def calculate_total_cost_usd(val_usd):
        comm = val_usd * COMMISSION_RATE
        slip = val_usd * SLIPPAGE_RATE
        return (comm + slip) * 2

    @staticmethod
    def calculate_total_cost_jpy(val_usd, fx):
        return TransactionCostModel.calculate_total_cost_usd(val_usd) * fx + (val_usd * FX_SPREAD_RATE * fx) * 2

# ---------------------------
# Position sizing
# ---------------------------
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

# ---------------------------
# Backtest - ENHANCED VERSION
# ---------------------------
def simulate_past_performance_v3(df, sector, lookback_years=3):
    """Enhanced backtest with more realistic simulation"""
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
            return {'winrate':0, 'net_expectancy':0, 'message':'LowData', 'trades':0}

        end_date = close.index[-1]
        start_date = end_date - pd.DateOffset(years=lookback_years)
        mask = close.index >= start_date
        close = close.loc[mask]
        high = high.loc[mask]
        low = low.loc[mask]
        if len(close) < 60:
            return {'winrate':0, 'net_expectancy':0, 'message':'ShortWindow', 'trades':0}

        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=7).mean().dropna()
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']

        wins = 0; losses = 0; total_r = 0.0; samples = 0
        
        # More lenient entry conditions for better signal frequency
        for i in range(50, len(close)-40):
            try:
                window_high = high.iloc[i-5:i].max()
                pivot = window_high * 1.002
                if high.iloc[i] < pivot:
                    continue
                ma50 = close.rolling(50, min_periods=10).mean().iloc[i]
                ma200 = close.rolling(200, min_periods=50).mean().iloc[i] if i >= 200 else None
                
                # Relaxed trend condition
                if ma200 is not None and close.iloc[i] < ma50 * 0.97:  # Allow slight pullbacks
                    continue
                elif ma200 is None and close.iloc[i] < ma50 * 0.95:
                    continue
                    
                stop_dist = atr.iloc[i] * ATR_STOP_MULT if i < len(atr) else atr.iloc[-1] * ATR_STOP_MULT
                entry = pivot
                target = entry + stop_dist * reward_mult
                outcome = None
                
                # Check outcome over 30 days
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
        if total < 5:  # Reduced from 8 to allow more data
            return {'winrate':0, 'net_expectancy':0, 'message':f'LowSample:{total}', 'trades':total}
        wr = (wins / total) * 100
        ev = total_r / total
        
        msg = f"WR{wr:.0f}%/EV{ev:+.2f}R/{total}T"
        return {'winrate':wr, 'net_expectancy':ev, 'message':msg, 'trades':total}

    except Exception as e:
        logger.exception("Backtest error: %s", e)
        return {'winrate':0, 'net_expectancy':0, 'message':'Error', 'trades':0}

# ---------------------------
# Strategic Analyzer - ENHANCED
# ---------------------------
class StrategicAnalyzerV3:
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_pos_val_usd, vix, sec_exposures, cap_usd, is_bull):
        try:
            df = ensure_df(df)
            
            # Normalize columns
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

            close = df['Close'].dropna() if 'Close' in df.columns else None
            high = df['High'].dropna() if 'High' in df.columns else None
            low = df['Low'].dropna() if 'Low' in df.columns else None
            vol = df['Volume'].dropna() if 'Volume' in df.columns else None

            if close is None or high is None or low is None or len(close) < 220:
                return None, "❌DATA"

            curr = float(close.iloc[-1])
            
            # Volume filter
            if vol is not None and len(vol) > 0:
                avg_vol = float(vol.rolling(20, min_periods=5).mean().iloc[-1])
                avg_vol_usd = avg_vol * curr
                if avg_vol_usd < MIN_DAILY_VOLUME_USD:
                    return None, "❌VOL"

            # Price filter - require reasonable price (not penny stock)
            if curr < 5.0:
                return None, "❌PRICE"

            # Calculate max shares buyable
            max_shares = max_pos_val_usd / curr if curr > 0 else 0

            # ATR calculation
            tr = pd.concat([
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14, min_periods=7).mean().iloc[-1]
            atr_pct = atr14 / curr if curr > 0 else 0.0

            # Moving averages
            ma20 = safe_rolling_last(close, 20, min_periods=10, default=None)
            ma50 = safe_rolling_last(close, 50, min_periods=20, default=None)
            ma200 = safe_rolling_last(close, 200, min_periods=100, default=None)

            if ma20 is None or ma50 is None:
                return None, "❌DATA"

            # === ENHANCED SCORING SYSTEM ===
            score = 0
            reasons = []
            score_details = {}

            # Stage analysis (30 points max)
            stage_score = 0
            if curr > ma20:
                stage_score += 10; reasons.append("MA20+")
            if ma20 > ma50:
                stage_score += 10; reasons.append("MA50+")
            if ma200 is not None and ma50 > ma200:
                stage_score += 10; reasons.append("MA200+")
            elif ma200 is None and curr > ma50:
                stage_score += 5; reasons.append("MA200?")
            score += stage_score
            score_details['stage'] = stage_score

            # VCP pattern (25 points max)
            vcp_score = 0
            try:
                recent_high = high.iloc[-60:].max() if len(high) >= 60 else high.max()
                recent_low = low.iloc[-60:].max() if len(low) >= 60 else low.min()
                range_pct = ((recent_high - recent_low) / recent_low) * 100 if recent_low > 0 else 100
                
                last_high = high.iloc[-20:].max()
                last_low = low.iloc[-20:].min()
                last_range = ((last_high - last_low) / last_low) * 100 if last_low > 0 else 100
                
                tightness = last_range / range_pct if range_pct > 0 else 999
                
                # Dynamic tightness threshold based on VIX
                if TIGHTNESS_DYNAMIC:
                    if vix < 15:
                        tight_thresh = 2.5  # Very relaxed in calm markets
                    elif vix < 18:
                        tight_thresh = 2.0
                    elif vix < 25:
                        tight_thresh = 1.5
                    else:
                        tight_thresh = 1.2
                else:
                    tight_thresh = 1.5
                
                if tightness <= tight_thresh:
                    vcp_score = 25; reasons.append("VCP++")
                elif tightness <= tight_thresh * 1.5:
                    vcp_score = 15; reasons.append("VCP+")
                elif tightness <= tight_thresh * 2:
                    vcp_score = 5; reasons.append("VCP")
                else:
                    return None, "❌LOOSE"
                    
                score += vcp_score
                score_details['vcp'] = vcp_score
                
            except Exception:
                return None, "❌CALC"

            # Volume pattern (15 points max)
            vol_score = 0
            try:
                vol50 = safe_rolling_last(vol, 50, min_periods=10, default=np.nan)
                vol20 = safe_rolling_last(vol, 20, min_periods=5, default=np.nan)
                
                if not pd.isna(vol50) and vol.iloc[-1] < vol50 * 0.8:
                    vol_score += 10; reasons.append("VolDry+")
                elif not pd.isna(vol50) and vol.iloc[-1] < vol50:
                    vol_score += 5; reasons.append("VolDry")
                    
                score += vol_score
                score_details['volume'] = vol_score
            except Exception:
                pass

            # Momentum (20 points max)
            mom_score = 0
            try:
                mom5 = safe_rolling_last(close, 5, min_periods=3, default=np.nan)
                mom20 = safe_rolling_last(close, 20, min_periods=10, default=np.nan)
                
                if not pd.isna(mom5) and not pd.isna(mom20):
                    mom_ratio = mom5 / mom20
                    if mom_ratio > 1.03:
                        mom_score = 20; reasons.append("Mom++")
                    elif mom_ratio > 1.01:
                        mom_score = 10; reasons.append("Mom+")
                        
                score += mom_score
                score_details['momentum'] = mom_score
            except Exception:
                pass

            # Trend strength (20 points max)
            trend_score = 0
            try:
                if ma200 is not None:
                    slope = ((ma50 - ma200) / ma200) * 100
                    if slope > 5:
                        trend_score = 20; reasons.append("Trend++")
                    elif slope > 2:
                        trend_score = 15; reasons.append("Trend+")
                    elif slope > 0:
                        trend_score = 10; reasons.append("Trend")
                else:
                    if curr > ma50 * 1.02:
                        trend_score = 10; reasons.append("Trend?")
                        
                score += trend_score
                score_details['trend'] = trend_score
            except Exception:
                pass

            # Backtest with enhanced version
            bt = simulate_past_performance_v3(df, sector)
            winrate = bt.get('winrate', 0) / 100.0
            
            # Bonus points for strong backtest (10 points max)
            bt_score = 0
            if winrate > 0.55:
                bt_score = 10; reasons.append("BT++")
            elif winrate > 0.45:
                bt_score = 5; reasons.append("BT+")
            score += bt_score
            score_details['backtest'] = bt_score

            # Position sizing
            try:
                pos_val_usd, frac = PositionSizer.calculate_position(
                    cap_usd, winrate, 2.0, atr_pct, vix, 
                    float(sec_exposures.get(sector, 0.0))
                )
            except Exception as e:
                logger.exception("PositionSizer error for %s: %s", ticker, e)
                pos_val_usd, frac = 0.0, 0.0

            # Convert to shares
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
                'score_details': score_details,
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
                'vol': int(vol.iloc[-1]) if vol is not None and not pd.isna(vol.iloc[-1]) else 0
            }
            return result, "✅PASS"

        except Exception as e:
            logger.exception("Analyze error for %s: %s", ticker, e)
            return None, "❌ERROR"

# ---------------------------
# Messaging (LINE optional)
# ---------------------------
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
# Main mission - ENHANCED
# ---------------------------
def run_mission():
    fx = get_current_fx_rate()
    vix = get_vix()
    is_bull, market_status, market_dist = check_market_trend()
    logger.info("Market: %s | VIX: %.1f | FX: ¥%.2f", market_status, vix, fx)

    initial_cap_usd = jpy_to_usd(INITIAL_CAPITAL_JPY, fx)
    trading_cap_usd = initial_cap_usd * TRADING_RATIO

    results = []
    stats = {
        "Earnings":0, "Sector":0, "Trend":0, "Price":0, 
        "Loose":0, "Vol":0, "Data":0, "Pass":0, "Error":0
    }
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

            res, reason = StrategicAnalyzerV3.analyze_ticker(
                ticker, df_t, sector, max_pos_val_usd, vix, sec_exposures, trading_cap_usd, is_bull
            )

            if res:
                res['is_earnings'] = earnings_flag
                res['is_sector_weak'] = sector_flag
                results.append((ticker, res))
                if not earnings_flag and not sector_flag:
                    stats["Pass"] += 1
                    # Update sector exposure
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
                elif "VOL" in reason:
                    stats["Vol"] += 1
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
    
    # Multi-tier filtering
    tier_results = {}
    for tier_name, threshold in SCORE_THRESHOLDS.items():
        tier_results[tier_name] = [
            r for r in all_sorted 
            if r[1]['score'] >= threshold 
            and not r[1].get('is_earnings', False) 
            and not r[1].get('is_sector_weak', False)
        ]

    # === BUILD REPORT ===
    report_lines = []
    report_lines.append("SENTINEL v26.0 TARGET-DRIVEN")
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    report_lines.append(f"Mkt: {market_status}")
    report_lines.append(f"VIX: {vix:.1f} | FX: ¥{fx:.2f}")
    report_lines.append("="*40)
    
    # TARGET TRACKING
    report_lines.append("【TARGET STATUS】")
    report_lines.append(f"Annual Goal: +{ANNUAL_TARGET_RETURN*100:.1f}%")
    report_lines.append(f"Monthly Need: +{MONTHLY_TARGET_RETURN*100:.2f}%")
    report_lines.append(f"Capital: ¥{INITIAL_CAPITAL_JPY:,} → ${initial_cap_usd:.0f}")
    report_lines.append(f"Trading: ${trading_cap_usd:.0f} ({TRADING_RATIO*100:.0f}%)")
    report_lines.append("")
    
    report_lines.append("【STATISTICS】")
    report_lines.append(f"Analyzed: {len(TICKERS)} tickers")
    report_lines.append(f"Blocked by Earnings: {stats['Earnings']}")
    report_lines.append(f"Blocked by Sector:   {stats['Sector']}")
    report_lines.append(f"Blocked by Trend:    {stats['Trend']}")
    report_lines.append(f"Blocked by Loose:    {stats['Loose']}")
    report_lines.append(f"Blocked by Volume:   {stats['Vol']}")
    report_lines.append(f"VCP/Score Pass:      {len(all_sorted)}")
    report_lines.append(f"Data/Internal Error: {stats['Data']}/{stats['Error']}")
    report_lines.append("="*40)
    
    # MULTI-TIER SIGNALS
    report_lines.append("【SIGNALS BY TIER】")
    for tier_name in ['strict', 'standard', 'relaxed', 'aggressive']:
        threshold = SCORE_THRESHOLDS[tier_name]
        count = len(tier_results[tier_name])
        report_lines.append(f"{tier_name.upper():<12} (≥{threshold}pt): {count} signals")
    report_lines.append("")
    
    # Show standard tier signals (default)
    active_tier = 'standard'
    passed = tier_results[active_tier]
    
    report_lines.append(f"【BUY SIGNALS - {active_tier.upper()} TIER】")
    if not passed:
        report_lines.append(f"No candidates at {active_tier} tier ({SCORE_THRESHOLDS[active_tier]}+ pts).")
        # Suggest relaxed tier if available
        if len(tier_results['relaxed']) > 0:
            report_lines.append(f"→ Consider RELAXED tier: {len(tier_results['relaxed'])} candidates available")
    else:
        for i, (ticker, r) in enumerate(passed[:MAX_NOTIFICATIONS], 1):
            pos_usd = r['pos_usd']
            price = r['price']
            est_shares = r['est_shares']
            roundtrip_cost_usd = TransactionCostModel.calculate_total_cost_usd(pos_usd)
            shares_str = f"{est_shares:.4f}" if ALLOW_FRACTIONAL else f"{int(est_shares)}"
            
            report_lines.append(f"★ [{i}] {ticker} {r['score']}pt ({r['reasons']})")
            report_lines.append(f"   Entry: ${r['pivot']:.2f} / Current: ${price:.2f}")
            report_lines.append(f"   Shares: {shares_str} / Pos(USD): ${pos_usd:,.0f}")
            report_lines.append(f"   BT: {r['bt']['message']} / Tight:{r['tightness']:.2f}")
            
            # Score breakdown
            details = r['score_details']
            breakdown = " | ".join([f"{k}:{v}" for k,v in details.items()])
            report_lines.append(f"   Score: {breakdown}")

    report_lines.append("\n【TOP 15 ANALYSIS (ALL)】")
    for i, (ticker, r) in enumerate(all_sorted[:15], 1):
        tag = "✅OK"
        if r.get('is_earnings'): 
            tag = "❌EARN"
        elif r.get('is_sector_weak'): 
            tag = "❌SEC"
        elif r['score'] < SCORE_THRESHOLDS['standard']: 
            tag = f"⚠️{r['score']}pt"
            
        shares_str = f"{r.get('est_shares', 0):.2f}" if ALLOW_FRACTIONAL else f"{int(r.get('est_shares', 0))}"
        report_lines.append(f"{i:2}. {ticker:5} {r['score']:3}pt | {tag}")
        report_lines.append(f"    {r['reasons']} | T:{r['tightness']:.2f} WR:{r['bt']['winrate']:.0f}% Sh:{shares_str}")

    # ACTIONABLE INSIGHTS
    report_lines.append("\n【INSIGHTS】")
    if vix < 18:
        report_lines.append("✓ Low VIX environment - tightness relaxed to 2.0")
    if market_dist > 5:
        report_lines.append("✓ Strong bull market - momentum plays favored")
    if len(tier_results['standard']) == 0 and len(tier_results['relaxed']) > 0:
        report_lines.append("⚠ No standard signals - consider relaxed tier")
    if len(all_sorted) > 0 and len(passed) == 0:
        top_score = all_sorted[0][1]['score']
        report_lines.append(f"⚠ Top score: {top_score}pt (need {SCORE_THRESHOLDS[active_tier]}pt)")

    final_report = "\n".join(report_lines)
    logger.info("\n%s", final_report)
    send_line(final_report)

if __name__ == "__main__":
    run_mission()
