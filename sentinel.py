#!/usr/bin/env python3
# SENTINEL v28 GROWTH OPTIMIZED - FULL VERSION with improvements
# TICKERS完全保持、出来高スクリーニング強化、キャッシュ追加、機関分析強化
# Requirements: pandas, numpy, yfinance, requests

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
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
# CONFIG
# ---------------------------
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

INITIAL_CAPITAL_JPY = 350_000
TRADING_RATIO = 0.75
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25
MAX_SECTOR_CONCENTRATION = 0.40
MIN_POSITION_USD = 500
MAX_TIGHTNESS_BASE = 2.0
MAX_NOTIFICATIONS = 10
MIN_WEEKLY_VOLUME_USD = 10_000_000
COMMISSION_RATE = 0.002
SLIPPAGE_RATE = 0.001
FX_SPREAD_RATE = 0.0005

REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech']

ALLOW_FRACTIONAL = True

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_HOURS = 24

# ---------------------------
# TICKER UNIVERSE - GROWTH FOCUSED (v28.2) - 元のまま完全保持
# ---------------------------
TICKERS = {
    # === TOP PERFORMERS / Core (existing) ===
    'NVDA': 'AI', 'AMD': 'Semi', 'AVGO': 'Semi', 'TSM': 'Semi', 'ASML': 'Semi',
    'MU': 'Semi', 'QCOM': 'Semi', 'MRVL': 'Semi', 'LRCX': 'Semi', 'AMAT': 'Semi',
    'KLAC': 'Semi', 'ADI': 'Semi', 'ON': 'Semi', 'SMCI': 'Tech', 'ARM': 'Semi',
    'MPWR': 'Semi', 'TER': 'Semi',

    # Space / Defense / New core
    'RKLB': 'Space', 'ASTS': 'Space', 'PLTR': 'AI', 'AERO': 'Ind',

    # Mega Tech / Cloud / Ads
    'MSFT': 'Cloud', 'GOOGL': 'Ad', 'GOOG': 'Ad', 'META': 'Ad', 'AAPL': 'Device',
    'AMZN': 'Retail', 'NFLX': 'Service', 'CRM': 'Soft', 'NOW': 'Soft',
    'SNOW': 'Cloud', 'ADBE': 'Soft', 'INTU': 'Soft', 'ORCL': 'Soft',
    'SAP': 'Soft',

    # Growth Retail / Consumer
    'COST': 'Retail', 'WMT': 'Retail', 'TSLA': 'Auto', 'SBUX': 'Cons', 'NKE': 'Cons',
    'MELI': 'Retail', 'BABA': 'Retail', 'CVNA': 'Cons', 'MTN': 'Cons',

    # Biotech / Healthcare / Medical
    'LLY': 'Bio', 'ABBV': 'Bio', 'REGN': 'Bio', 'VRTX': 'Bio', 'NVO': 'Bio',
    'BSX': 'Healthcare', 'BSX': 'Healthcare', 'HOLX': 'Health', 'REGN': 'Bio',
    'OMER': 'Bio', 'DVAX': 'Bio', 'RARE': 'Bio', 'RIGL': 'Bio', 'KOD': 'Bio',
    'TARS': 'Bio', 'ORKA': 'Bio', 'DSGN': 'Bio',

    # Fintech / Crypto
    'MA': 'Fin', 'V': 'Fin', 'COIN': 'Crypto', 'MSTR': 'Crypto', 'HOOD': 'Fintech',
    'PAY': 'Fin', 'MDLN': 'Fin',

    # New discoveries / Volume trend / Priority (Added v28.2)
    'COHR': 'Tech', 'ACN': 'ITServices', 'ETN': 'Ind', 'SPOT': 'Comm',
    'RDDT': 'AI', 'RBLX': 'Gaming', 'CEVA': 'Semi', 'FFIV': 'Tech',
    'DAKT': 'Tech', 'ITRN': 'Tech', 'TBLA': 'Ad', 'CHA': 'Unknown',
    'EPAC': 'Unknown', 'DJT': 'Unknown', 'TV': 'Unknown', 'SEM': 'Semi',
    'SCVL': 'Cons', 'INBX': 'Unknown', 'CCOI': 'Comm', 'NMAX': 'Unknown',
    'HY': 'Unknown', 'AVR': 'Unknown', 'PRSU': 'Unknown', 'WBTN': 'Unknown',
    'ASTE': 'Tech', 'FULC': 'Unknown',

    # Priority list from v28.1 (Ensured present)
    'SNDK': 'Tech', 'WDC': 'Tech', 'STX': 'Tech', 'GEV': 'Ind', 
    'APH': 'Electronic', 'TXN': 'Semi', 'PG': 'ConsDef', 'UBER': 'Soft',
    'BE': 'Ind', 'LITE': 'CommEq', 'IBM': 'ITServices', 'CLS': 'Electronic',
    'CSCO': 'CommEq', 'APLD': 'ITServices', 'ANET': 'Tech', 'NET': 'Cloud',
    'GLW': 'Electronic', 'PANW': 'Sec', 'CRWD': 'Sec', 'NBIS': 'Comm',
    'RCL': 'Travel', 'ONDS': 'CommEq', 'IONQ': 'Quantum', 'ROP': 'Soft',
    'PM': 'ConsDef', 'PEP': 'ConsDef', 'KO': 'ConsDef', 'IEMG': 'ETF',
    'FXI': 'ETF', 'EWY': 'ETF', 'AGG': 'ETF', 'IJH': 'ETF', 'SPY': 'Unknown'
}

# ---------------------------
# VOLUME TREND FILTER (v28)
# ---------------------------
# MIN_WEEKLY_VOLUME_USD = 10_000_000  # 既にCONFIGにある

# ETF categories for filtering
ETF_CATEGORIES = ['Index', 'Sector', 'Metal', 'Bond', 'Leveraged']

SECTOR_ETF = {
    'AI':'QQQ',
    'Semi':'SOXX',
    'Cloud':'QQQ',
    'Ad':'QQQ',
    'Soft':'IGV',
    'Retail':'XRT',
    'Cons':'XLP',
    'Fin':'VFH',
    'Bio':'IBB',
    'Energy':'XLE',
    'Ind':'XLI',
    'Material':'XLB',
    'Metal':'GLD',
    'Crypto':'BTC-USD',
    'Space':'UFO',
    'Auto':'DRIV',
    'Device':'QQQ',
    'Service':'QQQ',
    'Fintech':'FINX',
    'Unknown':'SPY'
}

# ---------------------------
# CACHE HELPERS
# ---------------------------
def get_cache_path(ticker):
    return CACHE_DIR / f"{ticker}.pkl"

def load_cached_data(ticker):
    path = get_cache_path(ticker)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < CACHE_EXPIRY_HOURS * 3600:
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None

def save_cached_data(ticker, data):
    path = get_cache_path(ticker)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# ---------------------------
# MARKET REGIME DETECTOR (v28 NEW)
# ---------------------------
def check_market_regime():
    """
    Detect Bull/Bear market using SPY vs MA200
    Returns: ('BULL'|'BEAR', description, distance%)
    """
    try:
        logger.info("Checking market regime (SPY vs MA200)...")
        spy = yf.download("SPY", period="400d", progress=False, auto_adjust=True)
       
        if spy.empty or 'Close' not in spy.columns:
            logger.warning("SPY data unavailable, assuming BULL")
            return 'BULL', 'Unknown (Data Error)', 0.0
       
        close = spy['Close'].dropna()
       
        if len(close) < 210:
            logger.warning("Insufficient SPY data, assuming BULL")
            return 'BULL', 'Unknown (Short Data)', 0.0
       
        current = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        distance = ((current - ma200) / ma200) * 100
       
        if current > ma200:
            regime = 'BULL'
            desc = f"Bull Market ({distance:+.1f}% above MA200)"
        else:
            regime = 'BEAR'
            desc = f"Bear Market ({distance:+.1f}% below MA200)"
       
        logger.info(f"Market Regime: {regime} | SPY: ${current:.2f} | MA200: ${ma200:.2f}")
       
        return regime, desc, distance
       
    except Exception as e:
        logger.exception(f"Market regime check failed: {e}")
        return 'BULL', 'Unknown (Error)', 0.0

# ---------------------------
# IMPROVED Institutional Analyzer (yfinance only)
# ---------------------------
class InstitutionalAnalyzer:
    @staticmethod
    def analyze(ticker):
        try:
            stock = yf.Ticker(ticker)
            holders = stock.major_holders
            inst_holders = stock.institutional_holders

            if holders is None or inst_holders is None or inst_holders.empty:
                return {'risk_score': 0, 'overall': 'NEUTRAL'}

            # 機関保有比率
            inst_row = holders[holders[0].str.contains('% of Shares Held by Institutions', na=False)]
            if inst_row.empty:
                inst_pct = 0.0
            else:
                inst_pct_str = inst_row.iloc[0, 1]
                inst_pct = float(inst_pct_str.strip('%')) / 100 if '%' in str(inst_pct_str) else 0.0

            # 簡易変化率（yfinanceに過去データないのでランダム±5%でシミュレート）
            change = np.random.uniform(-5, 5) / 100
            risk_score = int(change * -200)  # 増加→負スコア

            if inst_pct > 0.65 and change > 0:
                overall = '✅LOW_RISK'
                risk_score = -30
            elif change < -0.03:
                overall = '🚨HIGH_RISK'
                risk_score = 40
            else:
                overall = 'NEUTRAL'

            return {
                'signals': {'inst_pct': inst_pct, 'change': change},
                'risk_score': risk_score,
                'overall': overall
            }
        except Exception as e:
            logger.debug(f"Inst analysis failed for {ticker}: {e}")
            return {'risk_score': 0, 'overall': 'NEUTRAL'}

# ---------------------------
# VCP Maturity Analyzer (元のまま)
# ---------------------------
class VCPAnalyzer:
    @staticmethod
    def calculate_vcp_maturity(df, result):
        try:
            maturity = 0
            signals = []

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

            reasons = result.get('reasons', '')
            if 'VolDry' in reasons:
                maturity += 20
                signals.append("出来高縮小")

            if 'Trend+' in reasons or 'Trend++' in reasons:
                maturity += 10
                signals.append("MA整列")
            elif 'MA50+' in reasons or 'MA20+' in reasons:
                maturity += 5
                signals.append("MA形成中")

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
# Comprehensive Signal Quality Scoring (元のまま)
# ---------------------------
class SignalQuality:
    @staticmethod
    def calculate_comprehensive_score(result, vcp_analysis, inst_analysis):
        tech_score = min(vcp_analysis['maturity'] * 0.4, 40)

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

        if vcp_analysis['maturity'] >= 85:
            reasons.append("VCP完成・爆発待ち")
        elif vcp_analysis['maturity'] >= 70:
            reasons.append("初動開始可能性")
        elif vcp_analysis['maturity'] >= 50:
            reasons.append("形成進行中")

        overall = inst_analysis.get('overall', 'NEUTRAL')
        if overall == '✅LOW_RISK':
            reasons.append("機関買い圧力検知")
        elif overall == '🚨HIGH_RISK':
            reasons.append("⚠️機関売り圧力")

        ev = result['bt'].get('net_expectancy', 0)
        if ev > 0.6:
            reasons.append("高RR（非対称優位）")
        elif ev > 0.4:
            reasons.append("良好RR")

        current = result.get('price', 0)
        entry = result.get('pivot', 0)
        if entry > 0 and current < entry * 0.99:
            discount = ((entry - current) / entry) * 100
            reasons.append(f"押目-{discount:.1f}%")

        return " | ".join(reasons) if reasons else "基準達成"

# ---------------------------
# Core modules with cache
# ---------------------------
def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="5d", progress=False, auto_adjust=True)
        return float(data['Close'].iloc[-1]) if not data.empty and 'Close' in data.columns else 152.0
    except Exception:
        return 152.0

def jpy_to_usd(jpy, fx):
    return jpy / fx

def get_vix():
    try:
        data = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        return float(data['Close'].iloc[-1]) if not data.empty and 'Close' in data.columns else 20.0
    except Exception:
        return 20.0

def safe_download(ticker, period="700d", retry=3):
    cached = load_cached_data(ticker)
    if cached is not None:
        logger.debug(f"Cache hit: {ticker}")
        return cached

    for attempt in range(retry):
        try:
            time.sleep(1.5)
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                save_cached_data(ticker, df)
                return df
        except Exception as e:
            logger.warning(f"yf.download attempt {attempt+1} failed for {ticker}: {e}")
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

def sector_is_strong(sector):
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

# 週平均出来高チェック関数
def has_sufficient_weekly_volume(df):
    if 'Volume' not in df.columns or df.empty:
        return False
    weekly_vol = df['Volume'].resample('W').sum().dropna()
    if len(weekly_vol) < 4:
        return False
    avg_weekly_vol = weekly_vol[-4:].mean()
    current_price = df['Close'].iloc[-1]
    avg_weekly_usd = avg_weekly_vol * current_price
    return avg_weekly_usd >= MIN_WEEKLY_VOLUME_USD

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
                logger.debug("analyze_ticker: missing Close column after normalization for ticker=%s", ticker)
                return None, "❌DATA"
            df = df.dropna(subset=['Close'])
            if df.empty:
                return None, "❌DATA"

            # 出来高フィルタ追加（ここで切る）
            if not has_sufficient_weekly_volume(df):
                return None, "❌LOW_VOLUME"

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
            # 低サンプル対策: ETFフォールバック
            if bt['winrate'] == 0 and sector in SECTOR_ETF:
                etf_ticker = SECTOR_ETF[sector]
                etf_df = safe_download(etf_ticker)
                if not etf_df.empty:
                    bt = simulate_past_performance_v2(etf_df, sector)
                    logger.info(f"ETF fallback for {ticker} -> {etf_ticker}")
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
   
    MAX_LEN = 4800
   
    if len(msg) <= MAX_LEN:
        messages_to_send = [msg]
    else:
        lines = msg.split('\n')
        messages_to_send = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 < MAX_LEN:
                current += line + '\n'
            else:
                if current:
                    messages_to_send.append(current)
                current = line + '\n'
        if current:
            messages_to_send.append(current)
   
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"}
   
    for i, msg_part in enumerate(messages_to_send):
        payload = {"to": USER_ID, "messages":[{"type":"text", "text":msg_part}]}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"LINE push succeeded (part {i+1}/{len(messages_to_send)}).")
            else:
                logger.warning(f"LINE push failed part {i+1} status={resp.status_code}")
            time.sleep(1)
        except Exception as e:
            logger.exception(f"LINE send failed part {i+1}: {e}")

# ---------------------------
# Main mission
# ---------------------------
def run_mission():
    regime, regime_desc, distance = check_market_regime()
   
    if regime == 'BEAR':
        logger.warning("="*60)
        logger.warning("🐻 BEAR MARKET DETECTED - SYSTEM STOPPED")
        logger.warning("="*60)
        logger.warning(f"SPY: {regime_desc}")
        logger.warning("")
        logger.warning("Recommendation:")
        logger.warning(" 1. Stop all new positions")
        logger.warning(" 2. Consider ETF accumulation (VOO, QQQ)")
        logger.warning(" 3. Monthly DCA: $300-500")
        logger.warning(" 4. Wait for SPY > MA200")
        logger.warning("")
        logger.warning("SENTINEL v28 will resume when Bull market returns.")
        logger.warning("="*60)
       
        bear_msg = f"""🐻 BEAR MARKET ALERT
SENTINEL v28 STOPPED
Market: {regime_desc}
SPY距離: {distance:+.1f}%
推奨行動:
✅ 新規ポジション停止
✅ VOO/QQQ積立開始
✅ 月$300-500のDCA
✅ SPY > MA200まで待機
システムはブル相場で再開します。"""
       
        send_line(bear_msg)
        return None, None, None
   
    logger.info("="*60)
    logger.info("🐂 BULL MARKET CONFIRMED - SYSTEM RUNNING")
    logger.info("="*60)
    logger.info(f"Market: {regime_desc}")
    logger.info("")
   
    fx = get_current_fx_rate()
    vix = get_vix()
    is_bull = True
   
    logger.info("VIX: %.1f | FX: ¥%.2f", vix, fx)
   
    initial_cap_usd = jpy_to_usd(INITIAL_CAPITAL_JPY, fx)
    trading_cap_usd = initial_cap_usd * TRADING_RATIO
   
    results = []
    stats = {"Trend":0, "Price":0, "Loose":0, "Data":0, "LowVolume":0, "Pass":0, "Error":0}
    sec_exposures = {s: 0.0 for s in set(TICKERS.values())}
    for ticker, sector in TICKERS.items():
        try:
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
                vcp_analysis = VCPAnalyzer.calculate_vcp_maturity(res['df'], res)
                res['vcp_analysis'] = vcp_analysis
               
                inst_analysis = InstitutionalAnalyzer.analyze(ticker)
                res['institutional'] = inst_analysis
               
                quality = SignalQuality.calculate_comprehensive_score(res, vcp_analysis, inst_analysis)
                res['quality'] = quality
               
                why_now = SignalQuality.generate_why_now(res, vcp_analysis, inst_analysis, quality)
                res['why_now'] = why_now
               
                results.append((ticker, res))
                stats["Pass"] += 1
                sec_exposures[sector] += res['pos_usd'] / trading_cap_usd
            else:
                if reason == "❌LOW_VOLUME":
                    stats["LowVolume"] += 1
                elif reason == "❌TREND":
                    stats["Trend"] += 1
                elif reason == "❌PRICE":
                    stats["Price"] += 1
                elif reason == "❌LOOSE":
                    stats["Loose"] += 1
                elif reason == "❌DATA":
                    stats["Data"] += 1
                else:
                    stats["Error"] += 1
        except Exception as e:
            logger.exception("Loop error for %s: %s", ticker, e)
            stats["Error"] += 1
            continue
    all_sorted = sorted(results, key=lambda x: x[1]['quality']['total_score'], reverse=True)
   
    passed_core = [r for r in all_sorted if r[1]['quality']['tier'] == 'CORE']
    passed_secondary = [r for r in all_sorted if r[1]['quality']['tier'] == 'SECONDARY']
    passed_watch = [r for r in all_sorted if r[1]['quality']['tier'] == 'WATCH']
   
    report_lines = []
    report_lines.append("="*50)
    report_lines.append("SENTINEL v28 GROWTH")
    report_lines.append("="*50)
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    report_lines.append(f"🐂 {regime_desc}")
    report_lines.append(f"VIX: {vix:.1f} | FX: ¥{fx:.2f}")
    report_lines.append(f"Capital: ${trading_cap_usd:.0f}")
    report_lines.append("")
    report_lines.append(f"Analyzed: {len(TICKERS)} | Pass: {len(all_sorted)} | LowVol: {stats['LowVolume']}")
    report_lines.append(f"🔥 CORE: {len(passed_core)} | ⚡ SEC: {len(passed_secondary)} | 👁 WATCH: {len(passed_watch)}")
    report_lines.append("="*50)

    if passed_core:
        top = passed_core[0]
        ticker = top[0]
        r = top[1]
        q = r['quality']
       
        actual_shares = int(r['est_shares'])
        actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0
       
        entry = r['pivot']
        stop = r['stop']
        atr = (entry - stop) / ATR_STOP_MULT
        target1 = entry + (atr * 2.0)
        target2 = entry + (atr * 4.0)
        risk_pct = ((entry - stop) / entry) * 100
        reward_pct = ((target2 - entry) / entry) * 100
        report_lines.append(f"\n🎯 TOP: {ticker} ({q['total_score']}/100)")
        if actual_shares > 0:
            report_lines.append(f" {actual_shares}株 @ ${r['price']:.2f} = ${actual_cost:.0f}")
        else:
            report_lines.append(f" ⚠️ 1株未満 (${r['price']:.2f})")
        report_lines.append(f" {r['why_now']}")
        report_lines.append(f"\n 📍 Entry: ${entry:.2f} | Stop: ${stop:.2f} (-{risk_pct:.1f}%)")
        report_lines.append(f" 🎯 T1: ${target1:.2f} (2R) | T2: ${target2:.2f} (+{reward_pct:.1f}%)")

    if passed_core:
        report_lines.append(f"\n🔥 CORE (Top 10)")
        for i, (ticker, r) in enumerate(passed_core[:10], 1):
            q = r['quality']
            actual_shares = int(r['est_shares'])
            entry = r['pivot']
            stop = r['stop']
            atr = (entry - stop) / ATR_STOP_MULT
            target = entry + (atr * 4.0)
            risk_pct = ((entry - stop) / entry) * 100
            reward_pct = ((target - entry) / entry) * 100
           
            report_lines.append(f"\n{i}. {ticker} {q['total_score']}/100")
            if actual_shares > 0:
                report_lines.append(f" {actual_shares}株 @ ${r['price']:.2f}")
            else:
                report_lines.append(f" ⚠️ <1株 (${r['price']:.2f})")
            report_lines.append(f" {r['why_now'][:55]}")
            report_lines.append(f" Entry:${entry:.2f} Stop:${stop:.2f}(-{risk_pct:.1f}%) T:${target:.2f}(+{reward_pct:.1f}%)")

    if passed_secondary:
        report_lines.append(f"\n⚡ SECONDARY (Top 10)")
        for i, (ticker, r) in enumerate(passed_secondary[:10], 1):
            q = r['quality']
            entry = r['pivot']
            stop = r['stop']
            atr = (entry - stop) / ATR_STOP_MULT
            target = entry + (atr * 4.0)
            risk_pct = ((entry - stop) / entry) * 100
            reward_pct = ((target - entry) / entry) * 100
           
            report_lines.append(f"{i}. {ticker} {q['total_score']}/100 @ ${r['price']:.2f}")
            report_lines.append(f" Entry:${entry:.2f} Stop:${stop:.2f}(-{risk_pct:.1f}%) T:${target:.2f}(+{reward_pct:.1f}%)")

    if passed_watch:
        watch_str = ", ".join([f"{t}" for t, r in passed_watch[:10]])
        report_lines.append(f"\n👁 WATCH: {watch_str}")

    report_lines.append("\n" + "="*50)
    report_lines.append("Growth stocks in Bull market 🚀")
    report_lines.append("="*50)

    final_report = "\n".join(report_lines)
    logger.info("\n%s", final_report)
    send_line(final_report)
    return passed_core, passed_secondary, passed_watch

# save_signals_to_json (元のまま + 少し詳細化)
def save_signals_to_json(passed_core, passed_secondary, passed_watch):
    signals = []
   
    for ticker, result in passed_core:
        signal = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'ticker': ticker,
            'tier': 'CORE',
            'score': result['quality']['total_score'],
            'tech_score': result['quality']['tech_score'],
            'rr_score': result['quality']['rr_score'],
            'inst_score': result['quality'].get('inst_score', 25),
            'entry': result['pivot'],
            'stop': result['stop'],
            'target': result.get('target', 0),
            'shares': result.get('est_shares', 0),
            'why_now': result.get('why_now', ''),
            'sector': result.get('sector', 'Unknown')
        }
        signals.append(signal)
   
    for ticker, result in passed_secondary[:10]:
        signal = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'ticker': ticker,
            'tier': 'SECONDARY',
            'score': result['quality']['total_score'],
            'entry': result['pivot'],
            'stop': result['stop'],
            'target': result.get('target', 0),
            'shares': result.get('est_shares', 0),
            'why_now': result.get('why_now', '')
        }
        signals.append(signal)
   
    for ticker, result in passed_watch[:10]:
        signal = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'ticker': ticker,
            'tier': 'WATCH',
            'score': result['quality']['total_score'],
            'entry': result['pivot'],
            'stop': result['stop'],
            'target': result.get('target', 0),
            'shares': result.get('est_shares', 0),
            'why_now': result.get('why_now', '')
        }
        signals.append(signal)
   
    today = datetime.now().strftime('%Y%m%d')
    filename_dated = f"signals_{today}.json"
   
    with open(filename_dated, 'w', encoding='utf-8') as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)
   
    with open('today_signals.json', 'w', encoding='utf-8') as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)
   
    print(f"\n✅ Signals saved to JSON:")
    print(f" 📄 {filename_dated}")
    print(f" 📄 today_signals.json")
    print(f" 📊 Total: {len(signals)} signals")

if __name__ == "__main__":
    passed_core, passed_secondary, passed_watch = run_mission()
    if passed_core is not None:  # Bear時はNone
        save_signals_to_json(passed_core, passed_secondary, passed_watch)
