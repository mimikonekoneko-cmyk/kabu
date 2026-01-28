#!/usr/bin/env python3
# SENTINEL v25.1 - Revised Full Version
# Goal: Target annual return ~10% (parameterized)
# Requirements: pandas, numpy, yfinance, requests

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import time
import logging

warnings.filterwarnings('ignore')

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("SENTINEL")

# ---------------------------
# CONFIG
# ---------------------------
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

# Capital (JPY) and conversion
INITIAL_CAPITAL_JPY = 350_000
TRADING_RATIO = 0.70
TARGET_ANNUAL_RETURN = 0.10

# Risk & sizing
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25  # fraction of capital
MAX_SECTOR_CONCENTRATION = 0.40

# Filters (base)
MIN_SCORE = 60          # lowered to increase candidate pool
MIN_WINRATE = 0.55
MIN_EXPECTANCY = 0.20
MAX_TIGHTNESS_BASE = 1.5
MAX_NOTIFICATIONS = 5

# Liquidity
MIN_DAILY_VOLUME_USD = 10_000_000

# Transaction costs
COMMISSION_RATE = 0.002  # per side (fraction)
SLIPPAGE_RATE = 0.001    # per side (fraction)
FX_SPREAD_RATE = 0.0005  # when converting USD<->JPY

# Reward multipliers
REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech']

# Moving averages
MA_SHORT, MA_LONG = 50, 200

# Tickers (same universe)
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
# Utilities: currency conversion (USD base)
# ---------------------------
def get_current_fx_rate():
    """Return JPY per 1 USD (e.g., 152.6). If fails, fallback to 152.0"""
    try:
        data = yf.download("JPY=X", period="5d", progress=False)
        if data is None or data.empty:
            return 152.0
        return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.warning("FX fetch failed: %s", e)
        return 152.0

def jpy_to_usd(jpy, fx):
    return jpy / fx

def usd_to_jpy(usd, fx):
    return usd * fx

# ---------------------------
# Market indicators
# ---------------------------
def get_vix():
    try:
        data = yf.download("^VIX", period="5d", progress=False)
        if data is None or data.empty:
            return 20.0
        return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)
        return 20.0

def check_market_trend():
    """Return (is_bull, status_str, dist_percent) using SPY vs 200MA"""
    try:
        spy = yf.download("SPY", period="400d", progress=False)
        close = spy['Close'].dropna()
        if len(close) < 210:
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
def safe_download(tickers, period="600d", group_by='ticker', threads=True, retry=2):
    """Robust wrapper around yf.download with retries"""
    for attempt in range(retry):
        try:
            df = yf.download(list(tickers), period=period, progress=False, group_by=group_by, threads=threads)
            return df
        except Exception as e:
            logger.warning("yf.download attempt %d failed: %s", attempt+1, e)
            time.sleep(1 + attempt)
    raise RuntimeError("Failed to download market data")

def ensure_df(df):
    """Ensure df is a DataFrame with expected columns"""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.copy()

# ---------------------------
# Earnings check (robust)
# ---------------------------
def is_earnings_near(ticker, days_window=2):
    """Return True if earnings within +/- days_window days. Use yfinance calendar cautiously."""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None:
            return False
        # calendar may be DataFrame or dict-like
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # first column often contains Earnings Date
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

# ---------------------------
# Sector strength
# ---------------------------
def sector_is_strong(sector):
    """Check ETF trend by slope of 200MA over recent window"""
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf:
            return True
        df = yf.download(etf, period="300d", progress=False)
        if df is None or df.empty:
            return True
        close = df['Close'].dropna()
        if len(close) < 220:
            return True
        ma200 = close.rolling(200).mean().dropna()
        if len(ma200) < 12:
            return True
        # slope over last 10 points
        slope = (ma200.iloc[-1] - ma200.iloc[-10]) / ma200.iloc[-10]
        return slope >= 0.0
    except Exception:
        return True

# ---------------------------
# Transaction cost model (clear)
# ---------------------------
class TransactionCostModel:
    @staticmethod
    def calculate_total_cost_usd(val_usd):
        """
        Return round-trip cost in USD for a trade of val_usd USD.
        Commission and slippage are per side; multiply by 2 for round-trip.
        """
        comm = val_usd * COMMISSION_RATE
        slip = val_usd * SLIPPAGE_RATE
        return (comm + slip) * 2

    @staticmethod
    def calculate_total_cost_jpy(val_usd, fx):
        """Return round-trip cost in JPY when starting from USD value."""
        return TransactionCostModel.calculate_total_cost_usd(val_usd) * fx + (val_usd * FX_SPREAD_RATE * fx) * 2

# ---------------------------
# Position sizing
# ---------------------------
class PositionSizer:
    @staticmethod
    def calculate_position(cap_usd, winrate, rr, atr_pct, vix, sec_exp):
        """
        cap_usd: available capital in USD
        winrate: 0-1
        rr: reward/risk ratio (e.g., 2.0)
        atr_pct: ATR as percent of price (e.g., 0.03 for 3%)
        vix: current VIX
        sec_exp: current sector exposure fraction
        Returns: (position_value_usd, fraction)
        """
        # Kelly fraction (conservative): f = (WR - (1-WR)/RR)
        try:
            if rr <= 0:
                return 0.0, 0.0
            kelly = max(0.0, (winrate - (1 - winrate) / rr))
            # scale down Kelly for robustness
            kelly = min(kelly * 0.5, MAX_POSITION_SIZE)
            # volatility and market adjustments
            v_f = 0.7 if atr_pct > 0.05 else 0.85 if atr_pct > 0.03 else 1.0
            m_f = 0.7 if vix > 30 else 0.85 if vix > 20 else 1.0
            s_f = 0.7 if sec_exp > MAX_SECTOR_CONCENTRATION else 1.0
            final_frac = min(kelly * v_f * m_f * s_f, MAX_POSITION_SIZE)
            pos_val = cap_usd * final_frac
            return pos_val, final_frac
        except Exception:
            return 0.0, 0.0

# ---------------------------
# Backtest / performance simulation (robust)
# ---------------------------
def simulate_past_performance_v2(df, sector, lookback_years=3):
    """
    Simplified, robust backtest:
    - Use last `lookback_years` years
    - Identify breakout days where price exceeds recent 5-day high (pivot)
    - Use ATR(14) for stop distance
    - Track outcomes within 30 trading days
    Returns dict with winrate (0-100), net_expectancy, message
    """
    try:
        df = ensure_df(df)
        close = df['Close'].dropna()
        high = df['High'].dropna()
        low = df['Low'].dropna()
        if len(close) < 60:
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
        atr = tr.rolling(14).mean().dropna()
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']

        wins = 0
        losses = 0
        total_r = 0.0
        samples = 0

        # iterate over potential pivot points (exclude last 40 days to avoid lookahead)
        for i in range(50, len(close)-40):
            window_high = high.iloc[i-5:i].max()
            pivot = window_high * 1.002
            # require price near pivot (breakout day)
            if high.iloc[i] < pivot:
                continue
            # require trend-like condition but relaxed
            ma50 = close.rolling(50).mean().iloc[i]
            ma200 = close.rolling(200).mean().iloc[i] if i >= 200 else None
            if ma200 is not None and not (close.iloc[i] > ma50 or ma50 > ma200):
                continue
            stop_dist = atr.iloc[i] * ATR_STOP_MULT if i < len(atr) else atr.iloc[-1] * ATR_STOP_MULT
            entry = pivot
            target = entry + stop_dist * reward_mult
            # simulate next 30 days
            outcome = None
            for j in range(1, 31):
                if i + j >= len(close):
                    break
                if high.iloc[i+j] >= target:
                    outcome = 'win'
                    break
                if low.iloc[i+j] <= entry - stop_dist:
                    outcome = 'loss'
                    break
            if outcome is None:
                # if neither hit, use last close relative to entry
                last_close = close.iloc[min(i+30, len(close)-1)]
                pnl = (last_close - entry) / stop_dist
                if pnl > 0:
                    wins += 1
                    total_r += min(pnl, reward_mult)
                else:
                    losses += 1
                    total_r -= abs(pnl)
                samples += 1
            else:
                samples += 1
                if outcome == 'win':
                    wins += 1
                    total_r += reward_mult
                else:
                    losses += 1
                    total_r -= 1.0

        total = wins + losses
        if total < 8:
            return {'winrate':0, 'net_expectancy':0, 'message':f'LowSample:{total}'}
        wr = (wins / total)
        ev = total_r / total
        return {'winrate':wr*100, 'net_expectancy':ev - 0.05, 'message':f"WR{wr*100:.0f}% EV{ev:.2f}"}
    except Exception as e:
        logger.exception("Backtest error: %s", e)
        return {'winrate':0, 'net_expectancy':0, 'message':'BT Error'}

# ---------------------------
# Analyzer
# ---------------------------
class StrategicAnalyzerV2:
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_position_value_usd, vix, sec_exposures, cap_usd, market_is_bull):
        """
        Analyze a single ticker. All monetary values in USD.
        Returns (result_dict, reason_str)
        """
        try:
            df = ensure_df(df)
            close = df['Close'].dropna()
            high = df['High'].dropna()
            low = df['Low'].dropna()
            vol = df['Volume'].dropna()
            if len(close) < 60:
                return None, "❌DATA"

            curr = float(close.iloc[-1])
            # Price filter: avoid extremely high-priced tickers relative to capital per position
            if curr > max_position_value_usd * 0.5:
                return None, "❌PRICE"

            # Trend: relaxed rule (either price > MA50 OR MA50 > MA200)
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
            if ma200 is not None:
                if not (curr > ma50 or ma50 > ma200):
                    return None, "❌TREND"
            else:
                if not (curr > ma50):
                    return None, "❌TREND"

            # ATR and tightness
            tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr14 / curr if curr > 0 else 0.0
            tightness = (high.iloc[-5:].max() - low.iloc[-5:].min()) / (atr14 if atr14 > 0 else 1.0)

            # dynamic tightness threshold based on market regime
            max_tightness = MAX_TIGHTNESS_BASE
            if market_is_bull and vix < 20:
                max_tightness = 2.2
            elif vix > 25:
                max_tightness = 1.4

            if tightness > max_tightness:
                return None, "❌LOOSE"

            # Score components
            score = 0
            reasons = []
            if tightness < 0.8:
                score += 30; reasons.append("VCP+++")
            elif tightness < 1.2:
                score += 20; reasons.append("VCP+")
            # Volume dryness
            if vol.iloc[-1] < vol.rolling(50).mean().iloc[-1]:
                score += 15; reasons.append("VolDry")
            # Momentum
            if (close.rolling(5).mean().iloc[-1] / close.rolling(20).mean().iloc[-1]) > 1.02:
                score += 20; reasons.append("Mom+")
            # Trend strength
            if ma200 is not None and ((ma50 - ma200) / ma200) > 0.03:
                score += 20; reasons.append("Trend+")
            elif ma200 is None and (curr > ma50):
                score += 10; reasons.append("Trend?")

            # Backtest
            bt = simulate_past_performance_v2(df, sector)
            winrate = bt['winrate'] / 100.0
            # position sizing
            pos_val_usd, frac = PositionSizer.calculate_position(cap_usd, winrate, 2.0, atr_pct, vix, sec_exposures.get(sector, 0.0))

            # pivot and stop
            pivot = high.iloc[-5:].max() * 1.002
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
                'tightness': tightness,
                'price': curr,
                'atr_pct': atr_pct,
                'vol': int(vol.iloc[-1])
            }
            return result, "✅PASS"
        except Exception as e:
            logger.exception("Analyze error for %s: %s", ticker, e)
            return None, "❌ERROR"

# ---------------------------
# Messaging (LINE) - optional
# ---------------------------
def send_line(msg):
    logger.info("LINE message prepared (not sent in dry-run).")
    if not ACCESS_TOKEN or not USER_ID:
        logger.debug("LINE credentials missing; skipping send.")
        return
    try:
        requests.post("https://api.line.me/v2/bot/message/push",
                      headers={"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"},
                      json={"to":USER_ID, "messages":[{"type":"text", "text":msg}]}, timeout=10)
    except Exception as e:
        logger.warning("LINE send failed: %s", e)

# ---------------------------
# Main mission
# ---------------------------
def run_mission():
    fx = get_current_fx_rate()
    vix = get_vix()
    is_bull, market_status, _ = check_market_trend()
    logger.info("Market: %s | VIX: %.1f | FX: ¥%.2f", market_status, vix, fx)

    # Convert capital to USD for internal calc
    initial_cap_usd = jpy_to_usd(INITIAL_CAPITAL_JPY, fx)
    trading_cap_usd = initial_cap_usd * TRADING_RATIO

    # Download data
    try:
        all_data = safe_download(list(TICKERS.keys()), period="700d", group_by='ticker', threads=True)
    except Exception as e:
        logger.error("Data download failed: %s", e)
        return

    results = []
    stats = {"Earnings":0, "Sector":0, "Trend":0, "Price":0, "Loose":0, "Data":0, "Pass":0}

    # sector exposures placeholder (could be computed from portfolio)
    sec_exposures = {s: 0.0 for s in set(TICKERS.values())}

    for ticker, sector in TICKERS.items():
        try:
            # check earnings
            earnings_flag = is_earnings_near(ticker, days_window=2)
            if earnings_flag:
                stats["Earnings"] += 1

            # sector strength
            sector_flag = not sector_is_strong(sector)
            if sector_flag:
                stats["Sector"] += 1

            # prepare df slice
            try:
                df_t = all_data[ticker].dropna()
            except Exception:
                # sometimes yf returns single-level DataFrame
                try:
                    df_t = yf.download(ticker, period="700d", progress=False)
                except Exception:
                    df_t = pd.DataFrame()

            if df_t is None or df_t.empty:
                stats["Data"] += 1
                continue

            # max position value: cap * MAX_POSITION_SIZE (USD)
            max_pos_val_usd = trading_cap_usd * MAX_POSITION_SIZE

            res, reason = StrategicAnalyzerV2.analyze_ticker(
                ticker, df_t, sector, max_pos_val_usd, vix, sec_exposures, trading_cap_usd, is_bull
            )

            if res:
                res['is_earnings'] = earnings_flag
                res['is_sector_weak'] = sector_flag
                results.append((ticker, res))
                if not earnings_flag and not sector_flag:
                    stats["Pass"] += 1
            else:
                if "TREND" in reason:
                    stats["Trend"] += 1
                elif "PRICE" in reason:
                    stats["Price"] += 1
                elif "LOOSE" in reason:
                    stats["Loose"] += 1
                elif "DATA" in reason:
                    stats["Data"] += 1

        except Exception as e:
            logger.exception("Loop error for %s: %s", ticker, e)
            continue

    # Sort by score
    all_sorted = sorted(results, key=lambda x: x[1]['score'], reverse=True)

    # Filter passed candidates
    passed = [r for r in all_sorted if r[1]['score'] >= MIN_SCORE and not r[1]['is_earnings'] and not r[1]['is_sector_weak']]

    # Build report
    report_lines = []
    report_lines.append("SENTINEL v25.1 DIAGNOSTIC")
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    report_lines.append(f"Mkt: {market_status}")
    report_lines.append(f"VIX: {vix:.1f} | FX: ¥{fx:.2f}")
    report_lines.append("="*40)
    report_lines.append("【STATISTICS】")
    report_lines.append(f"Analyzed: {len(TICKERS)} tickers")
    report_lines.append(f"Blocked by Earnings: {stats['Earnings']}")
    report_lines.append(f"Blocked by Sector:   {stats['Sector']}")
    report_lines.append(f"Blocked by Trend:    {stats['Trend']}")
    report_lines.append(f"VCP/Score Pass:      {len(all_sorted)}")
    report_lines.append("="*40)
    report_lines.append("【BUY SIGNALS】")

    if not passed:
        report_lines.append("No candidates passed all strict filters.")
    else:
        for i, (ticker, r) in enumerate(passed[:MAX_NOTIFICATIONS], 1):
            # compute estimated shares and costs
            pos_usd = r['pos_usd']
            est_shares = int(pos_usd / r['price']) if r['price'] > 0 else 0
            roundtrip_cost_usd = TransactionCostModel.calculate_total_cost_usd(pos_usd)
            report_lines.append(f"★ [{i}] {ticker} {r['score']}pt")
            report_lines.append(f"   Entry: ${r['pivot']:.2f} / Price: ${r['price']:.2f} / Shares est: {est_shares}")
            report_lines.append(f"   Pos(USD): ${pos_usd:,.2f} / RoundtripCost(USD): ${roundtrip_cost_usd:,.2f}")
            report_lines.append(f"   BT: {r['bt']['message']} Tight:{r['tightness']:.2f}")

    report_lines.append("\n【ANALYSIS TOP 10 (RAW)】")
    for i, (ticker, r) in enumerate(all_sorted[:10], 1):
        tag = "✅OK"
        if r.get('is_earnings'): tag = "❌EARN"
        elif r.get('is_sector_weak'): tag = "❌SEC"
        elif r['score'] < MIN_SCORE: tag = "❌SCOR"
        report_lines.append(f"{i}. {ticker:5} {r['score']}pt | {tag}")
        report_lines.append(f"   Tight:{r['tightness']:.2f} WR:{r['bt']['winrate']:.0f}% PosUSD:{r['pos_usd']:.0f}")

    final_report = "\n".join(report_lines)
    logger.info("\n%s", final_report)

    # Optionally send via LINE
    send_line(final_report)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_mission()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user\n")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}\n")
        import traceback
        traceback.print_exc()