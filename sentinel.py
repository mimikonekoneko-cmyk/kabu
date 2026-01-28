import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# 警告の抑制
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & GLOBAL CONSTANTS
# ============================================================================

# LINE Authentication
ACCESS_TOKEN = (
    os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or 
    os.getenv("LINECHANNELACCESSTOKEN") or
    os.getenv("ACCESS_TOKEN")
)

USER_ID = (
    os.getenv("LINE_USER_ID") or 
    os.getenv("LINEUSER_ID") or
    os.getenv("USER_ID")
)

# Capital Management (JPY)
INITIAL_CAPITAL = 350000  
TRADING_RATIO = 0.70

# Risk Management
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25  
MAX_SECTOR_CONCENTRATION = 0.40  

# Filter Thresholds
MIN_SCORE = 75          
MIN_WINRATE = 55        
MIN_EXPECTANCY = 0.45   
MAX_TIGHTNESS = 1.5     
MAX_NOTIFICATIONS = 5

# Liquidity Filter
MIN_DAILY_VOLUME_USD = 10_000_000  

# Reward Multipliers
REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech']

# Transaction Costs
COMMISSION_RATE = 0.002  
SLIPPAGE_RATE = 0.001  
FX_SPREAD_RATE = 0.0005  

# Performance Metrics
TARGET_ANNUAL_RETURN = 0.10
PERFORMANCE_LOG_PATH = Path("/tmp/sentinel_performance.json")

# Moving Average Periods
MA_SHORT, MA_LONG = 50, 200

# ============================================================================
# TICKER UNIVERSE (92 TICKERS)
# ============================================================================

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

# ============================================================================
# CORE FUNCTIONS (NO REDUCTION)
# ============================================================================

def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 152.0
    except: return 152.0

def get_vix():
    try:
        data = yf.download("^VIX", period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 20.0
    except: return 20.0

def check_market_trend():
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        close = spy['Close'].squeeze()
        curr, ma200 = float(close.iloc[-1]), float(close.rolling(200).mean().iloc[-1])
        dist = ((curr - ma200) / ma200) * 100
        return curr > ma200, f"{'Bull' if curr > ma200 else 'Bear'} ({dist:+.1f}%)", dist
    except: return True, "Unknown", 0

def is_earnings_near(ticker):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty): return False
        date_val = cal.get('Earnings Date', [None])[0] if isinstance(cal, dict) else cal.iloc[0, 0]
        if date_val is None: return False
        days_until = (pd.to_datetime(date_val).date() - datetime.now().date()).days
        return abs(days_until) <= 5
    except: return False

def sector_is_strong(sector):
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf: return True
        df = yf.download(etf, period="250d", progress=False)
        ma200 = df['Close'].squeeze().rolling(200).mean()
        return ma200.iloc[-1] > ma200.iloc[-10]
    except: return True

class TransactionCostModel:
    @staticmethod
    def calculate_total_cost(val_usd, fx):
        comm = val_usd * COMMISSION_RATE
        slip = val_usd * (0.0005 if val_usd < 500 else 0.001)
        fx_c = val_usd * FX_SPREAD_RATE
        return (comm + slip + fx_c) * 2 * fx

class PositionSizer:
    @staticmethod
    def calculate_position(cap, wr, rr, atr_p, vix, sec_exp):
        kelly = max(0, min(((rr * wr - (1 - wr)) / rr) / 2, 0.25))
        v_f = 0.7 if atr_p > 5.0 else 0.85 if atr_p > 3.0 else 1.0
        m_f = 0.6 if vix > 30 else 0.8 if vix > 20 else 1.0
        s_f = 0.7 if sec_exp > 0.30 else 1.0
        final = min(kelly * v_f * m_f * s_f, MAX_POSITION_SIZE)
        return cap * final, final

def simulate_past_performance_v2(df, sector):
    try:
        close, high, low = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze()
        tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        wins, losses, total_r = 0, 0, 0
        for i in range(max(200, len(df)-500), len(df)-10):
            if not (close.iloc[i] > close.iloc[i-50:i].mean() > close.iloc[i-200:i].mean()): continue
            pivot = high.iloc[i-5:i].max() * 1.002
            stop_dist = atr.iloc[i] * 2.0
            if high.iloc[i] >= pivot:
                entry, target, highest = pivot, pivot + (stop_dist * reward_mult), high.iloc[i]
                for j in range(1, 30):
                    if i+j >= len(df): break
                    c_h, c_l, c_c = high.iloc[i+j], low.iloc[i+j], close.iloc[i+j]
                    highest = max(highest, c_h)
                    c_stop = entry if c_c > entry + (stop_dist/2) else entry - stop_dist
                    if c_h >= target: wins += 1; total_r += reward_mult; break
                    if c_l <= c_stop: losses += 1; total_r -= (entry-c_stop)/stop_dist; break
        total = wins + losses
        if total < 5: return {'winrate':0, 'net_expectancy':0, 'message':f"LowSample:{total}"}
        wr, ev = (wins/total)*100, total_r/total
        return {'winrate':wr, 'net_expectancy':ev - 0.05, 'message':f"WR{wr:.0f}% EV{ev:.2f}"}
    except: return {'winrate':0, 'net_expectancy':0, 'message':"BT Error"}

class StrategicAnalyzerV2:
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_p, vix, sec_exposures, cap):
        try:
            close, high, low, vol = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
            curr = float(close.iloc[-1])
            if curr > max_p: return None, "❌PRICE"
            
            # Trend
            ma50, ma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
            if not (curr > ma50 > ma200): return None, "❌TREND"

            # Tightness
            tr = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            tightness = (high.iloc[-5:].max() - low.iloc[-5:].min()) / atr14
            if tightness > MAX_TIGHTNESS: return None, "❌LOOSE"

            score, reasons = 0, []
            if tightness < 0.8: score += 35; reasons.append("VCP+++")
            elif tightness < 1.1: score += 25; reasons.append("VCP+")
            if vol.iloc[-1] < vol.rolling(50).mean().iloc[-1]: score += 20; reasons.append("VolDry")
            if (close.rolling(5).mean().iloc[-1] / close.rolling(20).mean().iloc[-1]) > 1.02: score += 20; reasons.append("Mom+")
            if ((ma50-ma200)/ma200) > 0.1: score += 25; reasons.append("Trend+")

            bt = simulate_past_performance_v2(df, sector)
            pos_size, _ = PositionSizer.calculate_position(cap, bt['winrate']/100, 2.0, (atr14/curr)*100, vix, sec_exposures.get(sector,0))
            
            pivot = high.iloc[-5:].max() * 1.002
            return {
                'score': score, 'reasons': ' '.join(reasons), 'pivot': pivot, 
                'stop': pivot - (atr14*2.0), 'sector': sector, 'bt': bt, 
                'pos_jpy': pos_size, 'tightness': tightness, 'price': curr
            }, "✅PASS"
        except: return None, "❌ERROR"

# ============================================================================
# MISSION CONTROL
# ============================================================================

def send_line(msg):
    print(f"\n--- REPORT ---\n{msg}\n--------------")
    if not ACCESS_TOKEN or not USER_ID: return
    requests.post("https://api.line.me/v2/bot/message/push", 
                  headers={"Content-Type":"application/json","Authorization":f"Bearer {ACCESS_TOKEN}"},
                  json={"to":USER_ID, "messages":[{"type":"text", "text":msg}]}, timeout=10)

def run_mission():
    is_bull, market_status, _ = check_market_trend()
    fx, vix = get_current_fx_rate(), get_vix()
    trading_cap = INITIAL_CAPITAL * TRADING_RATIO
    
    all_data = yf.download(list(TICKERS.keys()), period="600d", progress=False, group_by='ticker', threads=True)
    
    results, stats = [], {"Earnings":0, "Sector":0, "Trend":0, "Price":0, "Loose":0, "Pass":0}
    
    for ticker, sector in TICKERS.items():
        # 戦略的カウント
        if is_earnings_near(ticker): stats["Earnings"] += 1; earnings_flag = True
        else: earnings_flag = False
            
        if not sector_is_strong(sector): stats["Sector"] += 1; sector_flag = True
        else: sector_flag = False

        res, reason = StrategicAnalyzerV2.analyze_ticker(ticker, all_data[ticker], sector, (trading_cap/fx)*0.9, vix, {}, trading_cap)
        
        if res:
            res['is_earnings'] = earnings_flag
            res['is_sector_weak'] = sector_flag
            results.append((ticker, res))
            if not earnings_flag and not sector_flag: stats["Pass"] += 1
        elif "TREND" in reason: stats["Trend"] += 1
        elif "PRICE" in reason: stats["Price"] += 1
        elif "LOOSE" in reason: stats["Loose"] += 1

    # Sorting
    all_sorted = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    
    # 厳選（決算・セクターNGを排除）
    passed = [r for r in all_sorted if r[1]['score'] >= MIN_SCORE and not r[1]['is_earnings'] and not r[1]['is_sector_weak']]
    
    report = [
        "SENTINEL v25.1 DIAGNOSTIC", f"{datetime.now().strftime('%m/%d %H:%M')}", "",
        f"Mkt: {market_status}", f"VIX: {vix:.1f} | FX: ¥{fx:.2f}",
        "=" * 30, "【STATISTICS】",
        f"Analyzed: {len(TICKERS)} tickers",
        f"Blocked by Earnings: {stats['Earnings']}",
        f"Blocked by Sector:   {stats['Sector']}",
        f"Blocked by Trend:    {stats['Trend']}",
        f"VCP/Score Pass:      {len(all_sorted)}",
        "=" * 30, "【BUY SIGNALS】"
    ]
    
    if not passed: report.append("No candidates passed all strict filters.")
    else:
        for i, (ticker, r) in enumerate(passed[:MAX_NOTIFICATIONS], 1):
            report.append(f"★ [{i}] {ticker} {r['score']}pt")
            report.append(f"   Entry: ${r['pivot']:.2f} / BT: {r['bt']['message']}")

    report.append("\n【ANALYSIS TOP 10 (RAW)】")
    for i, (ticker, r) in enumerate(all_sorted[:10], 1):
        tag = "✅OK"
        if r['is_earnings']: tag = "❌EARN"
        elif r['is_sector_weak']: tag = "❌SEC"
        elif r['score'] < MIN_SCORE: tag = "❌SCOR"
        
        report.append(f"{i}. {ticker:5} {r['score']}pt | {tag}")
        report.append(f"   Tight:{r['tightness']:.2f} WR:{r['bt']['winrate']:.0f}%")

    send_line("\n".join(report))

if __name__ == "__main__":
    run_mission()
