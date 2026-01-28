import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Authentication
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

# Capital Management
INITIAL_CAPITAL = 350000  # JPY
QUARTERLY_CONTRIBUTION = 30000  # JPY
TRADING_RATIO = 0.70
HOLDING_RATIO = 0.30

# Risk Management
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25
MAX_CONCURRENT_POSITIONS = 4
MAX_SECTOR_CONCENTRATION = 0.40

# Filter Thresholds - v25.1 BALANCED
MIN_SCORE = 75          
MIN_WINRATE = 55        
MIN_EXPECTANCY = 0.45   
MAX_TIGHTNESS = 1.5     
MAX_NOTIFICATIONS = 5

# Liquidity Filter
MIN_DAILY_VOLUME_USD = 10_000_000

# Reward Multipliers
REWARD_MULTIPLIERS = {
    'aggressive': 2.5,
    'stable': 2.0
}

AGGRESSIVE_SECTORS = [
    'Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 
    'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech'
]

# Transaction Costs
COMMISSION_RATE = 0.002
SLIPPAGE_RATE = 0.001
FX_SPREAD_RATE = 0.0005
TOTAL_COST_RATE = (COMMISSION_RATE + SLIPPAGE_RATE + FX_SPREAD_RATE) * 2

# Performance Tracking
TARGET_ANNUAL_RETURN = 0.10
PERFORMANCE_LOG_PATH = Path("/tmp/sentinel_performance.json")

# ============================================================================
# TICKER UNIVERSE (92 TICKERS)
# ============================================================================

TICKERS = {
    'NVDA':'AI','AMD':'Semi','AVGO':'Semi','TSM':'Semi','ASML':'Semi','MU':'Semi',
    'ARM':'Semi','INTC':'Semi','QCOM':'Semi','ON':'Semi','LRCX':'Semi','AMAT':'Semi',
    'MSFT':'Cloud','GOOGL':'Ad','META':'Ad','PLTR':'AI','NOW':'Soft','CRM':'Soft',
    'ADBE':'Soft','SNOW':'Cloud','DDOG':'Cloud','WDAY':'Soft','TEAM':'Soft',
    'ANET':'Cloud','ZS':'Sec','MDB':'Cloud','SHOP':'Retail','PANW':'Sec',
    'CRWD':'Sec','FTNT':'Sec','NET':'Sec','OKTA':'Sec','AAPL':'Device',
    'TSLA':'Auto','AMZN':'Retail','NFLX':'Service','COST':'Retail','WMT':'Retail',
    'TJX':'Retail','TGT':'Retail','NKE':'Cons','LULU':'Cons','SBUX':'Cons',
    'PEP':'Cons','KO':'Cons','PG':'Cons','ELF':'Cons','CELH':'Cons','MELI':'Retail',
    'V':'Fin','MA':'Fin','PYPL':'Fintech','SQ':'Fintech','JPM':'Bank','GS':'Bank',
    'MS':'Bank','AXP':'Fin','BLK':'Fin','COIN':'Crypto','SOFI':'Fintech','NU':'Fintech',
    'LLY':'Bio','UNH':'Health','ABBV':'Bio','ISRG':'Health','VRTX':'Bio','MRK':'Bio',
    'PFE':'Bio','AMGN':'Bio','HCA':'Health','TDOC':'Health','GE':'Ind','CAT':'Ind',
    'DE':'Ind','BA':'Ind','ETN':'Power','VRT':'Power','TT':'Ind','PH':'Ind',
    'TDG':'Ind','XOM':'Energy','CVX':'Energy','MPC':'Energy','UBER':'Platform',
    'BKNG':'Travel','ABNB':'Travel','MAR':'Travel','RCL':'Travel','DKNG':'Bet',
    'RBLX':'Service','DASH':'Service','SMCI':'AI'
}

SECTOR_ETF = {
    'Energy':'XLE', 'Semi':'SOXX', 'Bank':'XLF', 'Retail':'XRT',
    'Soft':'IGV', 'AI':'QQQ', 'Fin':'VFH', 'Device':'QQQ',
    'Cloud':'QQQ', 'Ad':'QQQ', 'Service':'QQQ', 'Sec':'HACK',
    'Cons':'XLP', 'Bio':'IBB', 'Health':'XLV', 'Ind':'XLI',
    'Auto':'CARZ', 'Crypto':'BTC-USD', 'Power':'XLI', 'Platform':'QQQ',
    'Travel':'XLY', 'Bet':'BETZ', 'Fintech':'ARKF'
}

MA_SHORT, MA_LONG = 50, 200

# ============================================================================
# CLASSES & ENGINES
# ============================================================================

class TransactionCostModel:
    @staticmethod
    def calculate_total_cost(position_value_usd, fx_rate):
        commission = position_value_usd * COMMISSION_RATE
        slippage_rate = 0.0005 if position_value_usd < 500 else 0.001
        slippage = position_value_usd * slippage_rate
        fx_cost = position_value_usd * FX_SPREAD_RATE
        return (commission + slippage + fx_cost) * 2 * fx_rate

    @staticmethod
    def adjust_expectancy_for_cost(gross_expectancy, avg_1r_pct, position_value_usd, fx_rate):
        cost_jpy = TransactionCostModel.calculate_total_cost(position_value_usd, fx_rate)
        cost_pct = (cost_jpy / (position_value_usd * fx_rate)) * 100
        cost_in_r = cost_pct / avg_1r_pct
        return gross_expectancy - cost_in_r, cost_in_r

class PositionSizer:
    @staticmethod
    def calculate_kelly_fraction(winrate, rr_ratio):
        if winrate <= 0 or winrate >= 1: return 0
        kelly = (rr_ratio * winrate - (1 - winrate)) / rr_ratio
        return max(0, min(kelly / 2, 0.25))

    @staticmethod
    def calculate_position_size(trading_capital, winrate, rr_ratio, atr_pct, vix, sector_exposure):
        fraction = PositionSizer.calculate_kelly_fraction(winrate, rr_ratio)
        v_factor = 0.7 if atr_pct > 5.0 else 0.85 if atr_pct > 3.0 else 1.0
        m_factor = 0.6 if vix > 30 else 0.8 if vix > 20 else 1.0
        s_factor = 0.7 if sector_exposure > 0.30 else 0.85 if sector_exposure > 0.20 else 1.0
        final_fraction = fraction * v_factor * m_factor * s_factor
        final_fraction = min(final_fraction, MAX_POSITION_SIZE)
        return trading_capital * final_fraction, {'kelly': fraction, 'final': final_fraction}

class TrailingStopManager:
    @staticmethod
    def calculate_stop(entry_price, current_price, highest, atr_val):
        initial_stop = entry_price - (atr_val * ATR_STOP_MULT)
        if current_price < entry_price + (atr_val * 0.5): return initial_stop
        elif current_price < entry_price + (atr_val * 1.0): return entry_price
        else: return max(highest - (atr_val * ATR_STOP_MULT), entry_price)

def simulate_past_performance_v2(df, sector):
    try:
        close, high, low = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze()
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        wins, losses, total_r = 0, 0, 0
        start_idx = max(MA_LONG, len(df) - 500)
        for i in range(start_idx, len(df) - 10):
            if not (close.iloc[i] > close.iloc[i-MA_SHORT:i].mean() > close.iloc[i-MA_LONG:i].mean()): continue
            pivot = high.iloc[i-5:i].max() * 1.002
            stop_dist = atr.iloc[i] * ATR_STOP_MULT
            if pd.isna(stop_dist) or stop_dist == 0: continue
            if high.iloc[i] >= pivot:
                entry, target, highest = pivot, pivot + (stop_dist * reward_mult), high.iloc[i]
                for j in range(1, 30):
                    if i + j >= len(df): break
                    c_high, c_low, c_close = high.iloc[i+j], low.iloc[i+j], close.iloc[i+j]
                    highest = max(highest, c_high)
                    c_stop = TrailingStopManager.calculate_stop(entry, c_close, highest, stop_dist/ATR_STOP_MULT)
                    if c_high >= target: wins += 1; total_r += reward_mult; break
                    if c_low <= c_stop: losses += 1; total_r -= (entry - c_stop)/stop_dist; break
        total = wins + losses
        if total < 10: return {'status':'insufficient', 'trades':total, 'winrate':0, 'expectancy':0, 'message':f'Sample:{total}'}
        wr, ev = (wins/total)*100, total_r/total
        net_ev, _ = TransactionCostModel.adjust_expectancy_for_cost(ev, 8.5, 500, 152.0)
        return {'status':'valid', 'winrate':wr, 'expectancy':ev, 'net_expectancy':net_ev, 'total':total, 'message':f"WR{wr:.0f}% NetEV{net_ev:.2f}R"}
    except: return {'status':'error', 'message':'BT Error'}

# ============================================================================
# STRATEGIC ANALYZER
# ============================================================================

class StrategicAnalyzerV2:
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_price_usd, vix, sector_exposures, trading_capital):
        if len(df) < MA_LONG + 50: return None
        close, high, low, volume = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
        curr_price = float(close.iloc[-1])
        if curr_price > max_price_usd: return None
        
        # Liquidity
        avg_vol_usd = volume.rolling(50).mean().iloc[-1] * curr_price
        if avg_vol_usd < MIN_DAILY_VOLUME_USD: return None

        # Trend & VCP
        ma50, ma200 = close.rolling(MA_SHORT).mean().iloc[-1], close.rolling(MA_LONG).mean().iloc[-1]
        if not (curr_price > ma50 > ma200): return None
        
        tr = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        tightness = float((high.iloc[-5:].max() - low.iloc[-5:].min()) / atr14)
        if tightness > MAX_TIGHTNESS: return None

        # Scoring
        score, reasons = 0, []
        if tightness < 0.8: score += 35; reasons.append("VCP+++")
        elif tightness < 1.0: score += 30; reasons.append("VCP++")
        
        vol_ratio = volume.iloc[-1] / volume.rolling(50).mean().iloc[-1]
        if 0.5 <= vol_ratio <= 0.85: score += 20; reasons.append("VolDry")
        
        mom = ((close.rolling(5).mean().iloc[-1] / close.rolling(20).mean().iloc[-1]) - 1) * 100
        if mom > 2.0: score += 20; reasons.append("Mom++")
        
        trend_s = ((ma50 - ma200) / ma200) * 100
        if trend_s > 10: score += 25; reasons.append("Trend+")
        
        bt = simulate_past_performance_v2(df, sector)
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        pos_size, s_factors = PositionSizer.calculate_position_size(
            trading_capital, bt.get('winrate',50)/100, reward_mult, (atr14/curr_price)*100, vix, sector_exposures.get(sector,0)
        )

        pivot = high.iloc[-5:].max() * 1.002
        return {
            'score': score, 'reasons': ' '.join(reasons), 'price': curr_price,
            'pivot': pivot, 'stop': pivot - (atr14*ATR_STOP_MULT),
            'target': pivot + (atr14*ATR_STOP_MULT*reward_mult), 'sector': sector,
            'bt': bt, 'position_size_jpy': pos_size, 'sizing_factors': s_factors,
            'liquidity_usd': avg_vol_usd
        }

# ============================================================================
# UTILITIES & MAIN
# ============================================================================

def get_market_data():
    try:
        spy = yf.download("SPY", period="250d", progress=False)['Close'].squeeze()
        ma200 = spy.rolling(200).mean().iloc[-1]
        curr = spy.iloc[-1]
        is_bull = curr > ma200
        status = f"{'Bull' if is_bull else 'Bear'} (+{((curr/ma200)-1)*100:.1f}% vs MA200)"
        fx = yf.download("JPY=X", period="1d", progress=False)['Close'].squeeze().iloc[-1]
        vix = yf.download("^VIX", period="1d", progress=False)['Close'].squeeze().iloc[-1]
        return is_bull, status, float(fx), float(vix)
    except: return True, "Market OK", 152.0, 20.0

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID: 
        print(f"\n--- LINE PREVIEW ---\n{msg}\n--------------------")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages": [{"type": "text", "text": msg}]}
    requests.post(url, headers=headers, json=payload, timeout=10)

def run_mission():
    is_bull, market_status, fx_rate, vix = get_market_data()
    trading_capital = INITIAL_CAPITAL * TRADING_RATIO
    
    # Download & Screen
    all_data = yf.download(list(TICKERS.keys()), period="600d", progress=False, group_by='ticker', threads=True)
    
    results = []
    sector_exposures = {}
    for ticker, sector in TICKERS.items():
        try:
            res = StrategicAnalyzerV2.analyze_ticker(ticker, all_data[ticker], sector, (trading_capital/fx_rate)*0.9, vix, sector_exposures, trading_capital)
            if res:
                results.append((ticker, res))
                if res['score'] >= MIN_SCORE:
                    sector_exposures[sector] = sector_exposures.get(sector,0) + (res['position_size_jpy']/trading_capital)
        except: continue

    # Sort
    all_sorted = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    passed = [r for r in all_sorted if r[1]['score'] >= MIN_SCORE and r[1]['bt'].get('winrate',0) >= MIN_WINRATE and r[1]['bt'].get('net_expectancy',0) >= MIN_EXPECTANCY]
    
    # Report
    report = [
        "SENTINEL v25.1 Final", f"{datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
        f"Market: {market_status}", f"VIX: {vix:.1f} | FX: ¥{fx_rate:.2f}",
        "=" * 35, "【SEARCH RESULTS】"
    ]
    
    if not passed:
        report.append("No candidates passed strict filters.")
    else:
        for i, (ticker, r) in enumerate(passed[:MAX_NOTIFICATIONS], 1):
            shares = int(r['position_size_jpy'] / fx_rate / r['pivot'])
            report.append(f"★買付 [{i}] {ticker} {r['score']}pt")
            report.append(f"推奨: {shares}株 (¥{r['position_size_jpy']:,.0f})")
            report.append(f"Entry: ${r['pivot']:.2f} / Stop: ${r['stop']:.2f}")
            report.append(f"BT: {r['bt']['message']}")
            report.append("-" * 20)

    report.append("\n【ANALYSIS TOP 10 (Raw Scores)】")
    for i, (ticker, r) in enumerate(all_sorted[:10], 1):
        # フィルタ不合格の理由判定
        reason = "✅PASS"
        if r['score'] < MIN_SCORE: reason = "❌SCORE"
        elif r['bt'].get('winrate',0) < MIN_WINRATE: reason = "❌WR"
        elif r['bt'].get('net_expectancy',0) < MIN_EXPECTANCY: reason = "❌EV"
        
        report.append(f"{i}. {ticker:5} : {r['score']}pt | {reason}")
        report.append(f"   WR:{r['bt'].get('winrate',0):.0f}% EV:{r['bt'].get('net_expectancy',0):.2f}R")

    report.append("=" * 35)
    send_line("\n".join(report))
    print("Mission complete.")

if __name__ == "__main__":
    run_mission()
