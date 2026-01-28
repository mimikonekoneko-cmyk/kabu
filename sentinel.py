import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG (環境設定)
# ============================================================================

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

# 運用資金とリスク設定
BUDGET_JPY = 350000        # 四半期ごとに入金したらここを増やす
RISK_PCT_PER_TRADE = 0.01  # 1トレードの許容損失額（総資金の1%）

# ============================================================================
# CORE PARAMETERS (戦略パラメータ)
# ============================================================================

MA_SHORT, MA_LONG = 50, 200
MIN_SCORE = 65 
MIN_WINRATE = 40 
MIN_EXPECTANCY = 0.2 
MAX_NOTIFICATIONS = 8
ATR_STOP_MULT = 2.0

REWARD_MULTIPLIERS = {
    'aggressive': 3.0,
    'stable': 2.5
}

AGGRESSIVE_SECTORS = [
    'Semi', 'AI', 'Soft', 'Sec', 'EV', 'Crypto', 
    'Cloud', 'Ad', 'Service', 'Platform', 'Bet'
]

# ============================================================================
# TICKER UNIVERSE (1株$1000以下 / 厳選約80銘柄)
# ============================================================================

TICKERS = {
    # --- Technology & AI ---
    'NVDA':'AI','ARM':'Semi','MU':'Semi','AMD':'Semi','TSM':'Semi','SMCI':'AI',
    'INTC':'Semi','QCOM':'Semi','ON':'Semi','LRCX':'Semi','AMAT':'Semi',
    'MSFT':'Cloud','GOOGL':'Ad','META':'Ad','AAPL':'Device','PLTR':'AI',
    'PANW':'Sec','CRWD':'Sec','FTNT':'Sec','NET':'Sec',
    'NOW':'Soft','CRM':'Soft','TEAM':'Soft','ADBE':'Soft','APP':'AI','ANET':'Cloud',
    'SNOW':'Cloud','WDAY':'Soft','DBX':'Soft','DDOG':'Cloud',
    
    # --- Consumer & Retail ---
    'AMZN':'Retail','TSLA':'EV','NFLX':'Service','COST':'Retail','WMT':'Retail',
    'TJX':'Retail','TGT':'Retail','NKE':'Cons','LULU':'Cons','SBUX':'Cons',
    'PEP':'Cons','KO':'Cons','PG':'Cons','ELF':'Cons','CELH':'Cons','MELI':'Retail',
    
    # --- Finance & Fintech ---
    'V':'Fin','MA':'Fin','PYPL':'Fin','SQ':'Fin','JPM':'Bank','GS':'Bank',
    'AXP':'Fin','BLK':'Fin','MS':'Bank','COIN':'Crypto','HOOD':'Crypto',
    'SOFI':'Fin','NU':'Fin','UPST':'Fin',
    
    # --- Health & Bio ---
    'LLY':'Bio','UNH':'Health','ABBV':'Bio','ISRG':'Health','VRTX':'Bio',
    'MRK':'Bio','PFE':'Bio','AMGN':'Bio','HCA':'Health','TDOC':'Health',
    
    # --- Industrials, Energy & Others ---
    'GE':'Ind','CAT':'Ind','DE':'Ind','BA':'Ind','XOM':'Energy','CVX':'Energy',
    'MPC':'Energy','VRT':'Power','ETN':'Power','TT':'Ind','PH':'Ind',
    'UBER':'Platform','BKNG':'Travel','ABNB':'Travel','DKNG':'Bet','MAR':'Travel',
    'RCL':'Travel','TDG':'Ind','RYAAY':'Travel'
}

SECTOR_ETF = {
    'Energy':'XLE', 'Semi':'SOXX', 'Bank':'XLF', 'Retail':'XRT',
    'Soft':'IGV', 'AI':'QQQ', 'Fin':'VFH', 'Device':'QQQ',
    'Cloud':'QQQ', 'Ad':'QQQ', 'Service':'QQQ', 'Sec':'HACK',
    'Cons':'XLP', 'Bio':'IBB', 'Health':'XLV', 'Ind':'XLI',
    'EV':'IDRV', 'Crypto':'BTC-USD', 'Power':'XLI', 'Platform':'QQQ',
    'Travel':'XLY', 'Bet':'BETZ'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_environment():
    if not ACCESS_TOKEN or not USER_ID:
        print("Error: LINE Credentials are missing!")
        return False
    return True

def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 152.0
    except:
        return 152.0

def check_market_trend():
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        if spy.empty or len(spy) < 200:
            return True, "Data Limited"
        close = spy['Close'].squeeze()
        current = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        return current > ma200, f"Bull (${current:.0f} > MA200)" if current > ma200 else f"Bear (${current:.0f} < MA200)"
    except:
        return True, "Check Skipped"

def is_earnings_near(ticker):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
            return False
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            date_val = cal['Earnings Date'][0]
        else:
            date_val = cal.iloc[0, 0]
        earnings_date = pd.to_datetime(date_val).date()
        days_until = (earnings_date - datetime.now().date()).days
        return 0 <= days_until <= 7
    except:
        return False

def sector_is_strong(sector):
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf: return True
        df = yf.download(etf, period="250d", progress=False)
        close = df['Close'].squeeze()
        ma200 = close.rolling(200).mean()
        return ma200.iloc[-1] > ma200.iloc[-10]
    except:
        return True

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def simulate_past_performance(df, sector, atr_mult=ATR_STOP_MULT):
    try:
        close, high, low = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze()
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        wins, losses, total_r = 0, 0, 0
        start_idx = max(MA_LONG, len(df) - 500)
        for i in range(start_idx, len(df) - 10):
            ma50_at_i = close.iloc[i-MA_SHORT:i].mean()
            ma200_at_i = close.iloc[i-MA_LONG:i].mean()
            if not (close.iloc[i] > ma50_at_i > ma200_at_i): continue
            
            pivot = high.iloc[i-5:i].max() * 1.002
            stop_dist = atr.iloc[i] * atr_mult
            if pd.isna(stop_dist) or stop_dist == 0: continue
            
            stop, target = pivot - stop_dist, pivot + (stop_dist * reward_mult)
            if high.iloc[i] >= pivot:
                for j in range(1, 21):
                    if i + j >= len(df): break
                    if high.iloc[i+j] >= target:
                        wins += 1; total_r += reward_mult; break
                    if low.iloc[i+j] <= stop:
                        losses += 1; total_r -= 1.0; break
        
        total = wins + losses
        if total < 10: return {'status': 'insufficient', 'message': f'Sample:{total}', 'winrate': 0, 'expectancy': 0}
        return {'status': 'valid', 'winrate': (wins/total)*100, 'expectancy': total_r/total, 'message': f"WR{(wins/total)*100:.0f}% EV{total_r/total:.2f}R"}
    except:
        return {'status': 'error', 'message': 'Error', 'winrate': 0, 'expectancy': 0}

# ============================================================================
# STRATEGIC ANALYZER
# ============================================================================

class StrategicAnalyzer:
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_price_usd):
        if len(df) < MA_LONG + 50: return None
        close, high, low, volume = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
        current_price = float(close.iloc[-1])
        if current_price > max_price_usd: return None
        
        ma50 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = close.rolling(MA_LONG).mean().iloc[-1]
        if not (current_price > ma50 > ma200): return None
        
        tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        tightness = float((high.iloc[-5:].max() - low.iloc[-5:].min()) / atr14)
        if tightness > 3.0: return None
        
        score = 10 # Base
        reasons = ["Base+10"]
        
        # VCP Score
        if tightness < 1.0: score += 30; reasons.append("VCP++30")
        elif tightness < 1.5: score += 20; reasons.append("VCP+20")
        
        # Volume Score
        vol_avg = volume.rolling(50).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg
        if 0.5 <= vol_ratio <= 0.9: score += 15; reasons.append("VolDry+15")
        
        # Momentum
        ma5, ma20 = close.rolling(5).mean().iloc[-1], close.rolling(20).mean().iloc[-1]
        if ma5 > ma20 * 1.02: score += 20; reasons.append("Mom++20")
        
        bt = simulate_past_performance(df, sector)
        if score < MIN_SCORE or (bt['status'] == 'valid' and (bt['winrate'] < MIN_WINRATE or bt['expectancy'] < MIN_EXPECTANCY)):
            return None
        
        pivot = high.iloc[-5:].max() * 1.002
        stop = pivot - (atr14 * ATR_STOP_MULT)
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        target = pivot + ((pivot - stop) * reward_mult)
        
        return {'score': score, 'reasons': ' '.join(reasons), 'price': current_price, 'pivot': pivot, 'stop': stop, 'target': target, 'sector': sector, 'bt': bt}

# ============================================================================
# LINE NOTIFICATION
# ============================================================================

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID: return False
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages": [{"type": "text", "text": msg}]}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        return res.status_code == 200
    except: return False

# ============================================================================
# MAIN
# ============================================================================

def run_mission():
    print(f"\n--- SENTINEL v22.3 | {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")
    if not check_environment(): return

    is_bull, market_status = check_market_trend()
    fx_rate = get_current_fx_rate()
    max_price_usd = (BUDGET_JPY / fx_rate) * 0.95
    risk_budget_jpy = BUDGET_JPY * RISK_PCT_PER_TRADE

    if not is_bull:
        msg = f"SENTINEL v22.3\nMarket Bearish: {market_status}"
        send_line(msg); return

    ticker_list = list(TICKERS.keys())
    all_data = yf.download(ticker_list, period="600d", progress=False, group_by='ticker')
    
    results = []
    for ticker, sector in TICKERS.items():
        if is_earnings_near(ticker) or not sector_is_strong(sector): continue
        try:
            res = StrategicAnalyzer.analyze_ticker(ticker, all_data[ticker], sector, max_price_usd)
            if res: results.append((ticker, res))
        except: continue

    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]

    report = [
        "SENTINEL v22.3",
        f"Market: {market_status}",
        f"FX: {fx_rate:.1f} | Budget: ¥{BUDGET_JPY:,}",
        f"Risk/Trade: ¥{int(risk_budget_jpy)}",
        "-" * 20
    ]

    if not results:
        report.append("No candidates.")
    else:
        for i, (ticker, r) in enumerate(results, 1):
            loss_per_share_jpy = (r['pivot'] - r['stop']) * fx_rate
            # 推奨株数 = リスク予算 ÷ 1株あたりの損切り額
            shares_to_buy = int(risk_budget_jpy // loss_per_share_jpy)
            # 予算による物理的な上限
            max_can_buy = int((BUDGET_JPY / fx_rate) // r['pivot'])
            final_shares = min(shares_to_buy, max_can_buy)

            loss_pct = (1 - r['stop'] / r['pivot']) * 100
            gain_pct = (r['target'] / r['pivot'] - 1) * 100

            report.append(f"[{i}] {ticker} ({r['sector']}) {r['score']}pt")
            report.append(f"推奨株数: {final_shares}株")
            report.append(f"Entry: ${r['pivot']:.2f} / Stop: ${r['stop']:.2f} (-{loss_pct:.1f}%)")
            report.append(f"Target: ${r['target']:.2f} (+{gain_pct:.1f}%)")
            report.append(f"BT: {r['bt']['message']}")
            report.append("-" * 20)

    full_report = "\n".join(report)
    print(full_report)
    send_line(full_report)

if __name__ == "__main__":
    run_mission()
