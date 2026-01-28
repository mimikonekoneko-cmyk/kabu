import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
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

BUDGET_JPY = 350000

# ============================================================================
# CORE PARAMETERS
# ============================================================================

MA_SHORT, MA_LONG = 50, 200
MIN_SCORE = 75
MIN_WINRATE = 45
MIN_EXPECTANCY = 0.3
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
# TICKER UNIVERSE
# ============================================================================

TICKERS = {
    'NVDA':'AI','AVGO':'Semi','ARM':'Semi','MU':'Semi','AMD':'Semi','SMCI':'AI','TSM':'Semi','ASML':'Semi',
    'AAPL':'Device','MSFT':'Cloud','GOOGL':'Ad','META':'Ad','AMZN':'Retail','TSLA':'EV','NFLX':'Service',
    'PLTR':'AI','PANW':'Sec','CRWD':'Sec','NET':'Sec','NOW':'Soft','CRM':'Soft','TEAM':'Soft','ADBE':'Soft',
    'COST':'Retail','WMT':'Retail','TJX':'Retail','ELF':'Cons','PEP':'Cons','KO':'Cons','PG':'Cons',
    'V':'Fin','MA':'Fin','JPM':'Bank','GS':'Bank','AXP':'Fin','BLK':'Fin','MS':'Bank','COIN':'Crypto',
    'LLY':'Bio','UNH':'Health','ABBV':'Bio','ISRG':'Health','VRTX':'Bio',
    'GE':'Ind','CAT':'Ind','DE':'Ind','XOM':'Energy','CVX':'Energy','MPC':'Energy','BA':'Ind',
    'UBER':'Platform','BKNG':'Travel','ABNB':'Travel','DKNG':'Bet','LULU':'Cons','VRT':'Power'
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
    """Check environment variables"""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)
    
    if ACCESS_TOKEN:
        masked_token = ACCESS_TOKEN[:10] + "..." if len(ACCESS_TOKEN) > 10 else "***"
        print(f"ACCESS_TOKEN: {masked_token}")
    else:
        print("ACCESS_TOKEN: Not set")
    
    if USER_ID:
        masked_user = USER_ID[:5] + "..." if len(USER_ID) > 5 else "***"
        print(f"USER_ID: {masked_user}")
    else:
        print("USER_ID: Not set")
    
    print("="*60 + "\n")
    
    return bool(ACCESS_TOKEN and USER_ID)

def get_current_fx_rate():
    """Get current USD/JPY rate"""
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                return float(close.iloc[-1, 0])
            return float(close.iloc[-1])
        return 155.0
    except:
        return 155.0

def check_market_trend():
    """Check overall market trend using SPY"""
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        if spy.empty or len(spy) < 200:
            return True, "Data Limited"
        
        close = spy['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        current = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        
        if current > ma200:
            return True, f"Bull (${current:.0f} > MA200)"
        else:
            return False, f"Bear (${current:.0f} < ${ma200:.0f})"
    except:
        return True, "Check Skipped"

def is_earnings_near(ticker):
    """Check if earnings announcement is within 5 days"""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
            return False
        
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            date_val = cal['Earnings Date']
            if isinstance(date_val, list):
                date_val = date_val[0]
        else:
            date_val = cal.iloc[0, 0]
        
        earnings_date = pd.to_datetime(date_val).date()
        days_until = (earnings_date - datetime.now().date()).days
        
        return abs(days_until) <= 5
    except:
        return False

def sector_is_strong(sector):
    """Check if sector ETF is in uptrend"""
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf:
            return True
        
        df = yf.download(etf, period="250d", progress=False)
        if df.empty or len(df) < 200:
            return True
        
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        ma200 = close.rolling(200).mean()
        return ma200.iloc[-1] > ma200.iloc[-10]
    except:
        return True

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def simulate_past_performance(df, sector, atr_mult=ATR_STOP_MULT):
    """
    Backtest the strategy on historical data
    No look-ahead bias
    """
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        wins = 0
        losses = 0
        total_r = 0
        
        start_idx = max(MA_LONG, len(df) - 500)
        end_idx = len(df) - 10
        
        for i in range(start_idx, end_idx):
            if i < MA_LONG:
                continue
            
            ma50_at_i = close.iloc[i-MA_SHORT:i].mean()
            ma200_at_i = close.iloc[i-MA_LONG:i].mean()
            
            if not (close.iloc[i] > ma50_at_i > ma200_at_i):
                continue
            
            pivot = high.iloc[i-5:i].max() * 1.002
            stop_dist = atr.iloc[i] * atr_mult
            
            if pd.isna(stop_dist) or stop_dist == 0:
                continue
            
            stop = pivot - stop_dist
            target = pivot + (stop_dist * reward_mult)
            
            if high.iloc[i] >= pivot:
                for j in range(1, 21):
                    if i + j >= len(df):
                        break
                    
                    if high.iloc[i+j] >= target:
                        wins += 1
                        total_r += reward_mult
                        break
                    
                    if low.iloc[i+j] <= stop:
                        losses += 1
                        total_r -= 1.0
                        break
        
        total_trades = wins + losses
        
        if total_trades < 10:
            return {
                'status': 'insufficient',
                'message': 'Insufficient samples',
                'trades': total_trades
            }
        
        winrate = (wins / total_trades) * 100
        expectancy = total_r / total_trades
        
        return {
            'status': 'valid',
            'winrate': winrate,
            'expectancy': expectancy,
            'wins': wins,
            'losses': losses,
            'total': total_trades,
            'message': f"WR{winrate:.0f}% ({wins}/{total_trades}) EV{expectancy:.2f}R"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error: {str(e)}'
        }

# ============================================================================
# STRATEGIC ANALYZER
# ============================================================================

class StrategicAnalyzer:
    
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_price_usd):
        """
        Analyze a ticker for entry opportunity
        100-point scoring system
        """
        if len(df) < MA_LONG + 50:
            return None
        
        try:
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
        except:
            return None
        
        current_price = float(close.iloc[-1])
        
        if current_price > max_price_usd:
            return None
        
        ma50 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = close.rolling(MA_LONG).mean().iloc[-1]
        
        if not (current_price > ma50 > ma200):
            return None
        
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(14).mean().iloc[-1]
        
        if atr14 == 0 or pd.isna(atr14):
            return None
        
        recent_range = high.iloc[-5:].max() - low.iloc[-5:].min()
        tightness = float(recent_range / atr14)
        
        if tightness > 3.0:
            return None
        
        score = 0
        reasons = []
        
        # VCP Tightness (max 30 points)
        if tightness < 1.0:
            score += 30
            reasons.append("VCP++30")
        elif tightness < 1.5:
            score += 20
            reasons.append("VCP+20")
        elif tightness < 2.0:
            score += 10
            reasons.append("VCP+10")
        else:
            score += 5
            reasons.append("VCP+5")
        
        # Volume Analysis (max 25 points)
        vol_avg = volume.rolling(50).mean().iloc[-1]
        
        if vol_avg > 0:
            vol_ratio = volume.iloc[-1] / vol_avg
            
            if 0.5 <= vol_ratio <= 0.9:
                score += 15
                reasons.append("VolDry+15")
            elif 0.9 < vol_ratio <= 1.1:
                score += 10
                reasons.append("VolStable+10")
            
            recent_vol_max = volume.iloc[-3:].max()
            if recent_vol_max > vol_avg * 2.0:
                score += 10
                reasons.append("Accumulation++10")
            elif recent_vol_max > vol_avg * 1.5:
                score += 5
                reasons.append("Accumulation+5")
        
        # Momentum (max 20 points)
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        
        if ma5 > ma20 * 1.02:
            score += 20
            reasons.append("Momentum++20")
        elif ma5 > ma20 * 1.01:
            score += 15
            reasons.append("Momentum+15")
        elif ma5 > ma20:
            score += 10
            reasons.append("Momentum+10")
        
        # Trend Strength (max 15 points)
        trend_strength = (ma50 - ma200) / ma200 * 100
        if trend_strength > 10:
            score += 15
            reasons.append("Trend++15")
        elif trend_strength > 5:
            score += 10
            reasons.append("Trend+10")
        else:
            score += 5
            reasons.append("Trend+5")
        
        # Baseline (10 points)
        score += 10
        reasons.append("Base+10")
        
        # Calculate entry/exit levels
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        pivot = high.iloc[-5:].max() * 1.002
        stop_dist = atr14 * ATR_STOP_MULT
        stop_loss = pivot - stop_dist
        target = pivot + (stop_dist * reward_mult)
        
        # Run backtest
        bt_result = simulate_past_performance(df, sector)
        
        # Filter by backtest results
        if bt_result['status'] == 'valid':
            if bt_result['winrate'] < MIN_WINRATE:
                return None
            if bt_result['expectancy'] < MIN_EXPECTANCY:
                return None
        elif bt_result['status'] == 'error':
            return None
        
        return {
            'score': score,
            'reasons': ' '.join(reasons),
            'price': current_price,
            'pivot': pivot,
            'stop': stop_loss,
            'target': target,
            'sector': sector,
            'tightness': tightness,
            'bt': bt_result
        }

# ============================================================================
# LINE NOTIFICATION
# ============================================================================

def send_line(msg):
    """Send LINE notification with detailed logging"""
    
    print("\n" + "="*60)
    print("LINE Notification")
    print("="*60)
    
    if not ACCESS_TOKEN:
        print("ERROR: ACCESS_TOKEN not set")
        print("Check environment variables:")
        print("  - LINE_CHANNEL_ACCESS_TOKEN")
        print("  - LINECHANNELACCESSTOKEN")
        print("  - ACCESS_TOKEN")
        print("="*60)
        print("Message content:")
        print("-"*60)
        print(msg)
        print("="*60 + "\n")
        return False
    
    if not USER_ID:
        print("ERROR: USER_ID not set")
        print("Check environment variables:")
        print("  - LINE_USER_ID")
        print("  - LINEUSER_ID")
        print("  - USER_ID")
        print("="*60)
        print("Message content:")
        print("-"*60)
        print(msg)
        print("="*60 + "\n")
        return False
    
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    payload = {
        "to": USER_ID,
        "messages": [{"type": "text", "text": msg}]
    }
    
    try:
        print(f"Sending to: {url}")
        print(f"USER_ID: {USER_ID[:10]}...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("SUCCESS: LINE notification sent")
            print("="*60 + "\n")
            return True
        else:
            print(f"FAILED: LINE notification")
            print(f"Response: {response.text}")
            print("="*60 + "\n")
            return False
            
    except requests.exceptions.Timeout:
        print("ERROR: Timeout (no response within 10 seconds)")
        print("="*60 + "\n")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        print("="*60 + "\n")
        return False

# ============================================================================
# MAIN MISSION
# ============================================================================

def run_mission():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("SENTINEL v22.1 - Production Ready")
    print("="*60)
    print(f"Launch Time: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Check environment
    env_ok = check_environment()
    
    if not env_ok:
        print("WARNING: LINE credentials not set. Console output only.\n")
    
    # Market check
    print("Checking market conditions...")
    is_bull, market_status = check_market_trend()
    
    if not is_bull:
        msg = (
            f"SENTINEL v22.1\n"
            f"Market conditions unfavorable. Standby mode.\n"
            f"\n"
            f"Market Status: {market_status}\n"
            f"Time: {datetime.now().strftime('%Y/%m/%d %H:%M')}"
        )
        print(msg)
        send_line(msg)
        return
    
    print(f"Market Status: {market_status}\n")
    
    # Get FX rate
    print("Fetching FX rate...")
    fx_rate = get_current_fx_rate()
    max_price_usd = (BUDGET_JPY / fx_rate) * 0.9
    
    print(f"FX Rate: JPY {fx_rate:.2f}/USD")
    print(f"Max Price: ${max_price_usd:.2f}\n")
    
    # Download data
    print(f"Downloading data for {len(TICKERS)} tickers...")
    ticker_list = list(TICKERS.keys())
    
    try:
        all_data = yf.download(
            ticker_list,
            period="600d",
            progress=False,
            group_by='ticker',
            threads=True
        )
        print("Download complete\n")
    except Exception as e:
        print(f"ERROR: Data download failed - {e}")
        return
    
    # Analyze tickers
    print("Starting ticker screening...\n")
    
    results = []
    analyzed_count = 0
    filtered_count = 0
    
    for ticker, sector in TICKERS.items():
        analyzed_count += 1
        
        if is_earnings_near(ticker):
            print(f"SKIP {ticker}: Earnings nearby")
            continue
        
        if not sector_is_strong(sector):
            print(f"SKIP {ticker}: Weak sector")
            continue
        
        try:
            if len(ticker_list) > 1:
                df_ticker = all_data[ticker]
            else:
                df_ticker = all_data
            
            result = StrategicAnalyzer.analyze_ticker(
                ticker, df_ticker, sector, max_price_usd
            )
            
            if result:
                if result['score'] >= MIN_SCORE:
                    results.append((ticker, result))
                    print(f"OK {ticker}: {result['score']} points - Added")
                else:
                    filtered_count += 1
                    print(f"LOW {ticker}: {result['score']} points - Filtered")
            else:
                filtered_count += 1
                
        except Exception as e:
            print(f"ERROR {ticker}: Analysis failed - {e}")
            continue
    
    # Sort results
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    print(f"\n{'='*60}")
    print(f"Screening Results")
    print(f"{'='*60}")
    print(f"Analyzed: {analyzed_count}")
    print(f"Candidates: {len(results)}")
    print(f"Filtered: {filtered_count}")
    print(f"{'='*60}\n")
    
    # Generate report
    report_lines = [
        "SENTINEL v22.1",
        f"{datetime.now().strftime('%Y/%m/%d %H:%M')}",
        f"Market: {market_status}",
        f"FX: JPY{fx_rate:.2f}/USD",
        "-" * 30
    ]
    
    if not results:
        report_lines.append("No candidates match criteria")
        report_lines.append("")
        report_lines.append(f"Analyzed: {analyzed_count}")
        report_lines.append(f"Filtered: {filtered_count}")
    else:
        for i, (ticker, r) in enumerate(results, 1):
            loss_pct = (1 - r['stop'] / r['pivot']) * 100
            gain_pct = (r['target'] / r['pivot'] - 1) * 100
            risk_reward = gain_pct / loss_pct
            
            bt_info = r['bt']['message'] if r['bt']['status'] == 'valid' else r['bt']['message']
            
            report_lines.append(f"[{i}] {ticker} ({r['sector']}) {r['score']}pt")
            report_lines.append(f"{r['reasons']}")
            report_lines.append(f"BT: {bt_info}")
            report_lines.append(f"Price: ${r['price']:.2f}")
            report_lines.append(f"Entry: ${r['pivot']:.2f}")
            report_lines.append(f"Stop: ${r['stop']:.2f} (-{loss_pct:.1f}%)")
            report_lines.append(f"Target: ${r['target']:.2f} (+{gain_pct:.1f}%)")
            report_lines.append(f"RR: 1:{risk_reward:.1f}")
            report_lines.append("-" * 30)
    
    full_report = "\n".join(report_lines)
    
    # Output
    print("\n" + "="*60)
    print("Final Report")
    print("="*60)
    print(full_report)
    print("="*60 + "\n")
    
    # Send LINE notification
    send_success = send_line(full_report)
    
    if send_success:
        print("All processes completed successfully\n")
    else:
        print("Analysis completed (LINE notification failed)\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_mission()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user\n")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}\n")
        import traceback
        traceback.print_exc()