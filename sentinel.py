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
QUARTERLY_CONTRIBUTION = 30000  
TRADING_RATIO = 0.70
HOLDING_RATIO = 0.30

# Risk Management Parameters
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25  
MAX_CONCURRENT_POSITIONS = 4
MAX_SECTOR_CONCENTRATION = 0.40  

# Filter Thresholds - v25.1 PROFESSIONAL STRATEGY
MIN_SCORE = 75          
MIN_WINRATE = 55        
MIN_EXPECTANCY = 0.45   
MAX_TIGHTNESS = 1.5     
MAX_NOTIFICATIONS = 5

# Liquidity Filter
MIN_DAILY_VOLUME_USD = 10_000_000  

# Reward Multipliers by Strategy Type
REWARD_MULTIPLIERS = {
    'aggressive': 2.5,
    'stable': 2.0
}

# Sector Categorization for Multipliers
AGGRESSIVE_SECTORS = [
    'Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 
    'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech'
]

# Transaction Costs Model Constants
COMMISSION_RATE = 0.002  
SLIPPAGE_RATE = 0.001  
FX_SPREAD_RATE = 0.0005  
TOTAL_COST_RATE = (COMMISSION_RATE + SLIPPAGE_RATE + FX_SPREAD_RATE) * 2

# Performance Metrics
TARGET_ANNUAL_RETURN = 0.10
PERFORMANCE_LOG_PATH = Path("/tmp/sentinel_performance.json")
TRADE_LOG_PATH = Path("/tmp/sentinel_trades.json")

# Moving Average Periods
MA_SHORT = 50
MA_LONG = 200

# ============================================================================
# TICKER UNIVERSE v25.1 FINAL (92 TICKERS) - FULL LIST
# ============================================================================

TICKERS = {
    # SEMICONDUCTORS
    'NVDA':'AI', 'AMD':'Semi', 'AVGO':'Semi', 'TSM':'Semi', 'ASML':'Semi', 'MU':'Semi',
    'ARM':'Semi', 'INTC':'Semi', 'QCOM':'Semi', 'ON':'Semi', 'LRCX':'Semi', 'AMAT':'Semi',
    
    # AI & ENTERPRISE SOFTWARE
    'MSFT':'Cloud', 'GOOGL':'Ad', 'META':'Ad', 'PLTR':'AI', 'NOW':'Soft', 'CRM':'Soft',
    'ADBE':'Soft', 'SNOW':'Cloud', 'DDOG':'Cloud', 'WDAY':'Soft', 'TEAM':'Soft',
    'ANET':'Cloud', 'ZS':'Sec', 'MDB':'Cloud', 'SHOP':'Retail',
    
    # CYBERSECURITY
    'PANW':'Sec', 'CRWD':'Sec', 'FTNT':'Sec', 'NET':'Sec', 'OKTA':'Sec',
    
    # CONSUMER & RETAIL
    'AAPL':'Device', 'TSLA':'Auto', 'AMZN':'Retail', 'NFLX':'Service', 'COST':'Retail', 
    'WMT':'Retail', 'TJX':'Retail', 'TGT':'Retail', 'NKE':'Cons', 'LULU':'Cons', 
    'SBUX':'Cons', 'PEP':'Cons', 'KO':'Cons', 'PG':'Cons', 'ELF':'Cons', 'CELH':'Cons', 
    'MELI':'Retail',
    
    # FINANCE & FINTECH
    'V':'Fin', 'MA':'Fin', 'PYPL':'Fintech', 'SQ':'Fintech', 'JPM':'Bank', 'GS':'Bank',
    'MS':'Bank', 'AXP':'Fin', 'BLK':'Fin', 'COIN':'Crypto', 'SOFI':'Fintech', 'NU':'Fintech',
    
    # HEALTHCARE & BIOTECH
    'LLY':'Bio', 'UNH':'Health', 'ABBV':'Bio', 'ISRG':'Health', 'VRTX':'Bio', 'MRK':'Bio',
    'PFE':'Bio', 'AMGN':'Bio', 'HCA':'Health', 'TDOC':'Health',
    
    # INDUSTRIALS, ENERGY & POWER
    'GE':'Ind', 'CAT':'Ind', 'DE':'Ind', 'BA':'Ind', 'ETN':'Power', 'VRT':'Power', 
    'TT':'Ind', 'PH':'Ind', 'TDG':'Ind', 'XOM':'Energy', 'CVX':'Energy', 'MPC':'Energy',
    
    # TRAVEL & LEISURE
    'UBER':'Platform', 'BKNG':'Travel', 'ABNB':'Travel', 'MAR':'Travel', 'RCL':'Travel', 
    'DKNG':'Bet',
    
    # EMERGING & HIGH GROWTH
    'RBLX':'Service', 'DASH':'Service', 'SMCI':'AI'
}

SECTOR_ETF = {
    'Energy':'XLE', 'Semi':'SOXX', 'Bank':'XLF', 'Retail':'XRT',
    'Soft':'IGV', 'AI':'QQQ', 'Fin':'VFH', 'Device':'QQQ',
    'Cloud':'QQQ', 'Ad':'QQQ', 'Service':'QQQ', 'Sec':'HACK',
    'Cons':'XLP', 'Bio':'IBB', 'Health':'XLV', 'Ind':'XLI',
    'Auto':'CARZ', 'Crypto':'BTC-USD', 'Power':'XLI', 'Platform':'QQQ',
    'Travel':'XLY', 'Bet':'BETZ', 'Fintech':'ARKF'
}

# ============================================================================
# UTILITY FUNCTIONS & MARKET INTELLIGENCE
# ============================================================================

def check_environment():
    """環境変数の詳細チェックとログ出力"""
    print("\n" + "="*70)
    print("SENTINEL SYSTEM ENVIRONMENT CHECK")
    print("="*70)
    
    status = True
    if ACCESS_TOKEN:
        print(f"✓ LINE_ACCESS_TOKEN: {ACCESS_TOKEN[:15]}... [LOADED]")
    else:
        print("✗ LINE_ACCESS_TOKEN: MISSING")
        status = False
        
    if USER_ID:
        print(f"✓ LINE_USER_ID: {USER_ID[:10]}... [LOADED]")
    else:
        print("✗ LINE_USER_ID: MISSING")
        status = False
        
    print(f"✓ SYSTEM TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    return status

def get_current_fx_rate():
    """最新の為替レート(USD/JPY)を取得"""
    print("Fetching FX rate (USD/JPY)...")
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            close = data['Close']
            rate = float(close.iloc[-1, 0]) if isinstance(close, pd.DataFrame) else float(close.iloc[-1])
            print(f"✓ Current FX Rate: ¥{rate:.2f}")
            return rate
        return 152.0
    except Exception as e:
        print(f"! FX Fetch Error: {e}. Using default 152.0")
        return 152.0

def get_vix():
    """市場の恐怖指数(VIX)を取得"""
    print("Fetching VIX Index...")
    try:
        data = yf.download("^VIX", period="1d", progress=False)
        if not data.empty:
            close = data['Close']
            vix_val = float(close.iloc[-1, 0]) if isinstance(close, pd.DataFrame) else float(close.iloc[-1])
            print(f"✓ Current VIX: {vix_val:.2f}")
            return vix_val
        return 20.0
    except Exception as e:
        print(f"! VIX Fetch Error: {e}. Using default 20.0")
        return 20.0

def check_market_trend():
    """SPYの200日移動平均線に基づいた地合い判定"""
    print("Analyzing Market Trend (SPY)...")
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        if spy.empty or len(spy) < 200:
            return True, "Data Limited", 0
        
        close = spy['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        current = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        distance = ((current - ma200) / ma200) * 100
        
        if current > ma200:
            status = f"Bull (+{distance:.1f}% above MA200)"
            print(f"✓ Market Status: {status}")
            return True, status, distance
        else:
            status = f"Bear ({distance:.1f}% below MA200)"
            print(f"✓ Market Status: {status}")
            return False, status, distance
    except Exception as e:
        print(f"! Market Trend Error: {e}")
        return True, "Trend Check Skipped", 0

def is_earnings_near(ticker):
    """決算発表が前後5日以内にあるかチェック"""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
            return False
        
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            date_val = cal['Earnings Date'][0] if isinstance(cal['Earnings Date'], list) else cal['Earnings Date']
        else:
            date_val = cal.iloc[0, 0]
            
        earnings_date = pd.to_datetime(date_val).date()
        days_until = (earnings_date - datetime.now().date()).days
        return abs(days_until) <= 5
    except:
        return False

def sector_is_strong(sector):
    """セクター別ETFのトレンドが上昇中かチェック"""
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf: return True
        df = yf.download(etf, period="250d", progress=False)
        if df.empty or len(df) < 200: return True
        close = df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        ma200 = close.rolling(200).mean()
        # 200MAが10日前より上昇しているか
        return ma200.iloc[-1] > ma200.iloc[-10]
    except:
        return True

# ============================================================================
# FINANCIAL MODELS (TRANSACTION & POSITION)
# ============================================================================

class TransactionCostModel:
    """実戦的な取引コスト（手数料、滑り、為替）を算出"""
    @staticmethod
    def calculate_total_cost(position_value_usd, fx_rate):
        # 日本の証券会社を想定した手数料(約0.2%)
        commission = position_value_usd * COMMISSION_RATE
        
        # 流動性に応じたスリッページ予測
        if position_value_usd < 500:
            slippage_rate = 0.0005
        elif position_value_usd < 2000:
            slippage_rate = 0.001
        else:
            slippage_rate = 0.0015
            
        slippage = position_value_usd * slippage_rate
        fx_cost = position_value_usd * FX_SPREAD_RATE
        
        # 往復(Round Trip)コスト
        total_usd = (commission + slippage + fx_cost) * 2
        return total_usd, total_usd * fx_rate

    @staticmethod
    def adjust_expectancy_for_cost(gross_expectancy, avg_1r_pct, position_value_usd, fx_rate):
        """取引コストをR倍数(リスク単位)に換算して期待値を修正"""
        cost_usd, _ = TransactionCostModel.calculate_total_cost(position_value_usd, fx_rate)
        cost_pct = (cost_usd / position_value_usd) * 100
        cost_in_r = cost_pct / avg_1r_pct
        return gross_expectancy - cost_in_r, cost_in_r

class PositionSizer:
    """ケリー基準と外部要因を組み合わせた動的資金管理"""
    @staticmethod
    def calculate_kelly_fraction(winrate, rr_ratio):
        if winrate <= 0 or winrate >= 1: return 0
        # ケリー公式: (bp - q) / b  (b:RR比, p:勝率, q:敗率)
        kelly = (rr_ratio * winrate - (1 - winrate)) / rr_ratio
        # 安全のためハーフケリーを上限に調整
        return max(0, min(kelly / 2, 0.25))

    @staticmethod
    def calculate_position_size(trading_capital, winrate, rr_ratio, atr_pct, vix, sector_exposure):
        kelly_fraction = PositionSizer.calculate_kelly_fraction(winrate, rr_ratio)
        
        # 各種調整係数
        vol_f = 0.7 if atr_pct > 5.0 else 0.85 if atr_pct > 3.0 else 1.0
        mkt_f = 0.6 if vix > 30 else 0.8 if vix > 20 else 1.0
        sec_f = 0.7 if sector_exposure > 0.30 else 0.85 if sector_exposure > 0.20 else 1.0
        
        final_frac = min(kelly_fraction * vol_f * mkt_f * sec_f, MAX_POSITION_SIZE)
        position_jpy = trading_capital * final_frac
        
        return position_jpy, {
            'kelly': kelly_fraction,
            'vol_adj': vol_f,
            'mkt_adj': mkt_f,
            'sec_adj': sec_f,
            'final_fraction': final_frac
        }

# ============================================================================
# TRADING LOGIC & BACKTEST ENGINE
# ============================================================================

class TrailingStopManager:
    """トレーリングストップの多段階管理"""
    @staticmethod
    def calculate_stop(entry_price, current_price, highest_since_entry, atr_val):
        initial_stop = entry_price - (atr_val * ATR_STOP_MULT)
        # 第1段階：含み益がわずかな時は初期ストップ
        if current_price < entry_price + (atr_val * 0.5):
            return initial_stop, "Initial"
        # 第2段階：1ATR程度の利益で建値決済(Breakeven)に移動
        elif current_price < entry_price + (atr_val * 1.0):
            return entry_price, "Breakeven"
        # 第3段階：それ以降は高値から追従
        else:
            trailing_stop = highest_since_entry - (atr_val * ATR_STOP_MULT)
            return max(trailing_stop, entry_price), "Trailing"

def simulate_past_performance_v2(df, sector):
    """過去データを用いた詳細な戦略検証（トレーリングストップ込）"""
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        # 真のレンジ(TR)とATR
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        wins, losses, total_r = 0, 0, 0
        start_idx = max(MA_LONG, len(df) - 500)
        
        for i in range(start_idx, len(df) - 10):
            # トレンド条件
            if not (close.iloc[i] > close.iloc[i-MA_SHORT:i].mean() > close.iloc[i-MA_LONG:i].mean()):
                continue
            
            # エントリー判定（VCPピボット）
            pivot = high.iloc[i-5:i].max() * 1.002
            stop_dist = atr.iloc[i] * ATR_STOP_MULT
            
            if pd.isna(stop_dist) or stop_dist == 0: continue
            
            if high.iloc[i] >= pivot:
                entry = pivot
                target = pivot + (stop_dist * reward_mult)
                highest = high.iloc[i]
                
                # 30日間のトレース
                for j in range(1, 30):
                    if i + j >= len(df): break
                    c_high, c_low, c_close = high.iloc[i+j], low.iloc[i+j], close.iloc[i+j]
                    highest = max(highest, c_high)
                    
                    c_stop, _ = TrailingStopManager.calculate_stop(entry, c_close, highest, stop_dist/ATR_STOP_MULT)
                    
                    if c_high >= target:
                        wins += 1; total_r += reward_mult; break
                    if c_low <= c_stop:
                        losses += 1; total_r -= (entry - c_stop)/stop_dist; break
                        
        total = wins + losses
        if total < 10:
            return {'status':'insufficient', 'trades':total, 'winrate':0, 'expectancy':0, 'message':f"Sample:{total}"}
            
        wr = (wins / total) * 100
        ev = total_r / total
        # コスト考慮
        net_ev, _ = TransactionCostModel.adjust_expectancy_for_cost(ev, 8.5, 5000, 152.0)
        
        return {
            'status': 'valid',
            'winrate': wr,
            'expectancy': ev,
            'net_expectancy': net_ev,
            'total': total,
            'message': f"WR{wr:.0f}% NetEV{net_ev:.2f}R"
        }
    except Exception as e:
        return {'status': 'error', 'message': f"BT Error: {str(e)}"}

# ============================================================================
# STRATEGIC ANALYZER CORE
# ============================================================================

class StrategicAnalyzerV2:
    """個別銘柄の多角的分析とスコアリング"""
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_price_usd, vix, sector_exposures, trading_capital):
        if len(df) < MA_LONG + 50: return None
        
        try:
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
        except: return None
        
        curr_price = float(close.iloc[-1])
        if curr_price > max_price_usd: return None
        
        # 流動性フィルター
        avg_vol_usd = volume.rolling(50).mean().iloc[-1] * curr_price
        if avg_vol_usd < MIN_DAILY_VOLUME_USD: return None

        # トレンドフィルター
        ma50 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = close.rolling(MA_LONG).mean().iloc[-1]
        if not (curr_price > ma50 > ma200): return None
        
        # VCP Tightness (ボラティリティ収縮)
        tr = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        tightness = float((high.iloc[-5:].max() - low.iloc[-5:].min()) / atr14)
        if tightness > MAX_TIGHTNESS: return None

        # --- SCORING SYSTEM ---
        score, reasons = 0, []
        
        # 1. Tightness Score
        if tightness < 0.8: score += 35; reasons.append("VCP+++")
        elif tightness < 1.0: score += 30; reasons.append("VCP++")
        elif tightness < 1.2: score += 20; reasons.append("VCP+")
        else: score += 10; reasons.append("VCP")
        
        # 2. Volume Score (売り枯れ)
        vol_avg = volume.rolling(50).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg
        if 0.5 <= vol_ratio <= 0.85:
            score += 20; reasons.append("VolDry")
        
        # 3. Momentum Score
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        mom = ((ma5 / ma20) - 1) * 100
        if mom > 2.0: score += 20; reasons.append("Mom++")
        elif mom > 1.0: score += 10; reasons.append("Mom+")
        
        # 4. Trend Strength Score
        trend_s = ((ma50 - ma200) / ma200) * 100
        if trend_s > 15: score += 25; reasons.append("Trend++")
        elif trend_s > 5: score += 15; reasons.append("Trend+")
        
        # バックテストと資金管理の統合
        bt = simulate_past_performance_v2(df, sector)
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        atr_pct = (atr14 / curr_price) * 100
        pos_size, s_factors = PositionSizer.calculate_position_size(
            trading_capital, 
            bt.get('winrate', 50)/100, 
            reward_mult, 
            atr_pct, 
            vix, 
            sector_exposures.get(sector, 0)
        )
        
        pivot = high.iloc[-5:].max() * 1.002
        
        return {
            'score': score,
            'reasons': ' '.join(reasons),
            'price': curr_price,
            'pivot': pivot,
            'stop': pivot - (atr14 * ATR_STOP_MULT),
            'target': pivot + (atr14 * ATR_STOP_MULT * reward_mult),
            'sector': sector,
            'bt': bt,
            'position_size_jpy': pos_size,
            'sizing_factors': s_factors,
            'liquidity_usd': avg_vol_usd,
            'tightness': tightness,
            'atr_pct': atr_pct
        }

# ============================================================================
# PERFORMANCE TRACKING & NOTIFICATION
# ============================================================================

class PerformanceTracker:
    @staticmethod
    def load_performance_history():
        if PERFORMANCE_LOG_PATH.exists():
            with open(PERFORMANCE_LOG_PATH, 'r') as f: return json.load(f)
        return {'start_date': datetime.now().isoformat(), 'initial_capital': INITIAL_CAPITAL, 'quarters': []}

    @staticmethod
    def get_current_status():
        perf = PerformanceTracker.load_performance_history()
        start_date = datetime.fromisoformat(perf['start_date'])
        days = (datetime.now() - start_date).days
        target = TARGET_ANNUAL_RETURN * (days / 365)
        actual = sum(q.get('return', 0) for q in perf['quarters'])
        return {
            'on_track': actual >= target * 0.9,
            'ytd_return': actual,
            'target_return': target,
            'days_passed': days
        }

def send_line(msg):
    """LINE通知の実行"""
    print("\n" + "="*70)
    print("LINE NOTIFICATION PREVIEW")
    print("="*70)
    print(msg)
    print("="*70)
    
    if not ACCESS_TOKEN or not USER_ID:
        print("Credentials not set. Skipping LINE send.")
        return False
        
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages": [{"type": "text", "text": msg}]}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"LINE Send Error: {e}")
        return False

# ============================================================================
# MAIN MISSION EXECUTION
# ============================================================================

def run_mission():
    """SENTINEL v25.1 メインミッション"""
    start_time = datetime.now()
    print(f"SENTINEL v25.1 FINAL - Launch Sequence Initiated: {start_time}")
    
    # 1. 環境と市場状況の把握
    if not check_environment():
        print("Critical environment variables missing. Aborting.")
        return
        
    is_bull, market_status, _ = check_market_trend()
    fx_rate = get_current_fx_rate()
    vix = get_vix()
    perf = PerformanceTracker.get_current_status()
    
    trading_capital = INITIAL_CAPITAL * TRADING_RATIO
    max_price = (trading_capital / fx_rate) * 0.9
    
    # 2. データ一括ダウンロード
    print(f"Downloading historical data for {len(TICKERS)} tickers...")
    try:
        all_data = yf.download(list(TICKERS.keys()), period="600d", progress=False, group_by='ticker', threads=True)
        print("✓ Data download complete.")
    except Exception as e:
        print(f"✗ Critical Download Error: {e}")
        return

    # 3. 銘柄スクリーニング
    print("Starting comprehensive screening...")
    all_results = []
    sector_exposures = {}
    
    for ticker, sector in TICKERS.items():
        # フィルタ：決算近傍とセクター弱気
        if is_earnings_near(ticker):
            print(f"  - {ticker}: Skipped (Earnings Near)")
            continue
        if not sector_is_strong(sector):
            print(f"  - {ticker}: Skipped (Weak Sector)")
            continue
            
        try:
            res = StrategicAnalyzerV2.analyze_ticker(
                ticker, all_data[ticker], sector, max_price, vix, sector_exposures, trading_capital
            )
            if res:
                all_results.append((ticker, res))
                # セクター露出の更新(合格基準に関わらず統計用に記録可能だが、ここでは厳選銘柄のみ)
                if res['score'] >= MIN_SCORE:
                    sector_exposures[sector] = sector_exposures.get(sector, 0) + (res['position_size_jpy'] / trading_capital)
                    print(f"  + {ticker}: High Potential ({res['score']}pt)")
        except Exception as e:
            print(f"  ! {ticker}: Analysis Error - {e}")

    # 4. 分析とフィルタリングの最終化
    all_sorted = sorted(all_results, key=lambda x: x[1]['score'], reverse=True)
    
    # ★買付推奨：全ての厳しい基準をクリアしたもの
    passed = [
        r for r in all_sorted 
        if r[1]['score'] >= MIN_SCORE 
        and r[1]['bt'].get('winrate', 0) >= MIN_WINRATE 
        and r[1]['bt'].get('net_expectancy', 0) >= MIN_EXPECTANCY
    ]
    
    # 5. レポート作成
    report = [
        "SENTINEL v25.1 FINAL",
        f"Execution: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Market: {market_status}",
        f"VIX: {vix:.1f} | FX: ¥{fx_rate:.2f}",
        f"YTD: {perf['ytd_return']*100:.1f}% (Target: {perf['target_return']*100:.1f}%)",
        "=" * 35,
        "【BUY SIGNALS (STRICT)】"
    ]
    
    if not passed:
        report.append("No candidates passed all strict filters.")
        report.append("Focus on capital preservation.")
    else:
        for i, (ticker, r) in enumerate(passed[:MAX_NOTIFICATIONS], 1):
            shares = int(r['position_size_jpy'] / fx_rate / r['pivot'])
            report.append(f"★ [{i}] {ticker} ({r['sector']}) {r['score']}pt")
            report.append(f"Position: {shares}株 (¥{r['position_size_jpy']:,.0f})")
            report.append(f"Entry: ${r['pivot']:.2f} / Stop: ${r['stop']:.2f}")
            report.append(f"Target: ${r['target']:.2f} / BT: {r['bt']['message']}")
            report.append("-" * 20)

    # 上位10社の詳細分析（落選理由の可視化）
    report.append("\n【ANALYSIS TOP 10 (RAW)】")
    if not all_sorted:
        report.append("No tickers met baseline criteria.")
    else:
        for i, (ticker, r) in enumerate(all_sorted[:10], 1):
            # 判定ロジックの可視化
            status = "✅PASS"
            if r['score'] < MIN_SCORE: status = "❌SCORE"
            elif r['bt'].get('winrate', 0) < MIN_WINRATE: status = "❌WR"
            elif r['bt'].get('net_expectancy', 0) < MIN_EXPECTANCY: status = "❌EV"
            
            report.append(f"{i}. {ticker:5} : {r['score']}pt | {status}")
            report.append(f"   WR:{r['bt'].get('winrate',0):.0f}% EV:{r['bt'].get('net_expectancy',0):.2f}R Tight:{r['tightness']:.2f}")

    report.append("=" * 35)
    report.append(f"Processed {len(TICKERS)} tickers in {(datetime.now()-start_time).seconds}s")
    
    # 6. 送信
    send_line("\n".join(report))
    print("\nMission Complete.")

if __name__ == "__main__":
    try:
        run_mission()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nCRITICAL SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# END OF CODE - SENTINEL v25.1
# ============================================================================
