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
QUARTERLY_CONTRIBUTION = 30000  # JPY (every 3 months)
TRADING_RATIO = 0.70
HOLDING_RATIO = 0.30

# Risk Management
ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25  # 25% max per position
MAX_CONCURRENT_POSITIONS = 4
MAX_SECTOR_CONCENTRATION = 0.40  # 40% max per sector

# Filter Thresholds - v25.1 BALANCED
MIN_SCORE = 75          # Balanced for weekly opportunities
MIN_WINRATE = 55        # 55% is still excellent
MIN_EXPECTANCY = 0.45   # Net expectancy after costs
MAX_TIGHTNESS = 1.5     # Allow slightly looser VCP
MAX_NOTIFICATIONS = 5

# Liquidity Filter
MIN_DAILY_VOLUME_USD = 10_000_000  # $10M minimum daily volume

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
COMMISSION_RATE = 0.002  # 0.2% per trade
SLIPPAGE_RATE = 0.001  # 0.1% average slippage
FX_SPREAD_RATE = 0.0005  # 0.05% FX spread
TOTAL_COST_RATE = (COMMISSION_RATE + SLIPPAGE_RATE + FX_SPREAD_RATE) * 2  # Round trip

# Performance Tracking
TARGET_ANNUAL_RETURN = 0.10
PERFORMANCE_LOG_PATH = Path("/tmp/sentinel_performance.json")
TRADE_LOG_PATH = Path("/tmp/sentinel_trades.json")

# ============================================================================
# TICKER UNIVERSE v25.1 FINAL (92 TICKERS)
# ============================================================================

TICKERS = {
    # ========================================
    # SEMICONDUCTORS (12)
    # ========================================
    'NVDA':'AI',        # GPU leader
    'AMD':'Semi',       # CPU/GPU
    'AVGO':'Semi',      # Broadcom - diversified chips
    'TSM':'Semi',       # Foundry leader
    'ASML':'Semi',      # EUV equipment monopoly
    'MU':'Semi',        # Memory
    'ARM':'Semi',       # Mobile CPU architecture
    'INTC':'Semi',      # Legacy + recovery play
    'QCOM':'Semi',      # Mobile chips
    'ON':'Semi',        # Automotive/Industrial
    'LRCX':'Semi',      # Equipment
    'AMAT':'Semi',      # Equipment
    
    # ========================================
    # AI & ENTERPRISE SOFTWARE (15)
    # ========================================
    'MSFT':'Cloud',     # Azure + Office + AI
    'GOOGL':'Ad',       # Search + Cloud + AI
    'META':'Ad',        # Social + AI
    'PLTR':'AI',        # Data analytics
    'NOW':'Soft',       # IT workflow
    'CRM':'Soft',       # Salesforce
    'ADBE':'Soft',      # Creative Cloud
    'SNOW':'Cloud',     # Data warehouse
    'DDOG':'Cloud',     # Monitoring
    'WDAY':'Soft',      # HR software
    'TEAM':'Soft',      # Atlassian - collaboration
    'ANET':'Cloud',     # Data center networking
    'ZS':'Sec',         # Zscaler - cloud security
    'MDB':'Cloud',      # MongoDB
    'SHOP':'Retail',    # Shopify - E-commerce platform
    
    # ========================================
    # CYBERSECURITY (5)
    # ========================================
    'PANW':'Sec',       # Palo Alto - leader
    'CRWD':'Sec',       # CrowdStrike - EDR
    'FTNT':'Sec',       # Fortinet - firewall
    'NET':'Sec',        # Cloudflare - CDN/security
    'OKTA':'Sec',       # Identity management
    
    # ========================================
    # CONSUMER & RETAIL (17)
    # ========================================
    'AAPL':'Device',    # iPhone/Services
    'TSLA':'Auto',      # EV + Energy
    'AMZN':'Retail',    # E-commerce + AWS
    'NFLX':'Service',   # Streaming
    'COST':'Retail',    # Wholesale
    'WMT':'Retail',     # Walmart
    'TJX':'Retail',     # Off-price retail
    'TGT':'Retail',     # Target
    'NKE':'Cons',       # Nike - athletic
    'LULU':'Cons',      # Lululemon - athleisure
    'SBUX':'Cons',      # Starbucks
    'PEP':'Cons',       # Pepsi
    'KO':'Cons',        # Coca-Cola
    'PG':'Cons',        # Procter & Gamble
    'ELF':'Cons',       # e.l.f. Beauty
    'CELH':'Cons',      # Celsius - energy drinks
    'MELI':'Retail',    # MercadoLibre - LatAm e-commerce
    
    # ========================================
    # FINANCE & FINTECH (12)
    # ========================================
    'V':'Fin',          # Visa
    'MA':'Fin',         # Mastercard
    'PYPL':'Fintech',   # PayPal
    'SQ':'Fintech',     # Block (Square)
    'JPM':'Bank',       # JP Morgan
    'GS':'Bank',        # Goldman Sachs
    'MS':'Bank',        # Morgan Stanley
    'AXP':'Fin',        # American Express
    'BLK':'Fin',        # BlackRock - asset management
    'COIN':'Crypto',    # Coinbase
    'SOFI':'Fintech',   # SoFi - digital banking
    'NU':'Fintech',     # Nubank - LatAm neobank
    
    # ========================================
    # HEALTHCARE & BIOTECH (10)
    # ========================================
    'LLY':'Bio',        # Eli Lilly - GLP-1 drugs
    'UNH':'Health',     # UnitedHealth - insurance
    'ABBV':'Bio',       # AbbVie - pharma
    'ISRG':'Health',    # Intuitive Surgical - robotics
    'VRTX':'Bio',       # Vertex - rare disease
    'MRK':'Bio',        # Merck
    'PFE':'Bio',        # Pfizer
    'AMGN':'Bio',       # Amgen
    'HCA':'Health',     # HCA Healthcare
    'TDOC':'Health',    # Teladoc - telehealth
    
    # ========================================
    # INDUSTRIALS, ENERGY & POWER (12)
    # ========================================
    'GE':'Ind',         # GE Aerospace
    'CAT':'Ind',        # Caterpillar
    'DE':'Ind',         # Deere
    'BA':'Ind',         # Boeing
    'ETN':'Power',      # Eaton - electrical
    'VRT':'Power',      # Vertiv - data center power
    'TT':'Ind',         # Trane - HVAC
    'PH':'Ind',         # Parker Hannifin
    'TDG':'Ind',        # TransDigm - aerospace
    'XOM':'Energy',     # Exxon
    'CVX':'Energy',     # Chevron
    'MPC':'Energy',     # Marathon Petroleum
    
    # ========================================
    # TRAVEL & LEISURE (6)
    # ========================================
    'UBER':'Platform',  # Uber
    'BKNG':'Travel',    # Booking.com
    'ABNB':'Travel',    # Airbnb
    'MAR':'Travel',     # Marriott
    'RCL':'Travel',     # Royal Caribbean
    'DKNG':'Bet',       # DraftKings
    
    # ========================================
    # EMERGING & HIGH GROWTH (3)
    # ========================================
    'RBLX':'Service',   # Roblox - metaverse/gaming
    'DASH':'Service',   # DoorDash - food delivery
    'SMCI':'AI',        # Super Micro - AI infrastructure (高リスク)
}

# Total: 92 tickers

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
# UTILITY FUNCTIONS
# ============================================================================

def check_environment():
    """Check and display environment variables"""
    print("\n" + "="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)
    
    if ACCESS_TOKEN:
        print(f"✓ ACCESS_TOKEN: {ACCESS_TOKEN[:15]}...")
    else:
        print("✗ ACCESS_TOKEN: Not set")
    
    if USER_ID:
        print(f"✓ USER_ID: {USER_ID[:10]}...")
    else:
        print("✗ USER_ID: Not set")
    
    print("="*70 + "\n")
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
        return 152.0
    except:
        return 152.0

def get_vix():
    """Get current VIX (market volatility index)"""
    try:
        data = yf.download("^VIX", period="1d", progress=False)
        if not data.empty:
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                return float(close.iloc[-1, 0])
            return float(close.iloc[-1])
        return 20.0
    except:
        return 20.0

def check_market_trend():
    """Check overall market trend"""
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
            return True, f"Bull (+{distance:.1f}% above MA200)", distance
        else:
            return False, f"Bear ({distance:.1f}% below MA200)", distance
    except:
        return True, "Check Skipped", 0

def is_earnings_near(ticker):
    """Check if earnings within 5 days"""
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
    """Check sector strength"""
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
# TRANSACTION COST MODEL
# ============================================================================

class TransactionCostModel:
    """Model for calculating realistic transaction costs"""
    
    @staticmethod
    def calculate_total_cost(position_value_usd, fx_rate):
        """
        Calculate total transaction cost including commission, slippage, FX
        
        Returns: cost in USD and JPY
        """
        # Commission (0.2%)
        commission = position_value_usd * COMMISSION_RATE
        
        # Slippage (depends on position size)
        if position_value_usd < 500:  # Small order
            slippage_rate = 0.0005
        elif position_value_usd < 2000:  # Medium
            slippage_rate = 0.001
        else:  # Large
            slippage_rate = 0.0015
        
        slippage = position_value_usd * slippage_rate
        
        # FX spread
        fx_cost = position_value_usd * FX_SPREAD_RATE
        
        # Total cost (round trip = x2)
        total_usd = (commission + slippage + fx_cost) * 2
        total_jpy = total_usd * fx_rate
        
        return total_usd, total_jpy
    
    @staticmethod
    def adjust_expectancy_for_cost(gross_expectancy, avg_1r_pct, position_value_usd, fx_rate):
        """
        Adjust expectancy for transaction costs
        
        Returns: net expectancy and cost in R
        """
        cost_usd, _ = TransactionCostModel.calculate_total_cost(position_value_usd, fx_rate)
        cost_pct = (cost_usd / position_value_usd) * 100
        cost_in_r = cost_pct / avg_1r_pct
        
        net_expectancy = gross_expectancy - cost_in_r
        
        return net_expectancy, cost_in_r

# ============================================================================
# DYNAMIC POSITION SIZING
# ============================================================================

class PositionSizer:
    """Kelly Criterion based position sizing with safety factors"""
    
    @staticmethod
    def calculate_kelly_fraction(winrate, rr_ratio):
        """Calculate Kelly fraction"""
        if winrate <= 0 or winrate >= 1:
            return 0
        
        kelly = (rr_ratio * winrate - (1 - winrate)) / rr_ratio
        
        # Safety: use half Kelly
        safe_kelly = kelly / 2
        
        return max(0, min(safe_kelly, 0.25))  # Cap at 25%
    
    @staticmethod
    def calculate_position_size(
        trading_capital,
        winrate,
        rr_ratio,
        atr_pct,
        vix,
        sector_exposure
    ):
        """Calculate optimal position size with multiple adjustments"""
        
        # Base Kelly fraction
        kelly_fraction = PositionSizer.calculate_kelly_fraction(winrate, rr_ratio)
        
        # Volatility adjustment
        volatility_factor = 1.0
        if atr_pct > 5.0:  # High volatility stock
            volatility_factor = 0.7
        elif atr_pct > 3.0:
            volatility_factor = 0.85
        
        # Market environment adjustment (VIX)
        market_factor = 1.0
        if vix > 30:  # High fear
            market_factor = 0.6
        elif vix > 20:
            market_factor = 0.8
        
        # Sector concentration adjustment
        sector_factor = 1.0
        if sector_exposure > 0.30:  # Already 30%+ in this sector
            sector_factor = 0.7
        elif sector_exposure > 0.20:
            sector_factor = 0.85
        
        # Final position size
        position_fraction = kelly_fraction * volatility_factor * market_factor * sector_factor
        position_fraction = min(position_fraction, MAX_POSITION_SIZE)
        
        position_size = trading_capital * position_fraction
        
        return position_size, {
            'kelly': kelly_fraction,
            'vol_adj': volatility_factor,
            'mkt_adj': market_factor,
            'sec_adj': sector_factor,
            'final': position_fraction
        }

# ============================================================================
# TRAILING STOP MANAGER
# ============================================================================

class TrailingStopManager:
    """Manage trailing stops for open positions"""
    
    @staticmethod
    def calculate_stop(entry_price, current_price, highest_since_entry, atr, stage):
        """Calculate stop loss level"""
        
        initial_stop = entry_price - (atr * ATR_STOP_MULT)
        
        # Stage 1: Initial fixed stop
        if current_price < entry_price + (atr * 0.5):
            return initial_stop, "Initial"
        
        # Stage 2: Move to breakeven
        elif current_price < entry_price + (atr * 1.0):
            return entry_price, "Breakeven"
        
        # Stage 3: Trailing stop
        else:
            trailing_stop = highest_since_entry - (atr * ATR_STOP_MULT)
            return max(trailing_stop, entry_price), "Trailing"
    
    @staticmethod
    def should_exit(current_price, stop_level, target_level):
        """Check if position should be closed"""
        if current_price <= stop_level:
            return True, "Stop loss hit", "LOSS"
        
        if current_price >= target_level:
            return True, "Target reached", "WIN"
        
        return False, None, None

# ============================================================================
# ENHANCED BACKTEST ENGINE
# ============================================================================

def simulate_past_performance_v2(df, sector, atr_mult=ATR_STOP_MULT, use_trailing=True):
    """Enhanced backtest with trailing stops and transaction costs"""
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
            
            initial_stop = pivot - stop_dist
            target = pivot + (stop_dist * reward_mult)
            
            if high.iloc[i] >= pivot:
                entry_price = pivot
                highest = entry_price
                current_stop = initial_stop
                
                for j in range(1, 30):
                    if i + j >= len(df):
                        break
                    
                    current_price = close.iloc[i+j]
                    current_high = high.iloc[i+j]
                    current_low = low.iloc[i+j]
                    
                    # Update highest
                    if current_high > highest:
                        highest = current_high
                    
                    # Update stop if using trailing
                    if use_trailing:
                        current_stop, stage = TrailingStopManager.calculate_stop(
                            entry_price, current_price, highest, stop_dist / atr_mult, None
                        )
                    
                    # Check exit
                    if current_high >= target:
                        wins += 1
                        actual_gain = (target - entry_price) / stop_dist
                        total_r += actual_gain
                        break
                    
                    if current_low <= current_stop:
                        losses += 1
                        actual_loss = (entry_price - current_stop) / stop_dist
                        total_r -= actual_loss
                        break
        
        total_trades = wins + losses
        
        if total_trades < 10:
            return {
                'status': 'insufficient',
                'message': f'Sample:{total_trades}',
                'trades': total_trades,
                'winrate': 0,
                'expectancy': 0,
                'wins': 0,
                'losses': 0
            }
        
        winrate = (wins / total_trades) * 100
        expectancy = total_r / total_trades
        
        # Adjust for transaction costs
        avg_position = 50000  # Assume $500 average position
        fx_rate = 152.0
        net_expectancy, cost_in_r = TransactionCostModel.adjust_expectancy_for_cost(
            expectancy, 8.5, avg_position, fx_rate
        )
        
        return {
            'status': 'valid',
            'winrate': winrate,
            'expectancy': expectancy,
            'net_expectancy': net_expectancy,
            'cost_in_r': cost_in_r,
            'wins': wins,
            'losses': losses,
            'total': total_trades,
            'message': f"WR{winrate:.0f}% ({wins}/{total_trades}) GrossEV{expectancy:.2f}R NetEV{net_expectancy:.2f}R"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error: {str(e)}',
            'winrate': 0,
            'expectancy': 0,
            'net_expectancy': 0,
            'wins': 0,
            'losses': 0
        }

# ============================================================================
# ENHANCED STRATEGIC ANALYZER
# ============================================================================

class StrategicAnalyzerV2:
    """Enhanced analyzer with optimized scoring and filters"""
    
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_price_usd, vix, sector_exposures, trading_capital):
        """Comprehensive ticker analysis with liquidity filter"""
        
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
        
        # Price filter
        if current_price > max_price_usd:
            return None
        
        # === LIQUIDITY FILTER (NEW) ===
        avg_volume = volume.rolling(50).mean().iloc[-1]
        avg_dollar_volume = avg_volume * current_price
        
        if avg_dollar_volume < MIN_DAILY_VOLUME_USD:
            print(f"  SKIP {ticker}: Low liquidity (${avg_dollar_volume/1e6:.1f}M/day)")
            return None
        
        # Trend filter
        ma50 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = close.rolling(MA_LONG).mean().iloc[-1]
        
        if not (current_price > ma50 > ma200):
            return None
        
        # ATR calculation
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(14).mean().iloc[-1]
        
        if atr14 == 0 or pd.isna(atr14):
            return None
        
        atr_pct = (atr14 / current_price) * 100
        
        # VCP tightness
        recent_range = high.iloc[-5:].max() - low.iloc[-5:].min()
        tightness = float(recent_range / atr14)
        
        if tightness > MAX_TIGHTNESS:
            return None
        
        # === OPTIMIZED SCORING SYSTEM ===
        score = 0
        reasons = []
        
        # 1. VCP Tightness (0-35 points)
        if tightness < 0.8:
            score += 35
            reasons.append("VCP+++35")
        elif tightness < 1.0:
            score += 30
            reasons.append("VCP++30")
        elif tightness < 1.2:
            score += 20
            reasons.append("VCP+20")
        else:
            score += 10
            reasons.append("VCP+10")
        
        # 2. Volume Analysis (0-25 points)
        vol_avg = volume.rolling(50).mean().iloc[-1]
        
        if vol_avg > 0:
            vol_ratio = volume.iloc[-1] / vol_avg
            
            if 0.5 <= vol_ratio <= 0.7:
                score += 20
                reasons.append("VolDry++20")
            elif 0.7 < vol_ratio <= 0.85:
                score += 15
                reasons.append("VolDry+15")
            elif 0.85 < vol_ratio <= 1.0:
                score += 10
                reasons.append("VolStable+10")
            
            recent_vol_max = volume.iloc[-3:].max()
            if recent_vol_max > vol_avg * 3.0:
                score += 15
                reasons.append("Accum+++15")
            elif recent_vol_max > vol_avg * 2.0:
                score += 10
                reasons.append("Accum++10")
            elif recent_vol_max > vol_avg * 1.5:
                score += 5
                reasons.append("Accum+5")
        
        # 3. Momentum (0-20 points)
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        
        if ma5 <= ma20:
            return None  # Must have positive momentum
        
        mom_strength = ((ma5 / ma20) - 1) * 100
        
        if mom_strength > 3.0:
            score += 20
            reasons.append("Mom+++20")
        elif mom_strength > 2.0:
            score += 15
            reasons.append("Mom++15")
        elif mom_strength > 1.0:
            score += 10
            reasons.append("Mom+10")
        else:
            score += 5
            reasons.append("Mom+5")
        
        # 4. Trend Strength (0-20 points)
        trend_strength = ((ma50 - ma200) / ma200) * 100
        
        if trend_strength > 15:
            score += 20
            reasons.append("Trend++20")
        elif trend_strength > 10:
            score += 15
            reasons.append("Trend+15")
        elif trend_strength > 5:
            score += 10
            reasons.append("Trend+10")
        else:
            return None  # Weak trend
        
        # === CALCULATE ENTRY/EXIT ===
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        pivot = high.iloc[-5:].max() * 1.002
        stop_dist = atr14 * ATR_STOP_MULT
        stop_loss = pivot - stop_dist
        target = pivot + (stop_dist * reward_mult)
        
        # === RUN ENHANCED BACKTEST ===
        bt_result = simulate_past_performance_v2(df, sector, use_trailing=True)
        
        # === STRICT FILTERING ===
        if score < MIN_SCORE:
            return None
        
        if bt_result['status'] == 'valid':
            if bt_result['winrate'] < MIN_WINRATE:
                return None
            if bt_result['net_expectancy'] < MIN_EXPECTANCY:
                return None
        elif bt_result['status'] == 'insufficient':
            if bt_result['trades'] < 5:
                return None
        elif bt_result['status'] == 'error':
            return None
        
        # === DYNAMIC POSITION SIZING ===
        sector_exposure = sector_exposures.get(sector, 0)
        
        position_size, sizing_factors = PositionSizer.calculate_position_size(
            trading_capital=trading_capital,
            winrate=bt_result.get('winrate', 50) / 100,
            rr_ratio=reward_mult,
            atr_pct=atr_pct,
            vix=vix,
            sector_exposure=sector_exposure
        )
        
        return {
            'score': score,
            'reasons': ' '.join(reasons),
            'price': current_price,
            'pivot': pivot,
            'stop': stop_loss,
            'target': target,
            'sector': sector,
            'tightness': tightness,
            'atr_pct': atr_pct,
            'bt': bt_result,
            'reward_mult': reward_mult,
            'position_size_jpy': position_size,
            'sizing_factors': sizing_factors,
            'liquidity_usd': avg_dollar_volume
        }

# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Track and analyze system performance"""
    
    @staticmethod
    def load_performance_history():
        """Load historical performance data"""
        if PERFORMANCE_LOG_PATH.exists():
            with open(PERFORMANCE_LOG_PATH, 'r') as f:
                return json.load(f)
        return {
            'start_date': datetime.now().isoformat(),
            'initial_capital': INITIAL_CAPITAL,
            'quarters': []
        }
    
    @staticmethod
    def save_performance(data):
        """Save performance data"""
        with open(PERFORMANCE_LOG_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def get_current_status():
        """Get current performance status"""
        perf = PerformanceTracker.load_performance_history()
        
        if not perf['quarters']:
            return {
                'on_track': True,
                'ytd_return': 0,
                'target_return': 0,
                'message': "System just started"
            }
        
        start_date = datetime.fromisoformat(perf['start_date'])
        days_passed = (datetime.now() - start_date).days
        
        target_return = TARGET_ANNUAL_RETURN * (days_passed / 365)
        actual_return = sum(q.get('return', 0) for q in perf['quarters'])
        
        on_track = actual_return >= target_return * 0.9
        
        return {
            'on_track': on_track,
            'ytd_return': actual_return,
            'target_return': target_return,
            'difference': actual_return - target_return,
            'days_passed': days_passed
        }

# ============================================================================
# LINE NOTIFICATION
# ============================================================================

def send_line(msg):
    """Send LINE notification"""
    
    print("\n" + "="*70)
    print("LINE NOTIFICATION")
    print("="*70)
    
    if not ACCESS_TOKEN or not USER_ID:
        print("Credentials not set. Message:")
        print("-"*70)
        print(msg)
        print("="*70 + "\n")
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
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✓ Sent successfully")
            print("="*70 + "\n")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            print("="*70 + "\n")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print("="*70 + "\n")
        return False

# ============================================================================
# MAIN MISSION
# ============================================================================

def run_mission():
    """Main execution - SENTINEL v25.1 FINAL"""
    
    print("\n" + "="*70)
    print("SENTINEL v25.1 FINAL - PROFESSIONAL EDITION")
    print("="*70)
    print(f"Launch: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {len(TICKERS)} curated stocks")
    print(f"Target: {TARGET_ANNUAL_RETURN*100}% Annual Return")
    print("="*70 + "\n")
    
    # Environment check
    env_ok = check_environment()
    
    # Get market data
    print("Gathering market intelligence...")
    is_bull, market_status, spy_distance = check_market_trend()
    fx_rate = get_current_fx_rate()
    vix = get_vix()
    
    print(f"✓ Market: {market_status}")
    print(f"✓ VIX: {vix:.1f}")
    print(f"✓ FX: ¥{fx_rate:.2f}/USD")
    
    # Check performance
    perf_status = PerformanceTracker.get_current_status()
    print(f"✓ YTD: {perf_status['ytd_return']*100:.1f}% (Target: {perf_status['target_return']*100:.1f}%)")
    print(f"✓ Status: {'ON TRACK' if perf_status['on_track'] else 'BEHIND'}\n")
    
    # Market filter
    if not is_bull:
        msg = (
            f"SENTINEL v25.1\n"
            f"Market conditions unfavorable\n"
            f"\n"
            f"{market_status}\n"
            f"VIX: {vix:.1f}\n"
            f"System standby mode activated"
        )
        print(msg)
        send_line(msg)
        return
    
    # Calculate capital allocation
    trading_capital = INITIAL_CAPITAL * TRADING_RATIO
    max_price_usd = (trading_capital / fx_rate) * 0.9
    
    print(f"Capital Allocation:")
    print(f"  Trading: ¥{trading_capital:,.0f} (${trading_capital/fx_rate:,.0f})")
    print(f"  Max Price: ${max_price_usd:.2f}\n")
    
    # Download data
    print(f"Downloading {len(TICKERS)} tickers...")
    ticker_list = list(TICKERS.keys())
    
    try:
        all_data = yf.download(
            ticker_list,
            period="600d",
            progress=False,
            group_by='ticker',
            threads=True
        )
        print("✓ Download complete\n")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return
    
    # Screen tickers
    print("Screening candidates...\n")
    
    results = []
    sector_exposures = {}
    skipped_earnings = 0
    skipped_sector = 0
    skipped_liquidity = 0
    
    for ticker, sector in TICKERS.items():
        
        # Skip earnings
        if is_earnings_near(ticker):
            skipped_earnings += 1
            print(f"  SKIP {ticker}: Earnings within 5 days")
            continue
        
        # Skip weak sectors
        if not sector_is_strong(sector):
            skipped_sector += 1
            print(f"  SKIP {ticker}: Weak sector")
            continue
        
        try:
            if len(ticker_list) > 1:
                df_ticker = all_data[ticker]
            else:
                df_ticker = all_data
            
            result = StrategicAnalyzerV2.analyze_ticker(
                ticker=ticker,
                df=df_ticker,
                sector=sector,
                max_price_usd=max_price_usd,
                vix=vix,
                sector_exposures=sector_exposures,
                trading_capital=trading_capital
            )
            
            if result:
                results.append((ticker, result))
                
                # Update sector exposure
                sector_exposures[sector] = sector_exposures.get(sector, 0) + (
                    result['position_size_jpy'] / trading_capital
                )
                
                print(f"  ✓ {ticker}: {result['score']}pt "
                      f"WR{result['bt'].get('winrate',0):.0f}% "
                      f"EV{result['bt'].get('net_expectancy',0):.2f}R "
                      f"Size¥{result['position_size_jpy']:,.0f}")
                
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")
    
    # Sort and limit
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    print(f"\n{'='*70}")
    print(f"SCREENING COMPLETE")
    print(f"{'='*70}")
    print(f"Analyzed: {len(TICKERS)} tickers")
    print(f"Candidates: {len(results)}")
    print(f"Filtered: Earnings={skipped_earnings}, Sector={skipped_sector}")
    print(f"{'='*70}\n")
    
    # Generate report
    report = [
        "SENTINEL v25.1 Final",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Market: {market_status}",
        f"VIX: {vix:.1f} | FX: ¥{fx_rate:.2f}",
        f"YTD: {perf_status['ytd_return']*100:.1f}% (Target: {perf_status['target_return']*100:.1f}%)",
        "=" * 35
    ]
    
    if not results:
        report.append("No qualifying candidates")
        report.append("Filters: High quality + liquidity")
        report.append(f"Analyzed: {len(TICKERS)} stocks")
    else:
        for i, (ticker, r) in enumerate(results, 1):
            loss_pct = (1 - r['stop'] / r['pivot']) * 100
            gain_pct = (r['target'] / r['pivot'] - 1) * 100
            rr = gain_pct / loss_pct
            
            shares = int(r['position_size_jpy'] / fx_rate / r['pivot'])
            
            report.append(f"[{i}] {ticker} ({r['sector']}) {r['score']}pt")
            report.append(f"{r['reasons']}")
            report.append(f"BT: {r['bt']['message']}")
            report.append(f"")
            report.append(f"Price: ${r['price']:.2f}")
            report.append(f"Entry: ${r['pivot']:.2f}")
            report.append(f"Stop: ${r['stop']:.2f} (-{loss_pct:.1f}%)")
            report.append(f"Target: ${r['target']:.2f} (+{gain_pct:.1f}%)")
            report.append(f"RR: 1:{rr:.1f}")
            report.append(f"")
            report.append(f"Position: {shares} shares")
            report.append(f"Size: ¥{r['position_size_jpy']:,.0f}")
            report.append(f"Kelly: {r['sizing_factors']['kelly']:.1%}")
            report.append(f"Liquidity: ${r['liquidity_usd']/1e6:.1f}M/day")
            report.append("=" * 35)
    
    full_report = "\n".join(report)
    
    # Output
    print("="*70)
    print("FINAL REPORT")
    print("="*70)
    print(full_report)
    print("="*70 + "\n")
    
    # Send notification
    send_line(full_report)
    
    print("Mission complete\n")

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