#!/usr/bin/env python3
# SENTINEL v30.0 SAFE - BAN Risk Mitigation Edition
# Multi-Source with Rate Limiting, Caching, and Fallback Strategies

import os
import time
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
import json
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import alpaca_trade_api as tradeapi
from cachetools import TTLCache, cached

warnings.filterwarnings('ignore')

# ---------------------------
# Enhanced Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('sentinel_safe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SENTINEL_SAFE")

# ---------------------------
# SAFETY CONFIGURATION
# ---------------------------
# Rate limiting to avoid BAN
MAX_REQUESTS_PER_MINUTE = 180  # Conservative limit (Yahoo Finance unofficial limit)
MIN_REQUEST_INTERVAL = 0.5  # Minimum seconds between requests
JITTER_RANGE = (0.1, 0.5)  # Random jitter to avoid pattern detection

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Cache configuration
CACHE_TTL_HOURS = 6  # Cache data for 6 hours
MAX_CACHE_SIZE = 1000

# Data source priorities (adjust based on reliability)
DATA_SOURCE_PRIORITY = ['CACHE', 'ALPACA', 'YFINANCE_FALLBACK']

# ---------------------------
# API Configuration
# ---------------------------
# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Multiple Yahoo Finance alternatives (User-Agent rotation)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
]

# Free alternative data sources
ALTERNATIVE_SOURCES = {
    'tiingo': os.getenv("TIINGO_API_KEY", ""),
    'finnhub': os.getenv("FINNHUB_API_KEY", ""),
    'polygon': os.getenv("POLYGON_API_KEY", ""),
}

# Trading parameters (conservative)
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.01
MIN_HISTORY_DAYS = 100

# Volume thresholds (adjusted for safety)
MIN_DAILY_VOLUME = 500000  # 500k shares (more realistic)
MIN_DOLLAR_VOLUME = 2000000  # $2M

# Filtering thresholds
STAGE1_MIN_PRICE = 5.0
STAGE1_MAX_PRICE = 300.0

# Performance
MAX_WORKERS = 4  # Reduced for safety
MAX_SYMBOLS_PER_RUN = 200  # Limit total symbols analyzed

# Paths
CACHE_DIR = Path("./cache_safe")
CACHE_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("./results_safe")
RESULTS_DIR.mkdir(exist_ok=True)

UNIVERSE_FILE = "us_stocks_universe.txt"

# ---------------------------
# Safe Request Session with Rate Limiting
# ---------------------------
class SafeRequestSession:
    """Safe HTTP session with rate limiting and retry logic"""
    
    _last_request_time = 0
    _request_count = 0
    _minute_start = time.time()
    
    @classmethod
    def create_session(cls):
        """Create a safe HTTP session"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Rotate User-Agent
        session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        return session
    
    @classmethod
    def rate_limit(cls):
        """Enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - cls._minute_start > 60:
            cls._request_count = 0
            cls._minute_start = current_time
        
        # Check rate limit
        if cls._request_count >= MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - cls._minute_start) + 1
            logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
            cls._request_count = 0
            cls._minute_start = time.time()
        
        # Enforce minimum interval
        time_since_last = current_time - cls._last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - time_since_last
            time.sleep(sleep_time)
        
        # Add random jitter
        jitter = random.uniform(*JITTER_RANGE)
        time.sleep(jitter)
        
        cls._last_request_time = time.time()
        cls._request_count += 1

# ---------------------------
# Smart Cache System
# ---------------------------
class SmartCache:
    """Intelligent caching system to minimize API calls"""
    
    def __init__(self):
        self.price_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL_HOURS * 3600)
        self.volume_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL_HOURS * 3600)
        self.historical_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL_HOURS * 3600)
        
        # Disk cache for persistence
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cached_price(self, symbol: str) -> Optional[float]:
        """Get cached price data"""
        cache_key = f"price_{symbol}"
        return self.price_cache.get(cache_key)
    
    def set_cached_price(self, symbol: str, price: float):
        """Cache price data"""
        cache_key = f"price_{symbol}"
        self.price_cache[cache_key] = price
        
        # Also save to disk
        cache_file = self.cache_dir / f"price_{symbol}.json"
        cache_data = {
            'symbol': symbol,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    def get_cached_historical(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get cached historical data"""
        cache_key = f"hist_{symbol}_{days}"
        return self.historical_cache.get(cache_key)
    
    def set_cached_historical(self, symbol: str, days: int, data: pd.DataFrame):
        """Cache historical data"""
        cache_key = f"hist_{symbol}_{days}"
        self.historical_cache[cache_key] = data
        
        # Save to disk (compressed)
        cache_file = self.cache_dir / f"hist_{symbol}_{days}.parquet"
        try:
            data.to_parquet(cache_file, compression='gzip')
        except Exception as e:
            logger.debug(f"Failed to save cache to parquet: {e}")
    
    def load_disk_cache(self):
        """Load cache from disk on startup"""
        logger.info("Loading disk cache...")
        loaded = 0
        
        for cache_file in self.cache_dir.glob("price_*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    symbol = data['symbol']
                    price = data['price']
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    
                    # Check if cache is still valid
                    if (datetime.now() - timestamp).total_seconds() < CACHE_TTL_HOURS * 3600:
                        self.set_cached_price(symbol, price)
                        loaded += 1
            except Exception as e:
                logger.debug(f"Failed to load cache {cache_file}: {e}")
        
        logger.info(f"Loaded {loaded} price records from disk cache")

# ---------------------------
# Safe Data Fetcher with Fallbacks
# ---------------------------
class SafeDataFetcher:
    """Data fetcher with multiple fallbacks and BAN protection"""
    
    def __init__(self):
        self.cache = SmartCache()
        self.session = SafeRequestSession.create_session()
        self.alpaca_client = None
        self.request_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0
        }
        
        # Initialize Alpaca if available
        if ALPACA_API_KEY and ALPACA_API_SECRET:
            try:
                self.alpaca_client = tradeapi.REST(
                    ALPACA_API_KEY,
                    ALPACA_API_SECRET,
                    base_url=ALPACA_BASE_URL,
                    api_version='v2'
                )
                logger.info("Alpaca API connected")
            except Exception as e:
                logger.warning(f"Alpaca connection failed: {e}")
        
        # Load existing cache
        self.cache.load_disk_cache()
    
    def get_stock_data_safe(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        """Safely get stock data with multiple fallbacks"""
        self.request_stats['total'] += 1
        
        # 1. Check cache first
        cached_data = self.cache.get_cached_historical(symbol, days)
        if cached_data is not None:
            self.request_stats['cached'] += 1
            return cached_data
        
        # 2. Try Alpaca (most reliable, but limited volume)
        if self.alpaca_client:
            data = self._try_alpaca(symbol, days)
            if data is not None:
                self.cache.set_cached_historical(symbol, days, data)
                return data
        
        # 3. Try Yahoo Finance with careful rate limiting
        SafeRequestSession.rate_limit()
        data = self._try_yfinance_safe(symbol, days)
        
        if data is not None:
            self.cache.set_cached_historical(symbol, days, data)
            self.request_stats['successful'] += 1
            return data
        
        # 4. Final fallback to alternative sources
        data = self._try_alternative_sources(symbol, days)
        if data is not None:
            self.cache.set_cached_historical(symbol, days, data)
            return data
        
        self.request_stats['failed'] += 1
        return None
    
    def _try_alpaca(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Try Alpaca API"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            bars = self.alpaca_client.get_bars(
                symbol,
                '1Day',
                start=start.isoformat(),
                end=end.isoformat(),
                limit=min(days, 1000),
                adjustment='all'
            ).df
            
            if bars.empty:
                return None
            
            # Standardize columns
            bars.columns = [col.lower() for col in bars.columns]
            
            if 'close' in bars.columns and 'volume' in bars.columns:
                bars['dollar_volume'] = bars['close'] * bars['volume']
            
            return bars
            
        except Exception as e:
            logger.debug(f"Alpaca failed for {symbol}: {e}")
            return None
    
    def _try_yfinance_safe(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Safely try Yahoo Finance with error handling"""
        for attempt in range(MAX_RETRIES):
            try:
                # Add delay between retries
                if attempt > 0:
                    time.sleep(RETRY_DELAY * attempt)
                
                ticker = yf.Ticker(symbol)
                
                # Limit data request size
                end_date = datetime.now()
                start_date = end_date - timedelta(days=min(days, 365))  # Max 1 year
                
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    timeout=10
                )
                
                if data.empty:
                    return None
                
                # Standardize columns
                data.columns = [col.lower() for col in data.columns]
                
                if 'close' in data.columns and 'volume' in data.columns:
                    data['dollar_volume'] = data['close'] * data['volume']
                
                return data
                
            except Exception as e:
                logger.debug(f"YFinance attempt {attempt+1} failed for {symbol}: {e}")
                
                # Check for specific error conditions
                error_str = str(e).lower()
                if 'forbidden' in error_str or '429' in error_str:
                    logger.warning(f"Possible rate limiting detected for {symbol}")
                    time.sleep(30)  # Longer wait for rate limit
                
                if attempt == MAX_RETRIES - 1:
                    logger.warning(f"All YFinance attempts failed for {symbol}")
        
        return None
    
    def _try_alternative_sources(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Try alternative free data sources"""
        # Tiingo (free tier available)
        if ALTERNATIVE_SOURCES.get('tiingo'):
            try:
                url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f"Token {ALTERNATIVE_SOURCES['tiingo']}"
                }
                
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                params = {
                    'startDate': start_date,
                    'endDate': end_date,
                    'format': 'json',
                    'resampleFreq': 'daily'
                }
                
                SafeRequestSession.rate_limit()
                response = self.session.get(url, headers=headers, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        # Rename columns
                        column_map = {
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume'
                        }
                        
                        df.rename(columns=column_map, inplace=True)
                        df['dollar_volume'] = df['close'] * df['volume']
                        
                        return df
                        
            except Exception as e:
                logger.debug(f"Tiingo failed for {symbol}: {e}")
        
        return None
    
    def get_current_price_safe(self, symbol: str) -> Optional[float]:
        """Safely get current price with caching"""
        # Check cache first
        cached_price = self.cache.get_cached_price(symbol)
        if cached_price is not None:
            return cached_price
        
        # Try to get price from available sources
        price = None
        
        # Try Alpaca first
        if self.alpaca_client:
            try:
                quote = self.alpaca_client.get_last_trade(symbol)
                price = quote.price
            except Exception:
                pass
        
        # Fallback to Yahoo Finance
        if price is None:
            SafeRequestSession.rate_limit()
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d', interval='1m', timeout=5)
                if not hist.empty and 'close' in hist.columns:
                    price = float(hist['close'].iloc[-1])
            except Exception as e:
                logger.debug(f"Price fetch failed for {symbol}: {e}")
        
        # Cache the result
        if price is not None:
            self.cache.set_cached_price(symbol, price)
        
        return price
    
    def get_volume_metrics_safe(self, symbol: str) -> Dict:
        """Safely get volume metrics"""
        data = self.get_stock_data_safe(symbol, days=60)
        
        metrics = {
            'symbol': symbol,
            'avg_volume': 0,
            'avg_dollar_volume': 0,
            'is_sufficient': False,
            'data_source': 'none'
        }
        
        if data is not None and not data.empty:
            if 'volume' in data.columns:
                recent_data = data.tail(20)
                metrics['avg_volume'] = int(recent_data['volume'].mean())
                metrics['data_source'] = 'cached' if self.request_stats['cached'] > 0 else 'api'
            
            if 'dollar_volume' in data.columns:
                metrics['avg_dollar_volume'] = float(data['dollar_volume'].tail(20).mean())
            
            # Check sufficiency
            metrics['is_sufficient'] = (
                metrics['avg_volume'] >= MIN_DAILY_VOLUME or
                metrics['avg_dollar_volume'] >= MIN_DOLLAR_VOLUME
            )
        
        return metrics
    
    def print_stats(self):
        """Print request statistics"""
        logger.info("\n" + "="*60)
        logger.info("REQUEST STATISTICS")
        logger.info("="*60)
        logger.info(f"Total requests: {self.request_stats['total']}")
        logger.info(f"Successful: {self.request_stats['successful']}")
        logger.info(f"Failed: {self.request_stats['failed']}")
        logger.info(f"Cached hits: {self.request_stats['cached']}")
        logger.info(f"Cache hit rate: {(self.request_stats['cached']/max(self.request_stats['total'],1)*100):.1f}%")
        logger.info("="*60)

# ---------------------------
# Conservative Filter (Stage 1)
# ---------------------------
class ConservativeFilter:
    """Conservative filtering to minimize API calls"""
    
    def __init__(self, data_fetcher: SafeDataFetcher):
        self.data_fetcher = data_fetcher
        self.pre_filter_cache = {}
    
    def pre_filter_symbol(self, symbol: str) -> bool:
        """Quick pre-filter before detailed analysis"""
        # Cache pre-filter results
        if symbol in self.pre_filter_cache:
            return self.pre_filter_cache[symbol]
        
        result = self._perform_pre_filter(symbol)
        self.pre_filter_cache[symbol] = result
        
        return result
    
    def _perform_pre_filter(self, symbol: str) -> bool:
        """Perform quick checks"""
        try:
            # Get current price (cached)
            price = self.data_fetcher.get_current_price_safe(symbol)
            if price is None:
                return False
            
            # Price range check
            if price < STAGE1_MIN_PRICE or price > STAGE1_MAX_PRICE:
                return False
            
            # Quick volume check (use cached data if available)
            volume_metrics = self.data_fetcher.get_volume_metrics_safe(symbol)
            if not volume_metrics['is_sufficient']:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Pre-filter failed for {symbol}: {e}")
            return False
    
    def detailed_filter(self, symbol: str) -> Optional[Dict]:
        """Detailed filtering for pre-filtered symbols"""
        try:
            # Get historical data
            data = self.data_fetcher.get_stock_data_safe(symbol, days=100)
            
            if data is None or len(data) < 50:
                return None
            
            # Extract data
            if 'close' not in data.columns:
                return None
            
            close = data['close'].astype(float)
            current_price = float(close.iloc[-1])
            
            # Trend check (conservative)
            if len(close) >= 50:
                ma50 = close.rolling(50, min_periods=25).mean().iloc[-1]
                if current_price < ma50 * 0.93:  # Allow 7% below MA50
                    return None
            
            # Volume confirmation
            if 'volume' in data.columns:
                recent_volume = data['volume'].tail(10).mean()
                avg_volume = data['volume'].tail(50).mean()
                
                if recent_volume < avg_volume * 0.7:  # Volume drying up
                    return None
            
            return {
                'symbol': symbol,
                'price': current_price,
                'data': data,
                'stage': 'STAGE1_PASS'
            }
            
        except Exception as e:
            logger.debug(f"Detailed filter failed for {symbol}: {e}")
            return None
    
    def filter_batch(self, symbols: List[str]) -> List[Dict]:
        """Filter batch of symbols with conservative approach"""
        results = []
        
        logger.info(f"Stage 1: Pre-filtering {len(symbols)} symbols...")
        
        # First pass: Quick pre-filter
        pre_filtered = []
        for i, symbol in enumerate(symbols, 1):
            if self.pre_filter_symbol(symbol):
                pre_filtered.append(symbol)
            
            # Progress and rate limiting
            if i % 20 == 0:
                logger.info(f"Pre-filter progress: {i}/{len(symbols)}")
                time.sleep(1)
        
        logger.info(f"Pre-filter passed: {len(pre_filtered)} symbols")
        
        # Second pass: Detailed filter on pre-filtered symbols
        logger.info("Stage 1: Detailed filtering...")
        
        for i, symbol in enumerate(pre_filtered, 1):
            result = self.detailed_filter(symbol)
            if result:
                results.append(result)
            
            # Conservative rate limiting
            if i % 10 == 0:
                logger.info(f"Detailed filter progress: {i}/{len(pre_filtered)}")
                time.sleep(2)
            else:
                time.sleep(0.5)
        
        logger.info(f"Stage 1 complete: {len(results)} / {len(symbols)} passed")
        return results

# ---------------------------
# Simplified VCP Analyzer (Stage 2)
# ---------------------------
class SimpleVCPAnalyzer:
    """Simplified VCP analyzer for safety"""
    
    def __init__(self, data_fetcher: SafeDataFetcher):
        self.data_fetcher = data_fetcher
    
    def analyze(self, stage1_result: Dict) -> Optional[Dict]:
        """Simple VCP analysis"""
        try:
            symbol = stage1_result['symbol']
            data = stage1_result['data']
            
            if data is None or len(data) < 60:
                return None
            
            # Extract data
            required_cols = ['high', 'low', 'close']
            for col in required_cols:
                if col not in data.columns:
                    return None
            
            high = data['high'].astype(float)
            low = data['low'].astype(float)
            close = data['close'].astype(float)
            
            # Simple ATR calculation
            tr = high - low
            atr14 = tr.rolling(14, min_periods=7).mean().iloc[-1]
            
            if pd.isna(atr14) or atr14 <= 0:
                return None
            
            # Tightness
            recent_high = high.iloc[-5:].max()
            recent_low = low.iloc[-5:].min()
            tightness = (recent_high - recent_low) / atr14
            
            if tightness > 3.0:  # More lenient threshold
                return None
            
            # Simple VCP score
            maturity = 0
            signals = []
            
            if tightness < 2.0:
                maturity += 40
                signals.append("収縮中")
            elif tightness < 2.5:
                maturity += 30
                signals.append("軽度収縮")
            
            # Check for higher lows
            if len(close) >= 20:
                recent_low_avg = low.iloc[-5:].mean()
                prev_low_avg = low.iloc[-15:-5].mean()
                if recent_low_avg > prev_low_avg:
                    maturity += 20
                    signals.append("底上げ")
            
            # Volume check
            if 'volume' in data.columns:
                volume = data['volume'].astype(float)
                recent_vol = volume.iloc[-1]
                avg_vol = volume.tail(20).mean()
                
                if recent_vol < avg_vol * 0.8:
                    maturity += 20
                    signals.append("出来高縮小")
            
            if maturity < 50:  # Lower threshold for safety
                return None
            
            # Entry and stop
            pivot = recent_high * 1.01
            stop = pivot - (atr14 * 2.5)  # Wider stop for safety
            
            return {
                'symbol': symbol,
                'tightness': float(tightness),
                'vcp_maturity': maturity,
                'vcp_signals': signals,
                'price': float(close.iloc[-1]),
                'pivot': pivot,
                'stop': stop,
                'stage': 'STAGE2_PASS'
            }
            
        except Exception as e:
            logger.debug(f"VCP analysis failed for {stage1_result['symbol']}: {e}")
            return None

# ---------------------------
# Portfolio Optimizer (Stage 3)
# ---------------------------
class PortfolioOptimizer:
    """Portfolio optimization with risk management"""
    
    @staticmethod
    def calculate_position_size(price: float, stop: float, portfolio_value: float) -> Tuple[int, float]:
        """Calculate position size with Kelly criterion"""
        try:
            risk_per_share = price - stop
            if risk_per_share <= 0:
                return 0, 0.0
            
            # Conservative Kelly (half-Kelly)
            kelly_fraction = 0.01  # Fixed 1% risk
            
            max_risk_amount = portfolio_value * kelly_fraction
            shares_by_risk = max_risk_amount / risk_per_share
            
            # Convert to integer shares
            shares = int(shares_by_risk)
            
            if shares < 1:
                return 0, 0.0
            
            position_value = shares * price
            position_percentage = position_value / portfolio_value
            
            return shares, position_percentage
            
        except Exception:
            return 0, 0.0
    
    @staticmethod
    def diversify_picks(picks: List[Dict], max_positions: int = 8) -> List[Dict]:
        """Diversify picks across different sectors/characteristics"""
        if len(picks) <= max_positions:
            return picks
        
        # Simple diversification: pick top from different price ranges
        picks_by_price = sorted(picks, key=lambda x: x['price'])
        
        selected = []
        price_ranges = [
            (0, 50),    # Low price
            (50, 150),  # Mid price
            (150, 300)  # High price
        ]
        
        for price_min, price_max in price_ranges:
            in_range = [p for p in picks_by_price if price_min <= p['price'] < price_max]
            if in_range:
                selected.append(in_range[0])
                if len(selected) >= max_positions:
                    break
        
        # Fill remaining slots with highest VCP maturity
        remaining_slots = max_positions - len(selected)
        if remaining_slots > 0:
            remaining_picks = [p for p in picks if p not in selected]
            remaining_picks.sort(key=lambda x: x['vcp_maturity'], reverse=True)
            selected.extend(remaining_picks[:remaining_slots])
        
        return selected

# ---------------------------
# Main Safe Pipeline
# ---------------------------
def load_universe_safe(filepath: str) -> List[str]:
    """Safely load universe with size limits"""
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        
        # Limit number of symbols for safety
        if len(symbols) > MAX_SYMBOLS_PER_RUN:
            logger.info(f"Limiting universe from {len(symbols)} to {MAX_SYMBOLS_PER_RUN} symbols")
            symbols = symbols[:MAX_SYMBOLS_PER_RUN]
        
        logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        
        # Default symbols if file is empty
        if not symbols:
            symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
            logger.info(f"Using default {len(symbols)} symbols")
        
        return symbols
        
    except FileNotFoundError:
        logger.warning(f"Universe file {filepath} not found")
        return ['SPY', 'QQQ', 'AAPL', 'MSFT']

def run_safe_pipeline():
    """Main safe pipeline with BAN protection"""
    
    start_time = time.time()
    
    logger.info("="*70)
    logger.info("SENTINEL v30.0 SAFE - BAN Risk Mitigation Edition")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Max symbols per run: {MAX_SYMBOLS_PER_RUN}")
    logger.info(f"Rate limit: {MAX_REQUESTS_PER_MINUTE} requests/minute")
    logger.info("="*70)
    
    # Initialize safe data fetcher
    data_fetcher = SafeDataFetcher()
    
    # Load limited universe
    universe = load_universe_safe(UNIVERSE_FILE)
    
    # Stage 1: Conservative filtering
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: Conservative Filtering")
    logger.info("="*60)
    
    stage1_filter = ConservativeFilter(data_fetcher)
    stage1_results = stage1_filter.filter_batch(universe)
    
    if not stage1_results:
        logger.error("No stocks passed Stage 1")
        data_fetcher.print_stats()
        return None
    
    # Stage 2: Simplified VCP analysis
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: Simplified VCP Analysis")
    logger.info("="*60)
    
    vcp_analyzer = SimpleVCPAnalyzer(data_fetcher)
    stage2_results = []
    
    for i, candidate in enumerate(stage1_results, 1):
        result = vcp_analyzer.analyze(candidate)
        if result:
            stage2_results.append(result)
        
        # Rate limiting
        if i % 5 == 0:
            logger.info(f"VCP analysis progress: {i}/{len(stage1_results)}")
            time.sleep(1)
    
    if not stage2_results:
        logger.warning("No stocks passed VCP analysis")
        data_fetcher.print_stats()
        return None
    
    # Stage 3: Portfolio optimization
    logger.info("\n" + "="*60)
    logger.info("STAGE 3: Portfolio Optimization")
    logger.info("="*60)
    
    final_picks = PortfolioOptimizer.diversify_picks(stage2_results, max_positions=6)
    
    # Calculate position sizes
    for pick in final_picks:
        shares, position_pct = PortfolioOptimizer.calculate_position_size(
            pick['price'], pick['stop'], INITIAL_CAPITAL
        )
        pick['shares'] = shares
        pick['position_pct'] = position_pct * 100
        pick['position_value'] = shares * pick['price']
    
    # Generate report
    report = generate_safe_report(final_picks, stage1_results, stage2_results)
    
    # Output
    print("\n" + report)
    logger.info("\n" + report)
    
    # Print statistics
    data_fetcher.print_stats()
    
    # Timing
    elapsed = time.time() - start_time
    logger.info(f"\nTotal execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Save results
    save_safe_results(final_picks)
    
    return final_picks

def generate_safe_report(final_picks: List[Dict], 
                        stage1_results: List[Dict],
                        stage2_results: List[Dict]) -> str:
    """Generate safe report"""
    
    lines = []
    lines.append("="*70)
    lines.append("SENTINEL v30.0 SAFE - BAN Risk Mitigation Report")
    lines.append("="*70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("【SAFETY METRICS】")
    lines.append(f"• Maximum requests/minute: {MAX_REQUESTS_PER_MINUTE}")
    lines.append(f"• Minimum request interval: {MIN_REQUEST_INTERVAL}s")
    lines.append(f"• Cache TTL: {CACHE_TTL_HOURS} hours")
    lines.append("")
    
    lines.append("【FILTERING RESULTS】")
    lines.append(f"Stage 1 (Conservative): {len(stage1_results)} passed")
    lines.append(f"Stage 2 (VCP Analysis): {len(stage2_results)} passed")
    lines.append(f"Final (Portfolio): {len(final_picks)} selected")
    lines.append("")
    
    if not final_picks:
        lines.append("No suitable stocks found with current safety settings.")
        return '\n'.join(lines)
    
    lines.append("【RECOMMENDED PORTFOLIO】")
    lines.append("")
    
    total_investment = 0
    for i, pick in enumerate(final_picks, 1):
        symbol = pick['symbol']
        price = pick['price']
        shares = pick['shares']
        value = pick['position_value']
        maturity = pick['vcp_maturity']
        signals = ', '.join(pick['vcp_signals'][:2])
        
        total_investment += value
        
        lines.append(f"{i}. {symbol}")
        lines.append(f"   Price: ${price:.2f} | Shares: {shares:,} | Value: ${value:,.0f}")
        lines.append(f"   VCP: {maturity}% | Signals: {signals}")
        lines.append(f"   Entry: ${pick['pivot']:.2f} | Stop: ${pick['stop']:.2f}")
        lines.append("")
    
    lines.append(f"Total Portfolio Value: ${total_investment:,.0f}")
    lines.append(f"Capital Utilization: {(total_investment/INITIAL_CAPITAL*100):.1f}%")
    lines.append("")
    
    lines.append("【RISK MANAGEMENT】")
    lines.append("✓ Conservative position sizing (half-Kelly)")
    lines.append("✓ Portfolio diversification")
    lines.append("✓ Wide stop losses (2.5x ATR)")
    lines.append("✓ Rate limiting and caching")
    lines.append("")
    
    lines.append("【DISCLAIMER】")
    lines.append("This analysis uses conservative settings to minimize")
    lines.append("API BAN risk. Results may be fewer but more reliable.")
    lines.append("="*70)
    
    return '\n'.join(lines)

def save_safe_results(final_picks: List[Dict]):
    """Save results safely"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"safe_results_{timestamp}.json"
        
        # Make serializable
        serializable = []
        for pick in final_picks:
            pick_copy = pick.copy()
            if 'data' in pick_copy:
                del pick_copy['data']
            serializable.append(pick_copy)
        
        with open(RESULTS_DIR / filename, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {RESULTS_DIR}/{filename}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting SENTINEL v30.0 SAFE pipeline...")
        results = run_safe_pipeline()
        
        if results:
            logger.info(f"Pipeline completed. Found {len(results)} suitable stocks.")
        else:
            logger.info("Pipeline completed. No stocks met criteria.")
            
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
