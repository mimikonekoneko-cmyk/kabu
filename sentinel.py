#!/usr/bin/env python3
# backtest_v27_relaxed.py
# SENTINEL v27 バックテスト（フィルター緩和版）

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Backtest")

# ---------------------------
# CONFIG
# ---------------------------
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

TEST_TICKERS = [
    'NVDA', 'AMD', 'AVGO', 'TSM',
    'GOOGL', 'MSFT', 'AAPL', 'META',
    'JPM', 'GS', 'BAC', 'WFC',
    'ABBV', 'JNJ', 'LLY', 'PFE',
    'WMT', 'HD', 'COST',
    'RKLB'
]

INITIAL_CAPITAL = 350_000
ATR_STOP_MULT = 2.0
REWARD_MULT = 2.0
MAX_HOLD_DAYS = 60

# ---------------------------
# FUNCTIONS
# ---------------------------

def calculate_atr(df, period=14):
    """ATR計算"""
    try:
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        close = df['Close'].astype(float)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    except Exception:
        return pd.Series([np.nan] * len(df), index=df.index)


def detect_vcp_breakout_relaxed(df):
    """
    VCPブレイクアウト検出（緩和版）
    
    条件を緩和:
    1. MA50 > MA200（上昇トレンド）
    2. 価格が20日高値をブレイク
    3. 出来高が平均以上（1.0倍以上） ← 緩和
    """
    try:
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        volume = df['Volume'].astype(float)
        
        df['MA50'] = close.rolling(50).mean()
        df['MA200'] = close.rolling(200).mean()
        df['High20'] = high.rolling(20).max()
        df['AvgVol'] = volume.rolling(20).mean()
        
        signals = []
        
        for i in range(200, len(df)):
            # トレンドフィルター
            if pd.isna(df['MA50'].iloc[i]) or pd.isna(df['MA200'].iloc[i]):
                continue
            
            if df['MA50'].iloc[i] <= df['MA200'].iloc[i]:
                continue
            
            # ブレイクアウト検出（緩和）
            prev_high20 = df['High20'].iloc[i-1]
            current_close = close.iloc[i]
            current_volume = volume.iloc[i]
            avg_volume = df['AvgVol'].iloc[i]
            
            if pd.isna(prev_high20) or pd.isna(avg_volume):
                continue
            
            # 出来高条件を緩和（1.0倍以上）
            if (current_close > prev_high20 and
                current_volume > avg_volume * 1.0):
                
                signals.append({
                    'date': df.index[i],
                    'entry_price': current_close,
                    'atr': df['ATR'].iloc[i]
                })
        
        return signals
    
    except Exception as e:
        logger.error(f"Signal detection error: {e}")
        return []


def backtest_ticker(ticker, start_date, end_date):
    """1銘柄のバックテスト"""
    logger.info(f"Testing {ticker}...")
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < 250:
            logger.warning(f"{ticker}: Insufficient data")
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df['ATR'] = calculate_atr(df)
        
        # フィルター緩和版のシグナル検出
        signals = detect_vcp_breakout_relaxed(df)
        
        if not signals:
            logger.info(f"{ticker}: No signals")
            return None
        
        logger.info(f"{ticker}: {len(signals)} signals detected")
        
        trades = []
        
        for signal in signals:
            entry_date = signal['date']
            entry_price = signal['entry_price']
            atr = signal['atr']
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            stop_loss = entry_price - (atr * ATR_STOP_MULT)
            target = entry_price + (atr * ATR_STOP_MULT * REWARD_MULT)
            
            future_df = df[df.index > entry_date].head(MAX_HOLD_DAYS)
            
            if future_df.empty:
                continue
            
            exit_date = None
            exit_price = None
            result = None
            exit_reason = None
            
            for idx, row in future_df.iterrows():
                if row['Close'] <= stop_loss:
                    exit_date = idx
                    exit_price = stop_loss
                    result = 'LOSS'
                    exit_reason = 'STOP'
                    break
                
                if row['High'] >= target:
                    exit_date = idx
                    exit_price = target
                    result = 'WIN'
                    exit_reason = 'TARGET'
                    break
            
            if result is None:
                exit_date = future_df.index[-1]
                exit_price = future_df['Close'].iloc[-1]
                result = 'WIN' if exit_price > entry_price else 'LOSS'
                exit_reason = 'TIMEOUT'
            
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            hold_days = (exit_date - entry_date).days
            
            trades.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'result': result,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'hold_days': hold_days
            })
        
        return trades
    
    except Exception as e:
        logger.error(f"{ticker} error: {e}")
        return None


def run_backtest():
    """バックテスト実行"""
    logger.info("="*70)
    logger.info("SENTINEL v27 BACKTEST (RELAXED)")
    logger.info("="*70)
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Tickers: {len(TEST_TICKERS)}")
    logger.info("")
    
    all_trades = []
    
    for ticker in TEST_TICKERS:
        trades = backtest_ticker(ticker, START_DATE, END_DATE)
        if trades:
            all_trades.extend(trades)
        time.sleep(0.5)
    
    if not all_trades:
        logger.error("No trades generated")
        return
    
    df_trades = pd.DataFrame(all_trades)
    
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    losses = len(df_trades[df_trades['result'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100
    
    win_trades = df_trades[df_trades['result'] == 'WIN']['pnl_pct']
    loss_trades = df_trades[df_trades['result'] == 'LOSS']['pnl_pct']
    
    avg_win = win_trades.mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0
    avg_pnl = df_trades['pnl_pct'].mean()
    
    expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
    
    df_trades['cumulative_pnl'] = df_trades['pnl_pct'].cumsum()
    running_max = df_trades['cumulative_pnl'].expanding().max()
    drawdown = df_trades['cumulative_pnl'] - running_max
    max_drawdown = drawdown.min()
    
    exit_reasons = df_trades['exit_reason'].value_counts()
    
    # レポート
    print("\n" + "="*70)
    print("BACKTEST RESULTS - SENTINEL v27 (RELAXED)")
    print("="*70)
    print(f"Period:          {START_DATE} to {END_DATE}")
    print(f"Total Trades:    {total_trades}")
    print(f"")
    print(f"【PERFORMANCE】")
    print(f"Wins:            {wins} ({win_rate:.1f}%)")
    print(f"Losses:          {losses} ({100-win_rate:.1f}%)")
    print(f"")
    print(f"Average Win:     {avg_win:+.2f}%")
    print(f"Average Loss:    {avg_loss:+.2f}%")
    print(f"Average P&L:     {avg_pnl:+.2f}%")
    print(f"Expectancy:      {expectancy:+.2f}%")
    print(f"")
    print(f"【RISK】")
    print(f"Max Drawdown:    {max_drawdown:.2f}%")
    print(f"Avg Hold Days:   {df_trades['hold_days'].mean():.1f} days")
    print(f"")
    print(f"【EXIT REASONS】")
    for reason, count in exit_reasons.items():
        pct = (count / total_trades) * 100
        print(f"{reason:12} {count:3} ({pct:5.1f}%)")
    print("")
    
    print("="*70)
    print("TOP 10 TRADES")
    print("="*70)
    top10 = df_trades.nlargest(10, 'pnl_pct')
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"[{i:2}] {row['ticker']:6} {row['entry_date'].strftime('%Y-%m-%d')} "
              f"{row['pnl_pct']:+6.2f}% ({row['hold_days']:2}日) {row['exit_reason']}")
    print("")
    
    print("="*70)
    print("WORST 10 TRADES")
    print("="*70)
    worst10 = df_trades.nsmallest(10, 'pnl_pct')
    for i, (_, row) in enumerate(worst10.iterrows(), 1):
        print(f"[{i:2}] {row['ticker']:6} {row['entry_date'].strftime('%Y-%m-%d')} "
              f"{row['pnl_pct']:+6.2f}% ({row['hold_days']:2}日) {row['exit_reason']}")
    print("")
    
    print("="*70)
    print("PERFORMANCE BY TICKER")
    print("="*70)
    ticker_perf = df_trades.groupby('ticker').agg({
        'pnl_pct': ['count', 'mean', 'sum'],
        'result': lambda x: (x == 'WIN').sum() / len(x) * 100
    }).round(2)
    ticker_perf.columns = ['Trades', 'Avg%', 'Total%', 'WR%']
    ticker_perf = ticker_perf.sort_values('Total%', ascending=False)
    print(ticker_perf.to_string())
    print("")
    
    df_trades.to_csv('backtest_trades_relaxed.csv', index=False)
    logger.info("Trades saved to backtest_trades_relaxed.csv")
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    
    if expectancy > 0:
        print(f"✅ POSITIVE EXPECTANCY: {expectancy:+.2f}%")
        if win_rate >= 50:
            print(f"✅ Win Rate {win_rate:.1f}% は良好")
        else:
            print(f"⚠️  Win Rate {win_rate:.1f}% は低いが、RR比でカバー")
    else:
        print(f"❌ NEGATIVE EXPECTANCY: {expectancy:+.2f}%")
        print(f"   → システム改善が必要")
    
    total_return = df_trades['pnl_pct'].sum()
    print(f"\n総リターン: {total_return:+.2f}% ({total_trades}トレード)")
    print(f"月平均: {total_return/12:+.2f}%")
    print("="*70)
    
    return df_trades


if __name__ == "__main__":
    df_result = run_backtest()
