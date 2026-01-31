#!/usr/bin/env python3
# backtest_v27.py
# SENTINEL v27 完全バックテスト
# 期間: 2024年1月～12月（1年間）

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

# テスト銘柄（主要20銘柄）
TEST_TICKERS = [
    'NVDA', 'AMD', 'AVGO', 'TSM',        # Semi
    'GOOGL', 'MSFT', 'AAPL', 'META',     # Tech
    'JPM', 'GS', 'BAC', 'WFC',           # Finance
    'ABBV', 'JNJ', 'LLY', 'PFE',         # Health
    'WMT', 'HD', 'COST',                 # Retail
    'RKLB'                                # Space
]

# トレード設定
INITIAL_CAPITAL = 350_000  # JPY
TRADING_RATIO = 0.75
ATR_STOP_MULT = 2.0
REWARD_MULT = 2.0
MAX_HOLD_DAYS = 60  # 最大保有期間

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
    except Exception as e:
        logger.debug(f"ATR calculation error: {e}")
        return pd.Series([np.nan] * len(df), index=df.index)


def detect_vcp_breakout(df, lookback=20):
    """
    VCPブレイクアウト検出（簡易版）
    
    SENTINEL v27のロジックを簡略化:
    1. MA50 > MA200（上昇トレンド）
    2. 価格が20日高値をブレイク
    3. 出来高が平均の1.5倍以上（機関買い）
    4. Tightness < 2.0（収縮）
    """
    try:
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()
        df['MA200'] = close.rolling(200).mean()
        df['High20'] = high.rolling(20).max()
        df['Low20'] = low.rolling(20).min()
        df['AvgVol'] = volume.rolling(20).mean()
        
        # Tightness計算（5日レンジ / ATR）
        df['Range5'] = high.rolling(5).max() - low.rolling(5).min()
        df['Tightness'] = df['Range5'] / df['ATR']
        
        signals = []
        
        for i in range(200, len(df)):
            # トレンドフィルター
            if pd.isna(df['MA50'].iloc[i]) or pd.isna(df['MA200'].iloc[i]):
                continue
            
            if df['MA50'].iloc[i] <= df['MA200'].iloc[i]:
                continue
            
            # Tightnessフィルター
            if pd.isna(df['Tightness'].iloc[i]) or df['Tightness'].iloc[i] > 2.0:
                continue
            
            # ブレイクアウト検出
            prev_high20 = df['High20'].iloc[i-1]
            current_close = close.iloc[i]
            current_volume = volume.iloc[i]
            avg_volume = df['AvgVol'].iloc[i]
            
            if pd.isna(prev_high20) or pd.isna(avg_volume):
                continue
            
            if (current_close > prev_high20 and
                current_volume > avg_volume * 1.5):
                
                signals.append({
                    'date': df.index[i],
                    'entry_price': current_close,
                    'atr': df['ATR'].iloc[i],
                    'tightness': df['Tightness'].iloc[i]
                })
        
        return signals
    
    except Exception as e:
        logger.error(f"Signal detection error: {e}")
        return []


def backtest_ticker(ticker, start_date, end_date):
    """1銘柄のバックテスト"""
    logger.info(f"Testing {ticker}...")
    
    try:
        # データ取得
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < 250:
            logger.warning(f"{ticker}: Insufficient data")
            return None
        
        # MultiIndex対応
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # ATR計算
        df['ATR'] = calculate_atr(df)
        
        # シグナル検出
        signals = detect_vcp_breakout(df)
        
        if not signals:
            logger.info(f"{ticker}: No signals")
            return None
        
        logger.info(f"{ticker}: {len(signals)} signals detected")
        
        # 各シグナルをトレード
        trades = []
        
        for signal in signals:
            entry_date = signal['date']
            entry_price = signal['entry_price']
            atr = signal['atr']
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            # ストップとターゲット
            stop_loss = entry_price - (atr * ATR_STOP_MULT)
            target = entry_price + (atr * ATR_STOP_MULT * REWARD_MULT)
            
            # エントリー後の価格推移
            future_df = df[df.index > entry_date].head(MAX_HOLD_DAYS)
            
            if future_df.empty:
                continue
            
            # 勝敗判定
            exit_date = None
            exit_price = None
            result = None
            exit_reason = None
            
            for idx, row in future_df.iterrows():
                # ストップロス（終値ベース）
                if row['Close'] <= stop_loss:
                    exit_date = idx
                    exit_price = stop_loss
                    result = 'LOSS'
                    exit_reason = 'STOP'
                    break
                
                # ターゲット達成（高値ベース）
                if row['High'] >= target:
                    exit_date = idx
                    exit_price = target
                    result = 'WIN'
                    exit_reason = 'TARGET'
                    break
            
            # タイムアウト（最大保有期間経過）
            if result is None:
                exit_date = future_df.index[-1]
                exit_price = future_df['Close'].iloc[-1]
                result = 'WIN' if exit_price > entry_price else 'LOSS'
                exit_reason = 'TIMEOUT'
            
            # トレード記録
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            hold_days = (exit_date - entry_date).days
            
            trades.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'target': target,
                'result': result,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'hold_days': hold_days,
                'tightness': signal['tightness']
            })
        
        return trades
    
    except Exception as e:
        logger.error(f"{ticker} error: {e}")
        return None


def run_backtest():
    """バックテスト実行"""
    logger.info("="*70)
    logger.info("SENTINEL v27 BACKTEST")
    logger.info("="*70)
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Tickers: {len(TEST_TICKERS)}")
    logger.info(f"Strategy: VCP Breakout with Tightness Filter")
    logger.info("")
    
    all_trades = []
    
    for ticker in TEST_TICKERS:
        trades = backtest_ticker(ticker, START_DATE, END_DATE)
        if trades:
            all_trades.extend(trades)
        time.sleep(0.5)  # API制限対策
    
    if not all_trades:
        logger.error("No trades generated")
        return
    
    # 結果分析
    df_trades = pd.DataFrame(all_trades)
    
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    losses = len(df_trades[df_trades['result'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    win_trades = df_trades[df_trades['result'] == 'WIN']['pnl_pct']
    loss_trades = df_trades[df_trades['result'] == 'LOSS']['pnl_pct']
    
    avg_win = win_trades.mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0
    avg_pnl = df_trades['pnl_pct'].mean()
    
    expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
    
    # 最大ドローダウン計算
    df_trades['cumulative_pnl'] = df_trades['pnl_pct'].cumsum()
    running_max = df_trades['cumulative_pnl'].expanding().max()
    drawdown = df_trades['cumulative_pnl'] - running_max
    max_drawdown = drawdown.min()
    
    # 月次パフォーマンス
    df_trades['month'] = pd.to_datetime(df_trades['entry_date']).dt.to_period('M')
    monthly = df_trades.groupby('month').agg({
        'pnl_pct': ['mean', 'sum', 'count']
    })
    
    # Exit理由の集計
    exit_reasons = df_trades['exit_reason'].value_counts()
    
    # レポート
    print("\n" + "="*70)
    print("BACKTEST RESULTS - SENTINEL v27")
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
    print("MONTHLY PERFORMANCE")
    print("="*70)
    print(monthly.to_string())
    print("")
    
    # トップ/ワーストトレード
    print("="*70)
    print("TOP 5 TRADES")
    print("="*70)
    top5 = df_trades.nlargest(5, 'pnl_pct')[
        ['ticker', 'entry_date', 'pnl_pct', 'hold_days', 'exit_reason']
    ]
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"[{i}] {row['ticker']:6} {row['entry_date'].strftime('%Y-%m-%d')} "
              f"{row['pnl_pct']:+6.2f}% ({row['hold_days']:2}日) {row['exit_reason']}")
    print("")
    
    print("="*70)
    print("WORST 5 TRADES")
    print("="*70)
    worst5 = df_trades.nsmallest(5, 'pnl_pct')[
        ['ticker', 'entry_date', 'pnl_pct', 'hold_days', 'exit_reason']
    ]
    for i, (_, row) in enumerate(worst5.iterrows(), 1):
        print(f"[{i}] {row['ticker']:6} {row['entry_date'].strftime('%Y-%m-%d')} "
              f"{row['pnl_pct']:+6.2f}% ({row['hold_days']:2}日) {row['exit_reason']}")
    print("")
    
    # 銘柄別パフォーマンス
    print("="*70)
    print("PERFORMANCE BY TICKER")
    print("="*70)
    ticker_perf = df_trades.groupby('ticker').agg({
        'pnl_pct': ['count', 'mean', 'sum'],
        'result': lambda x: (x == 'WIN').sum() / len(x) * 100
    }).round(2)
    ticker_perf.columns = ['Trades', 'Avg P&L %', 'Total P&L %', 'Win Rate %']
    ticker_perf = ticker_perf.sort_values('Total P&L %', ascending=False)
    print(ticker_perf.to_string())
    print("")
    
    # CSV保存
    df_trades.to_csv('backtest_trades.csv', index=False)
    logger.info("Trades saved to backtest_trades.csv")
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    
    if expectancy > 0:
        print(f"✅ POSITIVE EXPECTANCY: {expectancy:+.2f}%")
        print(f"   → システムは期待値がプラス")
        if win_rate >= 50:
            print(f"   → 勝率 {win_rate:.1f}% も良好")
        else:
            print(f"   → 勝率 {win_rate:.1f}% は低いが、RR比が良い")
    else:
        print(f"❌ NEGATIVE EXPECTANCY: {expectancy:+.2f}%")
        print(f"   → システムの改善が必要")
    
    print("")
    print(f"推定年間リターン: {df_trades['pnl_pct'].sum():.2f}%")
    print(f"（{total_trades}トレード × {expectancy:.2f}% = 理論値）")
    print("="*70)
    
    return df_trades


if __name__ == "__main__":
    df_result = run_backtest()
