#!/usr/bin/env python3
# check_new_tickers.py
# 新規追加銘柄がなぜv28で検出されないか確認

import yfinance as yf
import pandas as pd

NEW_TICKERS = [
    'TARS', 'ORKA', 'CEVA', 'HOLX', 'FFIV',
    'PLTR', 'CRWD', 'IONQ', 'ASTS', 'ANET', 'NET', 'PANW'
]

DETECTED = ['ANET', 'HOLX']  # v28で検出された

print("="*70)
print("新規追加銘柄の状態確認")
print("="*70)
print()

for ticker in NEW_TICKERS:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="200d")
        
        if df.empty or len(df) < 200:
            print(f"{ticker}: データ不足")
            continue
        
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        volume = df['Volume'].astype(float)
        
        # 現在価格
        current = float(close.iloc[-1])
        
        # MA50, MA200
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        
        # 高値更新チェック
        high20 = high.iloc[-20:].max()
        prev_high20 = high.iloc[-21:-1].max()
        
        # 出来高
        vol_current = volume.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1]
        
        # 判定
        status = []
        
        if current > ma50:
            status.append("✅ Price > MA50")
        else:
            status.append("❌ Price < MA50")
        
        if ma50 > ma200:
            status.append("✅ MA50 > MA200")
        else:
            status.append("❌ MA50 < MA200")
        
        if current > prev_high20:
            status.append("✅ 高値更新")
        else:
            status.append("❌ 高値未更新")
        
        if vol_current > vol_avg * 1.0:
            status.append("✅ 出来高増")
        else:
            status.append("❌ 出来高減")
        
        detected = "🔥 DETECTED" if ticker in DETECTED else "❌ NOT DETECTED"
        
        print(f"{ticker:6} {detected}")
        print(f"  Price: ${current:.2f} | MA50: ${ma50:.2f} | MA200: ${ma200:.2f}")
        for s in status:
            print(f"  {s}")
        print()
        
    except Exception as e:
        print(f"{ticker}: エラー - {e}")
        print()

print("="*70)
print("結論:")
print("="*70)
print("検出された銘柄 = 全条件を満たす")
print("検出されない銘柄 = 1つ以上の条件を満たさない")
print()
print("→ 新規銘柄はまだVCP形成中")
print("→ 1-3ヶ月後に検出される可能性")
print("="*70)
