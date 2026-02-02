#!/usr/bin/env python3
# analyze_ticker.py
# 個別銘柄をv28と同じロジックで分析

"""
使い方:
    python analyze_ticker.py FULC
    python analyze_ticker.py TSM NVDA GOOG
    python analyze_ticker.py --all CORE

機能:
- v28と完全同一のロジック
- VCPパターン検出
- スコアリング
- バックテスト
- エントリー/ストップ/ターゲット計算
- 推奨株数計算
- 視覚的レポート
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime, timedelta

# ===========================
# v28と同じ定数
# ===========================
ATR_STOP_MULT = 2.0
ATR_TARGET_MULT = 4.0
MAX_TIGHTNESS_BASE = 1.5
MIN_VOLUME_DRY = 0.8
CAPITAL_JPY = 3_500_000
TRADING_CAPITAL_PCT = 0.75
FX_RATE = 154.73

# ===========================
# ヘルパー関数
# ===========================

def safe_rolling_last(series, window, min_periods=1, default=np.nan):
    """安全なローリング計算"""
    if len(series) < min_periods:
        return default
    try:
        result = series.rolling(window, min_periods=min_periods).mean().iloc[-1]
        return result if not pd.isna(result) else default
    except Exception:
        return default

def calculate_atr(df, period=14):
    """ATR計算"""
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    
    return float(atr) if not pd.isna(atr) else 0.0

# ===========================
# VCP検出（v28完全再現）
# ===========================

def detect_vcp_pattern(df):
    """VCPパターン検出"""
    
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float)
    
    # 収縮度計算
    atr14 = calculate_atr(df, 14)
    recent_range = high.iloc[-5:].max() - low.iloc[-5:].min()
    tightness = recent_range / atr14 if atr14 > 0 else 999
    
    # VCP判定
    score = 0
    reasons = []
    
    if tightness < 0.8:
        score += 30
        reasons.append("VCP+++")
        stage = "⚡初動圏"
        completion = 70
    elif tightness < 1.2:
        score += 20
        reasons.append("VCP+")
        stage = "👁形成中"
        completion = 50
    elif tightness < MAX_TIGHTNESS_BASE:
        score += 10
        reasons.append("VCP")
        stage = "⏳準備段階"
        completion = 30
    else:
        stage = "❌未形成"
        completion = 0
    
    # 出来高縮小
    vol50 = safe_rolling_last(volume, 50, min_periods=10)
    vol_dry = volume.iloc[-1] < vol50 * MIN_VOLUME_DRY
    if vol_dry:
        score += 15
        reasons.append("VolDry")
    
    # モメンタム
    mom5 = safe_rolling_last(close, 5, min_periods=3)
    mom20 = safe_rolling_last(close, 20, min_periods=10)
    if not pd.isna(mom5) and not pd.isna(mom20) and (mom5 / mom20) > 1.02:
        score += 20
        reasons.append("Mom+")
    
    # トレンド
    ma50 = safe_rolling_last(close, 50, min_periods=25)
    ma200 = safe_rolling_last(close, 200, min_periods=100)
    
    trend_ok = False
    if not pd.isna(ma50) and not pd.isna(ma200):
        if (ma50 - ma200) / ma200 > 0.03:
            score += 20
            reasons.append("Trend+")
            trend_ok = True
    
    return {
        'vcp_score': score,
        'tightness': tightness,
        'completion': completion,
        'stage': stage,
        'reasons': reasons,
        'vol_dry': vol_dry,
        'trend_ok': trend_ok,
        'ma50': ma50,
        'ma200': ma200
    }

# ===========================
# バックテスト（v28完全再現）
# ===========================

def simulate_backtest(df):
    """バックテスト実行"""
    
    trades = []
    
    for i in range(200, len(df) - 60):
        window = df.iloc[max(0, i-60):i]
        
        # VCP検出
        high = window['High'].astype(float)
        low = window['Low'].astype(float)
        close = window['Close'].astype(float)
        
        atr = calculate_atr(window, 14)
        recent_range = high.iloc[-5:].max() - low.iloc[-5:].min()
        tightness = recent_range / atr if atr > 0 else 999
        
        if tightness >= MAX_TIGHTNESS_BASE:
            continue
        
        # エントリー
        entry = high.iloc[-5:].max() * 1.002
        stop = entry - (atr * ATR_STOP_MULT)
        target = entry + (atr * ATR_TARGET_MULT)
        
        # シミュレート
        for j in range(i, min(i + 60, len(df))):
            current_high = df['High'].iloc[j]
            current_low = df['Low'].iloc[j]
            
            if current_low <= stop:
                pnl = ((stop - entry) / entry) * 100
                trades.append({'result': 'LOSS', 'pnl': pnl})
                break
            
            if current_high >= target:
                pnl = ((target - entry) / entry) * 100
                trades.append({'result': 'WIN', 'pnl': pnl})
                break
    
    if not trades:
        return {'win_rate': 50, 'expectancy': 0, 'total_trades': 0}
    
    wins = [t for t in trades if t['result'] == 'WIN']
    win_rate = (len(wins) / len(trades)) * 100
    expectancy = sum(t['pnl'] for t in trades) / len(trades)
    
    return {
        'win_rate': win_rate,
        'expectancy': expectancy,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(trades) - len(wins)
    }

# ===========================
# スコアリング（v28完全再現）
# ===========================

def calculate_comprehensive_score(vcp_result, rr_ratio, backtest):
    """総合スコア計算"""
    
    # テクニカルスコア
    tech_score = vcp_result['vcp_score']
    
    # RRスコア
    if rr_ratio >= 2.5:
        rr_score = 35
    elif rr_ratio >= 2.0:
        rr_score = 30
    elif rr_ratio >= 1.5:
        rr_score = 25
    else:
        rr_score = 20
    
    # バックテストスコア
    win_rate = backtest['win_rate']
    if win_rate >= 60:
        bt_score = 25
    elif win_rate >= 55:
        bt_score = 20
    elif win_rate >= 50:
        bt_score = 15
    else:
        bt_score = 10
    
    total = tech_score + rr_score + bt_score
    
    # Tier判定
    if total >= 75:
        tier = 'CORE'
        emoji = '🔥'
    elif total >= 60:
        tier = 'SECONDARY'
        emoji = '⚡'
    else:
        tier = 'WATCH'
        emoji = '👁'
    
    return {
        'total_score': total,
        'tech_score': tech_score,
        'rr_score': rr_score,
        'bt_score': bt_score,
        'tier': tier,
        'emoji': emoji
    }

# ===========================
# ポジションサイジング
# ===========================

def calculate_position_size(capital_usd, atr_pct, win_rate):
    """推奨株数計算"""
    
    # ケリー基準（簡易版）
    rr = 2.0  # 固定
    kelly_pct = (win_rate * rr - (1 - win_rate)) / rr
    kelly_pct = max(0, min(kelly_pct, 0.25))  # 0-25%
    
    # リスク調整
    risk_pct = atr_pct * ATR_STOP_MULT
    position_usd = (capital_usd * kelly_pct) * 0.5  # 安全係数
    
    return position_usd

# ===========================
# メイン分析関数
# ===========================

def analyze_ticker(ticker, capital_jpy=CAPITAL_JPY, fx_rate=FX_RATE):
    """個別銘柄分析（v28完全再現）"""
    
    print(f"\n{'='*70}")
    print(f"📊 {ticker} - Individual Analysis (v28 Logic)")
    print(f"{'='*70}\n")
    
    try:
        # データ取得
        print(f"📥 Fetching data for {ticker}...")
        df = yf.download(ticker, period="400d", progress=False, auto_adjust=True)
        
        if df.empty or len(df) < 200:
            print(f"❌ Insufficient data for {ticker}")
            return None
        
        print(f"✅ Data loaded: {len(df)} days")
        
        # 企業情報
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'Unknown')
        market_cap = info.get('marketCap', 0)
        
        # 現在価格
        current_price = float(df['Close'].iloc[-1])
        
        # VCP分析
        print(f"\n🔍 VCP Pattern Detection...")
        vcp = detect_vcp_pattern(df)
        
        # エントリー/ストップ/ターゲット
        high = df['High'].astype(float)
        pivot = high.iloc[-5:].max() * 1.002
        
        atr14 = calculate_atr(df, 14)
        stop = pivot - (atr14 * ATR_STOP_MULT)
        target = pivot + (atr14 * ATR_TARGET_MULT)
        
        risk_pct = ((pivot - stop) / pivot) * 100
        reward_pct = ((target - pivot) / pivot) * 100
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        # バックテスト
        print(f"📈 Running backtest...")
        backtest = simulate_backtest(df)
        
        # スコアリング
        score = calculate_comprehensive_score(vcp, rr_ratio, backtest)
        
        # ポジションサイジング
        capital_usd = (capital_jpy * TRADING_CAPITAL_PCT) / fx_rate
        atr_pct = atr14 / current_price
        position_usd = calculate_position_size(capital_usd, atr_pct, backtest['win_rate']/100)
        
        shares = int(position_usd / current_price) if current_price > 0 else 0
        position_cost = shares * current_price
        
        # レポート生成
        print(f"\n{'='*70}")
        print(f"{score['emoji']} {ticker} Analysis Report")
        print(f"{'='*70}\n")
        
        # 基本情報
        print(f"📍 Basic Info")
        print(f"   Sector: {sector}")
        print(f"   Market Cap: ${market_cap/1e9:.2f}B" if market_cap > 0 else "   Market Cap: N/A")
        print(f"   Current Price: ${current_price:.2f}")
        print()
        
        # スコア
        print(f"🎯 VCP Score: {score['total_score']}/100 ({score['tier']})")
        print(f"   Technical: {score['tech_score']}")
        print(f"   Risk/Reward: {score['rr_score']}")
        print(f"   Backtest: {score['bt_score']}")
        print()
        
        # VCPパターン
        print(f"📊 VCP Pattern Analysis")
        print(f"   Completion: {vcp['completion']}% {vcp['stage']}")
        print(f"   Tightness: {vcp['tightness']:.2f} (Target: <{MAX_TIGHTNESS_BASE})")
        print(f"   Volume Dry: {'✅ Yes' if vcp['vol_dry'] else '❌ No'}")
        print(f"   Trend: {'✅ MA50 > MA200' if vcp['trend_ok'] else '⚠️  Weak'}")
        print(f"   Signals: {', '.join(vcp['reasons'])}")
        print()
        
        # エントリー戦略
        print(f"💰 Entry Strategy")
        print(f"   Entry:  ${pivot:.2f}")
        print(f"   Stop:   ${stop:.2f} ({risk_pct:-.1f}%)")
        print(f"   Target: ${target:.2f} (+{reward_pct:.1f}%)")
        print(f"   R/R Ratio: 1:{rr_ratio:.1f}")
        print()
        
        # ポジションサイジング
        print(f"📈 Position Sizing")
        print(f"   Capital (USD): ${capital_usd:.0f}")
        print(f"   Recommended: {shares} shares = ${position_cost:.0f}")
        print(f"   Portfolio %: {(position_cost/capital_usd)*100:.1f}%")
        print()
        
        # バックテスト結果
        print(f"🔬 Backtest Results (200 days)")
        print(f"   Win Rate: {backtest['win_rate']:.1f}%")
        print(f"   Expectancy: {backtest['expectancy']:+.2f}%")
        print(f"   Total Trades: {backtest['total_trades']}")
        print(f"   Wins: {backtest['wins']} | Losses: {backtest['losses']}")
        print()
        
        # 推奨
        distance_from_entry = ((current_price - pivot) / pivot) * 100
        
        print(f"✅ Recommendation")
        if distance_from_entry < -10:
            recommendation = "⏳ WAIT (Too far from entry)"
            action = "Wait for pullback"
        elif distance_from_entry < -2:
            recommendation = "👀 WATCH (Near entry)"
            action = "Prepare to buy"
        elif distance_from_entry < 2:
            recommendation = "✅ BUY (At entry)"
            action = f"Buy {shares} shares"
        else:
            recommendation = "⚠️  EXTENDED (Above entry)"
            action = "Wait for pullback or skip"
        
        print(f"   Status: {recommendation}")
        print(f"   Action: {action}")
        print(f"   Distance from Entry: {distance_from_entry:+.1f}%")
        print()
        
        print(f"{'='*70}\n")
        
        return {
            'ticker': ticker,
            'score': score,
            'vcp': vcp,
            'current_price': current_price,
            'entry': pivot,
            'stop': stop,
            'target': target,
            'shares': shares,
            'backtest': backtest,
            'recommendation': recommendation
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        return None

# ===========================
# CLI
# ===========================

def main():
    """メイン処理"""
    
    parser = argparse.ArgumentParser(description='Analyze individual stocks with v28 logic')
    parser.add_argument('tickers', nargs='+', help='Stock ticker(s) to analyze')
    parser.add_argument('--capital', type=float, default=CAPITAL_JPY, help='Trading capital in JPY')
    parser.add_argument('--fx', type=float, default=FX_RATE, help='USD/JPY exchange rate')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SENTINEL v28 - Individual Ticker Analyzer")
    print(f"{'='*70}")
    print(f"\nCapital: ¥{args.capital:,.0f}")
    print(f"FX Rate: ¥{args.fx:.2f}")
    print(f"Tickers: {', '.join(args.tickers)}")
    
    results = []
    
    for ticker in args.tickers:
        result = analyze_ticker(ticker.upper(), args.capital, args.fx)
        if result:
            results.append(result)
    
    # サマリー
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"📊 Summary ({len(results)} tickers)")
        print(f"{'='*70}\n")
        
        results_sorted = sorted(results, key=lambda x: x['score']['total_score'], reverse=True)
        
        for i, r in enumerate(results_sorted, 1):
            print(f"{i}. {r['ticker']:6} {r['score']['total_score']:3}/100 {r['score']['emoji']} - {r['recommendation']}")
        
        print(f"\n{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # インタラクティブモード
        print("="*70)
        print("SENTINEL v28 - Individual Ticker Analyzer")
        print("="*70)
        print()
        print("Usage:")
        print("  python analyze_ticker.py FULC")
        print("  python analyze_ticker.py TSM NVDA GOOG")
        print("  python analyze_ticker.py FULC --capital 5000000 --fx 155.0")
        print()
        print("="*70)
        
        ticker = input("Enter ticker to analyze: ").strip().upper()
        if ticker:
            analyze_ticker(ticker)
    else:
        main()
