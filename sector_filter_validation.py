#!/usr/bin/env python3
# sector_filter_validation.py
# セクターフィルターの有効性を検証

"""
質問:
「セクターフィルターで除外した銘柄の中に、
 実は大化けした株があったのでは？」

検証:
1. 2024年の全出来高急増銘柄を取得
2. セクターでフィルター
3. 除外された銘柄の6ヶ月後のパフォーマンスを確認
4. 「見逃し」を定量化
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

# セクター定義
PRIORITY_SECTORS = {
    'Technology',
    'Communication Services',
    'Consumer Cyclical',
    'Healthcare',
    'Industrials',
}

EXCLUDE_SECTORS = {
    'Financial Services',
    'Real Estate',
    'Utilities',
    'Basic Materials',
}

def get_volume_surge_stocks_2024():
    """
    2024年に出来高急増した銘柄を検出
    （簡易版：主要銘柄のみ）
    """
    # サンプル銘柄リスト
    tickers = [
        # テック
        'NVDA', 'AMD', 'AVGO', 'PLTR', 'IONQ',
        # バイオ
        'FULC', 'ORKA', 'TARS', 'INBX',
        # 銀行（除外対象）
        'JPM', 'GS', 'BAC', 'WFC',
        # 小売（除外対象）
        'HD', 'WMT', 'COST',
        # エネルギー（除外対象）
        'XOM', 'CVX',
        # REIT（除外対象）
        'SPG', 'O', 'VICI',
    ]
    
    return tickers

def analyze_missed_opportunities():
    """
    セクターフィルターで除外した銘柄のパフォーマンス分析
    """
    tickers = get_volume_surge_stocks_2024()
    
    results = {
        'priority': [],      # 優先セクター
        'excluded': [],      # 除外セクター
        'other': []          # その他
    }
    
    for ticker in tickers:
        try:
            print(f"Analyzing {ticker}...")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('sector', 'Unknown')
            
            # 2024年のデータ取得
            df = stock.history(start="2024-01-01", end="2024-12-31")
            
            if df.empty:
                continue
            
            # パフォーマンス計算
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            performance = ((end_price - start_price) / start_price) * 100
            
            # 分類
            result = {
                'ticker': ticker,
                'sector': sector,
                'performance': performance,
                'start': start_price,
                'end': end_price
            }
            
            if sector in PRIORITY_SECTORS:
                results['priority'].append(result)
            elif sector in EXCLUDE_SECTORS:
                results['excluded'].append(result)
            else:
                results['other'].append(result)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue
    
    return results

def generate_report(results):
    """
    レポート生成
    """
    print("\n" + "="*70)
    print("セクターフィルター検証結果")
    print("="*70)
    
    # 優先セクター
    print("\n【優先セクター】")
    if results['priority']:
        df = pd.DataFrame(results['priority'])
        df = df.sort_values('performance', ascending=False)
        
        print(df[['ticker', 'sector', 'performance']].to_string(index=False))
        
        avg_perf = df['performance'].mean()
        winners = len(df[df['performance'] > 0])
        total = len(df)
        
        print(f"\n平均パフォーマンス: {avg_perf:+.2f}%")
        print(f"勝率: {winners}/{total} ({winners/total*100:.1f}%)")
    
    # 除外セクター
    print("\n【除外セクター（見逃しチェック）】")
    if results['excluded']:
        df = pd.DataFrame(results['excluded'])
        df = df.sort_values('performance', ascending=False)
        
        print(df[['ticker', 'sector', 'performance']].to_string(index=False))
        
        avg_perf = df['performance'].mean()
        winners = len(df[df['performance'] > 0])
        total = len(df)
        big_winners = len(df[df['performance'] > 50])
        
        print(f"\n平均パフォーマンス: {avg_perf:+.2f}%")
        print(f"勝率: {winners}/{total} ({winners/total*100:.1f}%)")
        print(f"大勝ち(+50%以上): {big_winners}銘柄")
        
        if big_winners > 0:
            print("\n⚠️  除外セクターに大勝ち銘柄あり！")
            print("   → セクターフィルター見直しの余地")
        else:
            print("\n✅ 除外セクターに大勝ちなし")
            print("   → セクターフィルターは正しかった")
    
    # その他
    print("\n【その他セクター】")
    if results['other']:
        df = pd.DataFrame(results['other'])
        df = df.sort_values('performance', ascending=False)
        
        print(df[['ticker', 'sector', 'performance']].to_string(index=False))
    
    # 総合評価
    print("\n" + "="*70)
    print("総合評価")
    print("="*70)
    
    if results['priority'] and results['excluded']:
        priority_avg = pd.DataFrame(results['priority'])['performance'].mean()
        excluded_avg = pd.DataFrame(results['excluded'])['performance'].mean()
        
        print(f"優先セクター平均: {priority_avg:+.2f}%")
        print(f"除外セクター平均: {excluded_avg:+.2f}%")
        print(f"差分: {priority_avg - excluded_avg:+.2f}%")
        
        if priority_avg > excluded_avg + 20:
            print("\n✅ 結論: セクターフィルターは有効")
            print("   優先セクターが明確に優位")
        elif priority_avg > excluded_avg:
            print("\n⚠️  結論: セクターフィルターは微妙に有効")
            print("   優位性は小さい")
        else:
            print("\n❌ 結論: セクターフィルターは逆効果")
            print("   除外セクターの方が良かった")

if __name__ == "__main__":
    print("セクターフィルター有効性検証")
    print("="*70)
    print()
    print("検証内容:")
    print("1. 2024年の主要銘柄パフォーマンス取得")
    print("2. セクター別に分類")
    print("3. 優先 vs 除外 を比較")
    print()
    
    results = analyze_missed_opportunities()
    generate_report(results)
    
    print("\n" + "="*70)
    print("推奨アクション")
    print("="*70)
    print()
    print("もし除外セクターに大勝ち銘柄が多ければ:")
    print("→ セクターフィルターを緩める")
    print("→ スコア調整のみにする")
    print()
    print("もし優先セクターが明確に優位なら:")
    print("→ 現状維持")
    print("→ バックテストを信じる")
