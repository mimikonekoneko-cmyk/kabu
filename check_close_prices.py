#!/usr/bin/env python3
# check_close_prices.py
# 朝のシグナル銘柄の終値をチェックしてLINE通知

import yfinance as yf
import json
import os
import requests
from datetime import datetime
from pathlib import Path

# LINE設定
ACCESS_TOKEN = os.getenv('LINE_ACCESS_TOKEN')
USER_ID = os.getenv('LINE_USER_ID')

def send_line(message):
    """LINE通知送信"""
    if not ACCESS_TOKEN or not USER_ID:
        print("LINE credentials not set")
        return
    
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    
    data = {
        'to': USER_ID,
        'messages': [{
            'type': 'text',
            'text': message
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print("✅ LINE notification sent")
    except Exception as e:
        print(f"❌ LINE notification failed: {e}")

def load_today_signals():
    """今日のシグナルを読み込み"""
    today = datetime.now().strftime('%Y%m%d')
    signal_file = f"signals_{today}.json"
    
    if not Path(signal_file).exists():
        print(f"⚠️  No signals file found: {signal_file}")
        return None
    
    with open(signal_file, 'r') as f:
        return json.load(f)

def check_close_prices(signals):
    """終値をチェック"""
    results = []
    
    for signal in signals:
        ticker = signal['ticker']
        entry = signal['entry']
        
        try:
            # 今日の終値取得
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d")
            
            if df.empty:
                print(f"⚠️  No data for {ticker}")
                continue
            
            close = float(df['Close'].iloc[-1])
            change = ((close - entry) / entry) * 100
            
            results.append({
                'ticker': ticker,
                'entry': entry,
                'close': close,
                'change': change,
                'score': signal.get('score', 0),
                'tier': signal.get('tier', 'UNKNOWN')
            })
            
        except Exception as e:
            print(f"❌ Error for {ticker}: {e}")
            continue
    
    return results

def generate_report(results):
    """レポート生成"""
    if not results:
        return "📊 今日のシグナルなし"
    
    # ソート
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # レポート作成
    lines = []
    lines.append("="*50)
    lines.append("📊 今日のシグナル終値レポート")
    lines.append("="*50)
    lines.append(datetime.now().strftime("%Y/%m/%d %H:%M"))
    lines.append("")
    
    # CORE
    core = [r for r in results_sorted if r['tier'] == 'CORE']
    if core:
        lines.append("🔥 CORE銘柄")
        for r in core:
            emoji = "📈" if r['change'] > 0 else "📉"
            lines.append(f"{emoji} {r['ticker']} ({r['score']}/100)")
            lines.append(f"   Entry: ${r['entry']:.2f}")
            lines.append(f"   Close: ${r['close']:.2f} ({r['change']:+.2f}%)")
            lines.append("")
    
    # SECONDARY
    secondary = [r for r in results_sorted if r['tier'] == 'SECONDARY']
    if secondary:
        lines.append("⚡ SECONDARY銘柄")
        for r in secondary[:5]:  # TOP5のみ
            emoji = "📈" if r['change'] > 0 else "📉"
            lines.append(f"{emoji} {r['ticker']} ({r['score']}/100)")
            lines.append(f"   Entry: ${r['entry']:.2f} → ${r['close']:.2f} ({r['change']:+.2f}%)")
    
    lines.append("")
    lines.append("="*50)
    
    # 統計
    gains = [r for r in results if r['change'] > 0]
    losses = [r for r in results if r['change'] <= 0]
    
    lines.append(f"📊 統計")
    lines.append(f"上昇: {len(gains)}銘柄 / 下落: {len(losses)}銘柄")
    
    if results:
        avg_change = sum(r['change'] for r in results) / len(results)
        lines.append(f"平均変動: {avg_change:+.2f}%")
    
    lines.append("="*50)
    
    return "\n".join(lines)

def main():
    """メイン処理"""
    print("="*70)
    print("終値チェック＆通知")
    print("="*70)
    
    # シグナル読み込み
    signals = load_today_signals()
    
    if not signals:
        message = "📊 今日はシグナルなし（またはファイル未検出）"
        send_line(message)
        return
    
    print(f"✅ {len(signals)} signals loaded")
    
    # 終値チェック
    results = check_close_prices(signals)
    
    print(f"✅ {len(results)} prices checked")
    
    # レポート生成
    report = generate_report(results)
    
    print("\n" + report)
    
    # LINE通知
    send_line(report)
    
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()
