#!/usr/bin/env python3
# SENTINEL SAFE v28 - Defensive Asset Screener
# v28の逆: リスクオフ時に強い銘柄を検出
#
# 検出対象:
# - 債券ETF（国債、社債）
# - 金・貴金属
# - 公益株（電力、ガス、水道）
# - 生活必需品（食品、日用品）
# - ディフェンシブ株（ヘルスケア、タバコ）
#
# トリガー:
# - VIX > 20
# - SPY < MA200（ベア相場）
# - グロース株の崩壊
#
# Philosophy: "リスクオフ時の避難先"

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests

# ---------------------------
# CONFIG
# ---------------------------
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

INITIAL_CAPITAL_JPY = 350_000
TRADING_RATIO = 0.75

# ---------------------------
# SAFE ASSET UNIVERSE
# ---------------------------
SAFE_TICKERS = {
    # === 債券ETF ===
    # 米国債（最も安全）
    'TLT': '米国債20年', 'IEF': '米国債7-10年', 'SHY': '米国債1-3年',
    'GOVT': '米国債総合', 'VGLT': '長期米国債',
    
    # 社債（ややリスク、利回り高）
    'LQD': '投資適格社債', 'HYG': 'ハイイールド債',
    
    # === 金・貴金属 ===
    'GLD': '金ETF', 'IAU': '金ETF2', 'SLV': '銀ETF',
    'GDXJ': '金鉱株Jr', 'GDX': '金鉱株',
    
    # === 公益株（ディフェンシブ） ===
    'XLU': '公益セクター', 'NEE': 'NextEra電力', 'DUK': 'Duke電力',
    'SO': 'Southern電力', 'D': 'Dominion電力',
    'AEP': 'American電力', 'EXC': 'Exelon',
    
    # === 生活必需品 ===
    'XLP': '生活必需品セクター', 'PG': 'P&G', 'KO': 'コカコーラ',
    'PEP': 'ペプシ', 'WMT': 'ウォルマート', 'COST': 'コストコ',
    'CL': 'コルゲート', 'KMB': 'キンバリー',
    
    # === ヘルスケア（ディフェンシブ） ===
    'JNJ': 'J&J', 'PFE': 'ファイザー', 'ABBV': 'アッヴィ',
    'MRK': 'メルク', 'BMY': 'ブリストル',
    
    # === タバコ（超ディフェンシブ） ===
    'MO': 'アルトリア', 'PM': 'フィリップモリス',
    
    # === REIT（配当狙い） ===
    'VNQ': 'REIT総合', 'O': 'リアルティ',
    
    # === その他ディフェンシブ ===
    'BRK.B': 'バークシャー', 'VOO': 'S&P500',
}

# ---------------------------
# Market Condition Checker
# ---------------------------
def check_risk_environment():
    """
    リスク環境チェック
    
    Returns:
        'RISK_OFF': リスクオフ（安全資産へ）
        'RISK_ON': リスクオン（グロース株へ）
        'NEUTRAL': 中立
    """
    
    try:
        # VIX取得
        vix = yf.Ticker('^VIX')
        vix_df = vix.history(period='5d')
        current_vix = float(vix_df['Close'].iloc[-1])
        
        # SPY MA200チェック
        spy = yf.Ticker('SPY')
        spy_df = spy.history(period='1y')
        spy_close = float(spy_df['Close'].iloc[-1])
        spy_ma200 = spy_df['Close'].rolling(200).mean().iloc[-1]
        
        # QQQ（ナスダック）チェック
        qqq = yf.Ticker('QQQ')
        qqq_df = qqq.history(period='3mo')
        qqq_close = float(qqq_df['Close'].iloc[-1])
        qqq_ma50 = qqq_df['Close'].rolling(50).mean().iloc[-1]
        
        # 判定
        risk_signals = []
        
        # VIX判定
        if current_vix > 30:
            risk_signals.append('VIX_PANIC')
        elif current_vix > 20:
            risk_signals.append('VIX_HIGH')
        
        # ベア相場判定
        if spy_close < spy_ma200:
            risk_signals.append('BEAR_MARKET')
        
        # ナスダック弱気判定
        if qqq_close < qqq_ma50 * 0.95:
            risk_signals.append('TECH_WEAK')
        
        # 総合判定
        if 'VIX_PANIC' in risk_signals or 'BEAR_MARKET' in risk_signals:
            env = 'RISK_OFF'
        elif 'VIX_HIGH' in risk_signals or 'TECH_WEAK' in risk_signals:
            env = 'NEUTRAL'
        else:
            env = 'RISK_ON'
        
        return {
            'environment': env,
            'vix': current_vix,
            'spy_vs_ma200': ((spy_close - spy_ma200) / spy_ma200) * 100,
            'signals': risk_signals
        }
        
    except Exception as e:
        print(f"Risk environment check error: {e}")
        return {
            'environment': 'NEUTRAL',
            'vix': 0,
            'spy_vs_ma200': 0,
            'signals': []
        }


# ---------------------------
# Safe Asset Screening
# ---------------------------
def screen_safe_assets():
    """
    安全資産スクリーニング
    
    評価基準:
    1. 安定性（ボラティリティの低さ）
    2. 相対的強さ（他の安全資産との比較）
    3. 配当利回り
    4. 最大ドローダウン
    """
    
    print("="*70)
    print("🛡️ SENTINEL SAFE v28 - Defensive Asset Screener")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    print()
    
    # リスク環境チェック
    risk_env = check_risk_environment()
    
    print(f"🌍 Market Environment: {risk_env['environment']}")
    print(f"📊 VIX: {risk_env['vix']:.1f}")
    print(f"📈 SPY vs MA200: {risk_env['spy_vs_ma200']:+.1f}%")
    if risk_env['signals']:
        print(f"⚠️  Signals: {', '.join(risk_env['signals'])}")
    print()
    
    # RISK_ONの時は終了
    if risk_env['environment'] == 'RISK_ON':
        print("✅ RISK_ON環境")
        print("   → グロース株（SENTINEL v28）を使用してください")
        print()
        return
    
    print(f"🛡️ {risk_env['environment']} 環境")
    print("   → 安全資産スクリーニング開始")
    print()
    
    results = []
    
    for ticker, name in SAFE_TICKERS.items():
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period='1y')
            
            if len(df) < 100:
                continue
            
            # 基本データ
            close = df['Close'].astype(float)
            current_price = float(close.iloc[-1])
            
            # 1. 安定性スコア（ボラティリティの低さ）
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            if volatility < 10:
                stability_score = 100
            elif volatility < 15:
                stability_score = 80
            elif volatility < 20:
                stability_score = 60
            elif volatility < 30:
                stability_score = 40
            else:
                stability_score = 20
            
            # 2. パフォーマンス（直近3ヶ月）
            perf_3m = ((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]) * 100 if len(close) >= 63 else 0
            
            if perf_3m > 5:
                perf_score = 100
            elif perf_3m > 2:
                perf_score = 80
            elif perf_3m > 0:
                perf_score = 60
            elif perf_3m > -5:
                perf_score = 40
            else:
                perf_score = 20
            
            # 3. 最大ドローダウン（小さいほど良い）
            cummax = close.expanding().max()
            drawdown = ((close - cummax) / cummax) * 100
            max_dd = drawdown.min()
            
            if max_dd > -5:
                dd_score = 100
            elif max_dd > -10:
                dd_score = 80
            elif max_dd > -15:
                dd_score = 60
            elif max_dd > -20:
                dd_score = 40
            else:
                dd_score = 20
            
            # 総合スコア
            total_score = int((stability_score * 0.4 + perf_score * 0.3 + dd_score * 0.3))
            
            # Tier判定
            if total_score >= 75:
                tier = 'TOP_SAFE'
                emoji = '🛡️'
            elif total_score >= 65:
                tier = 'SAFE'
                emoji = '✅'
            elif total_score >= 55:
                tier = 'MODERATE'
                emoji = '⚠️'
            else:
                tier = 'SKIP'
                emoji = '❌'
            
            if tier != 'SKIP':
                results.append({
                    'ticker': ticker,
                    'name': name,
                    'score': total_score,
                    'tier': tier,
                    'emoji': emoji,
                    'price': current_price,
                    'volatility': volatility,
                    'perf_3m': perf_3m,
                    'max_dd': max_dd,
                    'stability': stability_score,
                    'performance': perf_score,
                    'drawdown': dd_score
                })
        
        except Exception as e:
            continue
    
    # ソート
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # レポート
    print("="*70)
    print("🛡️ TOP SAFE ASSETS")
    print("="*70)
    print()
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i}. {r['emoji']} {r['ticker']:8} ({r['name']})")
        print(f"   Score: {r['score']}/100")
        print(f"   Price: ${r['price']:.2f}")
        print(f"   Volatility: {r['volatility']:.1f}% (低いほど安定)")
        print(f"   3M Perf: {r['perf_3m']:+.1f}%")
        print(f"   Max DD: {r['max_dd']:.1f}%")
        print()
    
    print("="*70)
    print(f"Total: {len(results)} safe assets")
    print("="*70)
    print()
    
    # 推奨アクション
    print("💡 推奨アクション:")
    print()
    
    if risk_env['environment'] == 'RISK_OFF':
        print("   🚨 RISK_OFF環境")
        print("   → グロース株を全売却")
        print("   → 安全資産に100%移行")
        print("   → TOP3の安全資産に分散")
        print()
    elif risk_env['environment'] == 'NEUTRAL':
        print("   ⚠️ NEUTRAL環境")
        print("   → グロース株を50%削減")
        print("   → 安全資産に50%移行")
        print("   → リスク分散")
        print()
    
    # 具体的な推奨
    if results:
        print("   推奨銘柄:")
        for r in results[:3]:
            print(f"   {r['emoji']} {r['ticker']} - {r['name']}")
        print()
    
    print("="*70)
    
    return results


# ---------------------------
# LINE Notification
# ---------------------------
def send_line_notification(message):
    """LINE通知送信"""
    
    if not ACCESS_TOKEN or not USER_ID:
        return
    
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    
    # 5000文字制限
    if len(message) > 4800:
        chunks = [message[i:i+4800] for i in range(0, len(message), 4800)]
        for chunk in chunks:
            payload = {
                'to': USER_ID,
                'messages': [{'type': 'text', 'text': chunk}]
            }
            requests.post(url, headers=headers, json=payload, timeout=30)
    else:
        payload = {
            'to': USER_ID,
            'messages': [{'type': 'text', 'text': message}]
        }
        requests.post(url, headers=headers, json=payload, timeout=30)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    results = screen_safe_assets()
    
    # LINE通知
    if results and ACCESS_TOKEN and USER_ID:
        message_lines = []
        message_lines.append("🛡️ SENTINEL SAFE v28")
        message_lines.append("="*40)
        message_lines.append(f"📅 {datetime.now().strftime('%m/%d %H:%M')}")
        message_lines.append("")
        
        risk_env = check_risk_environment()
        message_lines.append(f"🌍 {risk_env['environment']}")
        message_lines.append(f"VIX: {risk_env['vix']:.1f}")
        message_lines.append("")
        
        message_lines.append("🛡️ TOP SAFE ASSETS:")
        for r in results[:5]:
            message_lines.append(f"{r['emoji']} {r['ticker']} {r['score']}/100")
            message_lines.append(f"   ${r['price']:.2f} | 3M: {r['perf_3m']:+.1f}%")
        
        message_lines.append("")
        message_lines.append("="*40)
        
        message = "\n".join(message_lines)
        send_line_notification(message)
