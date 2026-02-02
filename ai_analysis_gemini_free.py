#!/usr/bin/env python3
# ai_analysis_gemini_free.py
# Google Gemini API（完全無料）でAI分析

"""
Google Gemini API:
- 完全無料
- 1日1,500リクエスト
- Claude Sonnetと同等性能
- API Key取得: https://makersuite.google.com/app/apikey

コスト: ¥0
"""

import json
import os
import requests
from datetime import datetime
from pathlib import Path

# Google Gemini API設定
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# LINE設定
LINE_ACCESS_TOKEN = os.getenv('LINE_ACCESS_TOKEN')
LINE_USER_ID = os.getenv('LINE_USER_ID')

def send_line(message):
    """LINE通知"""
    if not LINE_ACCESS_TOKEN or not LINE_USER_ID:
        print("LINE credentials not set")
        return
    
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}'
    }
    
    # 5000文字制限対応
    if len(message) > 4800:
        chunks = [message[i:i+4800] for i in range(0, len(message), 4800)]
        for chunk in chunks:
            data = {
                'to': LINE_USER_ID,
                'messages': [{'type': 'text', 'text': chunk}]
            }
            requests.post(url, headers=headers, json=data)
    else:
        data = {
            'to': LINE_USER_ID,
            'messages': [{'type': 'text', 'text': message}]
        }
        requests.post(url, headers=headers, json=data)

def load_signals():
    """今日のシグナルを読み込み"""
    today = datetime.now().strftime('%Y%m%d')
    signal_file = f"signals_{today}.json"
    
    if not Path(signal_file).exists():
        signal_file = "today_signals.json"
    
    if not Path(signal_file).exists():
        raise FileNotFoundError("No signals file found")
    
    with open(signal_file, 'r') as f:
        return json.load(f)

def create_analysis_prompt(signals):
    """Gemini用プロンプト作成"""
    
    prompt = f"""あなたは世界トップクラスの株式アナリストです。

以下は、VCPパターン検出システム「SENTINEL v28」が検出した今日のシグナルです。
全{len(signals)}銘柄の詳細データを提供しますので、深層分析を行ってください。

# シグナルデータ（JSON）

```json
{json.dumps(signals, indent=2)}
```

# 分析依頼

以下の観点で分析し、最終的にTOP3を推奨してください：

## 1. 個別銘柄分析
各銘柄について：
- セクタートレンド
- テクニカル評価（VCPスコアの妥当性）
- リスク要因

## 2. 相関分析
- セクター集中リスク
- ポートフォリオバランス

## 3. 最終推奨（TOP3）

以下の形式で：

```
【AI推奨 TOP3】

🥇 1位: [TICKER] (スコア: XX/100)
推奨理由:
- [理由1]
- [理由2]

リスク:
- [リスク要因]

エントリー: $XX.XX | Stop: $XX.XX | Target: $XX.XX

🥈 2位: ...
🥉 3位: ...
```

## 4. 推奨ポートフォリオ
資金配分の提案

---

**重要**: 
- 推測ではなく、データベースで分析
- リスクも必ず明示
- 最終判断は人間が行うことを前提

それでは分析をお願いします。
"""
    
    return prompt

def analyze_with_gemini(prompt):
    """
    Google Gemini APIで分析（完全無料）
    
    モデル: gemini-1.5-flash (無料)
    制限: 1日1,500リクエスト（十分）
    """
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
        }
    }
    
    print("🤖 Sending to Gemini API (FREE)...")
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    
    # テキスト抽出
    analysis = result['candidates'][0]['content']['parts'][0]['text']
    
    return analysis

def format_final_report(signals, ai_analysis):
    """最終レポート作成"""
    
    lines = []
    lines.append("="*50)
    lines.append("🤖 AI深層分析レポート (Powered by Gemini)")
    lines.append("="*50)
    lines.append(datetime.now().strftime("%Y/%m/%d %H:%M"))
    lines.append("")
    lines.append(f"📊 分析対象: {len(signals)}銘柄")
    lines.append("")
    lines.append("="*50)
    lines.append("")
    
    # AI分析結果
    lines.append(ai_analysis)
    
    lines.append("")
    lines.append("="*50)
    lines.append("⚠️  注意事項")
    lines.append("="*50)
    lines.append("- これは分析結果であり、投資助言ではありません")
    lines.append("- 最終判断はご自身で行ってください")
    lines.append("- リスク管理を徹底してください")
    lines.append("")
    lines.append("💰 コスト: ¥0 (Google Gemini Free)")
    lines.append("="*50)
    
    return "\n".join(lines)

def main():
    """メイン処理"""
    
    print("="*70)
    print("AI深層分析システム (Google Gemini - 完全無料版)")
    print("="*70)
    print()
    
    try:
        # シグナル読み込み
        print("📊 Loading signals...")
        signals = load_signals()
        print(f"✅ {len(signals)} signals loaded")
        print()
        
        # プロンプト作成
        print("📝 Creating analysis prompt...")
        prompt = create_analysis_prompt(signals)
        print(f"✅ Prompt created ({len(prompt)} chars)")
        print()
        
        # Gemini APIで分析
        print("🤖 Analyzing with Google Gemini API (FREE)...")
        ai_analysis = analyze_with_gemini(prompt)
        print("✅ Analysis complete")
        print()
        
        # レポート作成
        print("📄 Generating final report...")
        report = format_final_report(signals, ai_analysis)
        print("✅ Report generated")
        print()
        
        # 表示
        print(report)
        print()
        
        # LINE通知
        print("📱 Sending to LINE...")
        send_line(report)
        print("✅ LINE notification sent")
        print()
        
        # ファイル保存
        today = datetime.now().strftime('%Y%m%d')
        with open(f'ai_analysis_gemini_{today}.txt', 'w') as f:
            f.write(report)
        print(f"✅ Saved: ai_analysis_gemini_{today}.txt")
        
        print()
        print("💰 Cost: ¥0 (Completely FREE!)")
        
    except Exception as e:
        error_msg = f"❌ Error: {e}"
        print(error_msg)
        send_line(error_msg)
        raise
    
    print()
    print("="*70)
    print("✅ Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
