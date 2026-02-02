#!/usr/bin/env python3
# ai_deep_analysis.py
# v28のシグナルをClaude APIで深層分析

"""
フロー:
1. v28が全シグナルをJSON出力
2. Claude APIに全データを投げる
3. AIが深層分析
4. 最終推奨をLINE通知

AI分析内容:
- ニュース検索
- セクタートレンド
- 相関分析
- リスク評価
- 最終推奨（TOP3）
"""

import json
import os
import requests
from datetime import datetime
from pathlib import Path

# Claude API設定
CLAUDE_API_KEY = os.getenv('ANTHROPIC_API_KEY')
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# LINE設定
LINE_ACCESS_TOKEN = os.getenv('LINE_ACCESS_TOKEN')
LINE_USER_ID = os.getenv('LINE_USER_ID')

def send_line(message):
    """LINE通知"""
    if not LINE_ACCESS_TOKEN or not LINE_USER_ID:
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
    """
    Claude APIに投げるプロンプト作成
    
    重要: 全シグナルをJSONで渡す
    """
    
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
- 最近のニュース・材料（あれば）
- セクタートレンド
- テクニカル評価（VCPスコアの妥当性）
- リスク要因

## 2. 相関分析
- セクター集中リスク
- ポートフォリオバランス
- 分散効果

## 3. マクロ環境
- 現在の市場環境
- VIX・金利動向
- セクターローテーション

## 4. 最終推奨（TOP3）

以下の形式で：

```
【AI推奨 TOP3】

🥇 1位: [TICKER]
スコア: [v28スコア]/100
AI評価: [A+/A/A-/B+/B]
推奨理由:
- [理由1]
- [理由2]
- [理由3]

リスク:
- [リスク要因]

エントリー戦略:
- Entry: $XX.XX
- Stop: $XX.XX (-X.X%)
- Target: $XX.XX (+X.X%)

🥈 2位: ...
🥉 3位: ...
```

## 5. 回避すべき銘柄

もしあれば、理由とともに。

---

**重要**: 
- 推測ではなく、事実ベースで分析
- リスクも必ず明示
- 最終判断は人間が行うことを前提
- 投資助言ではなく、分析結果の提供

それでは分析をお願いします。
"""
    
    return prompt

def analyze_with_claude(prompt):
    """
    Claude APIで分析
    
    使用モデル: Claude Sonnet 4
    """
    
    if not CLAUDE_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    headers = {
        'x-api-key': CLAUDE_API_KEY,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }
    
    data = {
        'model': 'claude-sonnet-4-20250514',
        'max_tokens': 4000,
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }
    
    print("🤖 Sending to Claude API...")
    
    response = requests.post(
        CLAUDE_API_URL,
        headers=headers,
        json=data
    )
    
    response.raise_for_status()
    
    result = response.json()
    
    # テキスト抽出
    analysis = result['content'][0]['text']
    
    return analysis

def format_final_report(signals, ai_analysis):
    """
    最終レポート作成
    """
    
    lines = []
    lines.append("="*50)
    lines.append("🤖 AI深層分析レポート")
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
    lines.append("="*50)
    
    return "\n".join(lines)

def main():
    """
    メイン処理
    """
    
    print("="*70)
    print("AI深層分析システム")
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
        
        # Claude APIで分析
        print("🤖 Analyzing with Claude API...")
        ai_analysis = analyze_with_claude(prompt)
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
        with open(f'ai_analysis_{today}.txt', 'w') as f:
            f.write(report)
        print(f"✅ Saved: ai_analysis_{today}.txt")
        
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
