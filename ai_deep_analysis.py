#!/usr/bin/env python3
# ai_deep_analysis.py
# SENTINEL v28 signals → Gemini API 深層分析

import json
import os
import glob
import requests
from datetime import datetime

# ===== Gemini API 設定 =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# ===== LINE 設定 =====
# YAML側でどちらの名前を使っても動くように or で結合
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN") 
LINE_USER_ID = os.getenv("LINE_USER_ID")


#------------------------------------------------
# LINE 通知
#------------------------------------------------
def send_line(message: str):
    """LINE通知送信"""
    if not LINE_ACCESS_TOKEN or not LINE_USER_ID:
        print("⚠️  LINE credentials not set")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
    }

    # LINE 制限対策（5000文字）
    chunks = [message[i:i + 4800] for i in range(0, len(message), 4800)]
    for chunk in chunks:
        payload = {
            "to": LINE_USER_ID,
            "messages": [{"type": "text", "text": chunk}],
        }
        try:
            requests.post(url, headers=headers, json=payload, timeout=30)
        except Exception as e:
            print(f"⚠️  LINE error: {e}")


#------------------------------------------------
# signals 読み込み
#------------------------------------------------
def load_signals():
    """シグナルファイル読み込み"""
    print("🔍 Searching signals files...")
    print(f"📂 Current directory: {os.getcwd()}")
    
    # ファイル一覧確認
    all_files = os.listdir(".")
    print(f"📄 Files in directory: {len(all_files)}")
    
    # signals_*.json を検索
    candidates = sorted(glob.glob("signals_*.json"))
    
    if not candidates:
        print("⚠️  No signals_*.json found")
        print("   Trying today_signals.json...")
        
        if os.path.exists("today_signals.json"):
            print("✅ Found: today_signals.json")
            with open("today_signals.json", "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print("⚠️  No signals file found. Exit normally.")
            return []

    signal_file = candidates[-1]
    print(f"✅ Using signals file: {signal_file}")

    with open(signal_file, "r", encoding="utf-8") as f:
        return json.load(f)


#------------------------------------------------
# Gemini 用プロンプト生成
#------------------------------------------------
def create_analysis_prompt(signals):
    """プロンプト作成"""
    return f"""あなたは世界トップクラスの株式アナリストです。

以下は、VCPパターン検出システム「SENTINEL v28」が検出した
本日の株式シグナルです（全{len(signals)}銘柄）。

# シグナルデータ（JSON）

```json
{json.dumps(signals, indent=2, ensure_ascii=False)}
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


#------------------------------------------------
# Gemini API 呼び出し
#------------------------------------------------
def analyze_with_gemini(prompt: str) -> str:
    """Gemini APIで分析"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    
    headers = {"Content-Type": "application/json"}
    
    payload = {
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
    
    print("🤖 Sending to Gemini API...")
    
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


#------------------------------------------------
# レポート生成
#------------------------------------------------
def format_final_report(signals, ai_analysis):
    """最終レポート生成"""
    lines = []
    lines.append("=" * 60)
    lines.append("🤖 AI深層分析レポート（Powered by Gemini）")
    lines.append("=" * 60)
    lines.append(datetime.now().strftime("%Y/%m/%d %H:%M"))
    lines.append("")
    lines.append(f"📊 分析対象: {len(signals)}銘柄")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append(ai_analysis)
    lines.append("")
    lines.append("=" * 60)
    lines.append("⚠️  注意事項")
    lines.append("=" * 60)
    lines.append("- 本レポートは投資助言ではありません")
    lines.append("- 最終判断はご自身で行ってください")
    lines.append("- リスク管理を徹底してください")
    lines.append("")
    lines.append("💰 コスト: ¥0 (Google Gemini Free)")
    lines.append("=" * 60)
    
    return "\n".join(lines)


#------------------------------------------------
# メイン処理
#------------------------------------------------
def main():
    """メイン処理"""
    print("=" * 70)
    print("AI深層分析システム（Gemini - 完全無料版）")
    print("=" * 70)
    print()
    
    try:
        # シグナル読み込み
        print("📊 Loading signals...")
        signals = load_signals()
        
        if not signals:
            print("ℹ️  No signals today. Finish normally.")
            return
        
        print(f"✅ {len(signals)} signals loaded")
        print()
        
        # プロンプト作成
        print("📝 Creating analysis prompt...")
        prompt = create_analysis_prompt(signals)
        print(f"✅ Prompt created ({len(prompt)} chars)")
        print()
        
        # Gemini API で分析
        print("🤖 Analyzing with Gemini API (FREE)...")
        ai_analysis = analyze_with_gemini(prompt)
        print("✅ Analysis complete")
        print()
        
        # レポート生成
        print("📄 Generating final report...")
        report = format_final_report(signals, ai_analysis)
        print("✅ Report generated")
        print()
        
        # 表示
        print(report)
        print()
        
        # LINE 通知
        print("📱 Sending to LINE...")
        send_line(report)
        print("✅ LINE notification sent")
        print()
        
        # ファイル保存
        today = datetime.now().strftime("%Y%m%d")
        filename = f"ai_analysis_{today}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Saved: {filename}")
        print()
        
        print("💰 Cost: ¥0 (Completely FREE!)")
        
    except Exception as e:
        msg = f"❌ Error: {e}"
        print(msg)
        send_line(msg)
        raise
    
    print()
    print("=" * 70)
    print("✅ Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
