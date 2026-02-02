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
    "gemini-1.5-pro:generateContent"
)

# ===== LINE 設定 =====
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")


# -------------------------------------------------
# LINE 通知
# -------------------------------------------------
def send_line(message: str):
    if not LINE_ACCESS_TOKEN or not LINE_USER_ID:
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
    }

    # LINE 制限対策
    chunks = [message[i:i + 4800] for i in range(0, len(message), 4800)]
    for chunk in chunks:
        payload = {
            "to": LINE_USER_ID,
            "messages": [{"type": "text", "text": chunk}],
        }
        requests.post(url, headers=headers, json=payload, timeout=30)


# -------------------------------------------------
# signals 読み込み
# -------------------------------------------------
def load_signals():
    print("🔍 Searching signals files...")
    print("📂 CWD files:", os.listdir("."))

    candidates = sorted(glob.glob("signals_*.json"))

    if not candidates:
        print("⚠️ No signals file found. Exit normally.")
        return []

    signal_file = candidates[-1]
    print(f"✅ Using signals file: {signal_file}")

    with open(signal_file, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------
# Gemini 用プロンプト生成
# -------------------------------------------------
def create_analysis_prompt(signals):
    return f"""
あなたは世界トップクラスの株式アナリストです。

以下は、VCPパターン検出システム「SENTINEL v28」が検出した
本日の株式シグナルです（全{len(signals)}銘柄）。

```json
{json.dumps(signals, indent=2, ensure_ascii=False)}
【分析観点】
ニュース・材料
セクタートレンド
テクニカル評価
リスク評価
最終推奨 TOP3
【条件】
事実ベースで分析
リスクを必ず明示
投資助言ではなく分析結果として出力 """
-------------------------------------------------
Gemini API 呼び出し
-------------------------------------------------
def analyze_with_gemini(prompt: str) -> str: if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not set")
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
        "temperature": 0.3,
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
-------------------------------------------------
レポート整形
-------------------------------------------------
def format_final_report(signals, ai_analysis): lines = [] lines.append("=" * 60) lines.append("🤖 AI深層分析レポート（Gemini）") lines.append("=" * 60) lines.append(datetime.now().strftime("%Y/%m/%d %H:%M")) lines.append("") lines.append(f"📊 分析対象: {len(signals)}銘柄") lines.append("") lines.append(ai_analysis) lines.append("") lines.append("=" * 60) lines.append("⚠️ 注意事項") lines.append("- 本レポートは投資助言ではありません") lines.append("- 最終判断はご自身で行ってください") lines.append("=" * 60) return "\n".join(lines)
-------------------------------------------------
main
-------------------------------------------------
def main(): print("=" * 70) print("AI深層分析システム（Gemini）") print("=" * 70)
try:
    print("📊 Loading signals...")
    signals = load_signals()

    if not signals:
        print("ℹ️ No signals today. Finish.")
        return

    print(f"✅ {len(signals)} signals loaded")

    print("📝 Creating prompt...")
    prompt = create_analysis_prompt(signals)

    print("🤖 Analyzing with Gemini...")
    ai_analysis = analyze_with_gemini(prompt)

    print("📄 Generating report...")
    report = format_final_report(signals, ai_analysis)

    print(report)

    print("📱 Sending LINE...")
    send_line(report)

    today = datetime.now().strftime("%Y%m%d")
    filename = f"ai_analysis_{today}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Saved: {filename}")

except Exception as e:
    msg = f"❌ Error: {e}"
    print(msg)
    send_line(msg)
    raise

print("=" * 70)
print("✅ Complete")
print("=" * 70)
if name == "main": main()