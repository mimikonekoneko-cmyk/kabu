#!/usr/bin/env python3
# save_signals_json.py
# v28のシグナルをJSON形式で保存（GitHub Actions用）

"""
v28の最後に追加するコード

これを追加することで:
1. 朝のシグナルがJSONで保存される
2. GitHub Actionsがファイルを読める
3. 夜に終値チェックできる
"""

import json
from datetime import datetime

def save_signals_to_json(passed_core, passed_secondary):
    """
    シグナルをJSON保存
    
    v28の最後に追加:
    save_signals_to_json(passed_core, passed_secondary)
    """
    
    signals = []
    
    # CORE
    for ticker, result in passed_core:
        signals.append({
            'ticker': ticker,
            'tier': 'CORE',
            'score': result['quality']['total_score'],
            'entry': result['pivot'],
            'stop': result['stop'],
            'target': result.get('target', 0),
            'shares': result.get('est_shares', 0),
            'why_now': result.get('why_now', ''),
            'sector': result.get('sector', 'Unknown')
        })
    
    # SECONDARY (TOP10)
    for ticker, result in passed_secondary[:10]:
        signals.append({
            'ticker': ticker,
            'tier': 'SECONDARY',
            'score': result['quality']['total_score'],
            'entry': result['pivot'],
            'stop': result['stop'],
            'target': result.get('target', 0),
            'shares': result.get('est_shares', 0),
            'why_now': result.get('why_now', ''),
            'sector': result.get('sector', 'Unknown')
        })
    
    # 保存
    today = datetime.now().strftime('%Y%m%d')
    filename = f"signals_{today}.json"
    
    with open(filename, 'w') as f:
        json.dump(signals, f, indent=2)
    
    # GitHub Actions用に固定名でも保存
    with open('today_signals.json', 'w') as f:
        json.dump(signals, f, indent=2)
    
    print(f"✅ Signals saved: {filename}")
    print(f"   {len(signals)} signals ({len(passed_core)} CORE, {min(10, len(passed_secondary))} SEC)")

# ===========================
# v28への統合方法
# ===========================

"""
sentinel_v28_growth.py の最後に追加:

# 既存のコード
if __name__ == "__main__":
    run_mission()

# ↓ これを追加
    
    # シグナルをJSON保存（GitHub Actions用）
    from save_signals_json import save_signals_to_json
    
    # run_mission()内でpassed_core, passed_secondaryを取得
    # グローバル変数にするか、return値で受け取る
    
    # 例:
    # passed_core, passed_secondary = run_mission()
    # save_signals_to_json(passed_core, passed_secondary)
"""

# ===========================
# 使い方
# ===========================

if __name__ == "__main__":
    print("このファイルはv28に統合して使用します")
    print()
    print("手順:")
    print("1. sentinel_v28_growth.pyを修正")
    print("2. run_mission()がpassed_core, passed_secondaryを返すように")
    print("3. 最後にsave_signals_to_json()を呼び出す")
    print()
    print("または:")
    print("save_signals_json.pyの内容を直接v28にコピペ")
