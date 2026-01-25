import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime

# --- CONFIG (GitHub Secretsから読み込み) ---
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# --- 予算設定 ---
BUDGET_JPY = 200000      # 総予算 20万円

# --- テクニカルパラメータ ---
MA_SHORT, MA_LONG = 50, 200
MIN_SCORE = 85
MAX_NOTIFICATIONS = 8

TICKERS = {
    'NVDA':'AI','AVGO':'Semi','ARM':'Semi','MU':'Semi','AMD':'Semi','SMCI':'AI','TSM':'Semi','ASML':'Semi',
    'AAPL':'Device','MSFT':'Cloud','GOOGL':'Ad','META':'Ad','AMZN':'Retail','TSLA':'EV','NFLX':'Service',
    'PLTR':'AI','PANW':'Sec','CRWD':'Sec','NET':'Sec','NOW':'Soft','CRM':'Soft','TEAM':'Soft','ADBE':'Soft',
    'COST':'Retail','WMT':'Retail','TJX':'Retail','ELF':'Cons','PEP':'Cons','KO':'Cons','PG':'Cons',
    'V':'Fin','MA':'Fin','JPM':'Bank','GS':'Bank','AXP':'Fin','BLK':'Fin','MS':'Bank','COIN':'Crypto',
    'LLY':'Bio','UNH':'Health','ABBV':'Bio','ISRG':'Health','VRTX':'Bio',
    'GE':'Ind','CAT':'Ind','DE':'Ind','XOM':'Energy','CVX':'Energy','MPC':'Energy','BA':'Ind',
    'UBER':'Platform','BKNG':'Travel','ABNB':'Travel','DKNG':'Bet','LULU':'Cons','VRT':'Power'
}

def get_current_fx_rate():
    """USD/JPYの現在レートを取得"""
    try:
        # yfinanceでドル円レートを取得
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 155.0  # 取得失敗時のフォールバック
    except:
        return 155.0

class StrategicAnalyzer:
    @staticmethod
    def analyze_ticker(t, df, sector, max_price_usd):
        if len(df) < MA_LONG: return None
        
        c = df['Close']
        h, l, v = df['High'], df['Low'], df['Volume']
        current_price = float(c.iloc[-1])
        
        # 🟢 リアルタイム予算フィルター
        if current_price > max_price_usd:
            return None
        
        # トレンド分析
        ma50 = c.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = c.rolling(MA_LONG).mean().iloc[-1]
        ma200_prev = c.rolling(MA_LONG).mean().iloc[-10]
        
        if not (current_price > ma50 > ma200 and ma200 > ma200_prev):
            return None

        # 収縮度 (Tightness)
        tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        range_5d = h.iloc[-5:].max() - l.iloc[-5:].min()
        tightness = float(range_5d / atr14)
        if tightness > 3.0: return None
        
        # スコアリング
        vol_avg = v.rolling(50).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / vol_avg
        
        score = 60
        if tightness < 1.5: score += 25
        elif tightness < 2.0: score += 15
        if 0.7 <= vol_ratio <= 1.0: score += 15
        
        pivot = h.iloc[-5:].max() * 1.002
        stop_loss = pivot * 0.93
        target = pivot * 1.15
        
        return {
            "score": score, "price": current_price, "pivot": pivot,
            "stop": stop_loss, "target": target,
            "tightness": tightness, "vol_ratio": vol_ratio, "sector": sector
        }

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID:
        print("⚠️ LINE設定がありません")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages": [{"type": "text", "text": msg}]}
    requests.post(url, headers=headers, json=payload)

def run_mission():
    # 🛰️ 最新レートを取得
    current_fx = get_current_fx_rate()
    # 予算20万円の90%を、1銘柄あたりの上限（ドル）とする
    max_price_usd = (BUDGET_JPY / current_fx) * 0.9

    print(f"🛰️ 偵察開始... (FX: {current_fx:.2f}円, 予算上限: ${max_price_usd:.1f})")
    all_data = yf.download(list(TICKERS.keys()), period="300d", progress=False, group_by='ticker')
    
    results = []
    for t, sec in TICKERS.items():
        try:
            res = StrategicAnalyzer.analyze_ticker(t, all_data[t], sec, max_price_usd)
            if res and res['score'] >= MIN_SCORE:
                results.append((t, res))
        except: continue
    
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    # レポート構築
    report = [
        f"🛡️ Sentinel v17.2",
        f"📅 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        f"💵 $1 = {current_fx:.2f}円",
        f"💰 予算内上限: ${max_price_usd:.1f}",
        "─" * 15
    ]
    
    if not results:
        report.append("⚠️ 条件に合う銘柄なし。待機。")
    for i, (t, r) in enumerate(results, 1):
        report.append(f"[{i}] {t} ({r['sector']}) {r['score']}点\n現: ${r['price']:.2f} / 入: ${r['pivot']:.2f}\n止: ${r['stop']:.2f} / 目: ${r['target']:.2f}\n")

    full_msg = "\n".join(report)
    print(full_msg)
    send_line(full_msg)

if __name__ == "__main__":
    run_mission()
