import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime
import time

# --- CONFIG (GitHub Secretsから読み込み) ---
ACCESSTOKEN = os.getenv("LINECHANNELACCESSTOKEN")
USERID = os.getenv("LINEUSER_ID")
BUDGET_JPY = 350000 

# --- パラメータ ---
MA_SHORT, MA_LONG = 50, 200
MIN_SCORE = 85
MAX_NOTIFICATIONS = 8
ATR_STOP_MULT = 2.0

AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'EV', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet']

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

SECTOR_ETF = {
    'Energy': 'XLE', 'Semi': 'SOXX', 'Bank': 'XLF', 'Retail': 'XRT',
    'Soft': 'IGV', 'AI': 'QQQ', 'Fin': 'VFH', 'Device': 'QQQ',
    'Cloud': 'QQQ', 'Ad': 'QQQ', 'Service': 'QQQ', 'Sec': 'HACK',
    'Cons': 'XLP', 'Bio': 'IBB', 'Health': 'XLV', 'Ind': 'XLI',
    'EV': 'IDRV', 'Crypto': 'CRYPTO', 'Power': 'PWR'
}

# --- 機能関数 ---

def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 155.0
    except: return 155.0

def check_market_trend():
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        c = spy['Close'].squeeze()
        ma200 = c.rolling(200).mean().iloc[-1]
        return (c.iloc[-1] > ma200, "Bull" if c.iloc[-1] > ma200 else "Bear")
    except: return (True, "Unknown")

def is_earnings_near(ticker):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        date_val = cal['Earnings Date'][0] if isinstance(cal, dict) else cal.iloc[0,0]
        return abs((pd.to_datetime(date_val).date() - datetime.now().date()).days) <= 5
    except: return False

# --- バックテストエンジン ---

def simulate_past_performance(df, pivot, stop, target):
    """
    直近100日の中で、現在のロジックに近いエントリーがあった場合の成功率を検証
    """
    try:
        c = df['Close'].squeeze()
        h = df['High'].squeeze()
        l = df['Low'].squeeze()
        
        # 過去100日でエントリーポイント(pivot)を超えた回数と、その後の結果を簡易シミュレーション
        success, failure = 0, 0
        for i in range(len(df)-20, len(df)-5): # 直近の数サンプルを抽出
            if h.iloc[i] >= pivot:
                # エントリー後5日間でTargetかStopか
                for j in range(1, 6):
                    if i+j >= len(df): break
                    if h.iloc[i+j] >= target: success += 1; break
                    if l.iloc[i+j] <= stop: failure += 1; break
        
        total = success + failure
        return f"勝率 {int(success/total*100)}%" if total > 0 else "データ不足"
    except: return "検証不能"

# --- 分析クラス ---

class StrategicAnalyzer:
    @staticmethod
    def analyze_ticker(t, df, sector, max_p):
        if len(df) < MA_LONG: return None
        c, h, l, v = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
        
        curr_p = float(c.iloc[-1])
        if curr_p > max_p: return None

        ma50, ma200 = c.rolling(MA_SHORT).mean().iloc[-1], c.rolling(MA_LONG).mean().iloc[-1]
        if not (curr_p > ma50 > ma200): return None

        tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        tightness = float((h.iloc[-5:].max() - l.iloc[-5:].min()) / atr14)
        if tightness > 3.0: return None

        score = 65
        reasons = ["基礎65"]
        if tightness < 1.5: score += 20; reasons.append("VCPタイト+20")
        vol_avg = v.rolling(50).mean().iloc[-1]
        if 0.7 <= v.iloc[-1]/vol_avg <= 1.1: score += 15; reasons.append("売り枯れ+15")

        reward = 3.0 if sector in AGGRESSIVE_SECTORS else 1.8
        pivot = h.iloc[-5:].max() * 1.002
        stop = pivot - (atr14 * ATR_STOP_MULT)
        target = pivot + ((pivot - stop) * reward)

        # バックテスト実行
        bt_stat = simulate_past_performance(df, pivot, stop, target)

        return {
            "score": score, "reasons": " ".join(reasons), "price": curr_p, 
            "pivot": pivot, "stop": stop, "target": target, "sector": sector, 
            "velocity": "HIGH" if c.rolling(5).mean().iloc[-1] > c.rolling(20).mean().iloc[-1] else "SLOW",
            "bt": bt_stat
        }

# --- 実行メイン ---

def send_line(msg):
    if not ACCESSTOKEN: print(msg); return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESSTOKEN}"}
    payload = {"to": USERID, "messages": [{"type": "text", "text": msg}]}
    requests.post(url, headers=headers, json=payload)

def run_mission():
    is_bull, m_status = check_market_trend()
    if not is_bull:
        send_line(f"🛑 Sentinel待機: Market {m_status}"); return

    fx = get_current_fx_rate()
    max_p = (BUDGET_JPY / fx) * 0.9
    all_data = yf.download(list(TICKERS.keys()), period="300d", progress=False, group_by='ticker')
    
    results = []
    for t, sec in TICKERS.items():
        if is_earnings_near(t): continue
        try:
            res = StrategicAnalyzer.analyze_ticker(t, all_data[t], sec, max_p)
            if res and res['score'] >= MIN_SCORE: results.append((t, res))
        except: continue
    
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    report = [f"🛡️ Sentinel v21.1 BT-Exp\n📊 Market: {m_status}\n💵 $1 = {fx:.2f}円\n" + "─"*15]
    
    for i, (t, r) in enumerate(results[:MAX_NOTIFICATIONS], 1):
        lp, gp = (1 - r['stop']/r['pivot'])*100, (r['target']/r['pivot']-1)*100
        report.append(f"[{i}] {t} ({r['sector']}) {r['score']}点\n └ {r['reasons']}\n期待値: {r['bt']}\n入: ${r['pivot']:.2f}\n止: ${r['stop']:.2f} (-{lp:.1f}%)\n目: ${r['target']:.2f} (+{gp:.1f}%)")

    send_line("\n".join(report))

if __name__ == "__main__":
    run_mission()
