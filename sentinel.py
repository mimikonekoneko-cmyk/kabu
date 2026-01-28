import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
import time

# --- CONFIG ---
ACCESSTOKEN = os.getenv("LINECHANNELACCESSTOKEN")
USERID = os.getenv("LINEUSER_ID")
BUDGET_JPY = 350000 

# --- テクニカル / リスク管理パラメータ ---
MA_SHORT, MA_LONG = 50, 200
MIN_SCORE = 85
MAX_NOTIFICATIONS = 8
ATRSTOP_MULT = 2.0  # 損切りはATRの2倍離す

# セクター別・利確倍率
AGGRESSIVE_SECTORS = [
    'Semi', 'AI', 'Soft', 'Sec', 'EV', 'Crypto', 
    'Cloud', 'Ad', 'Service', 'Platform', 'Bet'
]

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

# --- ヘルパー関数 ---

def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 155.0
    except: return 155.0

def check_market_trend():
    """市場全体の健全性チェック (SPYがMA200より上か)"""
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        if spy.empty or len(spy) < 200: return True, "Data Limited"
        c = spy['Close'].squeeze()
        cur = float(c.iloc[-1])
        ma200 = float(c.rolling(200).mean().iloc[-1])
        return (True, "Bull Market") if cur > ma200 else (False, f"Bear Market (${cur:.0f}<MA200)")
    except: return True, "Check Skipped"

def is_earnings_near(ticker):
    """決算5日前後は回避"""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty): return False
        
        # 構造に合わせた取得
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            date_val = cal['Earnings Date'][0]
        else:
            date_val = cal.iloc[0,0]
            
        days = (pd.to_datetime(date_val).date() - datetime.now().date()).days
        return abs(days) <= 5
    except: return False

def sector_is_strong(sector):
    """セクターETFが上昇トレンドか"""
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf or etf == 'CRYPTO': return True
        df = yf.download(etf, period="250d", progress=False)
        c = df['Close'].squeeze()
        ma200 = c.rolling(200).mean()
        return ma200.iloc[-1] > ma200.iloc[-10]
    except: return True

# --- 分析エンジン ---

class StrategicAnalyzer:
    @staticmethod
    def analyze_ticker(t, df, sector, max_price_usd):
        if len(df) < MA_LONG: return None
        try:
            # yfinanceのマルチインデックス/シングルインデックス両対応
            c = df['Close'].squeeze()
            h = df['High'].squeeze()
            l = df['Low'].squeeze()
            v = df['Volume'].squeeze()
        except: return None

        current_price = float(c.iloc[-1])
        if current_price > max_price_usd: return None

        # トレンドフィルター
        ma50 = c.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = c.rolling(MA_LONG).mean().iloc[-1]
        if not (current_price > ma50 > ma200): return None

        # ATR & Tightness (VCP)
        tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        if atr14 == 0 or np.isnan(atr14): return None
        
        # VCP: 直近5日の高低差がATRに対してタイトか
        tightness = float((h.iloc[-5:].max() - l.iloc[-5:].min()) / atr14)
        if tightness > 3.0: return None

        # Speed (Velocity) 判定
        ma5 = c.rolling(5).mean().iloc[-1]
        ma20 = c.rolling(20).mean().iloc[-1]
        velocity = "HIGH" if ma5 > ma20 * 1.01 else "SLOW"

        # スコアリング
        score = 65
        reasons = ["基礎65"]
        
        if tightness < 1.5:
            score += 20; reasons.append("VCPタイト+20")
        elif tightness < 2.0:
            score += 10; reasons.append("VCP良好+10")
            
        vol_avg = v.rolling(50).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        if 0.7 <= vol_ratio <= 1.1:
            score += 15; reasons.append("売り枯れ+15")
            
        if v.iloc[-3:].max() > vol_avg * 1.5:
            score += 10; reasons.append("買い集め+10")

        # 戦略的出口設定
        reward_mult = 3.0 if sector in AGGRESSIVE_SECTORS else 1.8
        pivot = h.iloc[-5:].max() * 1.002
        stop_dist = atr14 * ATR_STOP_MULT
        stop_loss = pivot - stop_dist
        target = pivot + (stop_dist * reward_mult)

        return {
            "score": score, "reasons": " ".join(reasons),
            "price": current_price, "pivot": pivot,
            "stop": stop_loss, "target": target, "sector": sector, "velocity": velocity
        }

def send_line(msg):
    if not ACCESSTOKEN or not USERID:
        print("\n--- LINEメッセージ (未設定) ---\n", msg)
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESSTOKEN}"}
    payload = {"to": USERID, "messages": [{"type": "text", "text": msg}]}
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except:
        print("LINE送信エラー")

def run_mission():
    print(f"🛡️ Sentinel v21.1 - 起動中...")
    
    market_ok, market_status = check_market_trend()
    if not market_ok:
        msg = f"🛑 Sentinel: 市場環境悪化により待機\nStatus: {market_status}"
        print(msg); send_line(msg); return

    fx = get_current_fx_rate()
    max_p = (BUDGET_JPY / fx) * 0.9
    
    print(f"🛰️ 銘柄スキャン開始... FX: {fx:.2f}")
    ticker_list = list(TICKERS.keys())
    alldata = yf.download(ticker_list, period="300d", progress=False, group_by='ticker')
    
    results = []
    for t, sec in TICKERS.items():
        if is_earnings_near(t): continue
        if not sector_is_strong(sec): continue
        
        try:
            # yfinanceの一括ダウンロードデータの取得
            dft = alldata[t] if len(ticker_list) > 1 else alldata
            res = StrategicAnalyzer.analyze_ticker(t, dft, sec, max_p)
            if res and res['score'] >= MIN_SCORE:
                results.append((t, res))
        except Exception as e:
            continue
    
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    report = [
        f"🛡️ Sentinel v21.1",
        f"📅 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        f"📊 Market: {market_status}",
        f"💵 $1 = {fx:.2f}円",
        "─" * 15
    ]
    
    if not results:
        report.append("⚠️ 現在、射程圏内に銘柄はありません。")
    
    for i, (t, r) in enumerate(results, 1):
        loss_p = (1 - r['stop'] / r['pivot']) * 100
        gain_p = (r['target'] / r['pivot'] - 1) * 100
        report.append(
            f"[{i}] {t} ({r['sector']}) {r['score']}点\n"
            f" └ 根拠: {r['reasons']}\n"
            f"現: ${r['price']:.2f} 入: ${r['pivot']:.2f}\n"
            f"止: ${r['stop']:.2f} (-{loss_p:.1f}%) 目: ${r['target']:.2f} (+{gain_p:.1f}%)\n"
            f"⚡ Speed: {r['velocity']}"
        )

    full_msg = "\n".join(report)
    print("\n" + full_msg)
    send_line(full_msg)

if __name__ == "__main__":
    run_mission()
