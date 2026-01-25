import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime

# --- Messaging API CONFIG ---
# GitHubのSecretsに設定した変数から読み込みます
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# --- 戦略パラメータ (2022年有益データ反映済) ---
MA_SHORT = 50   
MA_LONG = 200   
VOL_SPIKE_RATIO = 1.15
TIGHTNESS_TIER1 = 2.5
TIGHTNESS_TIER2 = 3.5

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

class StrategicAnalyzer:
    @staticmethod
    def get_market_weather():
        try:
            m_data = yf.download(["SPY", "^VIX", "^TNX"], period="200d", progress=False)
            vix = m_data['Close']['^VIX'].iloc[-1]
            tnx = m_data['Close']['^TNX'].iloc[-1]
            spy = m_data['Close']['SPY']
            spy_ma200 = spy.rolling(200).mean().iloc[-1]
            spy_now = spy.iloc[-1]
            dist = (spy_now - spy_ma200) / spy_ma200 * 100
            if spy_now > spy_ma200 and vix < 22:
                return "☀️快晴", "積極参入。MA200上の強いトレンドです。"
            elif spy_now < spy_ma200 or vix > 28:
                return "⛈️荒天", "全休推奨。資金を守るのが今の仕事です。"
            else:
                return "🌥️曇天", "慎重に。個別銘柄のMA50保持を確認。"
        except: return "❔不明", "データ不足"

    @staticmethod
    def evaluate_tier(df):
        if len(df) < MA_LONG + 5: return 0, ["データ不足"]
        c = df['Close']
        ma50 = c.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = c.rolling(MA_LONG).mean().iloc[-1]
        ma200_prev = c.rolling(MA_LONG).mean().iloc[-5]
        if not (c.iloc[-1] > ma50 and c.iloc[-1] > ma200 and ma200 > ma200_prev):
            return 0, ["トレンドNG"]
        tr = pd.concat([(df['High']-df['Low']), (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        tightness = (float(df['High'].iloc[-5:].max() - df['Low'].iloc[-5:].min())) / tr.rolling(14).mean().iloc[-1]
        vol_spike = df['Volume'].iloc[-1] > df['Volume'].rolling(50).mean().iloc[-1] * VOL_SPIKE_RATIO
        dist_to_high = (df['High'].rolling(20).max().iloc[-1] - c.iloc[-1]) / c.iloc[-1]
        if tightness <= TIGHTNESS_TIER1 and vol_spike: return 1, ["Tier1:王道VCP"]
        if tightness <= TIGHTNESS_TIER2 and dist_to_high < 0.05: return 2, ["Tier2:前兆"]
        return 0, [f"条件未達(T:{tightness:.1f})"]

def send_line_message(msg):
    """Messaging APIを使用したプッシュ通知"""
    if not ACCESS_TOKEN or not USER_ID: return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    payload = {
        "to": USER_ID,
        "messages": [{"type": "text", "text": msg}]
    }
    try:
        requests.post(url, headers=headers, json=payload)
    except: pass

def run_mission():
    weather, advice = StrategicAnalyzer.get_market_weather()
    report = [f"🛡️ Sentinel 戦略報告\n天気: {weather}\n助言: {advice}\n" + "-"*15]
    hits = {1: [], 2: []}; rejects = {}
    all_data = yf.download(list(TICKERS.keys()), period="300d", progress=False, group_by='ticker')
    for t, sec in TICKERS.items():
        df = all_data[t]
        if df.empty or len(df) < 200: continue
        tier, reasons = StrategicAnalyzer.evaluate_tier(df)
        if tier > 0: hits[tier].append(f"{t}({sec})")
        else: rejects[reasons[0]] = rejects.get(reasons[0], 0) + 1
    report.append(f"🔥Tier1: {', '.join(hits[1]) if hits[1] else 'なし'}")
    report.append(f"⚡Tier2: {', '.join(hits[2]) if hits[2] else 'なし'}")
    send_line_message("\n".join(report))

if __name__ == "__main__":
    run_mission()
