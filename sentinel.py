import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os

# --- Messaging API CONFIG ---
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# --- ロジック定数 ---
MA_SHORT, MA_LONG = 50, 200
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
    def analyze_ticker(t, df, sector):
        if len(df) < MA_LONG: return None
        
        c = df['Close']
        h, l, v = df['High'], df['Low'], df['Volume']
        
        # 1. 守備力の判定 (2022年回避ロジック)
        ma50 = c.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = c.rolling(MA_LONG).mean().iloc[-1]
        ma200_prev = c.rolling(MA_LONG).mean().iloc[-10] # 2週間前比較
        
        trend_ok = c.iloc[-1] > ma50 and c.iloc[-1] > ma200 and ma200 > ma200_prev
        if not trend_ok: return None

        # 2. 攻撃力の判定 (VCPロジック)
        # タイトネス計算
        tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        tightness = (float(h.iloc[-5:].max() - l.iloc[-5:].min())) / tr.rolling(14).mean().iloc[-1]
        
        # 出来高確認
        vol_avg = v.rolling(50).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / vol_avg
        
        # INの目安 (直近5日の高値 + α)
        pivot = h.iloc[-5:].max() * 1.002 
        
        # スコアリング (最大100点)
        score = 60 # 基本点
        if tightness < 2.0: score += 20
        elif tightness < 3.0: score += 10
        if vol_ratio > 1.2: score += 20
        elif vol_ratio > 1.0: score += 10

        tier = 0
        if trend_ok and tightness <= TIGHTNESS_TIER1 and vol_ratio >= VOL_SPIKE_RATIO: tier = 1
        elif trend_ok and tightness <= TIGHTNESS_TIER2: tier = 2
        
        if tier == 0: return None

        return {
            "tier": tier, "score": score, "pivot": pivot, 
            "tightness": tightness, "vol_ratio": vol_ratio, "sector": sector
        }

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID: return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages": [{"type": "text", "text": msg}]}
    requests.post(url, headers=headers, json=payload)

def run_mission():
    all_data = yf.download(list(TICKERS.keys()), period="300d", progress=False, group_by='ticker')
    results = []
    
    for t, sec in TICKERS.items():
        res = StrategicAnalyzer.analyze_ticker(t, all_data[t], sec)
        if res: results.append((t, res))
    
    # スコア順にソート
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    
    report = ["🛡️ Sentinel v16.0 偵察報告", "----------------"]
    
    if not results:
        report.append("現在、112%ロジックに合致する銘柄はありません。2022年のような地固めを待つ時期です。")
    else:
        for t, r in results:
            t_icon = "🔥" if r['tier'] == 1 else "⚡"
            msg = f"{t_icon}{t} ({r['sector']})\n"
            msg += f" ├ 推奨スコア: {r['score']}点\n"
            msg += f" ├ IN目安: ${r['pivot']:.2f}超\n"
            msg += f" └ 根拠: 収縮度{r['tightness']:.1f} / 出来高{r['vol_ratio']:.1f}倍\n"
            if r['tightness'] < 2.5: msg += "   (※爆発寸前の非常にタイトな形状)"
            report.append(msg)

    send_line("\n".join(report))

if __name__ == "__main__":
    run_mission()
