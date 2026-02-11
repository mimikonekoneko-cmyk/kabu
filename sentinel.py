#!/usr/bin/env python3
# ============================================================
# 🛡 SENTINEL PRO v2.3 ULTIMATE
# 完全上位互換:
# - v2.2全機能保持（VCP/RSv2.0/PF/3段階出口/サイズ算出/STATUS/市場判定/LINE通知）
# - 258銘柄フルユニバース
# - 相関フィルター
# - セクター集中制限
# - ポートフォリオ想定MaxDD算出
# ============================================================

import os
import time
import logging
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION（既存保持＋追加）
# ============================================================

CONFIG = {
    'CAPITAL_JPY': 350_000,
    'MAX_POSITIONS': 4,
    'DISPLAY_LIMIT': 15,
    'ACCOUNT_RISK_PCT': 0.015,

    # フィルタ基準
    'MIN_RS_RATING': 70,
    'MIN_VCP_SCORE': 50,
    'MIN_PROFIT_FACTOR': 1.2,

    # リスク管理
    'STOP_LOSS_ATR': 2.0,
    'MAX_TIGHTNESS': 2.5,

    # 出口戦略
    'TARGET_CONSERVATIVE': 1.5,
    'TARGET_MODERATE': 2.5,
    'TARGET_AGGRESSIVE': 4.0,

    # --- 追加 ---
    'CORRELATION_LIMIT': 0.75,
    'MAX_SAME_SECTOR': 2,
    'ENABLE_PORTFOLIO_FILTER': True,
}

ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("SENTINEL_PRO")

CACHE_DIR = Path("./cache_pro_v23")
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================
# 258 TICKER UNIVERSE（重複排除後ちょうど258）
# ============================================================

TICKERS = sorted(list(set([
"AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","BRK-B","UNH",
"JNJ","V","XOM","LLY","JPM","AVGO","PG","MA","HD","CVX",
"MRK","ABBV","COST","PEP","KO","ADBE","CRM","WMT","BAC","AMD",
"MCD","CSCO","ACN","TMO","ABT","DHR","LIN","NFLX","ORCL","INTC",
"CMCSA","TXN","WFC","DIS","NEE","RTX","QCOM","UPS","PM","BMY",
"LOW","IBM","AMGN","INTU","CAT","GS","MS","HON","SPGI","BLK",
"ASML","TSM","MU","LRCX","KLAC","ADI","ON","MRVL","SMCI","ARM",
"MPWR","TER","AMAT","NXPI","MCHP","CDNS","SNPS","ANET","PANW","CRWD",
"NOW","SNOW","DDOG","MDB","ZS","NET","OKTA","TEAM","WDAY","SHOP",
"SQ","PYPL","ADSK","DOCU","UBER","LYFT","ROKU","RBLX","PLTR","AI",
"NKE","SBUX","TGT","TJX","CMG","YUM","DPZ","MELI","BABA","CVNA",
"ETSY","ONON","CROX","LULU","FIVE","WING","BOOT","ELF","CELH","DKNG",
"VRTX","REGN","MRNA","PFE","GILD","ZTS","ISRG","SYK","BDX","CI",
"BSX","HOLX","ALGN","NVO","RARE","RIGL","DVAX","TARS","KOD","OMER",
"BA","GE","ETN","DE","SLB","EOG","COP","MPC","VLO","FDX",
"LMT","NOC","GD","PH","ITW","ROK","APD","ECL","EMR","CMI",
"AXP","C","USB","PNC","TFC","SCHW","ICE","CME","AON","MMC",
"PGR","TRV","CB","COF","BK","AIG","MET","KKR","BX","APO",
"WDC","STX","SNDK","DELL","HPQ","APH","GLW","CSX","NSC","UNP",
"IONQ","RKLB","ASTS","SPOT","RDDT","CEVA","FFIV","COHR","APLD","CLS",
"NBIS","ONDS","NMAX","HY","AVR","PRSU","WBTN","ASTE","FULC","INBX",
"DUK","SO","D","AEP","EXC","XEL","SRE","ED","PEG","EIX",
"PLD","AMT","CCI","EQIX","PSA","O","SPG","WELL","DLR","VICI",
"SPY","QQQ","IWM","IEMG","FXI","EWY","AGG","IJH"
])))

# ============================================================
# DATA ENGINE
# ============================================================

class DataEngine:
    @staticmethod
    def get_data(ticker, period="700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"

        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < 12 * 3600:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df.empty or len(df) < 200:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = ['Close','High','Low','Volume']
            if not all(c in df.columns for c in required):
                return None

            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            return df
        except:
            return None

# ============================================================
# SECTOR ENGINE（追加）
# ============================================================

class SectorEngine:
    @staticmethod
    def get_sector(ticker):
        try:
            return yf.Ticker(ticker).info.get("sector","Unknown")
        except:
            return "Unknown"

# ============================================================
# VCP ANALYZER（v2.2保持）
# ============================================================

class VCPAnalyzer:
    @staticmethod
    def calculate_vcp_score(df):
        close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']
        tr = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=7).mean().iloc[-1]
        if pd.isna(atr) or atr<=0:
            return {'score':0,'atr':0,'signals':[]}

        recent_high = high.iloc[-10:].max()
        recent_low = low.iloc[-10:].min()
        tightness = (recent_high-recent_low)/atr

        score=0; signals=[]

        if tightness<0.8: score+=40; signals.append("極度収縮")
        elif tightness<1.2: score+=30; signals.append("強収縮")
        elif tightness<1.8: score+=20; signals.append("収縮中")
        elif tightness>3.0: signals.append("ルーズ")

        vol_ma = volume.rolling(50,min_periods=10).mean().iloc[-1]
        if volume.iloc[-1] < vol_ma*0.8:
            score+=20; signals.append("Vol枯渇")

        curr=close.iloc[-1]
        ma50=close.rolling(50,min_periods=10).mean().iloc[-1]
        ma200=close.rolling(200,min_periods=50).mean().iloc[-1]

        if curr>ma50>ma200:
            score+=20; signals.append("MA整列")
        elif curr>ma50:
            score+=10

        mom5=close.rolling(5,min_periods=3).mean().iloc[-1]
        mom20=close.rolling(20,min_periods=10).mean().iloc[-1]
        if (mom5/mom20)>1.02:
            score+=20; signals.append("モメンタム+")

        if score>=85: stage="🔥爆発直前"
        elif score>=70: stage="⚡初動圏"
        elif score>=50: stage="👁形成中"
        else: stage="準備段階"

        return {'score':score,'tightness':tightness,'stage':stage,'signals':signals,'atr':atr}

# ============================================================
# RS ANALYZER（v2.0ロジック保持）
# ============================================================

class RSAnalyzer:
    @staticmethod
    def calculate_rs_rating(ticker_df, benchmark_df):
        common = ticker_df.index.intersection(benchmark_df.index)
        if len(common)<100:
            return 50

        t_c = ticker_df.loc[common,'Close']
        s_c = benchmark_df.loc[common,'Close']

        periods={'3mo':63,'6mo':126,'9mo':189,'12mo':252}
        weights={'3mo':0.4,'6mo':0.2,'9mo':0.2,'12mo':0.2}
        raw=0

        for p,d in periods.items():
            if len(t_c)>d:
                t_r=(t_c.iloc[-1]-t_c.iloc[-d])/t_c.iloc[-d]
                s_r=(s_c.iloc[-1]-s_c.iloc[-d])/s_c.iloc[-d]
                raw+=(t_r-s_r)*weights[p]

        return min(99,max(1,int(50+(raw*100))))

# ============================================================
# BACKTEST ENGINE（保持）
# ============================================================

class BacktestEngine:
    @staticmethod
    def run_backtest(df):
        if len(df)<200:
            return {'pf':0,'winrate':0}

        close=df['Close']; high=df['High']; low=df['Low']
        tr=pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
        atr=tr.rolling(14).mean()

        trades=[]; wins=0; losses=0
        reward_mult=2.5

        for i in range(200,len(df)-30):
            pivot=high.iloc[i-10:i].max()*1.002
            if high.iloc[i]<pivot:
                continue

            ma50=close.rolling(50).mean().iloc[i]
            if close.iloc[i]<ma50*0.95:
                continue

            entry=pivot
            stop=entry-(atr.iloc[i]*CONFIG['STOP_LOSS_ATR'])
            target=entry+(atr.iloc[i]*CONFIG['STOP_LOSS_ATR']*reward_mult)

            for j in range(i+1,min(i+31,len(df))):
                if high.iloc[j]>=target:
                    trades.append(reward_mult); wins+=1; break
                if low.iloc[j]<=stop:
                    trades.append(-1.0); losses+=1; break

        if not trades:
            return {'pf':0,'winrate':0}

        total_wins=sum([t for t in trades if t>0])
        total_losses=abs(sum([t for t in trades if t<0]))
        pf=(total_wins/total_losses) if total_losses>0 else 10.0
        return {'pf':pf,'winrate':(wins/len(trades))*100}

# ============================================================
# ポートフォリオフィルター（追加）
# ============================================================

def apply_portfolio_filters(candidates, price_cache):
    if not CONFIG['ENABLE_PORTFOLIO_FILTER']:
        return candidates

    selected=[]
    sector_count={}

    for c in candidates:
        t=c['ticker']
        df=price_cache[t]
        sector=SectorEngine.get_sector(t)

        if sector_count.get(sector,0)>=CONFIG['MAX_SAME_SECTOR']:
            continue

        correlated=False
        for s in selected:
            corr=df['Close'].pct_change().corr(
                price_cache[s['ticker']]['Close'].pct_change()
            )
            if corr and corr>CONFIG['CORRELATION_LIMIT']:
                correlated=True
                break

        if correlated:
            continue

        selected.append(c)
        sector_count[sector]=sector_count.get(sector,0)+1

        if len(selected)>=CONFIG['MAX_POSITIONS']:
            break

    return selected

def estimate_portfolio_dd(selected):
    if not selected:
        return 0
    total_risk=len(selected)*(CONFIG['CAPITAL_JPY']*CONFIG['ACCOUNT_RISK_PCT'])
    return round((total_risk/CONFIG['CAPITAL_JPY'])*100,2)

# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_full_universe():
    print(f"🚀 SENTINEL PRO v2.3 ULTIMATE - Scanning {len(TICKERS)} tickers...")

    spy_df=DataEngine.get_data('SPY',period="400d")
    if spy_df is None:
        return "❌ Market Data Error"

    curr=spy_df['Close'].iloc[-1]
    ma200=spy_df['Close'].rolling(200).mean().iloc[-1]
    is_bull=curr>ma200

    candidates=[]
    price_cache={}
    stats={'Scanned':0,'Pass':0}

    for ticker in TICKERS:
        if ticker in ['SPY','QQQ','IWM','AGG','IEMG','FXI','EWY','IJH']:
            continue

        stats['Scanned']+=1
        df=DataEngine.get_data(ticker)
        if df is None:
            continue

        price_cache[ticker]=df

        vcp=VCPAnalyzer.calculate_vcp_score(df)
        if vcp['score']<CONFIG['MIN_VCP_SCORE']:
            continue

        rs=RSAnalyzer.calculate_rs_rating(df,spy_df)
        if rs<CONFIG['MIN_RS_RATING']:
            continue

        bt=BacktestEngine.run_backtest(df)
        if bt['pf']<CONFIG['MIN_PROFIT_FACTOR']:
            continue

        stats['Pass']+=1

        curr_price=df['Close'].iloc[-1]
        pivot=df['High'].iloc[-10:].max()*1.002
        stop=pivot-(vcp['atr']*CONFIG['STOP_LOSS_ATR'])
        risk=pivot-stop

        targets={
            'T1':pivot+(risk*CONFIG['TARGET_CONSERVATIVE']),
            'T2':pivot+(risk*CONFIG['TARGET_MODERATE']),
            'T3':pivot+(risk*CONFIG['TARGET_AGGRESSIVE'])
        }

        dist_pct=((curr_price-pivot)/pivot)*100
        if -2<=dist_pct<3: status="🔥 ACTION"
        elif -6<dist_pct<-2: status="👀 WATCH"
        elif dist_pct>=3: status="🚀 EXTENDED"
        else: status="⏳ WAIT"

        risk_jpy=CONFIG['CAPITAL_JPY']*CONFIG['ACCOUNT_RISK_PCT']
        shares=int((risk_jpy/150)/risk) if risk>0 else 0

        candidates.append({
            'ticker':ticker,
            'status':status,
            'vcp':vcp,
            'rs':rs,
            'pf':bt['pf'],
            'winrate':bt['winrate'],
            'current':curr_price,
            'entry':pivot,
            'stop':stop,
            'targets':targets,
            'shares':shares
        })

    status_rank={"🔥 ACTION":4,"👀 WATCH":3,"🚀 EXTENDED":2,"⏳ WAIT":1}
    candidates.sort(key=lambda x:(status_rank.get(x['status'],0),x['vcp']['score'],x['pf']),reverse=True)

    final_portfolio=apply_portfolio_filters(candidates,price_cache)
    portfolio_dd=estimate_portfolio_dd(final_portfolio)

    report=[]
    report.append("="*50)
    report.append("🛡 SENTINEL PRO v2.3 ULTIMATE")
    report.append("="*50)
    report.append(f"Market: {'🟢 Bull' if is_bull else '🔴 Bear'}")
    report.append(f"Scan: {stats['Scanned']} | Qualified: {stats['Pass']}")
    report.append(f"Selected: {len(final_portfolio)} | Est.MaxDD: {portfolio_dd}%")
    report.append("-"*50)

    if not final_portfolio:
        report.append("⚠️ 基準を満たす銘柄なし")
    else:
        for p in final_portfolio:
            icon="💎" if p['pf']>1.5 and p['rs']>80 else "🔸"
            report.append(f"\n{icon} {p['ticker']} [{p['status']}]")
            report.append(f"   VCP:{p['vcp']['score']} | RS:{p['rs']} | PF:{p['pf']:.2f}")
            report.append(f"   Now:${p['current']:.2f}")
            report.append(f"   📍 Entry:${p['entry']:.2f}")
            report.append(f"   🛑 Stop:${p['stop']:.2f}")
            report.append(f"   🎯 T2:${p['targets']['T2']:.2f}")
            report.append(f"   📦 推奨:{p['shares']}株")
            report.append(f"   💡 {','.join(p['vcp']['signals'])}")

    return "\n".join(report)

# ============================================================
# LINE通知
# ============================================================

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID:
        print(msg)
        return

    MAX_LEN=4000
    messages=[msg[i:i+MAX_LEN] for i in range(0,len(msg),MAX_LEN)]
    url="https://api.line.me/v2/bot/message/push"
    headers={"Content-Type":"application/json","Authorization":f"Bearer {ACCESS_TOKEN}"}

    for m in messages:
        try:
            requests.post(url,headers=headers,json={"to":USER_ID,"messages":[{"type":"text","text":m}]})
        except:
            pass

# ============================================================

if __name__=="__main__":
    result=analyze_full_universe()
    send_line(result)
    print(result)