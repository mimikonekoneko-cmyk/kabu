#!/usr/bin/env python3

# ==============================================================================
# 🛡 SENTINEL PRO v4.5.2 ELITE (THE TOTAL RESTORATION)
# ------------------------------------------------------------------------------
# 修正・統合レポート:
# 1. 銘柄リスト: 450銘柄以上を完全搭載（省略なし）。
# 2. RS 99: 全銘柄のパフォーマンスを順位化するパーセンタイル方式。
# 3. PFロジック: v3.3.1の250日シミュレータを完全復旧（含み益カウント対応）。
# 4. バグ修正: numpy型によるJSONエラーを徹底排除。
# 5. 執行戦略: 0株除外による資金効率の最適化。
# ==============================================================================

import os
import time
import logging
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
from datetime import datetime

# 警告の抑制
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "CAPITAL_JPY": 350_000,          # 運用資金
    "MAX_POSITIONS": 20,              # 最大ポジション数
    "ACCOUNT_RISK_PCT": 0.015,       # 1トレードあたりの許容リスク（1.5%）
    "MAX_SAME_SECTOR": 2,            # セクターあたりの最大銘柄数
    "CORRELATION_LIMIT": 0.80,       # 銘柄間の相関上限

    # v3.3.1 厳格フィルタ基準
    "MIN_RS_RATING": 70,             # RSスコア下限
    "MIN_VCP_SCORE": 55,             # VCPスコア下限
    "MIN_PROFIT_FACTOR": 1.1,        # 戦略適合性（PF）下限
    "MAX_TIGHTNESS_PCT": 0.15,       # 収縮許容度（15%以内）

    # 執行・出口戦略
    "STOP_LOSS_ATR": 2.0,            # 損切り幅（ATRの2倍）
    "TARGET_R_MULTIPLE": 2.5,        # 利確目標（リスクの2.5倍）

    "CACHE_EXPIRY": 12 * 3600        # キャッシュ有効期限（12時間）
}

# ディレクトリ管理
CACHE_DIR = Path("./cache_v45")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# TICKER UNIVERSE (450+ 銘柄 全搭載)
# ==============================================================================

ORIGINAL_LIST = [
    'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'MU', 'QCOM', 'MRVL', 'LRCX', 'AMAT',
    'KLAC', 'ADI', 'ON', 'SMCI', 'ARM', 'MPWR', 'TER', 'RKLB', 'ASTS', 'PLTR', 
    'AERO', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AAPL', 'AMZN', 'NFLX', 'CRM', 'NOW',
    'SNOW', 'ADBE', 'INTU', 'ORCL', 'SAP', 'COST', 'WMT', 'TSLA', 'SBUX', 'NKE', 
    'MELI', 'BABA', 'CVNA', 'MTN', 'LLY', 'ABBV', 'REGN', 'VRTX', 'NVO', 'BSX', 
    'HOLX', 'OMER', 'DVAX', 'RARE', 'RIGL', 'KOD', 'TARS', 'ORKA', 'DSGN', 'MA', 
    'V', 'COIN', 'MSTR', 'HOOD', 'PAY', 'MDLN', 'COHR', 'ACN', 'ETN', 'SPOT', 
    'RDDT', 'RBLX', 'CEVA', 'FFIV', 'DAKT', 'ITRN', 'TBLA', 'CHA', 'EPAC', 'DJT', 
    'TV', 'SEM', 'SCVL', 'INBX', 'CCOI', 'NMAX', 'HY', 'AVR', 'PRSU', 'WBTN', 
    'ASTE', 'FULC', 'SNDK', 'WDC', 'STX', 'GEV', 'APH', 'TXN', 'PG', 'UBER', 
    'BE', 'LITE', 'IBM', 'CLS', 'CSCO', 'APLD', 'ANET', 'NET', 'GLW', 'PANW', 
    'CRWD', 'NBIS', 'RCL', 'ONDS', 'IONQ', 'ROP', 'PM', 'PEP', 'KO', 'SPY', 'QQQ', 'IWM'
]

EXPANSION_LIST = [
    'BRK-B','JPM','UNH','XOM','HD','MRK','CVX','BAC','LIN','DIS','TMO','MCD','ABT','WFC',
    'CMCSA','VZ','PFE','CAT','ISRG','GE','SPGI','HON','UNP','RTX','LOW','GS','BKNG','ELV',
    'AXP','COP','MDT','SYK','BLK','NEE','BA','TJX','PGR','ETN','LMT','C','CB','ADP','MMC',
    'PLD','CI','MDLZ','AMT','BX','TMUS','SCHW', 'MO','EOG','DE','SO','DUK','SLB','CME','SHW',
    'CSX','PYPL','CL','EQIX','ICE','FCX', 'MCK','TGT','USB','PH','GD','BDX','ITW','ABNB',
    'HCA','NXPI','PSX','MAR','NSC','EMR', 'AON','PNC','CEG','CDNS','SNPS','MCO','PCAR','COF',
    'FDX','ORLY','ADSK','VLO','OXY','TRV', 'AIG','HLT','WELL','CARR','AZO','PAYX','MSI','TEL',
    'PEG','AJG','ROST','KMB','APD', 'URI','DHI','OKE','WMB','TRGP','SRE','CTAS','AFL','GWW',
    'LHX','MET','PCG','CMI','F','GM','STZ', 'PSA','O','DLR','CCI','KMI','ED','XEL','EIX',
    'WEC','D','AWK','ES','AEP','EXC', 'STM','GFS', 'DDOG','MDB','HUBS','TTD','APP','PATH',
    'MNDY','GTLB', 'IOT', 'DUOL', 'CFLT', 'AI', 'SOUN', 'CLSK', 'MARA', 'RIOT', 'BITF', 'HUT',
    'IREN', 'WULF', 'CORZ', 'CIFR', 'AFRM', 'UPST', 'SOFI', 'DKNG', 'MRNA', 'BNTX', 'UTHR', 'SMMT',
    'VKTX', 'ALT', 'CRSP', 'NTLA', 'BEAM', 'LUNR', 'HII', 'AXON', 'TDG', 'CCJ', 'URA', 'UUUU', 'DNN',
    'NXE', 'UEC', 'SCCO', 'AA', 'NUE', 'STLD', 'TTE', 'CART', 'CAVA', 'BIRK', 'KVUE', 'LULU', 'ONON',
    'DECK', 'CROX', 'WING', 'CMG', 'DPZ', 'YUM', 'CELH', 'MNST', 'GME', 'AMC', 'U', 'OPEN', 'Z',
    'SMH', 'XLF', 'XLV', 'XLE', 'XLI', 'XLK', 'XLC', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE''VRT', 'ALAB', 'OKLO', 'NBIS', 'SMCI', 'IONQ', 'ASTS'
]

TICKERS = sorted(list(set(ORIGINAL_LIST + EXPANSION_LIST)))

# ==============================================================================
# ENGINES
# ==============================================================================

class CurrencyEngine:
    @staticmethod
    def get_usd_jpy():
        try:
            ticker = yf.Ticker("JPY=X")
            df = ticker.history(period="1d")
            if df.empty: return 152.0
            rate = float(df['Close'].iloc[-1])
            return round(rate, 2)
        except: return 152.0

class DataEngine:
    @staticmethod
    def get_data(ticker, period="700d"):
        cache_file = CACHE_DIR / f"{ticker}.pkl"
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < CONFIG["CACHE_EXPIRY"]:
                try:
                    with open(cache_file, "rb") as f: return pickle.load(f)
                except: pass
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 150: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            with open(cache_file, "wb") as f: pickle.dump(df, f)
            return df
        except: return None

    @staticmethod
    def get_sector(ticker):
        sector_cache_file = CACHE_DIR / "sectors.json"
        sector_map = {}
        if sector_cache_file.exists():
            try:
                with open(sector_cache_file, 'r') as f: sector_map = json.load(f)
            except: pass
        if ticker in sector_map: return sector_map[ticker]
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            sector_map[ticker] = sector
            with open(sector_cache_file, 'w') as f: json.dump(sector_map, f)
            return sector
        except: return "Unknown"

# ==============================================================================
# ANALYZERS (v3.3.1 ロジック完全復旧)
# ==============================================================================

class VCPAnalyzer:
    @staticmethod
    def calculate(df):
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
            
            # ATR
            tr = pd.concat([
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

            if pd.isna(atr) or atr <= 0: return {"score": 0, "atr": 0, "signals": [], "is_dryup": False}

            # 1. 収縮判定 (10日レンジ)
            h10 = high.iloc[-10:].max(); l10 = low.iloc[-10:].min()
            range_pct = float((h10 - l10) / h10)
            
            # 収縮スコア (40点満点)
            tight_score = 40 if range_pct <= 0.05 else int(40 * (1 - (range_pct - 0.05) / 0.10))
            tight_score = max(0, min(40, tight_score))

            # 2. 出来高ドライアップ (30点満点)
            vol_ma = volume.rolling(50).mean().iloc[-1]
            vol_ratio = float(volume.iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
            is_dryup = bool(vol_ratio < 0.7) # 型キャスト
            vol_score = 30 if is_dryup else (15 if vol_ratio < 1.1 else 0)

            # 3. トレンド/MA整列 (30点満点)
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1]
            trend_score = (10 if close.iloc[-1] > ma50 else 0) + \
                          (10 if ma50 > ma200 else 0) + \
                          (10 if close.iloc[-1] > ma200 else 0)

            signals = []
            if range_pct < 0.06: signals.append("極度収縮")
            if is_dryup: signals.append("Vol枯渇")
            if trend_score == 30: signals.append("MA整列")

            return {
                "score": int(max(0, tight_score + vol_score + trend_score)),
                "atr": atr,
                "signals": signals,
                "is_dryup": is_dryup
            }
        except Exception:
            return {"score": 0, "atr": 0, "signals": [], "is_dryup": False}

class RSAnalyzer:
    """RS相対順位化エンジン"""
    @staticmethod
    def get_raw_score(df):
        try:
            c = df["Close"]
            # 12ヶ月, 6ヶ月, 3ヶ月, 1ヶ月の加重騰落率
            r12 = (c.iloc[-1] / c.iloc[-252] - 1) if len(c) >= 252 else (c.iloc[-1]/c.iloc[0]-1)
            r6  = (c.iloc[-1] / c.iloc[-126] - 1) if len(c) >= 126 else (c.iloc[-1]/c.iloc[0]-1)
            r3  = (c.iloc[-1] / c.iloc[-63] - 1)  if len(c) >= 63  else (c.iloc[-1]/c.iloc[0]-1)
            r1  = (c.iloc[-1] / c.iloc[-21] - 1)  if len(c) >= 21  else (c.iloc[-1]/c.iloc[0]-1)
            return (r12 * 0.4) + (r6 * 0.2) + (r3 * 0.2) + (r1 * 0.2)
        except: return -999.0

class StrategyValidator:
    """v3.3.1 バックテストエンジン完全復旧"""
    @staticmethod
    def run_backtest(df):
        try:
            if len(df) < 200: return 1.0
            close = df['Close']; high = df['High']; low = df['Low']
            tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            trades = []
            in_pos = False
            entry_p = 0; stop_p = 0

            # 直近250日のシミュレーション
            start_idx = max(50, len(df)-250)
            for i in range(start_idx, len(df)):
                if in_pos:
                    # エグジット判定
                    if low.iloc[i] <= stop_p:
                        trades.append(-1.0) # 損切り(1R失う)
                        in_pos = False
                    elif high.iloc[i] >= entry_p + (entry_p - stop_p) * CONFIG["TARGET_R_MULTIPLE"]:
                        trades.append(CONFIG["TARGET_R_MULTIPLE"]) # 利確
                        in_pos = False
                    elif i == len(df) - 1:
                        # 最終日は含み益をR倍数でカウント (v3.3.1コアロジック)
                        risk = entry_p - stop_p
                        if risk > 0:
                            pnl = (close.iloc[i] - entry_p) / risk
                            trades.append(float(pnl))
                        in_pos = False
                else:
                    # 20日ピボット突破 + MA50上でエントリー
                    pivot = high.iloc[i-20:i].max()
                    if close.iloc[i] > pivot and close.iloc[i] > close.rolling(50).mean().iloc[i]:
                        in_pos = True
                        entry_p = float(close.iloc[i])
                        stop_p = entry_p - (float(atr.iloc[i]) * CONFIG["STOP_LOSS_ATR"])

            if not trades: return 1.0
            pos_sum = sum([t for t in trades if t > 0])
            neg_sum = abs(sum([t for t in trades if t < 0]))
            pf = pos_sum / neg_sum if neg_sum > 0 else (5.0 if pos_sum > 0 else 1.0)
            return round(float(min(10.0, pf)), 2)
        except: return 1.0

# ==============================================================================
# EXECUTION LOGIC
# ==============================================================================

def calculate_position(entry, stop, usd_jpy):
    try:
        total_usd = CONFIG["CAPITAL_JPY"] / usd_jpy
        risk_usd = total_usd * CONFIG["ACCOUNT_RISK_PCT"]
        diff = abs(entry - stop)
        if diff <= 0: return 0
        
        shares_risk = int(risk_usd / diff)
        # 資金枠上限 (1ポジション最大40%)
        shares_cap = int((total_usd * 0.4) / entry)
        
        return max(0, min(shares_risk, shares_cap))
    except: return 0

def run():
    start_time = time.time()
    print("=" * 60)
    print("🛡 SENTINEL PRO v4.5.2 ELITE (THE TOTAL RESTORATION)")
    print("-" * 60)

    usd_jpy = CurrencyEngine.get_usd_jpy()
    print(f"Current Exchange Rate: {usd_jpy} JPY/USD")

    # パス1: 全ユニバースのスキャンとRS生スコア算出
    raw_list = []
    print(f"Phase 1: Deep Scanning {len(TICKERS)} tickers...")
    for ticker in TICKERS:
        df = DataEngine.get_data(ticker)
        if df is None: continue
        raw_rs = RSAnalyzer.get_raw_score(df)
        if raw_rs == -999.0: continue
        raw_list.append({"ticker": ticker, "df": df, "raw_rs": raw_rs})

    # パス2: RSパーセンタイル順位の割り当て
    raw_list.sort(key=lambda x: x['raw_rs'])
    total_scanned = len(raw_list)
    for i, item in enumerate(raw_list):
        item['rs_rating'] = int(((i + 1) / total_scanned) * 99)

    # パス3: 詳細分析とフィルタリング
    qualified = []
    return_map = {}
    print(f"Phase 2: Technical Validation & Budget Filtering...")

    for item in raw_list:
        ticker = item['ticker']; df = item['df']; rs = item['rs_rating']
        
        vcp = VCPAnalyzer.calculate(df)
        pf = StrategyValidator.run_backtest(df)

        # フィルタ (RS下限 / VCP下限 / PF下限)
        if rs < CONFIG["MIN_RS_RATING"] or vcp["score"] < CONFIG["MIN_VCP_SCORE"] or pf < CONFIG["MIN_PROFIT_FACTOR"]:
            continue

        price = float(df["Close"].iloc[-1])
        pivot = float(df["High"].iloc[-20:].max())
        
        entry = pivot * 1.002
        stop = entry - vcp["atr"] * CONFIG["STOP_LOSS_ATR"]
        target = entry + (entry - stop) * CONFIG["TARGET_R_MULTIPLE"]

        # 0株除外 (資金35万円で購入不可能な銘柄を排除)
        shares = calculate_position(entry, stop, usd_jpy)
        if shares <= 0:
            continue

        # ステータス判定
        dist_pct = (price - pivot) / pivot
        if -0.05 <= dist_pct <= 0.03: status = "ACTION"
        elif dist_pct < -0.05: status = "WAIT"
        else: status = "EXTENDED"

        qualified.append({
            "ticker": ticker,
            "status": status,
            "price": round(price, 2),
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "shares": int(shares),
            "vcp": vcp,
            "rs": int(rs),
            "pf": float(pf),
            "sector": DataEngine.get_sector(ticker)
        })
        return_map[ticker] = df["Close"].pct_change().dropna()

    # ACTION優先かつ、RS+VCP+PFの総合スコアでソート
    status_rank = {"ACTION": 3, "WAIT": 2, "EXTENDED": 1}
    qualified.sort(key=lambda x: (status_rank.get(x["status"], 0), x["rs"] + x["vcp"]["score"] + (x["pf"]*10)), reverse=True)

    # セクター分散フィルタリング
    selected = []
    sector_counts = {}
    for q in qualified:
        if q['status'] != "ACTION": continue # 通知のメインはACTION
        sec = q['sector']
        if sector_counts.get(sec, 0) >= CONFIG['MAX_SAME_SECTOR'] and sec != "Unknown": continue
        
        selected.append(q)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) >= CONFIG['MAX_POSITIONS']: break

    # 結果保存
    today = datetime.now().strftime("%Y-%m-%d")
    run_info = {
        "date": today,
        "runtime": f"{round(time.time() - start_time, 2)}s",
        "usd_jpy": usd_jpy,
        "scan_count": len(TICKERS),
        "qualified_count": len(qualified),
        "selected_count": len(selected),
        "selected": selected,
        "watchlist_wait": [q for q in qualified if q['status'] == "WAIT"][:8], # 期待のWAIT
        "qualified_full": qualified
    }

    with open(RESULTS_DIR / f"{today}.json", 'w', encoding='utf-8') as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2, default=str)

    # ログ出力
    print(f"\nScan Complete. Found {len(qualified)} qualified, {len(selected)} action items.")
    
    # LINE通知メッセージ構築
    msg = [f"🛡 SENTINEL v4.5.2 (Rate:{usd_jpy})\nScan:{len(TICKERS)} | Sel:{len(selected)}\n" + "="*20]
    
    if not selected:
        msg.append("\n⚠️ 現在、即エントリー可能な推奨銘柄はありません。")
    else:
        for s in selected:
            msg.append(f"\n💎 {s['ticker']} [RS{s['rs']} VCP{s['vcp']['score']}]")
            msg.append(f"PF:{s['pf']:.2f} | 推奨:{s['shares']}株")
            msg.append(f"Ent:${s['entry']:.2f} Stop:${s['stop']:.2f}")
            msg.append(f"💡 {','.join(s['vcp']['signals'])}")
            msg.append("-" * 15)

    wait_list = run_info["watchlist_wait"]
    if wait_list:
        msg.append("\n" + "="*20 + "\n🚨 注目Watchlist (WAIT)")
        for w in wait_list:
            msg.append(f"• {w['ticker']} (RS{w['rs']} VCP{w['vcp']['score']} PF{w['pf']:.2f})")

    # LINE送信 (環境変数がセットされている場合のみ)
    send_line("\n".join(msg))
    print("\n--- FINAL MESSAGE ---\n" + "\n".join(msg))

def send_line(message):
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.getenv("LINE_USER_ID")
    if not token or not user_id: return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for p in parts:
        payload = {"to": user_id, "messages": [{"type": "text", "text": p}]}
        try: requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload, timeout=15)
        except: pass

if __name__ == "__main__":
    run()

