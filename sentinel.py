import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime
import time

# --- CONFIG (環境変数から読み込み) ---
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# --- 予算設定 ---
BUDGET_JPY = 350000      # 総予算 35万円

# --- テクニカルパラメータ ---
MA_SHORT, MA_LONG = 50, 200
MIN_SCORE = 85
MAX_NOTIFICATIONS = 8

# ★v20.0: リスク管理パラメータ
ATR_STOP_MULT = 2.0      # 損切り幅 = ATRの何倍か (標準: 2.0)
RISK_REWARD_RATIO = 3.0  # リスクリワード比 (標準: 1:3)

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

MACRO_EVENTS = [
    # '2026-01-30',  # 必要に応じてマクロイベント日(FOMC等)を追加
]

# --- ヘルパー関数 ---

def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            close = data['Close']
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            return float(close.iloc[-1])
        return 155.0
    except:
        return 155.0

def is_macro_event_today():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        return today in MACRO_EVENTS
    except:
        return False

def check_market_trend():
    """
    ★v20.0 New: 市場全体の健全性をチェック (Market Filter)
    S&P500 (SPY) が200日移動平均線より上にあるかを確認
    """
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        if spy.empty or len(spy) < 200:
            return True, "Data Error (Allowed)" # データ不足時は一旦許可

        close = spy['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]

        current = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])

        if current < ma200:
            return False, f"Bear Market (SPY ${current:.0f} < MA200 ${ma200:.0f})"
        return True, "Bull Market"
    except Exception as e:
        print(f"Market Check Warning: {e}")
        return True, "Check Failed (Allowed)"

def is_earnings_near(ticker, days_window=5):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or cal.empty:
            return None
        
        # yfinanceのバージョンによる戻り値の差異を吸収
        if isinstance(cal, dict): # 新しいバージョン
            val = cal.get('Earnings Date')
            if val: val = val[0]
        else: # 古いバージョン (DataFrame)
            if 'Earnings Date' in cal.index:
                val = cal.loc['Earnings Date'].values[0]
            else:
                val = cal.iloc[0, 0]
                
        if isinstance(val, (list, tuple, np.ndarray)):
            val = val[0]
            
        if hasattr(val, 'to_pydatetime'):
            earnings_date = val.to_pydatetime()
        else:
            earnings_date = pd.to_datetime(val)
            
        days = (earnings_date.date() - datetime.now().date()).days
        return abs(days) <= days_window
    except Exception:
        return None

def sector_is_strong(sector):
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf or etf == 'CRYPTO': return None
        
        df = yf.download(etf, period="300d", progress=False)
        if df is None or df.empty or len(df) < 210: return None
        
        c = df['Close']
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        
        ma200 = c.rolling(200).mean()
        return ma200.iloc[-1] > ma200.iloc[-10]
    except Exception:
        return None

def basic_fundamental_check(ticker):
    try:
        info = yf.Ticker(ticker).info
        if not info: return None
        ocf = info.get("operatingCashflow")
        dte = info.get("debtToEquity")
        pm = info.get("profitMargins")
        
        if ocf is None and dte is None and pm is None: return None
        if ocf is not None and ocf <= 0: return False
        if dte is not None and dte > 300: return False
        if pm is not None and pm <= 0: return False
        return True
    except Exception:
        return None

# --- 分析エンジン ---

class StrategicAnalyzer:
    @staticmethod
    def analyze_ticker(t, df, sector, max_price_usd):
        if len(df) < MA_LONG: return None
        
        # データ整形 (MultiIndex対策)
        try:
            c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            if isinstance(h, pd.DataFrame): h = h.iloc[:, 0]
            if isinstance(l, pd.DataFrame): l = l.iloc[:, 0]
            if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]
        except:
            return None

        current_price = float(c.iloc[-1])
        if current_price > max_price_usd: return None

        # 1. トレンド判定 (Minervini Stage 2 Filter)
        ma50 = c.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = c.rolling(MA_LONG).mean().iloc[-1]
        ma200_prev = c.rolling(MA_LONG).mean().iloc[-10]
        
        if not (current_price > ma50 > ma200 and ma200 > ma200_prev):
            return None

        # 2. VCP & ATR計算 (Volatility Check)
        # TR (True Range) 計算
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        
        if atr14 == 0 or np.isnan(atr14): return None

        range_5d = h.iloc[-5:].max() - l.iloc[-5:].min()
        tightness = float(range_5d / atr14)
        
        # タイトネスが緩すぎるものは除外
        if tightness > 3.0: return None

        # 3. 出来高分析 (Supply/Demand)
        vol_avg = v.rolling(50).mean().iloc[-1]
        if vol_avg == 0: return None
        
        vol_ratio = v.iloc[-1] / vol_avg
        
        # ★v20.0 New: 出来高スパイク (Pocket Pivot)
        # 直近3日間のどこかで、平均出来高の1.5倍を超えた日があるか？
        recent_max_vol = v.iloc[-3:].max()
        has_volume_spike = recent_max_vol > (vol_avg * 1.5)

        # 4. スコアリング
        score = 60
        # VCP加点
        if tightness < 1.5: score += 25
        elif tightness < 2.0: score += 15
        
        # 売り枯れ加点
        if 0.7 <= vol_ratio <= 1.0: score += 15
        
        # ★v20.0 New: 買い集め加点
        if has_volume_spike: score += 10

        # 5. エントリー＆エグジット設計
        pivot = h.iloc[-5:].max() * 1.002 # 直近高値の0.2%上
        
        # ★v20.0 New: ATRベースの動的損切り
        stop_dist = atr14 * ATR_STOP_MULT
        stop_loss = pivot - stop_dist
        
        # ★v20.0 New: リスクリワード比に基づく利確目標
        target = pivot + (stop_dist * RISK_REWARD_RATIO)

        return {
            "score": score, "price": current_price, "pivot": pivot,
            "stop": stop_loss, "target": target,
            "tightness": tightness, "vol_ratio": vol_ratio, "sector": sector,
            "atr": atr14
        }

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID:
        print("⚠️ LINE設定なしのためコンソール出力のみ")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {"to": USER_ID, "messages": [{"type": "text", "text": msg}]}
    try:
        requests.post(url, headers=headers, json=payload)
    except Exception as e:
        print("LINE送信エラー:", e)

def build_portfolios(results, budget_jpy, fx_rate):
    if not results:
        return "📦 推奨ポートフォリオ: 対象銘柄なし"

    budget_usd = budget_jpy / fx_rate

    risk_profiles = {
        1: { "sector_weight": {'Fin':1.3, 'Energy':1.15, 'Semi':0.8, 'AI':0.7, 'Cons':1.2}, 
             "cash_buffer": 0.12, "max_per_asset": 0.25 },
        2: { "sector_weight": {'Fin':1.2, 'Energy':1.1, 'Semi':1.0, 'AI':0.9, 'Cons':0.8}, 
             "cash_buffer": 0.07, "max_per_asset": 0.35 },
        3: { "sector_weight": {'Fin':0.9, 'Energy':1.0, 'Semi':1.2, 'AI':1.2, 'Cons':0.7}, 
             "cash_buffer": 0.03, "max_per_asset": 0.50 }
    }

    weighted_base = []
    for t, r in results:
        weighted_base.append((t, r, r['score']))
    weighted_base.sort(key=lambda x: x[2], reverse=True)

    all_text_lines = []
    for risk in (1,2,3):
        cfg = risk_profiles[risk]
        sector_w = cfg["sector_weight"]
        cash_buf = cfg["cash_buffer"]
        max_asset_pct = cfg["max_per_asset"]

        weighted = []
        for t, r, base_score in weighted_base:
            sec = r['sector']
            w = base_score * sector_w.get(sec, 1.0)
            weighted.append((t, r, w))

        total_w = sum(w for _,_,w in weighted)
        if total_w == 0: continue

        usable_budget_usd = budget_usd * (1.0 - cash_buf)
        remaining = usable_budget_usd
        portfolio = []

        # 1巡目: 理想配分
        for t, r, w in weighted:
            ideal_usd = usable_budget_usd * (w / total_w)
            price = r['price'] # 予算計算は現在値で行う
            max_cost = usable_budget_usd * max_asset_pct
            max_shares = int(max_cost // price) if price > 0 else 0
            shares = int(ideal_usd // price)
            if shares > max_shares: shares = max_shares
            
            if shares > 0:
                cost = shares * price
                if cost <= remaining:
                    remaining -= cost
                    portfolio.append({"ticker":t,"shares":shares,"price":price,"cost":cost,
                                      "target": r['target'], "stop": r['stop'], "pivot": r['pivot']})

        # 2巡目: 余剰活用
        for t, r, w in weighted:
            price = r['price']
            entry = next((p for p in portfolio if p['ticker']==t), None)
            max_cost = usable_budget_usd * max_asset_pct
            max_shares = int(max_cost // price)
            current_shares = entry['shares'] if entry else 0
            can_buy = max_shares - current_shares
            
            if can_buy > 0:
                affordable = int(remaining // price)
                buy = min(can_buy, affordable)
                if buy > 0:
                    if entry:
                        entry['shares'] += buy
                        entry['cost'] += buy * price
                    else:
                        portfolio.append({"ticker":t,"shares":buy,"price":price,"cost":buy*price,
                                          "target": r['target'], "stop": r['stop'], "pivot": r['pivot']})
                    remaining -= buy * price

        # テキスト生成
        lines = []
        lines.append(f"[リスク{risk} {'保守' if risk==1 else ('中庸' if risk==2 else '攻め')}]\n")
        total_cost_jpy = 0
        for p in portfolio:
            cost_jpy = int(p['cost'] * fx_rate)
            total_cost_jpy += cost_jpy
            
            # 利益率計算 (Pivotからの上昇率)
            gain_pct = int((p['target'] / p['pivot'] - 1.0) * 100)
            loss_pct = int((1.0 - p['stop'] / p['pivot']) * 100)
            
            lines.append(f"{p['ticker']}: {p['shares']}株（¥{cost_jpy:,}）")
            lines.append(f"   入: ${p['pivot']:.2f} 止: ${p['stop']:.2f}(-{loss_pct}%) 目: ${p['target']:.2f}(+{gain_pct}%)")
            
        lines.append(f"使用額: ¥{total_cost_jpy:,}  現金バッファ: {int(cash_buf*100)}%")
        all_text_lines.append("\n".join(lines) + "\n")

    return "\n".join(all_text_lines)

def run_mission():
    print("🛡️ Sentinel v20.0 - System Initializing...")
    
    # ★v20.0 New: 市場全体の地合いチェック
    market_ok, market_status = check_market_trend()
    if not market_ok:
        msg = f"🛑 Sentinel Alert: 市場環境悪化 ({market_status})\n本日は全シグナルを停止し、資金を保全します。"
        print(msg)
        send_line(msg)
        return

    current_fx = get_current_fx_rate()
    max_price_usd = (BUDGET_JPY / current_fx) * 0.9

    print(f"🛰️ 偵察開始... (FX: {current_fx:.2f}円, 予算上限: ${max_price_usd:.1f})")
    print(f"📈 Market Status: {market_status}")

    macro_today = is_macro_event_today()
    if macro_today:
        msg = "⚠️ 本日は主要マクロイベント日のため、全シグナルを無効化します。"
        print(msg)
        send_line(msg)
        return

    # 一括ダウンロード
    tickers_list = list(TICKERS.keys())
    try:
        all_data = yf.download(tickers_list, period="300d", progress=False, group_by='ticker')
    except Exception as e:
        print(f"Data Download Error: {e}")
        return

    results = []
    print("🔍 Analyzing Tickers...")
    
    for t, sec in TICKERS.items():
        # 基本フィルタ
        if is_earnings_near(t) is True: continue
        if sector_is_strong(sec) is False: continue
        if basic_fundamental_check(t) is False: continue

        try:
            # データ抽出 (MultiIndexの処理)
            if len(tickers_list) > 1:
                df_t = all_data[t]
            else:
                df_t = all_data
            
            res = StrategicAnalyzer.analyze_ticker(t, df_t, sec, max_price_usd)
            if res and res['score'] >= MIN_SCORE:
                res['earnings_near'] = False
                res['sector_strength'] = True
                res['fund_ok'] = True
                results.append((t, res))
                print(f"   ✅ {t}: Score {res['score']}")
        except Exception as e:
            # print(f"   Skipped {t}: {e}") # デバッグ用
            continue
    
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    # レポート作成
    report = [
        f"🛡️ Sentinel v20.0",
        f"📅 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        f"💵 $1 = {current_fx:.2f}円",
        f"📊 Market: {market_status}",
        "─" * 15
    ]
    
    if not results:
        report.append("⚠️ 条件に合う銘柄なし。待機。")
    
    for i, (t, r) in enumerate(results, 1):
        # ATRベースの損切り幅を表示
        loss_pct = (1 - r['stop'] / r['pivot']) * 100
        gain_pct = (r['target'] / r['pivot'] - 1) * 100
        
        report.append(
            f"[{i}] {t} ({r['sector']}) {r['score']}点\n"
            f"現: ${r['price']:.2f}\n"
            f"入: ${r['pivot']:.2f} (逆指値推奨)\n"
            f"止: ${r['stop']:.2f} (-{loss_pct:.1f}%)\n"
            f"目: ${r['target']:.2f} (+{gain_pct:.1f}%)\n"
        )

    portfolio_text = build_portfolios(results, BUDGET_JPY, current_fx)
    report.append("─" * 15)
    report.append(portfolio_text)

    full_msg = "\n".join(report)
    print("\n" + full_msg)
    send_line(full_msg)

if __name__ == "__main__":
    run_mission()
