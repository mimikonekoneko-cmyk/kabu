import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime

# --- CONFIG ---
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")

# --- 予算設定 ---
BUDGET_JPY = 350000

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

# セクターETF精度アップ
SECTOR_ETF = {
    'Energy': 'XLE',
    'Semi': 'SOXX',
    'Bank': 'XLF',
    'Retail': 'XRT',
    'Soft': 'IGV',
    'AI': 'QQQ',
    'Fin': 'VFH',
    'Device': 'XLK',
    'Cloud': 'SKYY',
    'Ad': 'XLC',
    'Service': 'XLC',
    'Sec': 'HACK',
    'Cons': 'XLP',
    'Bio': 'IBB',
    'Health': 'XLV',
    'Ind': 'XLI',
    'EV': 'IDRV',
    'Crypto': 'CRYPTO',
    'Power': 'PWR',
    'Platform': 'QQQ',
    'Travel': 'JETS',
    'Bet': 'BETZ'
}

MACRO_EVENTS = [
    # '2026-01-30',
]

def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 155.0
    except:
        return 155.0

def is_macro_event_today():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        return today in MACRO_EVENTS
    except:
        return False

def market_is_risky():
    """
    市場全体のトレンド判定
    - SPY, QQQ がMA200割れ
    - VIXが高すぎる
    → Trueなら「リスク高いので新規シグナル停止」
    """
    try:
        spy = yf.download("SPY", period="250d", progress=False)
        qqq = yf.download("QQQ", period="250d", progress=False)
        vix = yf.download("^VIX", period="60d", progress=False)

        if spy.empty or qqq.empty:
            return False  # 判定不能なら止めない

        spy_ma200 = spy['Close'].rolling(200).mean()
        qqq_ma200 = qqq['Close'].rolling(200).mean()

        spy_trend_bad = spy['Close'].iloc[-1] < spy_ma200.iloc[-1]
        qqq_trend_bad = qqq['Close'].iloc[-1] < qqq_ma200.iloc[-1]

        vix_high = False
        if not vix.empty:
            vix_now = vix['Close'].iloc[-1]
            vix_high = vix_now >= 25  # ざっくり恐怖水準

        # どちらかの指数がMA200割れ ＋ VIX高め → リスク高い
        if (spy_trend_bad or qqq_trend_bad) and vix_high:
            return True
        return False
    except:
        return False

def is_earnings_near(ticker, days_window=5):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or cal.empty:
            return None
        try:
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
        except Exception:
            return None
        days = (earnings_date.date() - datetime.now().date()).days
        return abs(days) <= days_window
    except Exception:
        return None

def sector_is_strong(sector):
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf or etf == 'CRYPTO':
            return None
        df = yf.download(etf, period="300d", progress=False)
        if df is None or df.empty or len(df) < 210:
            return None
        ma200 = df['Close'].rolling(200).mean()
        return ma200.iloc[-1] > ma200.iloc[-10]
    except Exception:
        return None

def basic_fundamental_check(ticker):
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return None
        ocf = info.get("operatingCashflow")
        dte = info.get("debtToEquity")
        pm = info.get("profitMargins")
        if ocf is None and dte is None and pm is None:
            return None
        if ocf is not None and ocf <= 0:
            return False
        if dte is not None and dte > 300:
            return False
        if pm is not None and pm <= 0:
            return False
        return True
    except Exception:
        return None

class StrategicAnalyzer:
    @staticmethod
    def analyze_ticker(t, df, sector, max_price_usd):
        if len(df) < MA_LONG:
            return None

        c = df['Close']
        h, l, v = df['High'], df['Low'], df['Volume']
        current_price = float(c.iloc[-1])
        if current_price > max_price_usd:
            return None

        ma50 = c.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = c.rolling(MA_LONG).mean().iloc[-1]
        ma200_prev = c.rolling(MA_LONG).mean().iloc[-10]
        if not (current_price > ma50 > ma200 and ma200 > ma200_prev):
            return None

        tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        range_5d = h.iloc[-5:].max() - l.iloc[-5:].min()
        tightness = float(range_5d / atr14) if atr14 and atr14 != 0 else float('inf')
        if tightness > 3.0:
            return None

        vol_avg = v.rolling(50).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / vol_avg if vol_avg and vol_avg != 0 else 1.0

        score = 60
        if tightness < 1.5:
            score += 25
        elif tightness < 2.0:
            score += 15
        if 0.7 <= vol_ratio <= 1.0:
            score += 15

        # ATRベースの損切り・利確
        # ざっくり: 損切り 2ATR / 利確 4ATR
        if np.isnan(atr14) or atr14 <= 0:
            stop_loss = current_price * 0.93
            target = current_price * 1.15
        else:
            stop_loss = current_price - 2 * atr14
            target = current_price + 4 * atr14

        pivot = h.iloc[-5:].max() * 1.002

        return {
            "score": score,
            "price": current_price,
            "pivot": pivot,
            "stop": stop_loss,
            "target": target,
            "tightness": tightness,
            "vol_ratio": vol_ratio,
            "atr": float(atr14) if not np.isnan(atr14) else None,
            "sector": sector
        }

def send_line(msg):
    if not ACCESS_TOKEN or not USER_ID:
        print("⚠️ LINE設定がありません")
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
        1: {
            "sector_weight": {'Fin':1.3, 'Energy':1.15, 'Semi':0.8, 'AI':0.7, 'Cons':1.2, 'Retail':1.1},
            "cash_buffer": 0.12,
            "max_per_asset": 0.25,
            "target_mult": 1.10
        },
        2: {
            "sector_weight": {'Fin':1.2, 'Energy':1.1, 'Semi':1.0, 'AI':0.9, 'Cons':0.8, 'Retail':1.0},
            "cash_buffer": 0.07,
            "max_per_asset": 0.35,
            "target_mult": 1.15
        },
        3: {
            "sector_weight": {'Fin':0.9, 'Energy':1.0, 'Semi':1.2, 'AI':1.2, 'Cons':0.7, 'Retail':0.9},
            "cash_buffer": 0.03,
            "max_per_asset": 0.50,
            "target_mult": 1.20
        }
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
        target_mult = cfg["target_mult"]

        weighted = []
        for t, r, base_score in weighted_base:
            sec = r['sector']
            w = base_score * sector_w.get(sec, 1.0)
            weighted.append((t, r, w))

        total_w = sum(w for _,_,w in weighted)
        if total_w == 0:
            all_text_lines.append(f"[リスク{risk}] 計算不能（重み0）\n")
            continue

        usable_budget_usd = budget_usd * (1.0 - cash_buf)
        remaining = usable_budget_usd

        portfolio = []
        for t, r, w in weighted:
            ideal_usd = usable_budget_usd * (w / total_w)
            price = r['price']
            max_cost_for_asset = usable_budget_usd * max_asset_pct
            max_shares_allowed = int(max_cost_for_asset // price) if price > 0 else 0
            shares = int(ideal_usd // price)
            if shares > max_shares_allowed:
                shares = max_shares_allowed
            if shares > 0:
                cost = shares * price
                if cost <= remaining:
                    remaining -= cost
                    portfolio.append({
                        "ticker":t,
                        "shares":shares,
                        "price":price,
                        "cost":cost,
                        "target": r['pivot'] * target_mult
                    })

        for t, r, w in weighted:
            price = r['price']
            if price <= 0:
                continue
            entry = next((p for p in portfolio if p['ticker']==t), None)
            max_cost_for_asset = usable_budget_usd * max_asset_pct
            max_shares_allowed = int(max_cost_for_asset // price)
            current_shares = entry['shares'] if entry else 0
            can_buy = max_shares_allowed - current_shares
            if can_buy <= 0:
                continue
            affordable = int(remaining // price)
            buy = min(can_buy, affordable)
            if buy > 0:
                if entry:
                    entry['shares'] += buy
                    entry['cost'] += buy * price
                else:
                    portfolio.append({
                        "ticker":t,
                        "shares":buy,
                        "price":price,
                        "cost":buy*price,
                        "target": r['pivot'] * target_mult
                    })
                remaining -= buy * price

        if not portfolio:
            for t, r, w in weighted:
                price = r['price']
                if price <= 0:
                    continue
                if usable_budget_usd >= price:
                    portfolio.append({
                        "ticker":t,
                        "shares":1,
                        "price":price,
                        "cost":price,
                        "target": r['pivot'] * target_mult
                    })
                    remaining -= price
                    break

        lines = []
        lines.append(f"[リスク{risk} {'保守' if risk==1 else ('中庸' if risk==2 else '攻め')}]\n")
        total_cost_jpy = 0
        for p in portfolio:
            cost_jpy = int(p['cost'] * fx_rate)
            total_cost_jpy += cost_jpy
            gain_pct = int((p['target'] / p['price'] - 1.0) * 100)
            lines.append(
                f"{p['ticker']}: {p['shares']}株（¥{cost_jpy:,}） "
                f"売却推奨: ${p['target']:.2f}（+{gain_pct}%）"
            )
        lines.append(
            f"使用額: ¥{total_cost_jpy:,}  現金バッファ: {int(cash_buf*100)}% 残り: ¥{int(remaining * fx_rate):,}\n"
        )
        all_text_lines.append("\n".join(lines))

    return "\n".join(all_text_lines)

def run_mission():
    current_fx = get_current_fx_rate()
    max_price_usd = (BUDGET_JPY / current_fx) * 0.9

    print(f"🛰️ 偵察開始... (FX: {current_fx:.2f}円, 予算上限: ${max_price_usd:.1f})")

    macro_today = is_macro_event_today()
    market_risky = market_is_risky()

    all_data = yf.download(list(TICKERS.keys()), period="300d", progress=False, group_by='ticker')
    
    results = []
    for t, sec in TICKERS.items():
        if macro_today or market_risky:
            continue

        earnings_near = is_earnings_near(t)
        if earnings_near is True:
            continue

        sector_strength = sector_is_strong(sec)
        if sector_strength is False:
            continue

        fund_ok = basic_fundamental_check(t)
        if fund_ok is False:
            continue

        try:
            df_t = all_data[t]
            res = StrategicAnalyzer.analyze_ticker(t, df_t, sec, max_price_usd)
            if res and res['score'] >= MIN_SCORE:
                res['earnings_near'] = earnings_near if earnings_near is not None else None
                res['sector_strength'] = sector_strength if sector_strength is not None else None
                res['fund_ok'] = fund_ok if fund_ok is not None else None
                results.append((t, res))
        except Exception:
            continue
    
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    report = [
        f"🛡️ Sentinel v20.0",
        f"📅 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        f"💵 $1 = {current_fx:.2f}円",
        f"💰 予算内上限: ${(BUDGET_JPY / current_fx) * 0.9:.1f}",
        f"⚠️ マクロイベント日: {'Yes' if macro_today else 'No'}",
        f"⚠️ マーケットリスク: {'High' if market_risky else 'Normal'}",
        "─" * 15
    ]
    
    if macro_today or market_risky:
        report.append("⚠️ 地合いが悪いため、新規シグナルは全停止中。")
    elif not results:
        report.append("⚠️ 条件に合う銘柄なし。待機。")
    else:
        for i, (t, r) in enumerate(results, 1):
            earnings_label = 'Near' if r.get('earnings_near') is True else ('OK' if r.get('earnings_near') is False else '-')
            sector_label = 'Strong' if r.get('sector_strength') is True else ('Weak' if r.get('sector_strength') is False else '-')
            fund_label = 'OK' if r.get('fund_ok') is True else ('NG' if r.get('fund_ok') is False else '-')

            # スコア内訳の簡易表示
            tight = r['tightness']
            volr = r['vol_ratio']
            atr = r.get('atr')

            report.append(
                f"[{i}] {t} ({r['sector']}) {r['score']}点\n"
                f"現: ${r['price']:.2f} / 入候補: ${r['pivot']:.2f}\n"
                f"止: ${r['stop']:.2f} / 目: ${r['target']:.2f}\n"
                f"決算: {earnings_label}  セクター: {sector_label}  財務: {fund_label}\n"
                f"tight: {tight:.2f} / vol_ratio: {volr:.2f} / ATR: {atr:.2f if atr is not None else float('nan')}\n"
            )

        portfolio_text = build_portfolios(results, BUDGET_JPY, current_fx)
        report.append(portfolio_text)

    full_msg = "\n".join(report)
    print(full_msg)
    send_line(full_msg)

if __name__ == "__main__":
    run_mission()