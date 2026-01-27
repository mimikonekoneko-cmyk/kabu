import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta

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

# --- セクターETFマッピング（簡易） ---
SECTOR_ETF = {
    'Energy': 'XLE',
    'Semi': 'SOXX',
    'Bank': 'XLF',
    'Retail': 'XRT',
    'Soft': 'IGV',
    'AI': 'QQQ',
    'Fin': 'VFH',
    'Device': 'QQQ',
    'Cloud': 'QQQ',
    'Ad': 'QQQ',
    'Service': 'QQQ',
    'Sec': 'HACK',
    'Cons': 'XLP',
    'Bio': 'IBB',
    'Health': 'XLV',
    'Ind': 'XLI',
    'EV': 'IDRV',
    'Crypto': 'CRYPTO',  # placeholder; no ETF -> treated as None
    'Power': 'PWR'
}

# --- マクロイベント（手動リスト。必要に応じて更新） ---
MACRO_EVENTS = [
    # 例: 'YYYY-MM-DD'
    '2026-01-30',  # 例: FOMC（実際の日付はユーザーで更新してください）
    # ここに主要イベント日を追加
]

def get_current_fx_rate():
    """USD/JPYの現在レートを取得"""
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 155.0  # 取得失敗時のフォールバック
    except:
        return 155.0

# --- ファンダメンタル補助関数 ---

def is_macro_event_today():
    """今日が主要マクロイベント日かどうか"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        return today in MACRO_EVENTS
    except:
        return False

def is_earnings_near(ticker, days_window=5):
    """
    決算日が近いか判定。
    - True: 決算が近い（±days_window）
    - False: 決算が近くない
    - None: 情報取得できず
    """
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or cal.empty:
            return None
        # yfinance の calendar の形式は環境で異なるため柔軟に処理
        # calendar の最初のセルを取り、datetime に変換を試みる
        try:
            # pandas DataFrame で行ラベルが 'Earnings Date' の場合
            if 'Earnings Date' in cal.index:
                val = cal.loc['Earnings Date'].values[0]
            else:
                # それ以外は最初の要素を使う
                val = cal.iloc[0,0]
            # val が配列やリストの場合は最初の要素を使う
            if isinstance(val, (list, tuple, np.ndarray)):
                val = val[0]
            # pandas.Timestamp なら日付化
            if hasattr(val, 'to_pydatetime'):
                earnings_date = val.to_pydatetime()
            elif isinstance(val, datetime):
                earnings_date = val
            else:
                # 文字列ならパース
                earnings_date = pd.to_datetime(val)
        except Exception:
            return None

        days = (earnings_date.date() - datetime.now().date()).days
        if abs(days) <= days_window:
            return True
        return False
    except Exception:
        return None

def sector_is_strong(sector):
    """
    セクターETFのMA200が上向きか判定
    - True: 強い（MA200上向き）
    - False: 弱い（MA200下向き）
    - None: 情報取得できず / マッピングなし
    """
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf or etf == 'CRYPTO':
            return None
        df = yf.download(etf, period="300d", progress=False)
        if df is None or df.empty or len(df) < 210:
            return None
        ma200 = df['Close'].rolling(200).mean()
        # 上向き判定：直近値 > 10日前の値
        return ma200.iloc[-1] > ma200.iloc[-10]
    except Exception:
        return None

def basic_fundamental_check(ticker):
    """
    簡易的な財務チェック
    - True: OK
    - False: NG（危険）
    - None: 情報取得できず
    チェック項目（簡易）:
      - operatingCashflow > 0
      - debtToEquity not extremely high (例: <= 300)
      - profitMargins > 0
    """
    try:
        info = yf.Ticker(ticker).info
        # info が空なら None
        if not info:
            return None
        ocf = info.get("operatingCashflow")
        dte = info.get("debtToEquity")
        pm = info.get("profitMargins")

        # 値が None の場合は判定不能（None）
        if ocf is None and dte is None and pm is None:
            return None

        # 個別に判定。どれかがNGなら False
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
        tightness = float(range_5d / atr14) if atr14 and atr14 != 0 else float('inf')
        if tightness > 3.0: return None
        
        # スコアリング
        vol_avg = v.rolling(50).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / vol_avg if vol_avg and vol_avg != 0 else 1.0
        
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
    try:
        requests.post(url, headers=headers, json=payload)
    except Exception as e:
        print("LINE送信エラー:", e)

def run_mission():
    # 🛰️ 最新レートを取得
    current_fx = get_current_fx_rate()
    # 予算20万円の90%を、1銘柄あたりの上限（ドル）とする
    max_price_usd = (BUDGET_JPY / current_fx) * 0.9

    print(f"🛰️ 偵察開始... (FX: {current_fx:.2f}円, 予算上限: ${max_price_usd:.1f})")

    # マクロイベントチェック（全体）
    macro_today = is_macro_event_today()
    if macro_today:
        print("⚠️ 本日は主要マクロイベント日のため、全シグナルを無効化します。")

    all_data = yf.download(list(TICKERS.keys()), period="300d", progress=False, group_by='ticker')
    
    results = []
    for t, sec in TICKERS.items():
        # 1) マクロイベント日ならスキップ
        if macro_today:
            # レポート用に None を残す（スキップ）
            continue

        # 2) 決算回避
        earnings_near = is_earnings_near(t)
        if earnings_near is True:
            # 決算が近い -> スキップ
            continue

        # 3) セクター判断
        sector_strength = sector_is_strong(sec)
        if sector_strength is False:
            # セクターが弱い -> スキップ
            continue

        # 4) 財務健全性（簡易）
        fund_ok = basic_fundamental_check(t)
        if fund_ok is False:
            # 財務がNG -> スキップ
            continue

        # 5) テクニカル判定
        try:
            df_t = all_data[t]
            res = StrategicAnalyzer.analyze_ticker(t, df_t, sec, max_price_usd)
            # attach metadata for reporting
            if res:
                res['earnings_near'] = earnings_near if earnings_near is not None else None
                res['sector_strength'] = sector_strength if sector_strength is not None else None
                res['fund_ok'] = fund_ok if fund_ok is not None else None
                results.append((t, res))
        except Exception:
            continue
    
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    # レポート構築
    report = [
        f"🛡️ Sentinel v18.0",
        f"📅 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        f"💵 $1 = {current_fx:.2f}円",
        f"💰 予算内上限: ${max_price_usd:.1f}",
        f"⚠️ マクロイベント日: {'Yes' if macro_today else 'No'}",
        "─" * 15
    ]
    
    if not results:
        report.append("⚠️ 条件に合う銘柄なし。待機。")
    for i, (t, r) in enumerate(results, 1):
        earnings_label = 'Near' if r.get('earnings_near') is True else ('OK' if r.get('earnings_near') is False else '-')
        sector_label = 'Strong' if r.get('sector_strength') is True else ('Weak' if r.get('sector_strength') is False else '-')
        fund_label = 'OK' if r.get('fund_ok') is True else ('NG' if r.get('fund_ok') is False else '-')
        report.append(
            f"[{i}] {t} ({r['sector']}) {r['score']}点\n"
            f"現: ${r['price']:.2f} / 入: ${r['pivot']:.2f}\n"
            f"止: ${r['stop']:.2f} / 目: ${r['target']:.2f}\n"
            f"決算: {earnings_label}  セクター: {sector_label}  財務: {fund_label}\n"
        )

    full_msg = "\n".join(report)
    print(full_msg)
    send_line(full_msg)

if __name__ == "__main__":
    run_mission()