import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("SENTINEL")

---------------------------

CONFIG

---------------------------
ACCESSTOKEN = os.getenv("LINECHANNELACCESSTOKEN") or os.getenv("ACCESS_TOKEN")
USERID = os.getenv("LINEUSERID") or os.getenv("USERID")

INITIALCAPITALJPY = 350_000
TRADING_RATIO = 0.70
TARGETANNUALRETURN = 0.10

ATRSTOPMULT = 2.0
MAXPOSITIONSIZE = 0.25
MAXSECTORCONCENTRATION = 0.40

MIN_SCORE = 60
MIN_WINRATE = 0.55
MIN_EXPECTANCY = 0.20
MAXTIGHTNESSBASE = 1.5
MAX_NOTIFICATIONS = 5

MINDAILYVOLUMEUSD = 10000_000

COMMISSION_RATE = 0.002
SLIPPAGE_RATE = 0.001
FXSPREADRATE = 0.0005

REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
AGGRESSIVE_SECTORS = ['Semi', 'AI', 'Soft', 'Sec', 'Auto', 'Crypto', 'Cloud', 'Ad', 'Service', 'Platform', 'Bet', 'Fintech']

MASHORT, MALONG = 50, 200

ALLOW_FRACTIONAL = False  # True if broker supports fractional shares

---------------------------

TICKER UNIVERSE

---------------------------
TICKERS = {
    'NVDA':'AI', 'AMD':'Semi', 'AVGO':'Semi', 'TSM':'Semi', 'ASML':'Semi', 'MU':'Semi',
    'ARM':'Semi', 'INTC':'Semi', 'QCOM':'Semi', 'ON':'Semi', 'LRCX':'Semi', 'AMAT':'Semi',
    'MSFT':'Cloud', 'GOOGL':'Ad', 'META':'Ad', 'PLTR':'AI', 'NOW':'Soft', 'CRM':'Soft',
    'ADBE':'Soft', 'SNOW':'Cloud', 'DDOG':'Cloud', 'WDAY':'Soft', 'TEAM':'Soft',
    'ANET':'Cloud', 'ZS':'Sec', 'MDB':'Cloud', 'SHOP':'Retail', 'PANW':'Sec',
    'CRWD':'Sec', 'FTNT':'Sec', 'NET':'Sec', 'OKTA':'Sec', 'AAPL':'Device',
    'TSLA':'Auto', 'AMZN':'Retail', 'NFLX':'Service', 'COST':'Retail', 'WMT':'Retail',
    'TJX':'Retail', 'TGT':'Retail', 'NKE':'Cons', 'LULU':'Cons', 'SBUX':'Cons',
    'PEP':'Cons', 'KO':'Cons', 'PG':'Cons', 'ELF':'Cons', 'CELH':'Cons', 'MELI':'Retail',
    'V':'Fin', 'MA':'Fin', 'PYPL':'Fintech', 'SQ':'Fintech', 'JPM':'Bank', 'GS':'Bank',
    'MS':'Bank', 'AXP':'Fin', 'BLK':'Fin', 'COIN':'Crypto', 'SOFI':'Fintech', 'NU':'Fintech',
    'LLY':'Bio', 'UNH':'Health', 'ABBV':'Bio', 'ISRG':'Health', 'VRTX':'Bio', 'MRK':'Bio',
    'PFE':'Bio', 'AMGN':'Bio', 'HCA':'Health', 'TDOC':'Health', 'GE':'Ind', 'CAT':'Ind',
    'DE':'Ind', 'BA':'Ind', 'ETN':'Power', 'VRT':'Power', 'TT':'Ind', 'PH':'Ind',
    'TDG':'Ind', 'XOM':'Energy', 'CVX':'Energy', 'MPC':'Energy', 'UBER':'Platform',
    'BKNG':'Travel', 'ABNB':'Travel', 'MAR':'Travel', 'RCL':'Travel', 'DKNG':'Bet',
    'RBLX':'Service', 'DASH':'Service', 'SMCI':'AI'
}

SECTOR_ETF = {
    'Energy':'XLE', 'Semi':'SOXX', 'Bank':'XLF', 'Retail':'XRT', 'Soft':'IGV', 'AI':'QQQ',
    'Fin':'VFH', 'Device':'QQQ', 'Cloud':'QQQ', 'Ad':'QQQ', 'Service':'QQQ', 'Sec':'HACK',
    'Cons':'XLP', 'Bio':'IBB', 'Health':'XLV', 'Ind':'XLI', 'Auto':'CARZ', 'Crypto':'BTC-USD',
    'Power':'XLI', 'Platform':'QQQ', 'Travel':'XLY', 'Bet':'BETZ', 'Fintech':'ARKF'
}

---------------------------

Utilities: FX, market

---------------------------
def getcurrentfx_rate():
    try:
        data = yf.download("JPY=X", period="5d", progress=False)
        if data is None or data.empty:
            return 152.0
        return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.warning("FX fetch failed: %s", e)
        return 152.0

def jpytousd(jpy, fx):
    return jpy / fx

def usdtojpy(usd, fx):
    return usd * fx

def get_vix():
    try:
        data = yf.download("^VIX", period="5d", progress=False)
        if data is None or data.empty:
            return 20.0
        return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)
        return 20.0

def checkmarkettrend():
    try:
        spy = yf.download("SPY", period="400d", progress=False)
        close = spy['Close'].dropna()
        if len(close) < 210:
            return True, "Unknown", 0.0
        curr = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        dist = ((curr - ma200) / ma200) * 100
        return curr > ma200, f"{'Bull' if curr > ma200 else 'Bear'} ({dist:+.1f}%)", dist
    except Exception as e:
        logger.warning("Market trend check failed: %s", e)
        return True, "Unknown", 0.0

---------------------------

Data helpers

---------------------------
def safe_download(ticker, period="700d", retry=2):
    for attempt in range(retry):
        try:
            df = yf.download(ticker, period=period, progress=False)
            return df
        except Exception as e:
            logger.warning("yf.download attempt %d failed for %s: %s", attempt+1, ticker, e)
            time.sleep(1 + attempt)
    return pd.DataFrame()

def ensure_df(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.copy()

---------------------------

Earnings & sector

---------------------------
def isearningsnear(ticker, days_window=2):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None:
            return False
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            try:
                date_val = cal.iloc[0, 0]
            except Exception:
                return False
        elif isinstance(cal, dict):
            date_val = cal.get('Earnings Date', [None])[0]
        else:
            return False
        if date_val is None:
            return False
        ed = pd.todatetime(dateval).date()
        days_until = (ed - datetime.now().date()).days
        return abs(daysuntil) <= dayswindow
    except Exception:
        return False

def sectorisstrong(sector):
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf:
            return True
        df = yf.download(etf, period="300d", progress=False)
        if df is None or df.empty:
            return True
        close = df['Close'].dropna()
        if len(close) < 220:
            return True
        ma200 = close.rolling(200).mean().dropna()
        if len(ma200) < 12:
            return True
        slope = (ma200.iloc[-1] - ma200.iloc[-10]) / ma200.iloc[-10]
        return slope >= 0.0
    except Exception:
        return True

---------------------------

Transaction cost model

---------------------------
class TransactionCostModel:
    @staticmethod
    def calculatetotalcostusd(valusd):
        comm = valusd * COMMISSIONRATE
        slip = valusd * SLIPPAGERATE
        return (comm + slip) * 2

    @staticmethod
    def calculatetotalcostjpy(valusd, fx):
        return TransactionCostModel.calculatetotalcostusd(valusd)  fx + (valusd  FXSPREAD_RATE  fx)  2

---------------------------

Position sizing

---------------------------
class PositionSizer:
    @staticmethod
    def calculateposition(capusd, winrate, rr, atrpct, vix, secexp):
        try:
            if rr <= 0:
                return 0.0, 0.0
            kelly = max(0.0, (winrate - (1 - winrate) / rr))
            kelly = min(kelly * 0.5, MAXPOSITIONSIZE)
            vf = 0.7 if atrpct > 0.05 else 0.85 if atr_pct > 0.03 else 1.0
            m_f = 0.7 if vix > 30 else 0.85 if vix > 20 else 1.0
            sf = 0.7 if secexp > MAXSECTORCONCENTRATION else 1.0
            finalfrac = min(kelly  vf  mf * sf, MAXPOSITIONSIZE)
            posval = capusd * final_frac
            return posval, finalfrac
        except Exception:
            return 0.0, 0.0

---------------------------

Backtest

---------------------------
def simulatepastperformancev2(df, sector, lookbackyears=3):
    try:
        df = ensure_df(df)
        close = df['Close'].dropna()
        high = df['High'].dropna()
        low = df['Low'].dropna()
        if len(close) < 60:
            return {'winrate':0, 'net_expectancy':0, 'message':'LowData'}
        end_date = close.index[-1]
        startdate = enddate - pd.DateOffset(years=lookback_years)
        mask = close.index >= start_date
        close = close.loc[mask]
        high = high.loc[mask]
        low = low.loc[mask]
        if len(close) < 60:
            return {'winrate':0, 'net_expectancy':0, 'message':'ShortWindow'}
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().dropna()
        rewardmult = REWARDMULTIPLIERS['aggressive'] if sector in AGGRESSIVESECTORS else REWARDMULTIPLIERS['stable']
        wins = 0; losses = 0; total_r = 0.0; samples = 0
        for i in range(50, len(close)-40):
            window_high = high.iloc[i-5:i].max()
            pivot = window_high * 1.002
            if high.iloc[i] < pivot:
                continue
            ma50 = close.rolling(50).mean().iloc[i]
            ma200 = close.rolling(200).mean().iloc[i] if i >= 200 else None
            if ma200 is not None and not (close.iloc[i] > ma50 or ma50 > ma200):
                continue
            stopdist = atr.iloc[i]  ATRSTOPMULT if i < len(atr) else atr.iloc[-1]  ATRSTOP_MULT
            entry = pivot
            target = entry + stopdist * rewardmult
            outcome = None
            for j in range(1, 31):
                if i + j >= len(close):
                    break
                if high.iloc[i+j] >= target:
                    outcome = 'win'; break
                if low.iloc[i+j] <= entry - stop_dist:
                    outcome = 'loss'; break
            if outcome is None:
                last_close = close.iloc[min(i+30, len(close)-1)]
                pnl = (lastclose - entry) / stopdist
                if pnl > 0:
                    wins += 1; totalr += min(pnl, rewardmult)
                else:
                    losses += 1; total_r -= abs(pnl)
                samples += 1
            else:
                samples += 1
                if outcome == 'win':
                    wins += 1; totalr += rewardmult
                else:
                    losses += 1; total_r -= 1.0
        total = wins + losses
        if total < 8:
            return {'winrate':0, 'net_expectancy':0, 'message':f'LowSample:{total}'}
        wr = (wins / total)
        ev = total_r / total
        return {'winrate':wr100, 'net_expectancy':ev - 0.05, 'message':f"WR{wr100:.0f}% EV{ev:.2f}"}
    except Exception as e:
        logger.exception("Backtest error: %s", e)
        return {'winrate':0, 'net_expectancy':0, 'message':'BT Error'}

---------------------------

Analyzer (robust, shares-based)

---------------------------
class StrategicAnalyzerV2:
    @staticmethod
    def analyzeticker(ticker, df, sector, maxpositionvalueusd, vix, secexposures, capusd, marketisbull):
        try:
            if df is None or df.empty:
                return None, "❌DATA"
            df = ensure_df(df)
            # Ensure required columns
            for col in ['Close','High','Low','Volume']:
                if col not in df.columns:
                    if col == 'Volume':
                        df['Volume'] = 0
                    else:
                        df[col] = df['Close'] if 'Close' in df.columns else np.nan
            df = df.dropna(subset=['Close'])
            if df.empty:
                return None, "❌DATA"
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna(subset=['Close'])
            df[['High','Low','Close','Volume']] = df[['High','Low','Close','Volume']].ffill().bfill()
            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            vol = df['Volume'].astype(float)
            if len(close) < 60:
                return None, "❌DATA"
            curr = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else 0.0
            if curr <= 0:
                return None, "❌PRICE"
            # shares capacity
            maxshares = int(maxpositionvalueusd // curr)
            fractionalpossible = (maxpositionvalueusd / curr)
            if ALLOW_FRACTIONAL:
                cantrade = fractionalpossible >= 0.01
            else:
                cantrade = maxshares >= 1
            if not can_trade:
                return None, "❌PRICE"
            # Trend (relaxed)
            ma50 = close.rolling(50, min_periods=10).mean().iloc[-1]
            ma200 = close.rolling(200, min_periods=50).mean().iloc[-1] if len(close) >= 50 else None
            if ma200 is not None:
                if not (curr > ma50 or ma50 > ma200):
                    return None, "❌TREND"
            else:
                if not (curr > ma50):
                    return None, "❌TREND"
            # ATR and tightness
            tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14, min_periods=7).mean().iloc[-1]
            if pd.isna(atr14) or atr14 <= 0:
                atr14 = max((high - low).rolling(14, minperiods=7).mean().iloc[-1] if not pd.isna((high - low).rolling(14, minperiods=7).mean().iloc[-1]) else 0.0, 1e-6)
            atr_pct = atr14 / curr if curr > 0 else 0.0
            tightness = (high.iloc[-5:].max() - low.iloc[-5:].min()) / (atr14 if atr14 > 0 else 1.0)
            maxtightness = MAXTIGHTNESS_BASE
            if marketisbull and vix < 20:
                maxtightness = MAXTIGHTNESS_BASE * 1.4
            elif vix > 25:
                maxtightness = MAXTIGHTNESS_BASE * 0.9
            if tightness > max_tightness:
                return None, "❌LOOSE"
            # Score
            score = 0; reasons = []
            try:
                if tightness < 0.8:
                    score += 30; reasons.append("VCP+++")
                elif tightness < 1.2:
                    score += 20; reasons.append("VCP+")
                vol50 = vol.rolling(50, min_periods=10).mean().iloc[-1]
                if not pd.isna(vol50) and vol.iloc[-1] < vol50:
                    score += 15; reasons.append("VolDry")
                mom5 = close.rolling(5, min_periods=3).mean().iloc[-1]
                mom20 = close.rolling(20, min_periods=10).mean().iloc[-1]
                if not pd.isna(mom5) and not pd.isna(mom20) and (mom5 / mom20) > 1.02:
                    score += 20; reasons.append("Mom+")
                if ma200 is not None and ((ma50 - ma200) / ma200) > 0.03:
                    score += 20; reasons.append("Trend+")
                elif ma200 is None and (curr > ma50):
                    score += 10; reasons.append("Trend?")
            except Exception:
                pass
            # Backtest
            bt = simulatepastperformance_v2(df, sector)
            winrate = bt.get('winrate', 0) / 100.0
            posvalusd, frac = PositionSizer.calculateposition(capusd, winrate, 2.0, atrpct, vix, secexposures.get(sector, 0.0))
            # Convert to shares
            if ALLOW_FRACTIONAL:
                estshares = posval_usd / curr if curr > 0 else 0.0
            else:
                estshares = int(posval_usd // curr) if curr > 0 else 0
                if estshares < 1 and maxshares >= 1:
                    est_shares = 1
            if not ALLOWFRACTIONAL and estshares < 1:
                return None, "❌PRICE"
            if not ALLOWFRACTIONAL and estshares > max_shares:
                estshares = maxshares
            pivot = high.iloc[-5:].max() * 1.002
            stop = pivot - (atr14 * ATRSTOPMULT)
            result = {
                'score': int(score),
                'reasons': ' '.join(reasons),
                'pivot': pivot,
                'stop': stop,
                'sector': sector,
                'bt': bt,
                'posusd': posval_usd,
                'pos_frac': frac,
                'estshares': estshares,
                'tightness': tightness,
                'price': curr,
                'atrpct': atrpct,
                'vol': int(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0
            }
            return result, "✅PASS"
        except Exception as e:
            logger.exception("Analyze error for %s: %s", ticker, e)
            return None, "❌ERROR"

---------------------------

Messaging (LINE optional)

---------------------------
def send_line(msg):
    logger.info("LINE message prepared.")
    if not ACCESSTOKEN or not USERID:
        logger.debug("LINE credentials missing; skipping send.")
        return
    try:
        requests.post("https://api.line.me/v2/bot/message/push",
                      headers={"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"},
                      json={"to":USER_ID, "messages":[{"type":"text", "text":msg}]}, timeout=10)
    except Exception as e:
        logger.warning("LINE send failed: %s", e)

---------------------------

Main mission

---------------------------
def run_mission():
    fx = getcurrentfx_rate()
    vix = get_vix()
    isbull, marketstatus,  = checkmarket_trend()
    logger.info("Market: %s | VIX: %.1f | FX: ¥%.2f", market_status, vix, fx)
    initialcapusd = jpytousd(INITIALCAPITALJPY, fx)
    tradingcapusd = initialcapusd * TRADING_RATIO
    results = []
    stats = {"Earnings":0, "Sector":0, "Trend":0, "Price":0, "Loose":0, "Data":0, "Pass":0, "Error":0}
    sec_exposures = {s: 0.0 for s in set(TICKERS.values())}
    for ticker, sector in TICKERS.items():
        try:
            earningsflag = isearningsnear(ticker, dayswindow=2)
            if earnings_flag:
                stats["Earnings"] += 1
            sectorflag = not sectoris_strong(sector)
            if sector_flag:
                stats["Sector"] += 1
            dft = safedownload(ticker, period="700d")
            if dft is None or dft.empty:
                stats["Data"] += 1
                logger.debug("No data for %s", ticker)
                continue
            maxposvalusd = tradingcapusd * MAXPOSITION_SIZE
            res, reason = StrategicAnalyzerV2.analyze_ticker(
                ticker, dft, sector, maxposvalusd, vix, secexposures, tradingcapusd, isbull
            )
            if res:
                res['isearnings'] = earningsflag
                res['issectorweak'] = sector_flag
                results.append((ticker, res))
                if not earningsflag and not sectorflag:
                    stats["Pass"] += 1
            else:
                if "TREND" in reason:
                    stats["Trend"] += 1
                elif "PRICE" in reason:
                    stats["Price"] += 1
                elif "LOOSE" in reason:
                    stats["Loose"] += 1
                elif "DATA" in reason:
                    stats["Data"] += 1
                elif "ERROR" in reason:
                    stats["Error"] += 1
        except Exception as e:
            logger.exception("Loop error for %s: %s", ticker, e)
            stats["Error"] += 1
            continue
    all_sorted = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    passed = [r for r in allsorted if r[1]['score'] >= MINSCORE and not r[1]['isearnings'] and not r[1]['issector_weak']]
    report_lines = []
    report_lines.append("SENTINEL v25.1 DIAGNOSTIC")
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    reportlines.append(f"Mkt: {marketstatus}")
    report_lines.append(f"VIX: {vix:.1f} | FX: ¥{fx:.2f}")
    report_lines.append("="*40)
    report_lines.append("【STATISTICS】")
    report_lines.append(f"Analyzed: {len(TICKERS)} tickers")
    report_lines.append(f"Blocked by Earnings: {stats['Earnings']}")
    report_lines.append(f"Blocked by Sector:   {stats['Sector0, "Pass":0, "Error":0}
    sec_exposures = {s: 0.0 for s in set(TICKERS.values())}
    for ticker, sector in TICKERS.items():
        try:
            earningsflag = isearningsnear(ticker, dayswindow=2)
            if earnings_flag:
                stats["Earnings"] += 1
            sectorflag = not sectoris_strong(sector)
            if sector_flag:
                stats["Sector"] += 1
            dft = safedownload(ticker, period="700d")
            if dft is None or dft.empty:
                stats["Data"] += 1
                logger.debug("No data for %s", ticker)
                continue
            maxposvalusd = tradingcapusd * MAXPOSITION_SIZE
            res, reason = StrategicAnalyzerV2.analyze_ticker(
                ticker, dft, sector, maxposvalusd, vix, secexposures, tradingcapusd, isbull
            )
            if res:
                res['isearnings'] = earningsflag
                res['issectorweak'] = sector_flag
                results.append((ticker, res))
                if not earningsflag and not sectorflag:
                    stats["Pass"] += 1
            else:
                if "TREND" in reason:
                    stats["Trend"] += 1
                elif "PRICE" in reason:
                    stats["Price"] += 1
                elif "LOOSE" in reason:
                    stats["Loose"] += 1
                elif "DATA" in reason:
                    stats["Data"] += 1
                elif "ERROR" in reason:
                    stats["Error"] += 1
        except Exception as e:
            logger.exception("Loop error for %s: %s", ticker, e)
            stats["Error"] += 1
            continue
    all_sorted = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    passed = [r for r in allsorted if r[1]['score'] >= MINSCORE and not r[1]['isearnings'] and not r[1]['issector_weak']]
    report_lines = []
    report_lines.append("SENTINEL v25.1 DIAGNOSTIC")
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    reportlines.append(f"Mkt: {marketstatus}")
    report_lines.append(f"VIX: {vix:.1f} | FX: ¥{fx:.2f}")
    report_lines.append("="*40)
    report_lines.append("【STATISTICS】")
    report_lines.append(f"Analyzed: {len(TICKERS)} tickers")
    report_lines.append(f"Blocked by Earnings: {stats['Earnings']}")
    report_lines.append(f"Blocked by Sector:   {stats['Sector']}")
    report_lines.append(f"Blocked by Trend:    {stats['Trend']}")
    reportlines.append(f"VCP/Score Pass:      {len(allsorted)}")
    report_lines.append(f"Data Error:          {stats['Data']} / Internal Error: {stats['Error']}")
    report_lines.append("="*40)
    report_lines.append("【BUY SIGNALS】")
    if not passed:
        report_lines.append("No candidates passed all strict filters.")
    else:
        for i, (ticker, r) in enumerate(passed[:MAX_NOTIFICATIONS], 1):
            posusd = r['posusd']
            price = r['price']
            estshares = r['estshares']
            roundtripcostusd = TransactionCostModel.calculatetotalcostusd(posusd)
            sharesstr = f"{estshares:.4f}" if ALLOWFRACTIONAL else f"{int(estshares)}"
            report_lines.append(f"★ [{i}] {ticker} {r['score']}pt")
            reportlines.append(f"   Entry: ${r['pivot']:.2f} / Price: ${price:.2f} / Shares est: {sharesstr}")
            reportlines.append(f"   Pos(USD): ${posusd:,.2f} / RoundtripCost(USD): ${roundtripcostusd:,.2f}")
            report_lines.append(f"   BT: {r['bt']['message']} Tight:{r['tightness']:.2f}")
    report_lines.append("\n【ANALYSIS TOP 10 (RAW)】")
    for i, (ticker, r) in enumerate(all_sorted[:10], 1):
        tag = "✅OK"
        if r.get('is_earnings'): tag = "❌EARN"
        elif r.get('issectorweak'): tag = "❌SEC"
        elif r['score'] < MIN_SCORE: tag = "❌SCOR"
        report_lines.append(f"{i}. {ticker:5} {r['score']}pt | {tag}")
        reportlines.append(f"   Tight:{r['tightness']:.2f} WR:{r['bt']['winrate']:.0f}% PosUSD:{r['posusd']:.0f} Shares:{r.get('est_shares')}")
    finalreport = "\n".join(reportlines)
    logger.info("\n%s", final_report)
    sendline(finalreport)

if name == "main":
    run_mission()
