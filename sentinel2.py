#!/usr/bin/env python3
# SENTINEL v28_JP PRIORITIZED - 日本株向けETF/株式分割通知
# マルチ次元スコアリング with VCP成熟度と機関投資家分析
# 哲学: 「価格と出来高が原因、ニュースは結果」
# 目標: ニュース発表前に機関の買いを捉えて年間10%リターン
# 
# 要件: pandas, numpy, yfinance, requests, beautifulsoup4
# 使用法: python sentinel_v28_jp.py

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')

# ---------------------------
# ロギング設定
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("SENTINEL_JP")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler("sentinel_debug_jp.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(fh)

# ---------------------------
# 設定 (日本株向け)
# ---------------------------
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID") or os.getenv("USER_ID")

INITIAL_CAPITAL_JPY = 3_500_000  # 350万円
TRADING_RATIO = 0.75

ATR_STOP_MULT = 2.0
MAX_POSITION_SIZE = 0.25
MAX_SECTOR_CONCENTRATION = 0.40

# 最小ポジションサイズ（日本円ベース）
MIN_POSITION_JPY = 50_000  # 5万円以上

MAX_TIGHTNESS_BASE = 2.0
MAX_NOTIFICATIONS = 10
MIN_DAILY_VOLUME_JPY = 1_000_000_000  # 10億円以上

COMMISSION_RATE = 0.001  # 日本株手数料率（0.1%）
SLIPPAGE_RATE = 0.001
FX_SPREAD_RATE = 0.0005

REWARD_MULTIPLIERS = {'aggressive': 2.5, 'stable': 2.0}
# 日本株向けアグレッシブセクター
AGGRESSIVE_SECTORS = ['半導体', 'AI', 'ソフトウェア', 'セキュリティ', '自動車', 'クラウド', 'サービス', 'プラットフォーム', 'Fintech', '医療機器']

ALLOW_FRACTIONAL = True

CACHE_DIR = Path("./cache_jp")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------
# 日本株ティッカーユニバース
# スクリーナー結果と主要日本株を含む
# ---------------------------
TICKERS = {
    # 出来高増加トレンド銘柄（優先）
    '6197.T': 'ITサービス',  # ホソイ（出来高急増中）
    '4384.T': '小売',        # 出来高5.25倍増
    '3593.T': '小売',        # 出来高5.01倍増
    '212A.T': 'ITサービス',  # 出来高3.50倍増
    '6744.T': '電気機器',    # 出来高2.31倍増
    
    # 主要日本株成長株
    '7203.T': '自動車',      # トヨタ自動車
    '6758.T': 'エレクトロニクス',  # ソニーグループ
    '9984.T': '通信',        # ソフトバンクグループ
    '8035.T': '半導体',      # 東京エレクトロン
    '6861.T': '精密機器',    # キーエンス
    '6098.T': 'ITサービス',  # リクルート
    '9432.T': '通信',        # NTT
    '9433.T': '通信',        # KDDI
    '4063.T': '化学',        # 信越化学
    '6981.T': '電子部品',    # 村田製作所
    '7751.T': '精密機器',    # キヤノン
    '4901.T': '化学',        # 富士フイルム
    '4502.T': '製薬',        # 武田薬品工業
    '4519.T': '製薬',        # 中外製薬
    '7267.T': '自動車',      # ホンダ
    '7269.T': '自動車',      # スズキ
    '6501.T': '電機',        # 日立製作所
    '6503.T': '電機',        # 三菱電機
    '6506.T': '電機',        # 安川電機
    '6976.T': '電子部品',    # 太陽誘電
    '7733.T': '精密機器',    # オリンパス
    '7735.T': '精密機器',    # SCREENホールディングス
    '6723.T': '半導体',      # ルネサスエレクトロニクス
    '6702.T': '電機',        # 富士通
    '6752.T': '電機',        # パナソニックホールディングス
    '5801.T': '金属',        # 古河電気工業
    '5802.T': '金属',        # 住友電気工業
    '5803.T': '金属',        # フジクラ
    '5713.T': '非鉄金属',    # 住友金属鉱山
    '5016.T': '石油',        # 日揮ホールディングス
    '7974.T': '小売',        # 任天堂
    '7011.T': '重機',        # 三菱重工業
    '7012.T': '重機',        # 川崎重工業
    '7013.T': '重機',        # IHI
    '6323.T': 'ロボット',    # RORZE
    '6367.T': '機械',        # ダイキン工業
    '6479.T': '機械',        # ミネベアミツミ
    '6594.T': '電機',        # 日電産
    '3659.T': 'ITサービス',  # ネクソン
    '4307.T': 'ITサービス',  # 野村総合研究所
    '4689.T': 'ITサービス',  # ヤフー
    '4578.T': 'バイオ',      # 大塚製薬
    '4528.T': 'バイオ',      # 小野薬品工業
    '4583.T': 'バイオ',      # カイオム・バイオサイエンス
    '4592.T': 'バイオ',      # サンバイオ
    '7832.T': '小売',        # バンダイナムコホールディングス
    '7836.T': '小売',        # アバント
    '7976.T': '小売',        # ミツカン
    '8001.T': '商社',        # 伊藤忠商事
    '8002.T': '商社',        # 丸紅
    '8058.T': '商社',        # 三菱商事
    '8267.T': '小売',        # イオン
    '8306.T': '銀行',        # 三菱UFJフィナンシャル・グループ
    '8316.T': '銀行',        # 三井住友フィナンシャルグループ
    '8411.T': '銀行',        # みずほフィナンシャルグループ
    '8601.T': '証券',        # 大和証券グループ本社
    '8604.T': '証券',        # 野村ホールディングス
    '8697.T': '証券',        # 日本取引所グループ
    '8801.T': '不動産',      # 三井不動産
    '8802.T': '不動産',      # 三菱地所
    '8804.T': '不動産',      # 東京建物
    '9020.T': '鉄道',        # 東日本旅客鉄道
    '9021.T': '鉄道',        # 西日本旅客鉄道
    '9022.T': '鉄道',        # 東海旅客鉄道
    '9101.T': '海運',        # 日本郵船
    '9104.T': '海運',        # 商船三井
    '9107.T': '海運',        # 川崎汽船
    '9201.T': '航空',        # 日本航空
    '9202.T': '航空',        # 全日本空輸
    '9301.T': '倉庫',        # 三井倉庫ホールディングス
    '9437.T': '通信',        # NTTドコモ
    '9681.T': 'ITサービス',  # 東京エネルギーシステム
    '9735.T': 'サービス',    # セコム
    '9766.T': 'サービス',    # コナミホールディングス
    '9983.T': '小売',        # ファーストリテイリング
    '9994.T': '小売',        # ヤマダデンキ
}

# ETFカテゴリー（除外用）
ETF_CATEGORIES = ['インデックス', 'セクター', 'REIT', '債券', 'レバレッジ']

# ================================
# セクター → ETF マッピング（日本株向け）
# ================================

SECTOR_ETF = {
    'インデックス': '1321.T',  # TOPIX連動型上場投信
    'セクター': '1321.T',
    '半導体': 'SOXX',
    'エレクトロニクス': '1321.T',
    'AI': '1321.T',
    'ソフトウェア': '1321.T',
    'ITサービス': '1321.T',
    '通信': '1321.T',
    '自動車': '1321.T',
    '精密機器': '1321.T',
    '電気機器': '1321.T',
    '電子部品': '1321.T',
    '電機': '1321.T',
    '重機': '1321.T',
    '機械': '1321.T',
    'ロボット': '1321.T',
    '化学': '1321.T',
    '製薬': '1321.T',
    'バイオ': '1321.T',
    '医療機器': '1321.T',
    '金属': '1321.T',
    '非鉄金属': '1321.T',
    '石油': '1321.T',
    '小売': '1321.T',
    '商社': '1321.T',
    '銀行': '1321.T',
    '証券': '1321.T',
    '不動産': '1321.T',
    '鉄道': '1321.T',
    '海運': '1321.T',
    '航空': '1321.T',
    '倉庫': '1321.T',
    'サービス': '1321.T',
    '不明': '1321.T'
}

# ---------------------------
# VCP 成熟度アナライザー
# ---------------------------
class VCPAnalyzer:
    @staticmethod
    def calculate_vcp_maturity(df, result):
        try:
            maturity = 0
            signals = []

            # 1. ボラティリティ収縮 (40 pts)
            tightness = result.get('tightness', 999)
            if tightness < 1.0:
                maturity += 40
                signals.append("極度収縮")
            elif tightness < 1.5:
                maturity += 30
                signals.append("強収縮")
            elif tightness < 2.0:
                maturity += 20
                signals.append("収縮中")
            elif tightness < 2.5:
                maturity += 10
                signals.append("軽度収縮")

            # 2. 高値切り上げ (30 pts)
            if 'Close' in df.columns and len(df) >= 20:
                close = df['Close'].astype(float)
                recent_lows = close.iloc[-20:].rolling(5).min()

                if len(recent_lows) >= 10:
                    if recent_lows.iloc[-1] > recent_lows.iloc[-10] > recent_lows.iloc[-20]:
                        maturity += 30
                        signals.append("切上完了")
                    elif recent_lows.iloc[-1] > recent_lows.iloc[-10]:
                        maturity += 20
                        signals.append("切上中")
                    elif recent_lows.iloc[-1] >= recent_lows.iloc[-5]:
                        maturity += 10
                        signals.append("底固め")

            # 3. 出来高減少 (20 pts)
            reasons = result.get('reasons', '')
            if 'VolDry' in reasons or '出来高減少' in reasons:
                maturity += 20
                signals.append("出来高縮小")

            # 4. MA構造 (10 pts)
            if 'Trend+' in reasons or 'Trend++' in reasons:
                maturity += 10
                signals.append("MA整列")
            elif 'MA50+' in reasons or 'MA20+' in reasons:
                maturity += 5
                signals.append("MA形成中")

            # ステージ判定
            if maturity >= 85:
                stage = "🔥爆発直前"
                stage_en = "BREAKOUT_READY"
            elif maturity >= 70:
                stage = "⚡初動圏"
                stage_en = "EARLY_STAGE"
            elif maturity >= 50:
                stage = "👁形成中"
                stage_en = "FORMING"
            elif maturity >= 30:
                stage = "⏳準備段階"
                stage_en = "PREPARING"
            else:
                stage = "❌未成熟"
                stage_en = "IMMATURE"

            return {
                'maturity': maturity,
                'stage': stage,
                'stage_en': stage_en,
                'signals': signals
            }

        except Exception as e:
            logger.debug("VCP成熟度計算失敗: %s", e)
            return {
                'maturity': 0,
                'stage': '❌計算不可',
                'stage_en': 'UNKNOWN',
                'signals': []
            }

# ---------------------------
# 包括的なシグナル品質スコアリング
# ---------------------------
class SignalQuality:
    @staticmethod
    def calculate_comprehensive_score(result, vcp_analysis, inst_analysis):
        # テクニカルスコア (0-40) - VCP成熟度ベース
        tech_score = min(vcp_analysis['maturity'] * 0.4, 40)

        # リスクリターンスコア (0-30)
        ev = result['bt'].get('net_expectancy', 0)
        wr = result['bt'].get('winrate', 0) / 100.0

        rr_score = 0
        if ev > 0.6 and wr > 0.5:
            rr_score = 30
        elif ev > 0.4 and wr > 0.45:
            rr_score = 25
        elif ev > 0.3 and wr > 0.42:
            rr_score = 20
        elif ev > 0.2 and wr > 0.40:
            rr_score = 15
        elif ev > 0.1 and wr > 0.35:
            rr_score = 10
        elif ev > 0 and wr > 0.3:
            rr_score = 5

        # 機関投資家スコア (0-30)
        risk_score = inst_analysis.get('risk_score', 0)

        if risk_score < 0:
            inst_score = 30
        elif risk_score < 20:
            inst_score = 25
        elif risk_score < 40:
            inst_score = 20
        elif risk_score < 60:
            inst_score = 15
        else:
            inst_score = max(0, 15 - (risk_score - 60) // 10)

        total = tech_score + rr_score + inst_score

        # ティア分類
        if total >= 75:
            tier = 'コア'
            tier_emoji = '🔥'
            priority = 1
        elif total >= 60:
            tier = 'セカンダリー'
            tier_emoji = '⚡'
            priority = 2
        elif total >= 45:
            tier = 'ウォッチ'
            tier_emoji = '👁'
            priority = 3
        else:
            tier = '回避'
            tier_emoji = '❌'
            priority = 4

        return {
            'total_score': int(total),
            'tech_score': int(tech_score),
            'rr_score': int(rr_score),
            'inst_score': int(inst_score),
            'tier': tier,
            'tier_emoji': tier_emoji,
            'priority': priority
        }

    @staticmethod
    def generate_why_now(result, vcp_analysis, inst_analysis, quality):
        reasons = []

        # VCPステージ
        if vcp_analysis['maturity'] >= 85:
            reasons.append("VCP完成・爆発待ち")
        elif vcp_analysis['maturity'] >= 70:
            reasons.append("初動開始可能性")
        elif vcp_analysis['maturity'] >= 50:
            reasons.append("形成進行中")

        # 機関投資家分析
        overall = inst_analysis.get('overall', '中立')
        if overall == '✅低リスク':
            reasons.append("機関買い圧力検知")
        elif overall == '🚨高リスク':
            reasons.append("⚠️機関売り圧力")

        # リスクリターン品質
        ev = result['bt'].get('net_expectancy', 0)
        if ev > 0.6:
            reasons.append("高RR（非対称優位）")
        elif ev > 0.4:
            reasons.append("良好RR")

        # 価格アクション
        current = result.get('price', 0)
        entry = result.get('pivot', 0)
        if entry > 0 and current < entry * 0.99:
            discount = ((entry - current) / entry) * 100
            reasons.append(f"押目-{discount:.1f}%")

        return " | ".join(reasons) if reasons else "基準達成"

# ---------------------------
# 機関投資家モジュール（日本株向け簡略化）
# ---------------------------
class InsiderTracker:
    @staticmethod
    def get_insider_activity(ticker, days=30):
        try:
            cache_file = CACHE_DIR / f"insider_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)

            stock = yf.Ticker(ticker)
            insider_trades = stock.insider_transactions

            if insider_trades is None or insider_trades.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': '中立'}

            cutoff_date = datetime.now() - timedelta(days=days)
            recent = insider_trades[insider_trades.index >= cutoff_date]

            if recent.empty:
                return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': '中立'}

            buy_shares = recent[recent['Shares'] > 0]['Shares'].sum()
            sell_shares = abs(recent[recent['Shares'] < 0]['Shares'].sum())
            ratio = sell_shares / max(buy_shares, 1)

            if ratio > 5:
                signal = '🚨売り'
            elif ratio > 2:
                signal = '⚠️注意'
            elif ratio < 0.5:
                signal = '✅買い'
            else:
                signal = '中立'

            result = {'buy_shares': int(buy_shares), 'sell_shares': int(sell_shares), 'ratio': float(ratio), 'signal': signal}
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.debug("インサイダー追跡失敗 %s: %s", ticker, e)
            return {'buy_shares': 0, 'sell_shares': 0, 'ratio': 0, 'signal': '中立'}

class ShortInterestTracker:
    @staticmethod
    def get_short_interest(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            short_percent = info.get('shortPercentOfFloat', 0)
            if short_percent > 20:
                signal = '🚨高'
            elif short_percent > 10:
                signal = '⚠️上昇'
            else:
                signal = '正常'
            return {'short_percent': float(short_percent), 'signal': signal}
        except Exception:
            return {'short_percent': 0, 'signal': '不明'}

class InstitutionalOwnership:
    @staticmethod
    def get_institutional_holdings(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            inst_percent = info.get('heldPercentInstitutions', 0) * 100
            if inst_percent > 80:
                signal = '✅強'
            elif inst_percent < 40:
                signal = '⚠️弱'
            else:
                signal = '普通'
            return {'inst_percent': float(inst_percent), 'signal': signal}
        except Exception:
            return {'inst_percent': 0, 'signal': '不明'}

class OptionFlowAnalyzer:
    @staticmethod
    def get_put_call_ratio(ticker):
        try:
            stock = yf.Ticker(ticker)
            exp_dates = stock.options
            if not exp_dates:
                return {'put_call_ratio': 1.0, 'signal': '不明'}
            opt = stock.option_chain(exp_dates[0])
            calls = opt.calls
            puts = opt.puts
            if calls.empty or puts.empty:
                return {'put_call_ratio': 1.0, 'signal': '不明'}
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            ratio = put_volume / max(call_volume, 1)
            if ratio > 1.5:
                signal = '🐻弱気'
            elif ratio < 0.7:
                signal = '🐂強気'
            else:
                signal = '中立'
            return {'put_call_ratio': float(ratio), 'signal': signal}
        except Exception:
            return {'put_call_ratio': 1.0, 'signal': '不明'}

class MacroAnalyzer:
    @staticmethod
    def get_macro_environment():
        try:
            cache_file = CACHE_DIR / f"macro_{datetime.now().strftime('%Y%m%d')}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            # 日本市場向けマクロ指標
            # TOPIXのデータを使用
            topix_data = yf.download("^TPX", period="5d", progress=False)
            topix = float(topix_data['Close'].iloc[-1]) if not topix_data.empty and 'Close' in topix_data.columns else 2700.0
            
            # VIXの代わりに日本版VIX（簡易的）
            vix_data = yf.download("^VIX", period="5d", progress=False)
            vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty and 'Close' in vix_data.columns else 20.0
            
            # 日本国債10年利回り（簡易的）
            jgb_data = yf.download("1570.T", period="5d", progress=False)
            jgb_10y = float(jgb_data['Close'].iloc[-1]) if not jgb_data.empty and 'Close' in jgb_data.columns else 0.7
            
            rate_env = '⚠️上昇' if jgb_10y > 1.0 else '✅低金利'
            vol_env = '✅低ボラ' if vix < 20 else '⚠️上昇'
            
            result = {
                'topix': topix, 
                'jgb_10y': jgb_10y, 
                'vix': vix, 
                'rate_env': rate_env, 
                'vol_env': vol_env
            }
            
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            return result
        except Exception:
            return {'topix': 2700.0, 'jgb_10y': 0.7, 'vix': 20.0, 'rate_env': '不明', 'vol_env': '不明'}

class InstitutionalAnalyzer:
    @staticmethod
    def analyze(ticker):
        signals = {}
        alerts = []
        risk_score = 0

        insider = InsiderTracker.get_insider_activity(ticker)
        signals['insider'] = insider
        if insider['signal'] == '🚨売り':
            alerts.append(f"インサイダー売{insider['ratio']:.1f}倍")
            risk_score += 30
        elif insider['signal'] == '✅買い':
            risk_score -= 10

        short = ShortInterestTracker.get_short_interest(ticker)
        signals['short'] = short
        if short['signal'] == '🚨高':
            alerts.append(f"空売{short['short_percent']:.0f}%")
            risk_score += 20

        inst = InstitutionalOwnership.get_institutional_holdings(ticker)
        signals['institutional'] = inst
        if inst['signal'] == '⚠️弱':
            alerts.append(f"機関{inst['inst_percent']:.0f}%")
            risk_score += 10

        options = OptionFlowAnalyzer.get_put_call_ratio(ticker)
        signals['options'] = options
        if options['signal'] == '🐻弱気':
            alerts.append(f"P/C{options['put_call_ratio']:.2f}")
            risk_score += 15
        elif options['signal'] == '🐂強気':
            risk_score -= 10

        if risk_score > 60:
            overall = '🚨高リスク'
        elif risk_score > 30:
            overall = '⚠️注意'
        elif risk_score < 0:
            overall = '✅低リスク'
        else:
            overall = '中立'

        return {'signals': signals, 'alerts': alerts, 'risk_score': risk_score, 'overall': overall}

# ---------------------------
# コアモジュール（日本株向け）
# ---------------------------
def get_current_fx_rate():
    try:
        data = yf.download("JPY=X", period="5d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty and 'Close' in data.columns else 152.0
    except Exception:
        return 152.0

def get_vix():
    try:
        data = yf.download("^VIX", period="5d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty and 'Close' in data.columns else 20.0
    except Exception:
        return 20.0

def check_market_trend():
    try:
        # 日本市場のトレンドをTOPIXで確認
        topix = yf.download("^TPX", period="400d", progress=False)
        if topix.empty:
            return True, "不明", 0.0
        close = topix['Close'].dropna() if 'Close' in topix.columns else None
        if close is None or len(close) < 210:
            return True, "不明", 0.0
        curr = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        dist = ((curr - ma200) / ma200) * 100
        return curr > ma200, f"{'強気' if curr > ma200 else '弱気'} ({dist:+.1f}%)", dist
    except Exception:
        return True, "不明", 0.0

def safe_download(ticker, period="700d", retry=3):
    for attempt in range(retry):
        try:
            time.sleep(1.5)  # レート制限保護
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            return df.to_frame() if isinstance(df, pd.Series) else df
        except Exception as e:
            logger.warning("yf.download 試行 %d 失敗 %s: %s", attempt+1, ticker, e)
            time.sleep(3 + attempt * 2)
    return pd.DataFrame()

def ensure_df(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.copy() if df is not None else pd.DataFrame()

def safe_rolling_last(series, window, min_periods=1, default=np.nan):
    try:
        val = series.rolling(window, min_periods=min_periods).mean().iloc[-1]
        return float(val) if not pd.isna(val) else default
    except Exception:
        try:
            return float(series.iloc[-1])
        except Exception:
            return default

def is_earnings_near(ticker, days_window=2):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None:
            return False
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            date_val = cal.iloc[0, 0]
        elif isinstance(cal, dict):
            date_val = cal.get('Earnings Date', [None])[0]
        else:
            return False
        if date_val is None:
            return False
        ed = pd.to_datetime(date_val).date()
        days_until = (ed - datetime.now().date()).days
        return abs(days_until) <= days_window
    except Exception:
        return False

def sector_is_strong(sector):
    try:
        sector_key = str(sector[0]) if isinstance(sector, (pd.Series, np.ndarray, list, tuple)) and len(sector) > 0 else str(sector)
        etf = SECTOR_ETF.get(sector_key)
        if not etf:
            return True
        etf_sym = str(etf[0]) if isinstance(etf, (pd.Series, np.ndarray, list, tuple)) and len(etf) > 0 else str(etf)
        df = safe_download(etf_sym, period="300d", retry=2)
        if df is None or df.empty:
            return True
        if 'Close' not in df.columns:
            for c in df.columns:
                if 'close' in str(c).lower():
                    df['Close'] = df[c]
                    break
        if 'Close' not in df.columns:
            return True
        close = df['Close'].dropna()
        if len(close) < 220:
            return True
        ma200 = close.rolling(200, min_periods=50).mean().dropna()
        if len(ma200) < 12:
            return True
        last = float(ma200.iloc[-1])
        prev = float(ma200.iloc[-10])
        slope = (last - prev) / prev if prev != 0 else 0.0
        return bool(slope >= 0.0)
    except Exception as e:
        logger.exception("sector_is_strong エラー %s: %s", sector, e)
        return True

class TransactionCostModel:
    @staticmethod
    def calculate_total_cost_jpy(val_jpy):
        return (val_jpy * COMMISSION_RATE + val_jpy * SLIPPAGE_RATE) * 2

class PositionSizer:
    @staticmethod
    def calculate_position(cap_jpy, winrate, rr, atr_pct, vix, sec_exp):
        try:
            if rr <= 0:
                return 0.0, 0.0
            kelly = max(0.0, (winrate - (1 - winrate) / rr))
            kelly = min(kelly * 0.5, MAX_POSITION_SIZE)
            v_f = 0.7 if atr_pct > 0.05 else 0.85 if atr_pct > 0.03 else 1.0
            m_f = 0.7 if vix > 30 else 0.85 if vix > 20 else 1.0
            s_f = 0.7 if sec_exp > MAX_SECTOR_CONCENTRATION else 1.0
            final_frac = min(kelly * v_f * m_f * s_f, MAX_POSITION_SIZE)
            pos_val = cap_jpy * final_frac

            # 最小ポジションサイズを適用
            if pos_val > 0 and pos_val < MIN_POSITION_JPY:
                pos_val = MIN_POSITION_JPY
                final_frac = pos_val / cap_jpy

            return pos_val, final_frac
        except Exception:
            return 0.0, 0.0

def simulate_past_performance_v2(df, sector, lookback_years=3):
    try:
        df = ensure_df(df)
        if 'Close' not in df.columns:
            for c in df.columns:
                if 'close' in str(c).lower():
                    df['Close'] = df[c]; break
        if 'High' not in df.columns:
            for c in df.columns:
                if 'high' in str(c).lower():
                    df['High'] = df[c]; break
        if 'Low' not in df.columns:
            for c in df.columns:
                if 'low' in str(c).lower():
                    df['Low'] = df[c]; break
        close = df['Close'].dropna() if 'Close' in df.columns else pd.Series(dtype=float)
        high = df['High'].dropna() if 'High' in df.columns else pd.Series(dtype=float)
        low = df['Low'].dropna() if 'Low' in df.columns else pd.Series(dtype=float)
        if len(close) < 60 or len(high) < 60 or len(low) < 60:
            return {'winrate':0, 'net_expectancy':0, 'message':'データ不足'}
        end_date = close.index[-1]
        start_date = end_date - pd.DateOffset(years=lookback_years)
        mask = close.index >= start_date
        close = close.loc[mask]
        high = high.loc[mask]
        low = low.loc[mask]
        if len(close) < 60:
            return {'winrate':0, 'net_expectancy':0, 'message':'期間不足'}
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=7).mean().dropna()
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        wins = 0; losses = 0; total_r = 0.0; samples = 0
        for i in range(50, len(close)-40):
            try:
                window_high = high.iloc[i-5:i].max()
                pivot = window_high * 1.002
                if high.iloc[i] < pivot:
                    continue
                ma50 = close.rolling(50, min_periods=10).mean().iloc[i]
                ma200 = close.rolling(200, min_periods=50).mean().iloc[i] if i >= 200 else None
                if ma200 is not None and not (close.iloc[i] > ma50 or ma50 > ma200):
                    continue
                stop_dist = atr.iloc[i] * ATR_STOP_MULT if i < len(atr) else atr.iloc[-1] * ATR_STOP_MULT
                entry = pivot
                target = entry + stop_dist * reward_mult
                outcome = None
                for j in range(1, 31):
                    if i + j >= len(close):
                        break
                    if high.iloc[i+j] >= target:
                        outcome = '勝利'; break
                    if low.iloc[i+j] <= entry - stop_dist:
                        outcome = '敗北'; break
                if outcome is None:
                    last_close = close.iloc[min(i+30, len(close)-1)]
                    pnl = (last_close - entry) / stop_dist if stop_dist != 0 else 0
                    if pnl > 0:
                        wins += 1; total_r += min(pnl, reward_mult)
                    else:
                        losses += 1; total_r -= abs(pnl)
                    samples += 1
                else:
                    samples += 1
                    if outcome == '勝利':
                        wins += 1; total_r += reward_mult
                    else:
                        losses += 1; total_r -= 1.0
            except Exception:
                continue
        total = wins + losses
        if total < 8:
            return {'winrate':0, 'net_expectancy':0, 'message':f'サンプル不足:{total}'}
        wr = (wins / total)
        ev = total_r / total
        return {'winrate':wr*100, 'net_expectancy':ev - 0.05, 'message':f"勝率{wr*100:.0f}% 期待値{ev:.2f}"}
    except Exception as e:
        logger.exception("バックテストエラー: %s", e)
        return {'winrate':0, 'net_expectancy':0, 'message':'BTエラー'}

class StrategicAnalyzerV2:
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_position_value_jpy, vix, sec_exposures, cap_jpy, market_is_bull):
        try:
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return None, "❌データなし"
            df = ensure_df(df)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = [' '.join(map(str, c)).strip() for c in df.columns.values]
                except Exception:
                    pass
            if 'Close' not in df.columns:
                for c in df.columns:
                    if 'adj close' in str(c).lower() or 'adj_close' in str(c).lower():
                        df['Close'] = df[c]; break
                if 'Close' not in df.columns:
                    for c in df.columns:
                        if 'close' in str(c).lower():
                            df['Close'] = df[c]; break
            if 'High' not in df.columns:
                for c in df.columns:
                    if 'high' in str(c).lower():
                        df['High'] = df[c]; break
            if 'Low' not in df.columns:
                for c in df.columns:
                    if 'low' in str(c).lower():
                        df['Low'] = df[c]; break
            if 'Volume' not in df.columns:
                for c in df.columns:
                    if 'volume' in str(c).lower():
                        df['Volume'] = df[c]; break
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            if 'Close' not in df.columns:
                logger.debug("analyze_ticker: 銘柄=%s でClose列不足, cols=%s", ticker, list(df.columns))
                return None, "❌データなし"
            df = df.dropna(subset=['Close'])
            if df.empty:
                return None, "❌データなし"
            df[['High','Low','Close','Volume']] = df[['High','Low','Close','Volume']].ffill().bfill()
            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            vol = df['Volume'].astype(float)
            if len(close) < 60:
                return None, "❌データ不足"
            curr = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else 0.0
            if curr <= 0:
                return None, "❌価格不正"
            try:
                max_shares = int(max_position_value_jpy // curr)
            except Exception:
                max_shares = 0
            fractional_possible = (max_position_value_jpy / curr) if curr > 0 else 0.0
            if ALLOW_FRACTIONAL:
                can_trade = fractional_possible >= 0.01
            else:
                can_trade = max_shares >= 1
            if not can_trade:
                return None, "❌価格高"
            ma50 = safe_rolling_last(close, 50, min_periods=10, default=curr)
            ma200 = safe_rolling_last(close, 200, min_periods=50, default=None) if len(close) >= 50 else None
            if ma200 is not None:
                if not (curr > ma50 or ma50 > ma200):
                    return None, "❌トレンド弱"
            else:
                if not (curr > ma50):
                    return None, "❌トレンド弱"
            try:
                tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr14 = tr.rolling(14, min_periods=7).mean().iloc[-1]
            except Exception:
                atr14 = np.nan
            if pd.isna(atr14) or atr14 <= 0:
                try:
                    alt = (high - low).rolling(14, min_periods=7).mean().iloc[-1]
                    atr14 = max(alt if not pd.isna(alt) else 0.0, 1e-6)
                except Exception:
                    atr14 = 1e-6
            atr_pct = atr14 / curr if curr > 0 else 0.0
            try:
                tightness = (high.iloc[-5:].max() - low.iloc[-5:].min()) / (atr14 if atr14 > 0 else 1.0)
            except Exception:
                tightness = 999.0
            max_tightness = MAX_TIGHTNESS_BASE
            if market_is_bull and vix < 20:
                max_tightness = MAX_TIGHTNESS_BASE * 1.4
            elif vix > 25:
                max_tightness = MAX_TIGHTNESS_BASE * 0.9
            if tightness > max_tightness:
                return None, "❌ボラ高"
            score = 0; reasons = []
            try:
                if tightness < 0.8:
                    score += 30; reasons.append("VCP+++")
                elif tightness < 1.2:
                    score += 20; reasons.append("VCP+")
                vol50 = safe_rolling_last(vol, 50, min_periods=10, default=np.nan)
                if not pd.isna(vol50) and vol.iloc[-1] < vol50:
                    score += 15; reasons.append("出来高減少")
                mom5 = safe_rolling_last(close, 5, min_periods=3, default=np.nan)
                mom20 = safe_rolling_last(close, 20, min_periods=10, default=np.nan)
                if not pd.isna(mom5) and not pd.isna(mom20) and (mom5 / mom20) > 1.02:
                    score += 20; reasons.append("モメンタム上昇")
                if ma200 is not None and ((ma50 - ma200) / ma200) > 0.03:
                    score += 20; reasons.append("上昇トレンド")
                elif ma200 is None and (curr > ma50):
                    score += 10; reasons.append("トレンド形成中")
            except Exception:
                pass
            bt = simulate_past_performance_v2(df, sector)
            winrate = bt.get('winrate', 0) / 100.0
            try:
                pos_val_jpy, frac = PositionSizer.calculate_position(cap_jpy, winrate, 2.0, atr_pct, vix, float(sec_exposures.get(sector, 0.0)))
            except Exception as e:
                logger.exception("PositionSizerエラー %s: %s", ticker, e)
                pos_val_jpy, frac = 0.0, 0.0
            try:
                if ALLOW_FRACTIONAL:
                    est_shares = pos_val_jpy / curr if curr > 0 else 0.0
                else:
                    est_shares = int(pos_val_jpy // curr) if curr > 0 else 0
                    if est_shares < 1 and max_shares >= 1:
                        est_shares = 1
                if not ALLOW_FRACTIONAL and est_shares < 1:
                    return None, "❌価格高"
                if not ALLOW_FRACTIONAL and est_shares > max_shares:
                    est_shares = max_shares
            except Exception:
                return None, "❌価格高"
            pivot = high.iloc[-5:].max() * 1.002 if len(high) >= 5 else curr * 1.002
            stop = pivot - (atr14 * ATR_STOP_MULT)
            result = {
                'score': int(score),
                'reasons': ' '.join(reasons),
                'pivot': pivot,
                'stop': stop,
                'sector': sector,
                'bt': bt,
                'pos_jpy': pos_val_jpy,
                'pos_frac': frac,
                'est_shares': est_shares,
                'tightness': tightness,
                'price': curr,
                'atr_pct': atr_pct,
                'vol': int(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0,
                'df': df
            }
            return result, "✅合格"
        except Exception as e:
            logger.exception("分析エラー %s: %s", ticker, e)
            return None, "❌エラー"

def send_line(msg):
    logger.info("LINEメッセージ準備完了")
    if not ACCESS_TOKEN or not USER_ID:
        logger.debug("LINE認証情報不足；送信スキップ")
        return
    
    # 5000文字制限対応（4800文字で分割）
    MAX_LEN = 4800
    
    if len(msg) <= MAX_LEN:
        messages_to_send = [msg]
    else:
        lines = msg.split('\n')
        messages_to_send = []
        current = ""
        
        for line in lines:
            if len(current) + len(line) + 1 < MAX_LEN:
                current += line + '\n'
            else:
                if current:
                    messages_to_send.append(current)
                current = line + '\n'
        
        if current:
            messages_to_send.append(current)
    
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type":"application/json", "Authorization":f"Bearer {ACCESS_TOKEN}"}
    
    for i, msg_part in enumerate(messages_to_send):
        payload = {"to": USER_ID, "messages":[{"type":"text", "text":msg_part}]}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"LINE送信成功 (part {i+1}/{len(messages_to_send)})")
            else:
                logger.warning(f"LINE送信失敗 part {i+1} status={resp.status_code}")
            time.sleep(1)
        except Exception as e:
            logger.exception(f"LINE送信失敗 part {i+1}: {e}")

# ---------------------------
# ETFと株式を分割するヘルパー関数
# ---------------------------
def split_etf_stock(data_list):
    """(ticker, data)タプルのリストをETFと株式リストに分割"""
    etfs = []
    stocks = []

    for ticker, data in data_list:
        sector = data.get('sector', '')
        if sector in ETF_CATEGORIES:
            etfs.append((ticker, data))
        else:
            stocks.append((ticker, data))

    return etfs, stocks

# ---------------------------
# メインミッション - v28_JP PRIORITIZED with ETF/Stock split
# ---------------------------
def run_mission():
    # 日本市場向けマクロ環境
    macro = MacroAnalyzer.get_macro_environment()
    vix = macro['vix']
    is_bull, market_status, _ = check_market_trend()
    
    logger.info("市場: %s | 日本VIX: %.1f | TOPIX: %.0f", market_status, vix, macro['topix'])
    logger.info("日本国債10年: %.2f%% | %s %s", macro['jgb_10y'], macro['rate_env'], macro['vol_env'])
    
    initial_cap_jpy = INITIAL_CAPITAL_JPY
    trading_cap_jpy = initial_cap_jpy * TRADING_RATIO
    
    results = []
    stats = {"決算近":0, "セクター弱":0, "トレンド弱":0, "価格高":0, "ボラ高":0, "データ不足":0, "合格":0, "エラー":0}
    sec_exposures = {s: 0.0 for s in set(TICKERS.values())}

    for ticker, sector in TICKERS.items():
        try:
            earnings_flag = is_earnings_near(ticker, days_window=2)
            if earnings_flag:
                stats["決算近"] += 1
            
            try:
                sector_flag = not bool(sector_is_strong(sector))
            except Exception:
                logger.exception("セクターチェック失敗 %s", sector)
                sector_flag = False
            if sector_flag:
                stats["セクター弱"] += 1
            
            df_t = safe_download(ticker, period="700d")
            if df_t is None or df_t.empty:
                stats["データ不足"] += 1
                logger.debug("%s データなし", ticker)
                continue
            
            max_pos_val_jpy = trading_cap_jpy * MAX_POSITION_SIZE
            res, reason = StrategicAnalyzerV2.analyze_ticker(
                ticker, df_t, sector, max_pos_val_jpy, vix, sec_exposures, trading_cap_jpy, is_bull
            )
            
            if res:
                res['is_earnings'] = earnings_flag
                res['is_sector_weak'] = sector_flag
                
                vcp_analysis = VCPAnalyzer.calculate_vcp_maturity(res['df'], res)
                res['vcp_analysis'] = vcp_analysis
                
                inst_analysis = InstitutionalAnalyzer.analyze(ticker)
                res['institutional'] = inst_analysis
                
                quality = SignalQuality.calculate_comprehensive_score(res, vcp_analysis, inst_analysis)
                res['quality'] = quality
                
                why_now = SignalQuality.generate_why_now(res, vcp_analysis, inst_analysis, quality)
                res['why_now'] = why_now
                
                results.append((ticker, res))
                
                if not earnings_flag and not sector_flag:
                    stats["合格"] += 1
                    sec_exposures[sector] += res['pos_jpy'] / trading_cap_jpy
            else:
                if reason is None:
                    stats["エラー"] += 1
                elif "トレンド弱" in reason:
                    stats["トレンド弱"] += 1
                elif "価格高" in reason:
                    stats["価格高"] += 1
                elif "ボラ高" in reason:
                    stats["ボラ高"] += 1
                elif "データ不足" in reason:
                    stats["データ不足"] += 1
                elif "エラー" in reason:
                    stats["エラー"] += 1
                else:
                    stats["エラー"] += 1
                    
        except Exception as e:
            logger.exception("ループエラー %s: %s", ticker, e)
            stats["エラー"] += 1
            continue

    all_sorted = sorted(results, key=lambda x: x[1]['quality']['total_score'], reverse=True)
    
    passed_core = [r for r in all_sorted if r[1]['quality']['tier'] == 'コア' and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]
    passed_secondary = [r for r in all_sorted if r[1]['quality']['tier'] == 'セカンダリー' and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]
    passed_watch = [r for r in all_sorted if r[1]['quality']['tier'] == 'ウォッチ' and not r[1].get('is_earnings', False) and not r[1].get('is_sector_weak', False)]

    # ETFと株式に分割
    core_etfs, core_stocks = split_etf_stock(passed_core)
    secondary_etfs, secondary_stocks = split_etf_stock(passed_secondary)
    watch_etfs, watch_stocks = split_etf_stock(passed_watch)
    all_etfs, all_stocks = split_etf_stock(all_sorted)

    report_lines = []
    report_lines.append("="*50)
    report_lines.append("SENTINEL v28_JP PRIORITIZED - ETF/株式分割")
    report_lines.append("ニュース発表前に機関の買いを捉える")
    report_lines.append("="*50)
    report_lines.append(datetime.now().strftime("%m/%d %H:%M"))
    report_lines.append("")
    report_lines.append(f"市場: {market_status} | 日本VIX: {vix:.1f} | TOPIX: {macro['topix']:.0f}")
    report_lines.append(f"日本国債10年: {macro['jgb_10y']:.2f}% | {macro['rate_env']} {macro['vol_env']}")
    report_lines.append("")
    report_lines.append("【目標】年間10% / 月間0.8%")
    report_lines.append(f"資金: ¥{INITIAL_CAPITAL_JPY:,} | 取引資金: ¥{trading_cap_jpy:,}")
    report_lines.append("")
    report_lines.append("【統計】")
    report_lines.append(f"分析銘柄: {len(TICKERS)} | 合格: {len(all_sorted)}")
    report_lines.append(f"除外: 決算={stats['決算近']} セクター弱={stats['セクター弱']} トレンド弱={stats['トレンド弱']} ボラ高={stats['ボラ高']}")
    report_lines.append(f"エラー: データ不足={stats['データ不足']} 内部={stats['エラー']}")
    report_lines.append("="*50)

    report_lines.append("\n【優先シグナル】")
    report_lines.append(f"🔥 コア株式: {len(core_stocks)} | 🏆 コアETF: {len(core_etfs)}")
    report_lines.append(f"⚡ セカンダリー株式: {len(secondary_stocks)} | 🏅 セカンダリーETF: {len(secondary_etfs)}")
    report_lines.append(f"👁 ウォッチ株式: {len(watch_stocks)} | 📊 ウォッチETF: {len(watch_etfs)}")
    report_lines.append("")

    # 本日の最優先銘柄（株式のみ）
    if core_stocks:
        top = core_stocks[0]
        ticker = top[0]
        r = top[1]

        actual_shares = int(r['est_shares'])
        actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

        report_lines.append(f"🎯 本日の最優先銘柄（株式）: {ticker}")
        report_lines.append(f"   スコア: {r['quality']['total_score']}/100 (テク:{r['quality']['tech_score']} RR:{r['quality']['rr_score']} 機関:{r['quality']['inst_score']})")

        if actual_shares > 0:
            report_lines.append(f"   {actual_shares}株 @ ¥{r['price']:,.0f} = ¥{actual_cost:,.0f}")
        else:
            report_lines.append(f"   ⚠️ 1株未満 (¥{r['price']:,.0f})")

        report_lines.append(f"   理由: {r['why_now']}")
        report_lines.append("")

    # コア株式 - 即時検討
    if core_stocks:
        report_lines.append("🔥 コア株式 - 即時検討 (上位5)")
        for i, (ticker, r) in enumerate(core_stocks[:5], 1):
            q = r['quality']
            vcp = r['vcp_analysis']
            inst = r['institutional']

            actual_shares = int(r['est_shares'])
            actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

            report_lines.append(f"\n[{i}] {ticker} {q['total_score']}/100 | VCP:{vcp['maturity']}% {vcp['stage']}")
            report_lines.append(f"    テク:{q['tech_score']} RR:{q['rr_score']} 機関:{q['inst_score']} | リスク:{inst['risk_score']}")

            if actual_shares > 0:
                report_lines.append(f"    {actual_shares}株 @ ¥{r['price']:,.0f} = ¥{actual_cost:,.0f} | エントリー: ¥{r['pivot']:,.0f}")
            else:
                report_lines.append(f"    ⚠️ 1株未満 (¥{r['price']:,.0f}) | エントリー: ¥{r['pivot']:,.0f}")

            report_lines.append(f"    BT: {r['bt']['message']} | 収縮度:{r['tightness']:.2f}")
            report_lines.append(f"    💡 {r['why_now']}")
            if inst['alerts']:
                report_lines.append(f"    ⚠️  {' | '.join(inst['alerts'][:3])}")

    # コアETF - 即時検討
    if core_etfs:
        report_lines.append("\n🏆 コアETF - 即時検討 (上位5)")
        for i, (ticker, r) in enumerate(core_etfs[:5], 1):
            q = r['quality']
            vcp = r['vcp_analysis']

            actual_shares = int(r['est_shares'])
            actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

            report_lines.append(f"\n[{i}] {ticker} {q['total_score']}/100 | VCP:{vcp['maturity']}% {vcp['stage']}")

            if actual_shares > 0:
                report_lines.append(f"    {actual_shares}株 @ ¥{r['price']:,.0f} = ¥{actual_cost:,.0f} | エントリー: ¥{r['pivot']:,.0f}")
            else:
                report_lines.append(f"    ⚠️ 1株未満 (¥{r['price']:,.0f}) | エントリー: ¥{r['pivot']:,.0f}")

            report_lines.append(f"    {r['why_now']}")

    # セカンダリー株式
    if secondary_stocks:
        report_lines.append("\n⚡ セカンダリー株式 - 条件付き監視 (上位10)")
        for i, (ticker, r) in enumerate(secondary_stocks[:10], 1):
            q = r['quality']
            vcp = r['vcp_analysis']

            actual_shares = int(r['est_shares'])
            actual_cost = actual_shares * r['price'] if actual_shares > 0 else 0

            report_lines.append(f"\n[{i}] {ticker} {q['total_score']}/100 | VCP:{vcp['maturity']}% {vcp['stage']}")

            if actual_shares > 0:
                report_lines.append(f"    {actual_shares}株 @ ¥{r['price']:,.0f} = ¥{actual_cost:,.0f} | エントリー: ¥{r['pivot']:,.0f}")
            else:
                report_lines.append(f"    ⚠️ 1株未満 (¥{r['price']:,.0f}) | エントリー: ¥{r['pivot']:,.0f}")

            report_lines.append(f"    {r['why_now']}")

    # ウォッチリスト要約
    if watch_stocks:
        watch_str = ", ".join([f"{t} {r['quality']['total_score']}" for t, r in watch_stocks[:15]])
        report_lines.append("\n👁 ウォッチ株式 - 監視中 (上位15)")
        report_lines.append(f"    {watch_str}")

    if watch_etfs:
        etf_watch_str = ", ".join([f"{t} {r['quality']['total_score']}" for t, r in watch_etfs[:5]])
        report_lines.append("\n📊 ウォッチETF - 監視中 (上位5)")
        report_lines.append(f"    {etf_watch_str}")

    # トップ15個別株式包括分析
    report_lines.append("\n" + "="*50)
    report_lines.append("【トップ15個別株式 - 包括分析】")
    for i, (ticker, r) in enumerate(all_stocks[:15], 1):
        q = r['quality']
        vcp = r['vcp_analysis']
        tag = "✅OK"
        if r.get('is_earnings'): 
            tag = "❌決算"
        elif r.get('is_sector_weak'): 
            tag = "❌セクター"
        report_lines.append(f"\n{i:2}. {ticker:8} {q['total_score']:3}/100 {q['tier_emoji']} | {tag}")
        report_lines.append(f"    VCP:{vcp['maturity']:3}% {vcp['stage']} | 勝率:{r['bt']['winrate']:.0f}% 期待値:{r['bt']['net_expectancy']:+.2f}")
        report_lines.append(f"    {' '.join(vcp['signals'])}")
        report_lines.append(f"    {r['why_now']}")

    # トップ5 ETF包括分析
    report_lines.append("\n" + "="*50)
    report_lines.append("【トップ5 ETF - 包括分析】")
    for i, (ticker, r) in enumerate(all_etfs[:5], 1):
        q = r['quality']
        vcp = r['vcp_analysis']
        tag = "✅OK"
        if r.get('is_earnings'): 
            tag = "❌決算"
        elif r.get('is_sector_weak'): 
            tag = "❌セクター"
        report_lines.append(f"\n{i:2}. {ticker:8} {q['total_score']:3}/100 {q['tier_emoji']} | {tag}")
        report_lines.append(f"    VCP:{vcp['maturity']:3}% {vcp['stage']} | 勝率:{r['bt']['winrate']:.0f}% 期待値:{r['bt']['net_expectancy']:+.2f}")
        report_lines.append(f"    {' '.join(vcp['signals'])}")
        report_lines.append(f"    {r['why_now']}")

    report_lines.append("\n" + "="*50)
    report_lines.append("【哲学】")
    report_lines.append("✓ 価格と出来高が原因")
    report_lines.append("✓ ニュースは結果")
    report_lines.append("✓ ヘッドライン前に機関の買いを捉える")
    report_lines.append("="*50)

    final_report = "\n".join(report_lines)
    logger.info("\n%s", final_report)
    send_line(final_report)

if __name__ == "__main__":
    run_mission()
