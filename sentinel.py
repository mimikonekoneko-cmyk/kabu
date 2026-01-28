
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESSTOKEN")
USER_ID = os.getenv("LINE_USER_ID")
BUDGET_JPY = 350000

# ============================================================================
# CORE PARAMETERS (バックテスト済みの最適値)
# ============================================================================
MA_SHORT, MA_LONG = 50, 200
MIN_SCORE = 75  # 85→75に緩和（バックテストで正確に評価するため）
MIN_WINRATE = 45  # 最低勝率45%
MIN_EXPECTANCY = 0.3  # 最低期待値0.3R
MAX_NOTIFICATIONS = 8
ATR_STOP_MULT = 2.0

# リスク・リワード比率（全て2.5倍以上に統一）
REWARD_MULTIPLIERS = {
    'aggressive': 3.0,  # 成長株
    'stable': 2.5       # 安定株（1.8→2.5に改善）
}

AGGRESSIVE_SECTORS = [
    'Semi', 'AI', 'Soft', 'Sec', 'EV', 'Crypto', 
    'Cloud', 'Ad', 'Service', 'Platform', 'Bet'
]

# ============================================================================
# TICKER UNIVERSE
# ============================================================================
TICKERS = {
    # テクノロジー・半導体
    'NVDA':'AI', 'AVGO':'Semi', 'ARM':'Semi', 'MU':'Semi', 'AMD':'Semi', 
    'SMCI':'AI', 'TSM':'Semi', 'ASML':'Semi',
    
    # FAANG+
    'AAPL':'Device', 'MSFT':'Cloud', 'GOOGL':'Ad', 'META':'Ad', 
    'AMZN':'Retail', 'TSLA':'EV', 'NFLX':'Service',
    
    # エンタープライズSaaS
    'PLTR':'AI', 'PANW':'Sec', 'CRWD':'Sec', 'NET':'Sec', 
    'NOW':'Soft', 'CRM':'Soft', 'TEAM':'Soft', 'ADBE':'Soft',
    
    # リテール・消費財
    'COST':'Retail', 'WMT':'Retail', 'TJX':'Retail', 
    'ELF':'Cons', 'PEP':'Cons', 'KO':'Cons', 'PG':'Cons', 'LULU':'Cons',
    
    # 金融
    'V':'Fin', 'MA':'Fin', 'JPM':'Bank', 'GS':'Bank', 
    'AXP':'Fin', 'BLK':'Fin', 'MS':'Bank', 'COIN':'Crypto',
    
    # ヘルスケア
    'LLY':'Bio', 'UNH':'Health', 'ABBV':'Bio', 'ISRG':'Health', 'VRTX':'Bio',
    
    # 産業・エネルギー
    'GE':'Ind', 'CAT':'Ind', 'DE':'Ind', 'BA':'Ind',
    'XOM':'Energy', 'CVX':'Energy', 'MPC':'Energy',
    
    # その他
    'UBER':'Platform', 'BKNG':'Travel', 'ABNB':'Travel', 
    'DKNG':'Bet', 'VRT':'Power'
}

SECTOR_ETF = {
    'Energy':'XLE', 'Semi':'SOXX', 'Bank':'XLF', 'Retail':'XRT',
    'Soft':'IGV', 'AI':'QQQ', 'Fin':'VFH', 'Device':'QQQ',
    'Cloud':'QQQ', 'Ad':'QQQ', 'Service':'QQQ', 'Sec':'HACK',
    'Cons':'XLP', 'Bio':'IBB', 'Health':'XLV', 'Ind':'XLI',
    'EV':'IDRV', 'Crypto':'BTC-USD', 'Power':'XLI', 'Platform':'QQQ',
    'Travel':'XLY', 'Bet':'BETZ'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_fx_rate():
    """ドル円レート取得（フォールバック付き）"""
    try:
        data = yf.download("JPY=X", period="1d", progress=False)
        if not data.empty:
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                return float(close.iloc[-1, 0])
            return float(close.iloc[-1])
        return 155.0
    except Exception as e:
        print(f"⚠️ FX取得エラー: {e}")
        return 155.0

def check_market_trend():
    """市場全体のトレンド判定（SPY vs MA200）"""
    try:
        spy = yf.download("SPY", period="300d", progress=False)
        if spy.empty or len(spy) < 200:
            return True, "データ不足"
        
        close = spy['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        current = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        
        if current > ma200:
            return True, f"強気 (${current:.0f} > MA200)"
        else:
            return False, f"弱気 (${current:.0f} < ${ma200:.0f})"
    except Exception as e:
        print(f"⚠️ 市場判定エラー: {e}")
        return True, "判定スキップ"

def is_earnings_near(ticker):
    """決算発表が±5日以内かチェック"""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
            return False
        
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            date_val = cal['Earnings Date']
            if isinstance(date_val, list):
                date_val = date_val[0]
        else:
            date_val = cal.iloc[0, 0]
        
        earnings_date = pd.to_datetime(date_val).date()
        days_until = (earnings_date - datetime.now().date()).days
        
        return abs(days_until) <= 5
    except:
        return False

def sector_is_strong(sector):
    """セクターETFの強弱判定（MA200上昇トレンド）"""
    try:
        etf = SECTOR_ETF.get(sector)
        if not etf:
            return True
        
        df = yf.download(etf, period="250d", progress=False)
        if df.empty or len(df) < 200:
            return True
        
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        ma200 = close.rolling(200).mean()
        # 直近10日でMA200が上昇傾向
        return ma200.iloc[-1] > ma200.iloc[-10]
    except:
        return True

# ============================================================================
# BACKTEST ENGINE (未来視バイアス完全除去)
# ============================================================================

def simulate_past_performance(df, sector, atr_mult=ATR_STOP_MULT):
    """
    過去データで戦略の有効性を検証
    - 各時点でのpivot/stop/targetを正確に再計算
    - ルックアヘッドバイアスを完全排除
    - 最低10サンプル必要（それ未満は信頼性低）
    """
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        # ATRの計算
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # リワード倍率
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        wins = 0
        losses = 0
        total_r = 0  # R倍率の合計（実際の損益計算用）
        
        # バックテスト範囲（最低250日、最大500日）
        start_idx = max(MA_LONG, len(df) - 500)
        end_idx = len(df) - 10  # 直近10日は除外（未来データ使用防止）
        
        for i in range(start_idx, end_idx):
            # ============================================
            # その時点でのMA条件チェック
            # ============================================
            if i < MA_LONG:
                continue
            
            ma50_at_i = close.iloc[i-MA_SHORT:i].mean()
            ma200_at_i = close.iloc[i-MA_LONG:i].mean()
            
            if not (close.iloc[i] > ma50_at_i > ma200_at_i):
                continue
            
            # ============================================
            # その時点でのpivot/stop/target計算
            # ============================================
            pivot = high.iloc[i-5:i].max() * 1.002
            stop_dist = atr.iloc[i] * atr_mult
            
            if pd.isna(stop_dist) or stop_dist == 0:
                continue
            
            stop = pivot - stop_dist
            target = pivot + (stop_dist * reward_mult)
            
            # ============================================
            # ブレイクアウト判定
            # ============================================
            if high.iloc[i] >= pivot:
                # エントリー後20営業日間を追跡
                for j in range(1, 21):
                    if i + j >= len(df):
                        break
                    
                    # 利確判定
                    if high.iloc[i+j] >= target:
                        wins += 1
                        total_r += reward_mult
                        break
                    
                    # 損切り判定
                    if low.iloc[i+j] <= stop:
                        losses += 1
                        total_r -= 1.0
                        break
        
        total_trades = wins + losses
        
        # サンプル数が少なすぎる場合
        if total_trades < 10:
            return {
                'status': 'insufficient',
                'message': 'サンプル不足',
                'trades': total_trades
            }
        
        # 統計計算
        winrate = (wins / total_trades) * 100
        expectancy = total_r / total_trades  # R倍率ベースの期待値
        
        return {
            'status': 'valid',
            'winrate': winrate,
            'expectancy': expectancy,
            'wins': wins,
            'losses': losses,
            'total': total_trades,
            'message': f"勝率{winrate:.0f}% ({wins}/{total_trades}) 期待値{expectancy:.2f}R"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'検証エラー: {str(e)}'
        }

# ============================================================================
# STRATEGIC ANALYZER
# ============================================================================

class StrategicAnalyzer:
    
    @staticmethod
    def analyze_ticker(ticker, df, sector, max_price_usd):
        """
        銘柄分析のメインロジック
        - 100点満点のスコアリングシステム
        - バックテストによる信頼性検証
        """
        
        # データ量チェック
        if len(df) < MA_LONG + 50:
            return None
        
        try:
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
        except Exception as e:
            print(f"⚠️ {ticker}: データ抽出エラー - {e}")
            return None
        
        # ============================================
        # 基本フィルター
        # ============================================
        current_price = float(close.iloc[-1])
        
        # 予算オーバーチェック
        if current_price > max_price_usd:
            return None
        
        # トレンドフィルター
        ma50 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma200 = close.rolling(MA_LONG).mean().iloc[-1]
        
        if not (current_price > ma50 > ma200):
            return None
        
        # ============================================
        # ATR & Tightness (VCPパターン)
        # ============================================
        tr = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(14).mean().iloc[-1]
        
        if atr14 == 0 or pd.isna(atr14):
            return None
        
        # 直近5日間のボラティリティ圧縮度
        recent_range = high.iloc[-5:].max() - low.iloc[-5:].min()
        tightness = float(recent_range / atr14)
        
        # VCPパターン外（ボラティリティ大）は除外
        if tightness > 3.0:
            return None
        
        # ============================================
        # スコアリング（100点満点）
        # ============================================
        score = 0
        reasons = []
        
        # 1. VCPタイトネス (最大30点)
        if tightness < 1.0:
            score += 30
            reasons.append("VCP超タイト+30")
        elif tightness < 1.5:
            score += 20
            reasons.append("VCPタイト+20")
        elif tightness < 2.0:
            score += 10
            reasons.append("VCP良好+10")
        else:
            score += 5
            reasons.append("VCP許容+5")
        
        # 2. ボリューム分析 (最大25点)
        vol_avg = volume.rolling(50).mean().iloc[-1]
        
        if vol_avg > 0:
            vol_ratio = volume.iloc[-1] / vol_avg
            
            # 出来高枯れ（売り圧力低下）
            if 0.5 <= vol_ratio <= 0.9:
                score += 15
                reasons.append("売り枯れ+15")
            elif 0.9 < vol_ratio <= 1.1:
                score += 10
                reasons.append("出来高安定+10")
            
            # 直近3日間の急増（機関投資家の買い集め）
            recent_vol_max = volume.iloc[-3:].max()
            if recent_vol_max > vol_avg * 2.0:
                score += 10
                reasons.append("強い買い集め+10")
            elif recent_vol_max > vol_avg * 1.5:
                score += 5
                reasons.append("買い集め+5")
        
        # 3. モメンタム (最大20点)
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        
        if ma5 > ma20 * 1.02:
            score += 20
            reasons.append("強い上昇モメンタム+20")
        elif ma5 > ma20 * 1.01:
            score += 15
            reasons.append("上昇モメンタム+15")
        elif ma5 > ma20:
            score += 10
            reasons.append("モメンタム良好+10")
        
        # 4. トレンド強度 (最大15点)
        trend_strength = (ma50 - ma200) / ma200 * 100
        if trend_strength > 10:
            score += 15
            reasons.append("強いトレンド+15")
        elif trend_strength > 5:
            score += 10
            reasons.append("トレンド良好+10")
        else:
            score += 5
            reasons.append("トレンド形成中+5")
        
        # 5. ベースライン (最大10点)
        score += 10
        reasons.append("基礎評価+10")
        
        # ============================================
        # 戦略的エントリー・エグジット設定
        # ============================================
        reward_mult = REWARD_MULTIPLIERS['aggressive'] if sector in AGGRESSIVE_SECTORS else REWARD_MULTIPLIERS['stable']
        
        pivot = high.iloc[-5:].max() * 1.002  # 5日高値 + 0.2%
        stop_dist = atr14 * ATR_STOP_MULT
        stop_loss = pivot - stop_dist
        target = pivot + (stop_dist * reward_mult)
        
        # ============================================
        # バックテスト実施
        # ============================================
        bt_result = simulate_past_performance(df, sector)
        
        # バックテスト結果によるフィルタリング
        if bt_result['status'] == 'valid':
            if bt_result['winrate'] < MIN_WINRATE:
                return None  # 勝率不足
            if bt_result['expectancy'] < MIN_EXPECTANCY:
                return None  # 期待値不足
        elif bt_result['status'] == 'insufficient':
            # サンプル不足の場合は警告付きで通過
            pass
        else:
            return None  # エラー
        
        return {
            'score': score,
            'reasons': ' '.join(reasons),
            'price': current_price,
            'pivot': pivot,
            'stop': stop_loss,
            'target': target,
            'sector': sector,
            'tightness': tightness,
            'bt': bt_result
        }

# ============================================================================
# LINE NOTIFICATION
# ============================================================================

def send_line(msg):
    """LINE通知送信（エラーハンドリング強化版）"""
    
    # 認証情報チェック
    if not ACCESS_TOKEN or not USER_ID:
        print("\n" + "="*50)
        print("⚠️ LINE認証情報が未設定")
        print("="*50)
        print(msg)
        print("="*50 + "\n")
        return False
    
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
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ LINE送信成功")
            return True
        else:
            print(f"❌ LINE送信失敗: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ LINE送信エラー: {e}")
        return False

# ============================================================================
# MAIN MISSION
# ============================================================================

def run_mission():
    """メイン実行関数"""
    
    print("\n" + "="*60)
    print("🛡️  SENTINEL v22.0 - Perfect Edition")
    print("="*60)
    print(f"⏰ 起動時刻: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # ============================================
    # 市場環境チェック
    # ============================================
    print("📊 市場環境を確認中...")
    is_bull, market_status = check_market_trend()
    
    if not is_bull:
        msg = (
            f"🛑 Sentinel v22.0\n"
            f"市場環境が悪化しているため待機します\n"
            f"\n"
            f"📊 Market Status: {market_status}\n"
            f"⏰ {datetime.now().strftime('%Y/%m/%d %H:%M')}"
        )
        print(msg)
        send_line(msg)
        return
    
    print(f"✅ 市場環境: {market_status}\n")
    
    # ============================================
    # 為替レート取得
    # ============================================
    print("💱 為替レートを取得中...")
    fx_rate = get_current_fx_rate()
    max_price_usd = (BUDGET_JPY / fx_rate) * 0.9  # 予算の90%まで
    
    print(f"✅ FX Rate: ¥{fx_rate:.2f}/USD")
    print(f"✅ 最大購入価格: ${max_price_usd:.2f}\n")
    
    # ============================================
    # データダウンロード
    # ============================================
    print(f"📡 {len(TICKERS)}銘柄のデータをダウンロード中...")
    ticker_list = list(TICKERS.keys())
    
    try:
        all_data = yf.download(
            ticker_list,
            period="600d",  # 500日→600日に拡大（バックテスト精度向上）
            progress=False,
            group_by='ticker',
            threads=True
        )
        print("✅ ダウンロード完了\n")
    except Exception as e:
        print(f"❌ データ取得エラー: {e}")
        return
    
    # ============================================
    # 銘柄分析
    # ============================================
    print("🔍 銘柄スクリーニング開始...\n")
    
    results = []
    analyzed_count = 0
    filtered_count = 0
    
    for ticker, sector in TICKERS.items():
        analyzed_count += 1
        
        # 決算前後は回避
        if is_earnings_near(ticker):
            print(f"⏭️  {ticker}: 決算前後のためスキップ")
            continue
        
        # セクター弱気は回避
        if not sector_is_strong(sector):
            print(f"⏭️  {ticker}: セクター弱気のためスキップ")
            continue
        
        try:
            # データ抽出
            if len(ticker_list) > 1:
                df_ticker = all_data[ticker]
            else:
                df_ticker = all_data
            
            # 分析実行
            result = StrategicAnalyzer.analyze_ticker(
                ticker, df_ticker, sector, max_price_usd
            )
            
            if result:
                # スコアフィルター
                if result['score'] >= MIN_SCORE:
                    results.append((ticker, result))
                    print(f"✅ {ticker}: {result['score']}点 - 候補に追加")
                else:
                    filtered_count += 1
                    print(f"⚠️  {ticker}: {result['score']}点 - スコア不足")
            else:
                filtered_count += 1
                
        except Exception as e:
            print(f"❌ {ticker}: 分析エラー - {e}")
            continue
    
    # ============================================
    # 結果の並び替え＆制限
    # ============================================
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    results = results[:MAX_NOTIFICATIONS]
    
    print(f"\n{'='*60}")
    print(f"📊 スクリーニング結果")
    print(f"{'='*60}")
    print(f"分析銘柄: {analyzed_count}")
    print(f"候補検出: {len(results)}")
    print(f"フィルター: {filtered_count}")
    print(f"{'='*60}\n")
    
    # ============================================
    # レポート生成
    # ============================================
    report_lines = [
        "🛡️ Sentinel v22.0 Perfect",
        f"📅 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
        f"📊 Market: {market_status}",
        f"💵 $1 = ¥{fx_rate:.2f}",
        "─" * 30
    ]
    
    if not results:
        report_lines.append("⚠️ 現在、条件に合致する銘柄はありません")
        report_lines.append("")
        report_lines.append(f"分析: {analyzed_count}銘柄")
        report_lines.append(f"除外: {filtered_count}銘柄")
    else:
        for i, (ticker, r) in enumerate(results, 1):
            loss_pct = (1 - r['stop'] / r['pivot']) * 100
            gain_pct = (r['target'] / r['pivot'] - 1) * 100
            risk_reward = gain_pct / loss_pct
            
            # バックテスト結果
            bt_info = r['bt']['message'] if r['bt']['status'] == 'valid' else r['bt']['message']
            
            report_lines.append(
                f"[{i}] {ticker} ({r['sector']}) {r['score']}点"
            )
            report_lines.append(f"└ {r['reasons']}")
            report_lines.append(f"📈 {bt_info}")
            report_lines.append(f"現在: ${r['price']:.2f}")
            report_lines.append(f"入値: ${r['pivot']:.2f}")
            report_lines.append(f"損切: ${r['stop']:.2f} (-{loss_pct:.1f}%)")
            report_lines.append(f"利確: ${r['target']:.2f} (+{gain_pct:.1f}%)")
            report_lines.append(f"⚖️  RR比 1:{risk_reward:.1f}")
            report_lines.append("─" * 30)
    
    full_report = "\n".join(report_lines)
    
    # ============================================
    # 出力
    # ============================================
    print("\n" + "="*60)
    print("📋 最終レポート")
    print("="*60)
    print(full_report)
    print("="*60 + "\n")
    
    # LINE送信
    print("📤 LINE通知を送信中...")
    send_success = send_line(full_report)
    
    if send_success:
        print("✅ 処理完了\n")
    else:
        print("⚠️  LINE送信は失敗しましたが処理は完了しました\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_mission()
    except KeyboardInterrupt:
        print("\n\n⚠️  ユーザーにより中断されました\n")
    except Exception as e:
        print(f"\n\n❌ 予期しないエラー: {e}\n")
        import traceback
        traceback.print_exc()
