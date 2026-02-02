#!/usr/bin/env python3
# record_data.py
# Record SENTINEL v28 signals and trades for dashboard

import json
from datetime import datetime
from pathlib import Path

# ===========================
# CONFIGURATION
# ===========================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SIGNALS_FILE = DATA_DIR / "signals.json"
TRADES_FILE = DATA_DIR / "trades.json"
PERFORMANCE_FILE = DATA_DIR / "performance.json"

# ===========================
# SIGNAL RECORDING
# ===========================

def record_signal(ticker, score, tier, entry, stop, target, shares, why_now, sector):
    """
    Record a signal from v28
    
    Usage:
        record_signal('FULC', 87, 'CORE', 11.30, 9.77, 14.36, 46, 'VCP+++', 'Bio')
    """
    # Load existing signals
    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE, 'r') as f:
            signals = json.load(f)
    else:
        signals = []
    
    # Calculate percentages
    stop_pct = ((stop - entry) / entry) * 100
    target_pct = ((target - entry) / entry) * 100
    
    # Create signal record
    signal = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'ticker': ticker,
        'score': score,
        'tier': tier,
        'entry': entry,
        'stop': stop,
        'target': target,
        'stop_pct': stop_pct,
        'target_pct': target_pct,
        'shares': shares,
        'why_now': why_now,
        'sector': sector
    }
    
    # Append and save
    signals.append(signal)
    
    with open(SIGNALS_FILE, 'w') as f:
        json.dump(signals, f, indent=2)
    
    print(f"✅ Recorded signal: {ticker} @ ${entry:.2f}")
    return signal

def record_multiple_signals(signals_list):
    """
    Record multiple signals at once
    
    Usage:
        signals = [
            ('FULC', 87, 'CORE', 11.30, 9.77, 14.36, 46, 'VCP+++', 'Bio'),
            ('TSM', 83, 'CORE', 346.19, 325.30, 387.97, 1, 'High RR', 'Semi'),
        ]
        record_multiple_signals(signals)
    """
    for sig in signals_list:
        record_signal(*sig)
    
    print(f"✅ Recorded {len(signals_list)} signals")

# ===========================
# TRADE RECORDING
# ===========================

def record_entry(ticker, entry_price, shares, stop, target):
    """
    Record a trade entry
    
    Usage:
        record_entry('FULC', 10.73, 10, 8.50, 50.00)
    """
    # Load existing trades
    if TRADES_FILE.exists():
        with open(TRADES_FILE, 'r') as f:
            trades = json.load(f)
    else:
        trades = []
    
    # Create trade record
    trade = {
        'id': f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'ticker': ticker,
        'status': 'OPEN',
        'entry_date': datetime.now().strftime('%Y-%m-%d'),
        'entry_time': datetime.now().strftime('%H:%M:%S'),
        'entry': entry_price,
        'shares': shares,
        'stop': stop,
        'target': target,
        'current_price': entry_price,
        'exit_date': None,
        'exit': None,
        'pnl': 0.0,
        'exit_reason': None
    }
    
    # Append and save
    trades.append(trade)
    
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)
    
    print(f"✅ Recorded entry: {ticker} @ ${entry_price:.2f}")
    return trade

def update_trade_price(ticker, current_price):
    """
    Update current price for active trade
    
    Usage:
        update_trade_price('FULC', 12.50)
    """
    if not TRADES_FILE.exists():
        print("❌ No trades found")
        return
    
    with open(TRADES_FILE, 'r') as f:
        trades = json.load(f)
    
    # Find active trade
    updated = False
    for trade in trades:
        if trade['ticker'] == ticker and trade['status'] == 'OPEN':
            trade['current_price'] = current_price
            updated = True
    
    if updated:
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
        print(f"✅ Updated {ticker} price: ${current_price:.2f}")
    else:
        print(f"❌ No active trade found for {ticker}")

def record_exit(ticker, exit_price, reason='MANUAL'):
    """
    Record a trade exit
    
    Usage:
        record_exit('FULC', 14.50, 'TARGET')
    
    Reasons: 'TARGET', 'STOP', 'TIMEOUT', 'MANUAL'
    """
    if not TRADES_FILE.exists():
        print("❌ No trades found")
        return
    
    with open(TRADES_FILE, 'r') as f:
        trades = json.load(f)
    
    # Find and close trade
    closed = False
    for trade in trades:
        if trade['ticker'] == ticker and trade['status'] == 'OPEN':
            trade['status'] = 'CLOSED'
            trade['exit_date'] = datetime.now().strftime('%Y-%m-%d')
            trade['exit'] = exit_price
            trade['pnl'] = ((exit_price - trade['entry']) / trade['entry']) * 100
            trade['exit_reason'] = reason
            closed = True
            
            pnl = trade['pnl']
            print(f"✅ Closed {ticker}: {pnl:+.2f}% ({reason})")
    
    if closed:
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
        
        # Update performance
        update_performance()
    else:
        print(f"❌ No active trade found for {ticker}")

# ===========================
# PERFORMANCE CALCULATION
# ===========================

def update_performance():
    """Update performance metrics"""
    if not TRADES_FILE.exists():
        return
    
    with open(TRADES_FILE, 'r') as f:
        trades = json.load(f)
    
    # Filter closed trades
    closed = [t for t in trades if t['status'] == 'CLOSED']
    
    if not closed:
        return
    
    # Calculate metrics
    total_trades = len(closed)
    wins = [t for t in closed if t['pnl'] > 0]
    losses = [t for t in closed if t['pnl'] <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = sum(t['pnl'] for t in wins) / win_count if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / loss_count if losses else 0
    
    total_return = sum(t['pnl'] for t in closed)
    
    # Sharpe ratio (simplified)
    if closed:
        returns = [t['pnl'] for t in closed]
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return)**2 for r in returns) / len(returns)) ** 0.5
        sharpe_ratio = (avg_return / std_dev) if std_dev > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Max drawdown (simplified)
    cumulative = 0
    peak = 0
    max_dd = 0
    
    for trade in closed:
        cumulative += trade['pnl']
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    
    # Create performance record
    performance = {
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'wins': win_count,
        'losses': loss_count,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd
    }
    
    # Save
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(performance, f, indent=2)
    
    print(f"✅ Updated performance: {win_rate:.1f}% WR, {total_return:+.2f}% total")

# ===========================
# QUICK HELPERS
# ===========================

def show_status():
    """Show current status"""
    # Signals
    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE, 'r') as f:
            signals = json.load(f)
        today = datetime.now().strftime('%Y-%m-%d')
        today_signals = [s for s in signals if s['date'] == today]
        print(f"📊 Signals: {len(signals)} total, {len(today_signals)} today")
    else:
        print("📊 Signals: 0")
    
    # Trades
    if TRADES_FILE.exists():
        with open(TRADES_FILE, 'r') as f:
            trades = json.load(f)
        active = [t for t in trades if t['status'] == 'OPEN']
        closed = [t for t in trades if t['status'] == 'CLOSED']
        print(f"💼 Trades: {len(active)} active, {len(closed)} closed")
    else:
        print("💼 Trades: 0")
    
    # Performance
    if PERFORMANCE_FILE.exists():
        with open(PERFORMANCE_FILE, 'r') as f:
            perf = json.load(f)
        print(f"📈 Performance: {perf['total_return']:+.2f}% ({perf['win_rate']:.1f}% WR)")
    else:
        print("📈 Performance: N/A")

def clear_all_data():
    """Clear all data (CAUTION!)"""
    confirm = input("⚠️  Clear all data? Type 'YES' to confirm: ")
    if confirm == 'YES':
        for file in [SIGNALS_FILE, TRADES_FILE, PERFORMANCE_FILE]:
            if file.exists():
                file.unlink()
        print("✅ All data cleared")
    else:
        print("❌ Cancelled")

# ===========================
# EXAMPLE USAGE
# ===========================

if __name__ == "__main__":
    print("SENTINEL v28 Data Recorder")
    print("=" * 50)
    print()
    print("Example usage:")
    print()
    print("# Record today's signals")
    print("record_signal('FULC', 87, 'CORE', 11.30, 9.77, 14.36, 46, 'VCP+++', 'Bio')")
    print()
    print("# Record a trade entry")
    print("record_entry('FULC', 10.73, 10, 8.50, 50.00)")
    print()
    print("# Update price")
    print("update_trade_price('FULC', 12.50)")
    print()
    print("# Record exit")
    print("record_exit('FULC', 14.50, 'TARGET')")
    print()
    print("# Show status")
    print("show_status()")
    print()
    print("=" * 50)
    print()
    
    # Show current status
    show_status()
