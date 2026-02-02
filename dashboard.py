#!/usr/bin/env python3
# dashboard.py
# SENTINEL v28 Dashboard
# Real-time tracking and visualization

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="SENTINEL v28 Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .signal-card {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# DATA PATHS
# ===========================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SIGNALS_FILE = DATA_DIR / "signals.json"
TRADES_FILE = DATA_DIR / "trades.json"
PERFORMANCE_FILE = DATA_DIR / "performance.json"

# ===========================
# DATA LOADING
# ===========================

@st.cache_data(ttl=60)
def load_signals():
    """Load signal history"""
    if not SIGNALS_FILE.exists():
        return []
    
    with open(SIGNALS_FILE, 'r') as f:
        return json.load(f)

@st.cache_data(ttl=60)
def load_trades():
    """Load trade history"""
    if not TRADES_FILE.exists():
        return []
    
    with open(TRADES_FILE, 'r') as f:
        return json.load(f)

@st.cache_data(ttl=60)
def load_performance():
    """Load performance metrics"""
    if not PERFORMANCE_FILE.exists():
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    with open(PERFORMANCE_FILE, 'r') as f:
        return json.load(f)

# ===========================
# HELPER FUNCTIONS
# ===========================

def get_today_signals():
    """Get today's signals"""
    signals = load_signals()
    today = datetime.now().strftime('%Y-%m-%d')
    
    return [s for s in signals if s['date'] == today]

def get_active_trades():
    """Get currently active trades"""
    trades = load_trades()
    return [t for t in trades if t['status'] == 'OPEN']

def calculate_daily_performance():
    """Calculate daily performance"""
    trades = load_trades()
    
    # Group by date
    daily_pnl = {}
    for trade in trades:
        if trade['status'] == 'CLOSED':
            date = trade['exit_date']
            pnl = trade['pnl']
            
            if date in daily_pnl:
                daily_pnl[date] += pnl
            else:
                daily_pnl[date] = pnl
    
    # Convert to DataFrame
    if daily_pnl:
        df = pd.DataFrame([
            {'date': k, 'pnl': v} for k, v in sorted(daily_pnl.items())
        ])
        df['date'] = pd.to_datetime(df['date'])
        df['cumulative'] = df['pnl'].cumsum()
        return df
    
    return pd.DataFrame(columns=['date', 'pnl', 'cumulative'])

# ===========================
# MAIN DASHBOARD
# ===========================

def main():
    # Header
    st.title("🎯 SENTINEL v28 Dashboard")
    st.markdown("Real-time VCP Signal Tracking & Performance Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Date range
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Filters
        st.subheader("Filters")
        show_core_only = st.checkbox("CORE signals only", value=False)
        min_score = st.slider("Minimum Score", 0, 100, 70)
        
        # Refresh
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    perf = load_performance()
    signals = load_signals()
    trades = load_trades()
    today_signals = get_today_signals()
    active_trades = get_active_trades()
    
    # ===========================
    # KEY METRICS
    # ===========================
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{perf['total_return']:+.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Win Rate",
            f"{perf['win_rate']:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Total Trades",
            f"{perf['total_trades']}",
            delta=f"{perf['wins']}W / {perf['losses']}L"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{perf['sharpe_ratio']:.2f}",
            delta=None
        )
    
    # ===========================
    # TODAY'S SIGNALS
    # ===========================
    st.header("🔥 Today's Signals")
    
    if today_signals:
        for signal in today_signals:
            # Filter by settings
            if show_core_only and signal['tier'] != 'CORE':
                continue
            if signal['score'] < min_score:
                continue
            
            # Display signal
            with st.container():
                cols = st.columns([1, 2, 2, 2, 1])
                
                with cols[0]:
                    tier_emoji = "🔥" if signal['tier'] == 'CORE' else "⚡" if signal['tier'] == 'SECONDARY' else "👁"
                    st.markdown(f"### {tier_emoji} {signal['ticker']}")
                    st.markdown(f"**Score: {signal['score']}/100**")
                
                with cols[1]:
                    st.markdown("**Entry Strategy**")
                    st.markdown(f"Entry: ${signal['entry']:.2f}")
                    st.markdown(f"Shares: {signal.get('shares', 'N/A')}")
                
                with cols[2]:
                    st.markdown("**Exit Strategy**")
                    st.markdown(f"Stop: ${signal['stop']:.2f} ({signal['stop_pct']:.1f}%)")
                    st.markdown(f"Target: ${signal['target']:.2f} ({signal['target_pct']:.1f}%)")
                
                with cols[3]:
                    st.markdown("**Why Now**")
                    st.markdown(f"{signal['why_now'][:80]}...")
                
                with cols[4]:
                    if st.button("📋 Copy", key=f"copy_{signal['ticker']}"):
                        st.code(f"""
{signal['ticker']}
Entry: ${signal['entry']:.2f}
Stop: ${signal['stop']:.2f}
Target: ${signal['target']:.2f}
                        """)
                
                st.divider()
    else:
        st.info("No signals today. Run SENTINEL v28 to generate signals.")
    
    # ===========================
    # ACTIVE TRADES
    # ===========================
    st.header("💼 Active Trades")
    
    if active_trades:
        for trade in active_trades:
            current_price = trade.get('current_price', trade['entry'])
            current_pnl = ((current_price - trade['entry']) / trade['entry']) * 100
            
            cols = st.columns([1, 2, 2, 2])
            
            with cols[0]:
                st.markdown(f"### {trade['ticker']}")
                st.markdown(f"Entry: {trade['entry_date']}")
            
            with cols[1]:
                st.markdown(f"Shares: {trade['shares']}")
                st.markdown(f"Entry: ${trade['entry']:.2f}")
            
            with cols[2]:
                pnl_color = "positive" if current_pnl > 0 else "negative"
                st.markdown(f"<span class='{pnl_color}'>P&L: {current_pnl:+.2f}%</span>", unsafe_allow_html=True)
                st.markdown(f"Current: ${current_price:.2f}")
            
            with cols[3]:
                st.markdown(f"Stop: ${trade['stop']:.2f}")
                st.markdown(f"Target: ${trade['target']:.2f}")
            
            st.divider()
    else:
        st.info("No active trades. Waiting for entry signals.")
    
    # ===========================
    # PERFORMANCE CHART
    # ===========================
    st.header("📈 Cumulative Performance")
    
    daily_perf = calculate_daily_performance()
    
    if not daily_perf.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_perf['date'],
            y=daily_perf['cumulative'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="P&L (%)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No performance data yet. Complete some trades to see your performance.")
    
    # ===========================
    # TRADE HISTORY
    # ===========================
    st.header("📜 Trade History")
    
    closed_trades = [t for t in trades if t['status'] == 'CLOSED']
    
    if closed_trades:
        # Create DataFrame
        df_trades = pd.DataFrame(closed_trades)
        
        # Display table
        st.dataframe(
            df_trades[[
                'ticker', 'entry_date', 'exit_date', 'entry', 'exit',
                'shares', 'pnl', 'exit_reason'
            ]].sort_values('exit_date', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Win/Loss distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Win/Loss Distribution")
            
            wins = len([t for t in closed_trades if t['pnl'] > 0])
            losses = len([t for t in closed_trades if t['pnl'] <= 0])
            
            fig = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                marker_colors=['#00ff00', '#ff4444']
            )])
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("P&L Distribution")
            
            fig = go.Figure(data=[go.Histogram(
                x=[t['pnl'] for t in closed_trades],
                nbinsx=20,
                marker_color='#667eea'
            )])
            
            fig.update_layout(
                xaxis_title="P&L (%)",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades yet. Complete your first trade to see history.")
    
    # ===========================
    # SIGNAL HISTORY
    # ===========================
    with st.expander("📊 Signal History"):
        if signals:
            df_signals = pd.DataFrame(signals)
            
            # Group by date
            signals_by_date = df_signals.groupby('date').agg({
                'ticker': 'count',
                'score': 'mean'
            }).reset_index()
            
            signals_by_date.columns = ['Date', 'Signal Count', 'Avg Score']
            
            st.dataframe(
                signals_by_date.sort_values('Date', ascending=False),
                use_container_width=True,
                height=300
            )
        else:
            st.info("No signal history yet.")
    
    # ===========================
    # FOOTER
    # ===========================
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        SENTINEL v28 Dashboard | Real-time VCP Signal Tracking<br>
        Last updated: {}<br>
        <small>Not financial advice. Use at your own risk.</small>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
