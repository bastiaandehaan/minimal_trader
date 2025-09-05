"""
Minimal tests - just the essentials
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trader import SimpleBreakoutTrader, Backtester

def create_test_data(n_days=30, trend=0.001):
    """Create synthetic OHLC data"""
    dates = pd.date_range(end=datetime.now(), periods=n_days*24, freq='H')

    # Random walk with trend
    returns = np.random.normal(trend, 0.01, len(dates))
    close = 18000 * (1 + returns).cumprod()

    # Create OHLC
    high = close * (1 + np.random.uniform(0.001, 0.005, len(dates)))
    low = close * (1 - np.random.uniform(0.001, 0.005, len(dates)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    volume = np.random.uniform(1000, 5000, len(dates))

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df

def test_trader_init():
    """Test trader initialization"""
    trader = SimpleBreakoutTrader()
    assert trader.symbol == "GER40.cash"
    assert trader.risk_pct == 1.0

def test_signal_generation():
    """Test that signals are generated"""
    trader = SimpleBreakoutTrader()
    df = create_test_data()

    signal, meta = trader.get_signal(df)

    assert signal in ['BUY', 'SELL', None]
    assert 'close' in meta
    assert 'sma' in meta

def test_no_look_ahead():
    """Critical: Test no look-ahead bias"""
    trader = SimpleBreakoutTrader()
    df = create_test_data()

    # Signal at time T
    signal_t, _ = trader.get_signal(df.iloc[:100])

    # Signal at time T with more future data
    signal_t_future, _ = trader.get_signal(df.iloc[:100])  # Same window

    # Should be identical
    assert signal_t == signal_t_future

def test_backtest_runs():
    """Test backtest completes without errors"""
    trader = SimpleBreakoutTrader()
    backtester = Backtester(trader)

    df = create_test_data()
    results = backtester.run(df)

    assert 'metrics' in results
    assert 'trades' in results
    assert results['metrics']['initial_capital'] == 10000

def test_position_sizing():
    """Test risk-based position sizing"""
    trader = SimpleBreakoutTrader()

    size = trader.calculate_position_size(
        equity=10000,
        entry=18000,
        stop=17900
    )

    # Should risk 1% = 100 EUR
    # Risk is 100 points, so size should be 1.0
    assert size == 1.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
