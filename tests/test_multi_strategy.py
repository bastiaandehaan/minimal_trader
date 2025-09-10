"""Test suite for multi-strategy framework."""
import numpy as np
import pandas as pd
import pytest
from strategies.sma_cross import SMACrossStrategy
from strategies.breakout import BreakoutStrategy
from feeds.csv_feed import CSVFeed
from engine import MultiStrategyEngine, EngineConfig


def make_trending_data(n=500):
    """Create trending OHLC data that will trigger SMA crossovers."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")

    # Create data that starts below SMA then crosses above
    base = 18000
    # Start low, then trend up to create clear crossover
    close = np.concatenate([
        np.full(20, base - 50),  # Start well below
        np.linspace(base - 50, base + 200, n - 20)  # Strong uptrend
    ])
    
    high = close + np.random.uniform(2, 8, n)
    low = close - np.random.uniform(2, 8, n)
    open_ = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(1000, 5000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx)


def test_sma_strategy_signals():
    """Test SMA strategy generates signals."""
    strategy = SMACrossStrategy({'sma_period': 10, 'atr_period': 10})
    df = make_trending_data(100)
    df = strategy.calculate_indicators(df)

    # Check for at least one signal in trending data
    signals_found = False
    for i in range(strategy.required_bars, len(df)):
        signal, meta = strategy.get_signal(df, i)
        if signal.type.value != 0:  # BUY or SELL
            signals_found = True
            break

    assert signals_found, "No signals generated in trending data"


def test_multi_engine_runs():
    """Test multi-strategy engine executes."""
    # Create engine
    engine = MultiStrategyEngine(
        EngineConfig(initial_capital=10000, risk_per_trade=1.0, max_positions=1))

    # Add strategies
    engine.add_strategy(SMACrossStrategy({'sma_period': 10, 'atr_period': 10}),
        allocation=50.0)
    engine.add_strategy(BreakoutStrategy({'lookback_period': 10, 'atr_period': 10}),
        allocation=50.0)

    # Run backtest
    df = make_trending_data(200)
    results = engine.run_backtest(df)

    assert 'metrics' in results
    assert 'trades' in results
    assert results['metrics']['initial_capital'] == 10000


def test_no_look_ahead_bias():
    """Critical: Test no look-ahead in new architecture."""
    strategy = SMACrossStrategy({'sma_period': 5, 'atr_period': 5})
    df = make_trending_data(100)
    df = strategy.calculate_indicators(df)

    # Signal at bar 50 with data[:51]
    signal1, _ = strategy.get_signal(df.iloc[:51], 50)

    # Signal at bar 50 with full data (but only using [:51])
    signal2, _ = strategy.get_signal(df, 50)

    # Should be identical - no look-ahead
    assert signal1.type == signal2.type