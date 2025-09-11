"""Test suite for RSI Reversion Strategy."""
import numpy as np
import pandas as pd
import pytest
from strategies.rsi_reversion import RSIReversionStrategy
from strategies.abstract import SignalType
from engine import MultiStrategyEngine, EngineConfig


def create_oversold_data(n=100):
    """Create data that will trigger RSI oversold conditions."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="5min")

    # Create strong downtrend then reversal (RSI will be oversold)
    base = 18000
    close = np.concatenate([
        np.linspace(base, base - 200, 50),  # Strong down move
        np.linspace(base - 200, base - 150, 50)  # Bounce
    ])

    high = close + np.random.uniform(2, 5, n)
    low = close - np.random.uniform(2, 5, n)
    open_ = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(1000, 5000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx
    )


def create_overbought_data(n=100):
    """Create data that will trigger RSI overbought conditions."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="5min")

    # Create strong uptrend then reversal (RSI will be overbought)
    base = 18000
    close = np.concatenate([
        np.linspace(base, base + 200, 50),  # Strong up move
        np.linspace(base + 200, base + 150, 50)  # Pullback
    ])

    high = close + np.random.uniform(2, 5, n)
    low = close - np.random.uniform(2, 5, n)
    open_ = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(1000, 5000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx
    )


def test_rsi_calculation():
    """Test that RSI is calculated correctly."""
    strategy = RSIReversionStrategy({
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
        'atr_period': 14
    })

    df = create_oversold_data(100)
    df = strategy.calculate_indicators(df)

    # Check RSI exists and is in valid range
    assert 'rsi' in df.columns
    assert df['rsi'].dropna().between(0, 100).all()

    # After strong down move, RSI should be low
    assert df['rsi'].iloc[50] < 50  # Should be oversold area


def test_rsi_long_signal():
    """Test RSI generates LONG signal from oversold."""
    strategy = RSIReversionStrategy({
        'rsi_period': 5,  # Shorter for faster reaction
        'oversold': 30,
        'overbought': 70,
        'atr_period': 5,
        'use_next_open': False  # Immediate entry for testing
    })

    df = create_oversold_data(100)
    df = strategy.calculate_indicators(df)

    # Find a long signal
    long_found = False
    for i in range(strategy.required_bars, len(df)):
        signal, meta = strategy.get_signal(df, i)
        if signal.type == SignalType.BUY:
            long_found = True
            # Verify signal has all required fields
            assert signal.entry is not None
            assert signal.stop is not None
            assert signal.target is not None
            assert signal.stop < signal.entry < signal.target
            break

    assert long_found, "No LONG signal found in oversold conditions"


def test_rsi_short_signal():
    """Test RSI generates SHORT signal from overbought."""
    strategy = RSIReversionStrategy({
        'rsi_period': 5,  # Shorter for faster reaction
        'oversold': 30,
        'overbought': 70,
        'atr_period': 5,
        'use_next_open': False
    })

    df = create_overbought_data(100)
    df = strategy.calculate_indicators(df)

    # Find a short signal
    short_found = False
    for i in range(strategy.required_bars, len(df)):
        signal, meta = strategy.get_signal(df, i)
        if signal.type == SignalType.SELL:
            short_found = True
            # Verify signal has all required fields
            assert signal.entry is not None
            assert signal.stop is not None
            assert signal.target is not None
            assert signal.stop > signal.entry > signal.target  # Reversed for short
            break

    assert short_found, "No SHORT signal found in overbought conditions"


def test_rsi_no_lookahead():
    """Test RSI strategy has no look-ahead bias."""
    strategy = RSIReversionStrategy({
        'rsi_period': 5,
        'oversold': 30,
        'overbought': 70,
        'atr_period': 5
    })

    df = create_oversold_data(100)
    df = strategy.calculate_indicators(df)

    # Test at bar 50
    i = 50

    # Signal with data up to bar 50
    signal1, _ = strategy.get_signal(df.iloc[:i+1], i)

    # Signal with full data (but only using up to bar 50)
    signal2, _ = strategy.get_signal(df, i)

    # Should be identical - no look-ahead
    assert signal1.type == signal2.type
    if signal1.entry and signal2.entry:
        assert abs(signal1.entry - signal2.entry) < 0.01


def test_rsi_with_engine():
    """Test RSI strategy in full engine backtest."""
    engine = MultiStrategyEngine(
        EngineConfig(
            initial_capital=10000,
            risk_per_trade=1.0,
            max_positions=2,
            time_exit_bars=50,
            allow_shorts=True  # Important!
        )
    )

    strategy = RSIReversionStrategy({
        'rsi_period': 5,
        'oversold': 30,
        'overbought': 70,
        'atr_period': 5,
        'sl_multiplier': 1.5,
        'tp_multiplier': 2.0
    })

    engine.add_strategy(strategy, 100.0)

    # Create mixed market conditions
    df1 = create_oversold_data(100)
    df2 = create_overbought_data(100)
    df = pd.concat([df1, df2]).sort_index()

    results = engine.run_backtest(df)

    # Should have both long and short trades
    trades = results['trades']
    if not trades.empty:
        assert len(trades[trades['side'] == 'long']) > 0, "No long trades"
        assert len(trades[trades['side'] == 'short']) > 0, "No short trades"

    # Check metrics exist
    assert 'metrics' in results
    assert results['metrics']['initial_capital'] == 10000


def test_parameter_validation():
    """Test RSI parameter validation."""
    strategy = RSIReversionStrategy({
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
        'atr_period': 14,
        'sl_multiplier': 1.5,
        'tp_multiplier': 2.0
    })

    assert strategy.validate_params() == True

    # Test invalid params
    bad_strategy = RSIReversionStrategy({
        'rsi_period': 0,  # Invalid
        'oversold': 80,   # Higher than overbought
        'overbought': 70,
    })

    assert bad_strategy.validate_params() == False


if __name__ == "__main__":
    # Run tests manually
    test_rsi_calculation()
    test_rsi_long_signal()
    test_rsi_short_signal()
    test_rsi_no_lookahead()
    test_rsi_with_engine()
    test_parameter_validation()
    print("All RSI tests passed!")