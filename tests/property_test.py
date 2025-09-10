# minimal_trader/tests/test_rsi_reversion.py
import pandas as pd
import numpy as np

from strategies.rsi_reversion import RSIReversionStrategy
from strategies.abstract import SignalType

def _make_df(n=300, seed=7):
    rng = np.random.default_rng(seed)
    # synthetische random walk met OHLC
    close = np.cumsum(rng.normal(0, 1, size=n)) + 100
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.2, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, size=n))
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=idx)
    return df

def test_indicators_and_no_lookahead():
    strat = RSIReversionStrategy({'use_next_open': True})
    raw = _make_df()
    df = strat.calculate_indicators(raw)

    # indicators bestaan
    assert 'rsi' in df.columns and 'atr' in df.columns
    # geen NaN bij latere bars
    assert np.isfinite(df['rsi'].iloc[-1])

    # property: op laatste bar mag geen signaal worden gegeven als NEXT_OPEN vereist is
    i_last = len(df) - 1
    sig, meta = strat.get_signal(df, i_last)
    assert sig.type == SignalType.NONE

def test_signal_shapes_long_or_short_possible():
    strat = RSIReversionStrategy({'oversold': 60.0, 'overbought': 40.0})  # dwing signalen af
    df = strat.calculate_indicators(_make_df())
    # loop door bars en check of het object-consistent is
    seen_any = False
    for i in range(strat.required_bars, len(df)-1):
        sig, meta = strat.get_signal(df, i)
        assert 'timestamp' in meta
        assert sig.strategy.startswith("RSIRev_")
        if sig.type != SignalType.NONE:
            seen_any = True
            assert sig.entry is not None and sig.stop is not None and sig.target is not None
    assert seen_any
