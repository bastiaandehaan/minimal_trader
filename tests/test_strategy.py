import numpy as np
import pandas as pd
from strategies.sma_cross import SMACrossStrategy
from strategies.abstract import SignalType


def make_df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
    close = 18000 + np.cumsum(rng.normal(0, 5, n))
    high = close + rng.uniform(1, 5, n)
    low = close - rng.uniform(1, 5, n)
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_sma_crossover_buy_signal():
    strategy_params = {'sma_period': 5, 'atr_period': 5}
    strat = SMACrossStrategy(strategy_params)
    df = make_df(50)
    df = strat.calculate_indicators(df)
    i = len(df) - 1
    # force bullish cross
    df.iloc[i - 1, df.columns.get_loc("close")] = df["sma"].iloc[i - 1] - 1
    df.iloc[i, df.columns.get_loc("close")] = df["sma"].iloc[i - 1] + 2
    sig, _ = strat.get_signal(df, i)
    assert sig.type == SignalType.BUY


def test_no_lookahead():
    strategy_params = {'sma_period': 5, 'atr_period': 5}
    strat = SMACrossStrategy(strategy_params)
    df = make_df(60)
    df = strat.calculate_indicators(df)
    i = 30
    sig1, _ = strat.get_signal(df, i)
    sig2, _ = strat.get_signal(df.iloc[: i + 1], i)
    assert sig1.type == sig2.type
