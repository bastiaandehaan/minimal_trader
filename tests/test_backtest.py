import numpy as np
import pandas as pd
from strategy import Strategy, StrategyParams
from backtest import Backtest, ExecConfig


def synthetic(n=400, seed=123):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
    close = 18000 + np.cumsum(rng.normal(0, 3, n))
    high = close + rng.uniform(0.5, 3, n)
    low = close - rng.uniform(0.5, 3, n)
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_backtest_runs_and_produces_metrics():
    strat = Strategy(StrategyParams(sma_period=5, atr_period=5, volume_threshold=0.5))
    bt = Backtest(strat, ExecConfig(initial_capital=10_000, risk_pct=1.0, time_exit_bars=50))
    df = synthetic(400)
    res = bt.run(df)
    assert "metrics" in res and "trades" in res
    assert res["metrics"]["initial_capital"] == 10_000
