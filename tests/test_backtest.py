import numpy as np
import pandas as pd
from strategies.sma_cross import SMACrossStrategy
from engine import MultiStrategyEngine, EngineConfig


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
    strategy_params = {'sma_period': 5, 'atr_period': 5}
    strat = SMACrossStrategy(strategy_params)
    engine_config = EngineConfig(initial_capital=10000, risk_per_trade=1.0, time_exit_bars=50)
    engine = MultiStrategyEngine(engine_config)
    engine.add_strategy(strat, 100.0)
    df = synthetic(400)
    res = engine.run_backtest(df)
    assert "metrics" in res and "trades" in res
    assert res["metrics"]["initial_capital"] == 10000
