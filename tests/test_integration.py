import numpy as np
import pandas as pd
from strategies.sma_cross import SMACrossStrategy
from engine import MultiStrategyEngine, EngineConfig


def make_data(n=300):
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
    # Create data that will trigger SMA crossovers
    base = 18000
    # Start below SMA, then cross above to trigger a BUY signal
    close = np.concatenate([
        np.full(50, base - 10),  # Start below
        np.linspace(base - 10, base + 50, 100),  # Cross above (should trigger BUY)
        np.full(150, base + 40)  # Stay above
    ])
    high = close + 2
    low = close - 2
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(n, 2000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_pipeline_produces_some_trades():
    df = make_data(300)
    strategy_params = {'sma_period': 10, 'atr_period': 10}
    strat = SMACrossStrategy(strategy_params)
    engine_config = EngineConfig(initial_capital=10000, risk_per_trade=1.0, time_exit_bars=100)
    engine = MultiStrategyEngine(engine_config)
    engine.add_strategy(strat, 100.0)
    res = engine.run_backtest(df)
    assert res["metrics"]["total_trades"] >= 1
