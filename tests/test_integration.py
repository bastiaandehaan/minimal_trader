import numpy as np
import pandas as pd
from strategy import Strategy, StrategyParams
from backtest import Backtest, ExecConfig


def make_data(n=300):
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
    close = np.linspace(18000, 18200, n)  # gentle trend
    high = close + 2
    low = close - 2
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(n, 2000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_pipeline_produces_some_trades():
    df = make_data(300)
    strat = Strategy(StrategyParams(sma_period=10, atr_period=10, volume_threshold=0.8))
    bt = Backtest(strat, ExecConfig(initial_capital=10_000, risk_pct=1.0, time_exit_bars=100))
    res = bt.run(df)
    assert res["metrics"]["num_trades"] >= 1
