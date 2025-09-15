# strategies/simple_test.py (UPDATED VERSION)
from __future__ import annotations

import logging
from typing import Optional, Dict, Tuple

import pandas as pd

from .abstract import AbstractStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class SimpleTestStrategy(AbstractStrategy):
    """Enhanced test strategy: Multiple trades at regular intervals."""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        p = params or {}
        self.entry_bar = int(p.get("entry_bar", 100))
        self.trade_interval = int(p.get("trade_interval", 500))  # NEW
        self.max_trades = int(p.get("max_trades", 5))  # NEW
        self.trade_count = 0

    @property
    def name(self) -> str:
        return f"SimpleTest_interval{self.trade_interval}"

    @property
    def required_bars(self) -> int:
        return self.entry_bar + 1

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """No indicators needed."""
        return df

    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[Signal, Dict]:
        """Generate BUY signals at regular intervals."""

        # Check if this is a trade bar
        if i >= self.entry_bar and self.trade_count < self.max_trades:
            # Trade at entry_bar, then every trade_interval bars
            bars_since_start = i - self.entry_bar
            if bars_since_start % self.trade_interval == 0:
                self.trade_count += 1
                current_row = df.iloc[i]
                entry_price = float(current_row["close"])

                # Simple stop/target: 2% stop loss, 3% take profit
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.03

                logger.info(
                    f"SimpleTest: BUY #{self.trade_count} at bar {i}, price={entry_price:.2f}")

                return (Signal(type=SignalType.BUY, entry=entry_price, stop=stop_loss,
                    target=take_profit, reason=f"Trade #{self.trade_count} at bar {i}",
                    strategy=self.name, timestamp=current_row.name, ),
                        {"trade_number": self.trade_count, "entry_bar": i})

        return Signal(SignalType.NONE), {}

    def reset(self):
        """Reset strategy state for new backtest."""
        self.trade_count = 0