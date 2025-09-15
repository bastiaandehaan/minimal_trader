# strategies/rsi_reversion.py - CLEANED VERSION
from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from .abstract import AbstractStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI calculation."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use Wilder's smoothing (same as EMA with alpha = 1/period)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range calculation."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


class RSIReversionStrategy(AbstractStrategy):
    """Mean reversion strategy using RSI oversold/overbought levels."""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

        # Strategy parameters with defaults
        self.rsi_period = int(self.params.get('rsi_period', 14))
        self.oversold = float(self.params.get('oversold', 30.0))
        self.overbought = float(self.params.get('overbought', 70.0))
        self.atr_period = int(self.params.get('atr_period', 14))
        self.sl_multiplier = float(self.params.get('sl_multiplier', 2.0))
        self.tp_multiplier = float(self.params.get('tp_multiplier', 1.5))

        # Build name from parameters
        self._strategy_name = (f"RSI_p{self.rsi_period}_"
                               f"os{self.oversold}_ob{self.overbought}_"
                               f"atr{self.atr_period}")

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def required_bars(self) -> int:
        return max(self.rsi_period, self.atr_period) + 5  # Buffer for calculations

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and ATR indicators."""
        df = df.copy()

        # Calculate RSI
        df['rsi'] = _calculate_rsi(df['close'], self.rsi_period)

        # Calculate ATR if not present
        if 'atr' not in df.columns:
            df['atr'] = _calculate_atr(df, self.atr_period)

        return df

    def get_signal(self, df: pd.DataFrame, bar_idx: int) -> Signal:
        """Generate signal using only data up to bar_idx."""
        # Check if we have enough data
        if bar_idx < self.required_bars:
            return Signal(SignalType.NONE)

        # Get data slice up to current bar (no look-ahead)
        data = df.iloc[:bar_idx + 1]
        current_bar = data.iloc[-1]

        # Ensure indicators are calculated
        if 'rsi' not in data.columns or 'atr' not in data.columns:
            data = self.calculate_indicators(data)
            current_bar = data.iloc[-1]

        # Get current values
        current_rsi = current_bar['rsi']
        current_atr = current_bar['atr']
        current_close = current_bar['close']

        # Skip if indicators are invalid
        if pd.isna(current_rsi) or pd.isna(current_atr) or current_atr <= 0:
            return Signal(SignalType.NONE)

        # Check for oversold condition (buy signal)
        if current_rsi <= self.oversold:
            stop_loss = current_close - (self.sl_multiplier * current_atr)
            take_profit = current_close + (self.tp_multiplier * current_atr)

            return Signal(type=SignalType.BUY, entry=current_close, stop=stop_loss,
                target=take_profit,
                reason=f"RSI oversold: {current_rsi:.1f} <= {self.oversold}",
                confidence=min(1.0, (self.oversold - current_rsi) / 10.0),
                # Stronger when more oversold
                metadata={'rsi': current_rsi, 'atr': current_atr, 'bar_idx': bar_idx,
                    'timestamp': current_bar.name})

        # Check for overbought condition (sell signal)
        elif current_rsi >= self.overbought:
            stop_loss = current_close + (self.sl_multiplier * current_atr)
            take_profit = current_close - (self.tp_multiplier * current_atr)

            return Signal(type=SignalType.SELL, entry=current_close, stop=stop_loss,
                target=take_profit,
                reason=f"RSI overbought: {current_rsi:.1f} >= {self.overbought}",
                confidence=min(1.0, (current_rsi - self.overbought) / 10.0),
                # Stronger when more overbought
                metadata={'rsi': current_rsi, 'atr': current_atr, 'bar_idx': bar_idx,
                    'timestamp': current_bar.name})

        # No signal
        return Signal(SignalType.NONE,
            metadata={'rsi': current_rsi, 'atr': current_atr, 'bar_idx': bar_idx})

    def validate_params(self) -> bool:
        """Validate strategy parameters."""
        if self.rsi_period <= 0 or self.atr_period <= 0:
            logger.error(
                f"{self.name}: Invalid periods - RSI:{self.rsi_period}, ATR:{self.atr_period}")
            return False

        if self.oversold >= self.overbought:
            logger.error(
                f"{self.name}: Invalid RSI levels - oversold:{self.oversold} >= overbought:{self.overbought}")
            return False

        if self.sl_multiplier <= 0 or self.tp_multiplier <= 0:
            logger.error(
                f"{self.name}: Invalid multipliers - SL:{self.sl_multiplier}, TP:{self.tp_multiplier}")
            return False

        if not (0 < self.oversold < 50 < self.overbought < 100):
            logger.error(
                f"{self.name}: RSI levels out of logical range - OS:{self.oversold}, OB:{self.overbought}")
            return False

        return True