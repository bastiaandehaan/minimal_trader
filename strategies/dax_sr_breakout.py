# strategies/dax_trend_continuation.py
from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from .abstract import AbstractStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def _is_uptrend(ema_fast: float, ema_slow: float, min_separation: float = 20.0) -> bool:
    """Check if we're in a valid uptrend."""
    return ema_fast > ema_slow and (ema_fast - ema_slow) >= min_separation


def _is_downtrend(ema_fast: float, ema_slow: float,
                  min_separation: float = 20.0) -> bool:
    """Check if we're in a valid downtrend."""
    return ema_fast < ema_slow and (ema_slow - ema_fast) >= min_separation


def _is_pullback_to_support(df: pd.DataFrame, lookback: int = 3) -> bool:
    """Check if price pulled back to EMA21 support zone."""
    recent_data = df.tail(lookback)
    ema21_values = recent_data['ema_21']
    low_values = recent_data['low']

    # Price should have touched or come close to EMA21
    min_distance = (low_values - ema21_values).min()
    return min_distance <= 30.0  # Within 30 points of EMA21


def _is_bounce_to_resistance(df: pd.DataFrame, lookback: int = 3) -> bool:
    """Check if price bounced to EMA21 resistance zone."""
    recent_data = df.tail(lookback)
    ema21_values = recent_data['ema_21']
    high_values = recent_data['high']

    # Price should have touched or come close to EMA21
    min_distance = (ema21_values - high_values).min()
    return min_distance <= 30.0  # Within 30 points of EMA21


class DAXTrendContinuationStrategy(AbstractStrategy):
    """
    4H Trend Continuation Strategy optimized for DAX.

    Logic:
    - Uptrend: EMA8 > EMA21, wait for pullback to EMA21, then long
    - Downtrend: EMA8 < EMA21, wait for bounce to EMA21, then short
    - Conservative exits suitable for FTMO
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

        # Strategy parameters with 4H optimized defaults
        self.ema_fast_period = int(self.params.get('ema_fast', 8))  # ~1.5 days
        self.ema_slow_period = int(self.params.get('ema_slow', 21))  # ~3.5 days
        self.atr_period = int(self.params.get('atr_period', 14))  # ~2.5 days
        self.sl_multiplier = float(
            self.params.get('sl_multiplier', 2.5))  # Conservative for 4H
        self.tp_multiplier = float(
            self.params.get('tp_multiplier', 1.8))  # FTMO-friendly ratio
        self.min_trend_strength = float(
            self.params.get('min_trend_strength', 20.0))  # Points
        self.pullback_lookback = int(
            self.params.get('pullback_lookback', 3))  # Bars to check

        # Build strategy name
        self._strategy_name = (
            f"DAX_TrendCont_EMA{self.ema_fast_period}-{self.ema_slow_period}_"
            f"ATR{self.atr_period}_SL{self.sl_multiplier}")

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def required_bars(self) -> int:
        return max(self.ema_slow_period,
                   self.atr_period) + 10  # Buffer for calculations

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA and ATR indicators."""
        df = df.copy()

        # Calculate EMAs
        df['ema_8'] = _calculate_ema(df['close'], self.ema_fast_period)
        df['ema_21'] = _calculate_ema(df['close'], self.ema_slow_period)

        # Calculate ATR if not present
        if 'atr' not in df.columns:
            df['atr'] = _calculate_atr(df, self.atr_period)

        # Trend direction
        df['trend_direction'] = np.where(df['ema_8'] > df['ema_21'], 1,
                                         np.where(df['ema_8'] < df['ema_21'], -1, 0))

        # EMA separation (trend strength)
        df['ema_separation'] = abs(df['ema_8'] - df['ema_21'])

        return df

    def get_signal(self, df: pd.DataFrame, bar_idx: int) -> Signal:
        """Generate trend continuation signals."""
        # Check if we have enough data
        if bar_idx < self.required_bars:
            return Signal(SignalType.NONE)

        # Get data slice up to current bar (no look-ahead)
        data = df.iloc[:bar_idx + 1]

        # Ensure indicators are calculated
        if 'ema_8' not in data.columns or 'ema_21' not in data.columns:
            data = self.calculate_indicators(data)

        current_bar = data.iloc[-1]

        # Get current values
        current_close = current_bar['close']
        current_ema8 = current_bar['ema_8']
        current_ema21 = current_bar['ema_21']
        current_atr = current_bar['atr']
        ema_separation = current_bar['ema_separation']

        # Skip if indicators are invalid
        if any(pd.isna([current_ema8, current_ema21, current_atr])) or current_atr <= 0:
            return Signal(SignalType.NONE)

        # Check for minimum trend strength
        if ema_separation < self.min_trend_strength:
            return Signal(SignalType.NONE,
                          reason=f"Trend too weak: {ema_separation:.1f} < {self.min_trend_strength}")

        # LONG SIGNAL: Uptrend + pullback to EMA21 support
        if _is_uptrend(current_ema8, current_ema21, self.min_trend_strength):
            # Check if we had a pullback to EMA21 support
            if _is_pullback_to_support(data, self.pullback_lookback):
                # Confirm we're bouncing off support (price above EMA21)
                if current_close > current_ema21:
                    stop_loss = current_ema21 - (self.sl_multiplier * current_atr)
                    take_profit = current_close + (self.tp_multiplier * current_atr)

                    confidence = min(1.0,
                                     ema_separation / 100.0)  # Stronger trend = higher confidence

                    return Signal(type=SignalType.BUY, entry=current_close,
                        stop=stop_loss, target=take_profit,
                        reason=f"Uptrend continuation: EMA8({current_ema8:.0f}) > EMA21({current_ema21:.0f}), pullback bounce",
                        confidence=confidence,
                        metadata={'ema_8': current_ema8, 'ema_21': current_ema21,
                            'ema_separation': ema_separation, 'atr': current_atr,
                            'bar_idx': bar_idx, 'timestamp': current_bar.name,
                            'trend_type': 'uptrend_continuation'})

        # SHORT SIGNAL: Downtrend + bounce to EMA21 resistance
        elif _is_downtrend(current_ema8, current_ema21, self.min_trend_strength):
            # Check if we had a bounce to EMA21 resistance
            if _is_bounce_to_resistance(data, self.pullback_lookback):
                # Confirm we're rejecting resistance (price below EMA21)
                if current_close < current_ema21:
                    stop_loss = current_ema21 + (self.sl_multiplier * current_atr)
                    take_profit = current_close - (self.tp_multiplier * current_atr)

                    confidence = min(1.0,
                                     ema_separation / 100.0)  # Stronger trend = higher confidence

                    return Signal(type=SignalType.SELL, entry=current_close,
                        stop=stop_loss, target=take_profit,
                        reason=f"Downtrend continuation: EMA8({current_ema8:.0f}) < EMA21({current_ema21:.0f}), resistance rejection",
                        confidence=confidence,
                        metadata={'ema_8': current_ema8, 'ema_21': current_ema21,
                            'ema_separation': ema_separation, 'atr': current_atr,
                            'bar_idx': bar_idx, 'timestamp': current_bar.name,
                            'trend_type': 'downtrend_continuation'})

        # No signal - either no trend or no proper pullback/bounce
        return Signal(SignalType.NONE,
                      metadata={'ema_8': current_ema8, 'ema_21': current_ema21,
                          'ema_separation': ema_separation, 'atr': current_atr,
                          'trend_strength': 'weak' if ema_separation < self.min_trend_strength else 'strong'})

    def validate_params(self) -> bool:
        """Validate strategy parameters."""
        if self.ema_fast_period <= 0 or self.ema_slow_period <= 0:
            logger.error(f"{self.name}: Invalid EMA periods")
            return False

        if self.ema_fast_period >= self.ema_slow_period:
            logger.error(f"{self.name}: Fast EMA must be < Slow EMA")
            return False

        if self.atr_period <= 0:
            logger.error(f"{self.name}: Invalid ATR period")
            return False

        if self.sl_multiplier <= 0 or self.tp_multiplier <= 0:
            logger.error(f"{self.name}: Invalid stop/target multipliers")
            return False

        if self.min_trend_strength <= 0:
            logger.error(f"{self.name}: Invalid trend strength threshold")
            return False

        return True