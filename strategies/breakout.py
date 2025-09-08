"""Breakout Strategy - trades range breakouts."""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from strategies.abstract import AbstractStrategy, Signal, SignalType


class BreakoutStrategy(AbstractStrategy):
    """Range breakout with ATR stops."""

    def __init__(self, params: dict = None):
        default_params = {
            'lookback_period': 20,
            'atr_period': 14,
            'breakout_factor': 1.0,  # multiplier for range
            'sl_multiplier': 1.0,
            'tp_multiplier': 2.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    @property
    def name(self) -> str:
        return f"Breakout_{self.params['lookback_period']}"

    @property
    def required_bars(self) -> int:
        return max(self.params['lookback_period'], self.params['atr_period']) + 1

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add range and ATR indicators."""
        df = df.copy()

        # Rolling high/low for range
        df['range_high'] = df['high'].rolling(
            self.params['lookback_period']
        ).max()
        df['range_low'] = df['low'].rolling(
            self.params['lookback_period']
        ).min()
        df['range_mid'] = (df['range_high'] + df['range_low']) / 2

        # ATR for stops
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(
            self.params['atr_period'],
            min_periods=self.params['atr_period']
        ).mean()

        return df

    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[Signal, Dict]:
        """Detect range breakout at bar i."""
        if i < self.required_bars:
            return Signal(SignalType.NONE, strategy=self.name), {}

        current = df.iloc[i]
        prev = df.iloc[i-1]

        # Check if we have valid range
        if not (np.isfinite(current['range_high']) and np.isfinite(current['range_low'])):
            return Signal(SignalType.NONE, strategy=self.name), {}

        # Calculate breakout levels
        range_size = current['range_high'] - current['range_low']
        breakout_buffer = range_size * (self.params['breakout_factor'] - 1) / 2

        upper_breakout = current['range_high'] + breakout_buffer
        lower_breakout = current['range_low'] - breakout_buffer

        # Detect breakouts
        break_above = (prev['close'] <= upper_breakout) and (current['close'] > upper_breakout)
        break_below = (prev['close'] >= lower_breakout) and (current['close'] < lower_breakout)

        meta = {
            'close': float(current['close']),
            'range_high': float(current['range_high']),
            'range_low': float(current['range_low']),
            'atr': float(current['atr']) if np.isfinite(current['atr']) else 0.0,
            'timestamp': current.name
        }

        if break_above and meta['atr'] > 0:
            entry = meta['close']
            stop = entry - meta['atr'] * self.params['sl_multiplier']
            target = entry + meta['atr'] * self.params['tp_multiplier']

            return Signal(
                type=SignalType.BUY,
                entry=entry,
                stop=stop,
                target=target,
                reason=f"Breakout above {upper_breakout:.2f}",
                strategy=self.name,
                timestamp=meta['timestamp']
            ), meta

        return Signal(SignalType.NONE, strategy=self.name), meta