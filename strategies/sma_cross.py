"""Simple Moving Average Crossover Strategy."""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from strategies.abstract import AbstractStrategy, Signal, SignalType


class SMACrossStrategy(AbstractStrategy):
    """SMA crossover with ATR-based stops."""

    def __init__(self, params: dict = None):
        default_params = {'sma_period': 20, 'atr_period': 14, 'sl_multiplier': 1.5,
            'tp_multiplier': 2.5}
        if params:
            default_params.update(params)
        super().__init__(default_params)

    @property
    def name(self) -> str:
        return f"SMA_{self.params['sma_period']}"

    @property
    def required_bars(self) -> int:
        return max(self.params['sma_period'], self.params['atr_period']) + 1

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA and ATR indicators."""
        df = df.copy()

        # Simple Moving Average
        df['sma'] = df['close'].rolling(self.params['sma_period'],
            min_periods=self.params['sma_period']).mean()

        # Average True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.params['atr_period'],
            min_periods=self.params['atr_period']).mean()

        return df

    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[Signal, Dict]:
        """Detect SMA crossover at bar i."""
        if i < self.required_bars:
            return Signal(SignalType.NONE, strategy=self.name), {}

        current = df.iloc[i]
        prev = df.iloc[i - 1]

        # Check if indicators are valid
        if not (np.isfinite(current['sma']) and np.isfinite(prev['sma'])):
            return Signal(SignalType.NONE, strategy=self.name), {}

        # Detect crossovers
        cross_above = (prev['close'] <= prev['sma']) and (
                    current['close'] > current['sma'])
        cross_below = (prev['close'] >= prev['sma']) and (
                    current['close'] < current['sma'])

        # Metadata
        meta = {'close': float(current['close']), 'sma': float(current['sma']),
            'atr': float(current['atr']) if np.isfinite(current['atr']) else 0.0,
            'timestamp': current.name}

        # Generate signals
        if cross_above and meta['atr'] > 0:
            entry = meta['close']
            stop = entry - meta['atr'] * self.params['sl_multiplier']
            target = entry + meta['atr'] * self.params['tp_multiplier']

            return Signal(type=SignalType.BUY, entry=entry, stop=stop, target=target,
                reason=f"SMA cross above", strategy=self.name,
                timestamp=meta['timestamp']), meta

        elif cross_below:
            return Signal(type=SignalType.SELL, reason=f"SMA cross below",
                strategy=self.name, timestamp=meta['timestamp']), meta

        return Signal(SignalType.NONE, strategy=self.name), meta