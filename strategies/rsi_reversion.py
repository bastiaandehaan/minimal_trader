# strategies/rsi_reversion.py
from __future__ import annotations

import logging
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from .abstract import AbstractStrategy, Signal, SignalType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Classic Wilder RSI (EMA-based)."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


class RSIReversionStrategy(AbstractStrategy):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        p = params or {}
        self._name = f"RSIRev_rsi{p.get('rsi_period', 14)}_atr{p.get('atr_period', 14)}"
        self.rsi_period = int(p.get("rsi_period", 14))
        self.oversold = float(p.get("oversold", 30.0))
        self.overbought = float(p.get("overbought", 70.0))
        self.atr_period = int(p.get("atr_period", 14))
        self.sl_multiplier = float(p.get("sl_multiplier", 1.5))
        self.tp_multiplier = float(p.get("tp_multiplier", 2.0))
        # Engine handles NEXT_OPEN logic automatically

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def required_bars(self) -> int:
        return max(self.rsi_period, self.atr_period) + 1
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and ATR indicators to dataframe."""
        df = df.copy()
        
        # Calculate RSI
        df['rsi'] = _rsi(df['close'], self.rsi_period)
        
        # Calculate ATR if not present
        if 'atr' not in df.columns:
            tr_high_low = df['high'] - df['low']
            tr_high_close = (df['high'] - df['close'].shift()).abs()
            tr_low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=self.atr_period, min_periods=self.atr_period).mean()
        
        return df
    
    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[Signal, Dict]:
        """Generate signal at bar i using ONLY data up to bar i (no look-ahead)."""
        if i < self.required_bars:
            return Signal(SignalType.NONE), {}
        
        # Only use data up to current bar
        data_slice = df.iloc[:i+1]
        current_row = df.iloc[i]
        
        # Calculate indicators on data up to current bar
        data_with_indicators = self.calculate_indicators(data_slice)
        current_rsi = data_with_indicators['rsi'].iloc[-1]
        current_atr = data_with_indicators['atr'].iloc[-1]
        
        if pd.isna(current_rsi) or pd.isna(current_atr):
            return Signal(SignalType.NONE), {}
        
        entry_price = float(current_row['close'])
        
        # Long signal on oversold
        if current_rsi < self.oversold:
            stop_loss = entry_price - (self.sl_multiplier * current_atr)
            take_profit = entry_price + (self.tp_multiplier * current_atr)
            
            return Signal(
                type=SignalType.BUY,
                entry=entry_price,
                stop=stop_loss,
                target=take_profit,
                reason=f"RSI oversold: {current_rsi:.1f}",
                strategy=self.name,
                timestamp=current_row.name
            ), {'rsi': current_rsi, 'atr': current_atr}
        
        # Short signal on overbought
        elif current_rsi > self.overbought:
            stop_loss = entry_price + (self.sl_multiplier * current_atr)
            take_profit = entry_price - (self.tp_multiplier * current_atr)
            
            return Signal(
                type=SignalType.SELL,
                entry=entry_price,
                stop=stop_loss,
                target=take_profit,
                reason=f"RSI overbought: {current_rsi:.1f}",
                strategy=self.name,
                timestamp=current_row.name
            ), {'rsi': current_rsi, 'atr': current_atr}
        
        return Signal(SignalType.NONE), {'rsi': current_rsi, 'atr': current_atr}
    
    def validate_params(self) -> bool:
        """Validate strategy parameters."""
        if self.rsi_period <= 0:
            return False
        if self.atr_period <= 0:
            return False
        if self.oversold >= self.overbought:
            return False
        if self.sl_multiplier <= 0 or self.tp_multiplier <= 0:
            return False
        return True
    
