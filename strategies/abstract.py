"""Cleaned strategy abstract with single interface for both backtest and live."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import pandas as pd


class SignalType(Enum):
    NONE = 0
    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class Signal:
    type: SignalType
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    size: Optional[float] = None
    reason: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


class AbstractStrategy(ABC):
    """Unified strategy base for backtest and live trading."""

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self._name = None
        self._last_bar_processed = -1

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass

    @property
    @abstractmethod
    def required_bars(self) -> int:
        """Minimum bars needed before signals."""
        pass

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, bar_idx: int) -> Signal:
        """
        Generate signal at bar_idx using ONLY data [0:bar_idx+1].

        This is the ONLY signal interface - works for both backtest and live.
        For live: df contains recent bars, bar_idx is the latest bar.
        For backtest: df contains full dataset, bar_idx iterates through history.

        Args:
            df: OHLCV data with any pre-computed indicators
            bar_idx: Current bar index (0-based)

        Returns:
            Signal with type, prices, and metadata
        """
        pass

    def reset(self):
        """Reset strategy state for new run."""
        self._last_bar_processed = -1

    def validate_params(self) -> bool:
        """Validate strategy parameters."""
        return True

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicators to dataframe. Called once before signal generation.
        Override if strategy needs custom indicators.
        """
        return df

    def on_position_opened(self, position_id: str, entry_price: float):
        """Called when position is opened (live trading only)."""
        pass

    def on_position_closed(self, position_id: str, exit_price: float, pnl: float):
        """Called when position is closed (live trading only)."""
        pass

    def on_market_data_error(self, error: Exception):
        """Called when market data feed fails (live trading only)."""
        pass