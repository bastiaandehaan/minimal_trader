"""Abstract base class for all trading strategies."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict
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
    strategy: str = ""
    timestamp: Optional[pd.Timestamp] = None


class AbstractStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, params: dict = None):
        self.params = params or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        pass

    @property
    @abstractmethod
    def required_bars(self) -> int:
        """Minimum bars needed for warmup."""
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific indicators to dataframe."""
        pass

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[Signal, Dict]:
        """Generate signal at bar i. Returns (Signal, metadata)."""
        pass

    def validate_params(self) -> bool:
        """Validate strategy parameters."""
        return True