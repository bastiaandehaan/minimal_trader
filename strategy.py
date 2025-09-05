from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class SignalType(Enum):
    NONE = 0
    BUY = 1
    SELL = -1   # reversal/exit only; no short entries in v1


@dataclass(frozen=True)
class Signal:
    type: SignalType
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    reason: str = ""
    timestamp: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class StrategyParams:
    sma_period: int = 20
    atr_period: int = 14
    sl_mult: float = 1.5
    tp_mult: float = 2.5
    volume_threshold: float = 1.0  # x average


class Strategy:
    """Pure strategy: indicators + signal at bar i. No state, no side effects."""
    def __init__(self, params: StrategyParams = StrategyParams()):
        self.p = params

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # SMA
        df["sma"] = df["close"].rolling(self.p.sma_period, min_periods=self.p.sma_period).mean()

        # ATR (classic TR)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.p.atr_period, min_periods=self.p.atr_period).mean()

        # Volume ratio
        if "volume" in df.columns:
            df["volume_avg"] = df["volume"].rolling(self.p.sma_period, min_periods=self.p.sma_period).mean()
            df["volume_ratio"] = df["volume"] / df["volume_avg"]
        else:
            df["volume"] = 1000.0
            df["volume_avg"] = 1000.0
            df["volume_ratio"] = 1.0
        return df

    def get_signal_at(self, df: pd.DataFrame, i: int) -> Tuple[Signal, dict]:
        """Decide signal using only info up to index i (no look-ahead)."""
        if i < max(self.p.sma_period, self.p.atr_period):
            return Signal(SignalType.NONE), {}

        last = df.iloc[i]
        prev = df.iloc[i - 1]

        cross_above = (prev["close"] <= prev["sma"]) and (last["close"] > last["sma"])
        cross_below = (prev["close"] >= prev["sma"]) and (last["close"] < last["sma"])
        volume_ok = bool(last.get("volume_ratio", 0) >= self.p.volume_threshold)

        meta = {
            "close": float(last["close"]),
            "sma": float(last["sma"]),
            "atr": float(last["atr"]),
            "volume_ratio": float(last.get("volume_ratio", 1.0)),
            "timestamp": last.name if hasattr(last, "name") else None,
        }

        # Entry: only long for v1
        if cross_above and volume_ok and np.isfinite(last["atr"]) and last["atr"] > 0:
            entry = float(last["close"])
            stop = entry - float(last["atr"]) * self.p.sl_mult
            target = entry + float(last["atr"]) * self.p.tp_mult
            sig = Signal(
                type=SignalType.BUY,
                entry=entry,
                stop=stop,
                target=target,
                reason=f"BUY: cross↑ + vol {meta['volume_ratio']:.2f}x",
                timestamp=meta["timestamp"],
            )
            return sig, meta

        # Exit reversal signal (SELL only as exit trigger)
        if cross_below:
            sig = Signal(
                type=SignalType.SELL,
                reason="SELL: cross↓ (reversal/exit)",
                timestamp=meta["timestamp"],
            )
            return sig, meta

        return Signal(SignalType.NONE), meta
