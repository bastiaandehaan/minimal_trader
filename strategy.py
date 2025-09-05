from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class SignalType(Enum):
    NONE = 0
    BUY = 1
    SELL = -1  # reversal/exit only; no short entries in v1


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
    """Pure strategy: indicators + signal-at-index. No state, no side effects."""

    def __init__(self, params: StrategyParams = StrategyParams()):
        self.p = params

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMA, ATR, and volume ratios. Returns a copy."""
        df = df.copy()

        # SMA
        df["sma"] = df["close"].rolling(
            self.p.sma_period, min_periods=self.p.sma_period
        ).mean()

        # ATR (classic TR)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.p.atr_period, min_periods=self.p.atr_period).mean()

        # Volume ratio
        if "volume" in df.columns:
            df["volume_avg"] = df["volume"].rolling(
                self.p.sma_period, min_periods=self.p.sma_period
            ).mean()
            # Avoid division by zero
            df["volume_ratio"] = df["volume"] / df["volume_avg"].replace(0, np.nan)
            df["volume_ratio"] = df["volume_ratio"].fillna(0.0)
        else:
            df["volume"] = 1000.0
            df["volume_avg"] = 1000.0
            df["volume_ratio"] = 1.0

        return df

    def get_signal_at(self, df: pd.DataFrame, i: int) -> Tuple[Signal, dict]:
        """
        Decide signal at bar index i using only info up to i (no look-ahead).

        BUY when:
          - classic cross above SMA, OR
          - first bar after SMA becomes valid and close > SMA (first-window entry)
        AND volume_ratio >= threshold AND ATR is finite/positive.

        SELL (as exit/reversal only) on cross below SMA.
        """
        warmup = max(self.p.sma_period, self.p.atr_period)
        if i < warmup:
            return Signal(SignalType.NONE), {}

        last = df.iloc[i]
        prev = df.iloc[i - 1]

        prev_sma = prev["sma"]
        prev_close = prev["close"]

        # Classic cross rules
        cross_above = (
            np.isfinite(prev_sma) and (prev_close <= prev_sma) and (last["close"] > last["sma"])
        )
        cross_below = (
            np.isfinite(prev_sma) and (prev_close >= prev_sma) and (last["close"] < last["sma"])
        )

        # First-window entry:
        # index i-1 is the first bar with a valid SMA (prev.sma finite) and i-2 had no SMA yet
        prev2_sma = df["sma"].iloc[i - 2] if i - 2 >= 0 else np.nan
        first_window = np.isfinite(prev["sma"]) and not np.isfinite(prev2_sma)
        first_window_above = first_window and (last["close"] > last["sma"])

        # Volume check: inclusive so ratio==threshold passes
        volume_ok = bool(last.get("volume_ratio", 0.0) >= self.p.volume_threshold)

        meta = {
            "close": float(last["close"]),
            "sma": float(last["sma"]),
            "atr": float(last["atr"]),
            "volume_ratio": float(last.get("volume_ratio", 1.0)),
            "timestamp": last.name if hasattr(last, "name") else None,
        }

        # Entry: long only in v1
        if (cross_above or first_window_above) and volume_ok and np.isfinite(last["atr"]) and last["atr"] > 0:
            entry = float(last["close"])
            stop = entry - float(last["atr"]) * self.p.sl_mult
            target = entry + float(last["atr"]) * self.p.tp_mult
            sig = Signal(
                type=SignalType.BUY,
                entry=entry,
                stop=stop,
                target=target,
                reason=f"BUY: {'first-window' if first_window_above else 'cross'} + vol {meta['volume_ratio']:.2f}x",
                timestamp=meta["timestamp"],
            )
            return sig, meta

        # Exit trigger (reversal)
        if cross_below:
            sig = Signal(
                type=SignalType.SELL,
                reason="SELL: cross↓ (reversal/exit)",
                timestamp=meta["timestamp"],
            )
            return sig, meta

        return Signal(SignalType.NONE), meta
