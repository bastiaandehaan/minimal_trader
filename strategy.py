from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class SignalType(Enum):
    NONE = 0
    BUY = 1
    SELL = -1  # exit trigger only


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
    volume_threshold: float = 1.0  # set 0.0 if your CSV has no real volume


class Strategy:
    """Pure strategy: indicators + signal generation. No state."""

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
        df["atr"] = tr.rolling(
            self.p.atr_period, min_periods=self.p.atr_period
        ).mean()

        # Volume ratio (fallback naar 1.0 als volume ontbreekt)
        if "volume" in df.columns:
            df["volume_avg"] = df["volume"].rolling(
                self.p.sma_period, min_periods=self.p.sma_period
            ).mean()
            vr = df["volume"] / df["volume_avg"].replace(0, np.nan)
            df["volume_ratio"] = vr.fillna(0.0)
        else:
            df["volume"] = 1000.0
            df["volume_avg"] = 1000.0
            df["volume_ratio"] = 1.0

        return df

    def get_signal_at(self, df: pd.DataFrame, i: int) -> Tuple[Signal, dict]:
        """
        Generate signal at bar i using only data up to i (no look-ahead).
        BUY: SMA cross above + volume confirmation (>= threshold)
        SELL: SMA cross below (exit trigger)
        """
        warmup = max(self.p.sma_period, self.p.atr_period)
        if i < warmup:
            return Signal(SignalType.NONE), {}

        last = df.iloc[i]
        if not np.isfinite(last.get("sma", np.nan)):
            return Signal(SignalType.NONE), {}

        # 'First-window' entry (i == warmup): als de eerste bar met geldige SMA boven de SMA ligt,
        # beschouwen we dat als een geldige long setup zonder expliciete cross op i-1.
        if i == warmup:
            volume_ok = float(last.get("volume_ratio", 0.0)) >= self.p.volume_threshold
            if (
                last["close"] > last["sma"]
                and np.isfinite(last.get("atr", np.nan))
                and last["atr"] > 0
                and volume_ok
            ):
                entry = float(last["close"])
                stop = entry - float(last["atr"]) * self.p.sl_mult
                target = entry + float(last["atr"]) * self.p.tp_mult
                meta = {
                    "close": entry,
                    "sma": float(last["sma"]),
                    "atr": float(last["atr"]),
                    "volume_ratio": float(last.get("volume_ratio", 1.0)),
                    "timestamp": last.name if hasattr(last, "name") else None,
                }
                return (
                    Signal(
                        type=SignalType.BUY,
                        entry=entry,
                        stop=stop,
                        target=target,
                        reason=f"BUY: first-window above SMA + vol {meta['volume_ratio']:.2f}x",
                        timestamp=meta["timestamp"],
                    ),
                    meta,
                )

        # Vanaf hier i > warmup → klassieke cross check met vorige bar
        prev = df.iloc[i - 1]
        if not np.isfinite(prev.get("sma", np.nan)):
            return Signal(SignalType.NONE), {}

        cross_above = (prev["close"] <= prev["sma"]) and (last["close"] > last["sma"])
        cross_below = (prev["close"] >= prev["sma"]) and (last["close"] < last["sma"])
        volume_ok = float(last.get("volume_ratio", 0.0)) >= self.p.volume_threshold

        meta = {
            "close": float(last["close"]),
            "sma": float(last["sma"]),
            "atr": float(last["atr"]) if np.isfinite(last.get("atr", np.nan)) else 0.0,
            "volume_ratio": float(last.get("volume_ratio", 1.0)),
            "timestamp": last.name if hasattr(last, "name") else None,
        }

        if cross_above and volume_ok and meta["atr"] > 0:
            entry = meta["close"]
            stop = entry - meta["atr"] * self.p.sl_mult
            target = entry + meta["atr"] * self.p.tp_mult
            return (
                Signal(
                    type=SignalType.BUY,
                    entry=entry,
                    stop=stop,
                    target=target,
                    reason=f"BUY: SMA cross↑ + vol {meta['volume_ratio']:.2f}x",
                    timestamp=meta["timestamp"],
                ),
                meta,
            )

        if cross_below:
            return (
                Signal(
                    type=SignalType.SELL,
                    reason="SELL: SMA cross↓ (exit)",
                    timestamp=meta["timestamp"],
                ),
                meta,
            )

        return Signal(SignalType.NONE), meta
