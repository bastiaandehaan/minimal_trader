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
    volume_threshold: float = 1.0


class Strategy:
    """Pure strategy: indicators + signal generation. No state."""

    def __init__(self, params: StrategyParams = StrategyParams()):
        self.p = params

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SMA, ATR en een robuuste volume_ratio.
        - volume_ratio wordt geneutraliseerd (=1.0) als volume-data onbruikbaar is
          (bv. alles 0, avg ~ 0, of ratio volledig NaN/inf).
        """
        df = df.copy()

        # SMA
        df["sma"] = df["close"].rolling(
            self.p.sma_period, min_periods=self.p.sma_period
        ).mean()

        # ATR (classic True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.p.atr_period, min_periods=self.p.atr_period).mean()

        # Volume ratio — robuust tegen 0/ontbrekend volume
        if "volume" in df.columns:
            df["volume_avg"] = df["volume"].rolling(
                self.p.sma_period, min_periods=self.p.sma_period
            ).mean()

            ratio = (df["volume"] / df["volume_avg"]).replace([np.inf, -np.inf], np.nan)

            # Als volume waardeloos is (alles 0, avg ~ 0, of ratio volledig NaN) → neutraliseer filter
            if (
                df["volume"].fillna(0).sum() == 0
                or df["volume_avg"].fillna(0).sum() == 0
                or ratio.notna().sum() == 0
            ):
                df["volume_ratio"] = 1.0
            else:
                df["volume_ratio"] = ratio.fillna(1.0)
        else:
            # Geen volume → neutraliseer filter
            df["volume"] = 0.0
            df["volume_avg"] = 0.0
            df["volume_ratio"] = 1.0

        return df

    def get_signal_at(self, df: pd.DataFrame, i: int) -> Tuple[Signal, dict]:
        """
        Generate signal at bar i using only data up to i (no look-ahead).
        BUY: Classic SMA crossover with volume confirmation
        SELL: Crossover below SMA (exit trigger)
        """
        warmup = max(self.p.sma_period, self.p.atr_period)

        # Warm-up + 1 vorige bar nodig voor crossover-detectie
        if i <= warmup:
            return Signal(SignalType.NONE), {}

        last = df.iloc[i]
        prev = df.iloc[i - 1]

        # Beide SMA's moeten geldig zijn
        if not (np.isfinite(last["sma"]) and np.isfinite(prev["sma"])):
            return Signal(SignalType.NONE), {}

        # Crossover logica
        cross_above = (prev["close"] <= prev["sma"]) and (last["close"] > last["sma"])
        cross_below = (prev["close"] >= prev["sma"]) and (last["close"] < last["sma"])

        # Volume confirm — default 1.0 zodat afwezige volume_ratio niet blokkeert
        volume_ok = float(last.get("volume_ratio", 1.0)) >= self.p.volume_threshold

        # Metadata
        meta = {
            "close": float(last["close"]),
            "sma": float(last["sma"]),
            "atr": float(last["atr"]) if np.isfinite(last["atr"]) else 0.0,
            "volume_ratio": float(last.get("volume_ratio", 1.0)),
            "timestamp": last.name if hasattr(last, "name") else None,
        }

        # BUY signal
        if cross_above and volume_ok and np.isfinite(last["atr"]) and last["atr"] > 0:
            entry = float(last["close"])
            stop = entry - float(last["atr"]) * self.p.sl_mult
            target = entry + float(last["atr"]) * self.p.tp_mult

            return Signal(
                type=SignalType.BUY,
                entry=entry,
                stop=stop,
                target=target,
                reason=f"BUY: SMA cross↑ + vol {meta['volume_ratio']:.2f}x",
                timestamp=meta["timestamp"],
            ), meta

        # SELL signal (exit trigger)
        if cross_below:
            return Signal(
                type=SignalType.SELL,
                reason="SELL: SMA cross↓ (exit)",
                timestamp=meta["timestamp"],
            ), meta

        return Signal(SignalType.NONE), meta
