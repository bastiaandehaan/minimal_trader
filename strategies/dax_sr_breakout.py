import logging
import pandas as pd
from typing import Optional

from strategies.abstract import AbstractStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class DAXSRBreakoutStrategy(AbstractStrategy):
    """
    DAX Support/Resistance Breakout Strategy with improved
    risk/reward, pullback logic, and dynamic exits.
    """

    def __init__(self, params: dict):
        self.sl_multiplier: float = params.get("sl_multiplier", 1.2)
        self.tp_multiplier: float = params.get("tp_multiplier", 2.8)
        self.min_trend_strength: float = params.get("min_trend_strength", 30.0)
        self.atr_period: int = params.get("atr_period", 14)
        self.ema_fast_period: int = params.get("ema_fast_period", 8)
        self.ema_slow_period: int = params.get("ema_slow_period", 21)
        self.time_exit_bars: int = params.get("time_exit_bars", 200)

    def _is_pullback_to_support(self, df: pd.DataFrame, lookback: int = 3) -> bool:
        if len(df) < lookback + 1:
            return False
        recent = df.tail(lookback + 1)
        current_atr = recent["atr"].iloc[-1]

        min_dist = (recent["low"] - recent["ema_21"]).abs().min()
        if min_dist > 0.3 * current_atr:
            return False
        if recent["low"].iloc[-1] <= recent["low"].iloc[-2]:
            return False
        if recent["close"].iloc[-1] <= recent["close"].iloc[-2]:
            return False
        if recent["ema_8"].iloc[-1] <= recent["ema_21"].iloc[-1]:
            return False
        return True

    def _is_bounce_to_resistance(self, df: pd.DataFrame, lookback: int = 3) -> bool:
        if len(df) < lookback + 1:
            return False
        recent = df.tail(lookback + 1)
        current_atr = recent["atr"].iloc[-1]

        min_dist = (recent["high"] - recent["ema_21"]).abs().min()
        if min_dist > 0.3 * current_atr:
            return False
        if recent["high"].iloc[-1] >= recent["high"].iloc[-2]:
            return False
        if recent["close"].iloc[-1] >= recent["close"].iloc[-2]:
            return False
        if recent["ema_8"].iloc[-1] >= recent["ema_21"].iloc[-1]:
            return False
        return True

    def get_signal(self, df: pd.DataFrame, bar_idx: int = -1) -> Optional[Signal]:
        if len(df) < max(self.ema_fast_period, self.ema_slow_period, self.atr_period) + 5:
            return None

        current = df.iloc[bar_idx]
        trend_strength = abs(current["ema_8"] - current["ema_21"])

        # LONG
        if (
            current["ema_8"] > current["ema_21"]
            and trend_strength > self.min_trend_strength
            and self._is_pullback_to_support(df)
        ):
            entry = current["close"]
            atr = current["atr"]
            sl = entry - self.sl_multiplier * atr
            tp = entry + self.tp_multiplier * atr
            logger.info(f"LONG signal at {entry}, SL={sl}, TP={tp}")
            return Signal(
                type=SignalType.BUY,
                entry=entry,
                stop=sl,
                target=tp,
                reason="pullback-to-support"
            )

        # SHORT
        if (
            current["ema_8"] < current["ema_21"]
            and trend_strength > self.min_trend_strength
            and self._is_bounce_to_resistance(df)
        ):
            entry = current["close"]
            atr = current["atr"]
            sl = entry + self.sl_multiplier * atr
            tp = entry - self.tp_multiplier * atr
            logger.info(f"SHORT signal at {entry}, SL={sl}, TP={tp}")
            return Signal(
                type=SignalType.SELL,
                entry=entry,
                stop=sl,
                target=tp,
                reason="bounce-to-resistance"
            )

        return None

    def should_exit(self, position, current_bar: pd.Series) -> bool:
        atr = current_bar["atr"]
        if position.side == "long":
            if current_bar["ema_8"] < current_bar["ema_21"]:
                logger.info("Exit LONG due to trend reversal")
                return True
            if current_bar["close"] > position.entry_price + 2.5 * atr:
                logger.info("Exit LONG due to overextension")
                return True
        else:
            if current_bar["ema_8"] > current_bar["ema_21"]:
                logger.info("Exit SHORT due to trend reversal")
                return True
            if current_bar["close"] < position.entry_price - 2.5 * atr:
                logger.info("Exit SHORT due to overextension")
                return True
        return False
