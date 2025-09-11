# feeds/mt5_feed.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# MetaTrader5 is optioneel; import faalt netjes als het er niet is.
try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # ImportError of init-issues
    mt5 = None  # we laten load() hierop reageren met een duidelijke fout


@dataclass
class MT5Config:
    symbol: str = "GER40.cash"
    timeframe: str = "M1"  # "M1","M5","M15","H1",...
    bars: int = 10_000
    tz: str = "UTC"


def _tf_to_mt5(tf: str):
    """Map eenvoudige string timeframes naar MT5 constants (indien beschikbaar)."""
    if mt5 is None:
        return None
    tf = tf.upper()
    mapping = {
        "M1": getattr(mt5, "TIMEFRAME_M1", None),
        "M5": getattr(mt5, "TIMEFRAME_M5", None),
        "M15": getattr(mt5, "TIMEFRAME_M15", None),
        "M30": getattr(mt5, "TIMEFRAME_M30", None),
        "H1": getattr(mt5, "TIMEFRAME_H1", None),
        "H4": getattr(mt5, "TIMEFRAME_H4", None),
        "D1": getattr(mt5, "TIMEFRAME_D1", None),
    }
    return mapping.get(tf)


class MT5Feed:
    """Eenvoudige MT5 feed met veilige fallbacks en duidelijk logging."""

    def __init__(self, config: Optional[MT5Config] = None, **kwargs):
        # zowel MT5Feed(MT5Config(...)) als MT5Feed(symbol="...", ...) ondersteunen
        if config is None:
            config = MT5Config(**kwargs)
        self.cfg = config
        self._connected = False

    def connect(self) -> None:
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 module niet beschikbaar. Installeer 'MetaTrader5' "
                "of gebruik --csv in plaats van MT5 input."
            )
        if self._connected:
            return
        if not mt5.initialize():
            raise RuntimeError(f"MT5.initialize() faalde: {mt5.last_error()}")
        self._connected = True

        # Zorg dat symbool beschikbaar is
        sym = self.cfg.symbol
        if not mt5.symbol_select(sym, True):
            raise RuntimeError(f"Kon symbool niet selecteren in MT5: {sym}")

        info = mt5.symbol_info(sym)
        spread = getattr(info, "spread", None)
        logger.info("feeds.mt5_feed: Connected to MT5 - Symbol: %s, Spread: %s", sym, spread)

    def disconnect(self) -> None:
        if mt5 is not None and self._connected:
            mt5.shutdown()
        self._connected = False
        logger.info("feeds.mt5_feed: Disconnected from MT5")

    def load(self, resample: Optional[str] = None) -> pd.DataFrame:
        """Haal bars op uit MT5 en retourneer OHLCV DataFrame in UTC index.
        resample: optionele pandas-freq (bv '5min', '15min').
        """
        self.connect()
        try:
            tf_const = _tf_to_mt5(self.cfg.timeframe)
            if tf_const is None:
                raise ValueError(f"Onbekend timeframe voor MT5: {self.cfg.timeframe}")

            rates = mt5.copy_rates_from_pos(self.cfg.symbol, tf_const, 0, self.cfg.bars)
            if rates is None or len(rates) == 0:
                raise RuntimeError("MT5 gaf geen data terug (rates is leeg)")

            df = pd.DataFrame(rates)
            # MT5 time is epoch seconds (UTC).
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("time")[["open", "high", "low", "close", "tick_volume"]].rename(
                columns={"tick_volume": "volume"}
            )
            df.index = df.index.tz_convert(self.cfg.tz)

            if resample:
                df = df.resample(resample).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna()

            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            logger.info(
                "feeds.mt5_feed: Fetched %d bars from %s to %s",
                len(df),
                df.index[0],
                df.index[-1],
            )
            return df
        finally:
            self.disconnect()
