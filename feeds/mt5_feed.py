# feeds/mt5_feed.py
from __future__ import annotations

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MT5Feed:
    """Kleine, defensieve wrapper. Werkt alleen als MetaTrader5-module beschikbaar is."""

    def __init__(self, symbol: str, timeframe="M1", bars: int = 10_000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self._mt5 = None
        self._connected = False

    def connect(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None) -> bool:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception:
            logger.error("MT5 module not available; CSV mode only")
            return False
        self._mt5 = mt5
        if not mt5.initialize(login=login, password=password, server=server):
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            return False
        self._connected = True
        logger.info("Connected to MT5")
        return True

    def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    def fetch(self) -> pd.DataFrame:
        if not (self._mt5 and self._connected):
            raise RuntimeError("MT5 not connected")
        mt5 = self._mt5

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
        }
        tf = tf_map.get(self.timeframe, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, self.bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError("No MT5 rates returned")

        import numpy as np
        rates = pd.DataFrame(rates)
        rates["time"] = pd.to_datetime(rates["time"], unit="s", utc=True)
        df = rates.rename(
            columns={"time": "time", "open": "open", "high": "high", "low": "low", "close": "close", "tick_volume": "volume"}
        )[["time", "open", "high", "low", "close", "volume"]]

        df = df.set_index("time").sort_index()
        logger.info("feeds.mt5_feed: Fetched %d bars", len(df))
        return df
