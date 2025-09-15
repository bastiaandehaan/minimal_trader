# feeds/mt5_feed.py
from __future__ import annotations

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import MetaTrader5 as mt5
except Exception as e:
    mt5 = None
    logger.debug("MetaTrader5 import failed: %s", e)


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df.index.name = "time"
    return df


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    need = ["open", "high", "low", "close"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"MT5: missing column '{c}'")
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")]
    return df


class MT5Feed:
    """
    Past op main.run_backtest_mt5(): connect() -> fetch() -> disconnect()
    """

    def __init__(self, symbol: str, timeframe: str = "M1", bars: int = 10_000):
        if mt5 is None:
            raise ImportError("MetaTrader5 package niet beschikbaar. Installeer met: pip install MetaTrader5")
        self.symbol = symbol
        self.timeframe_str = timeframe.upper()
        self.bars = int(bars)
        self._connected = False

        # Map timeframe string -> MT5 const
        tfmap = {
            "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
            "M5": mt5.TIMEFRAME_M5, "M10": mt5.TIMEFRAME_M10, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
        }
        self.timeframe = tfmap.get(self.timeframe_str)
        if self.timeframe is None:
            raise ValueError(f"Unsupported timeframe: {self.timeframe_str}")

    @staticmethod
    def _as_int_or_none(x) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        s = str(x).strip()
        return int(s) if s.isdigit() else None

    def connect(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None) -> bool:
        # Probeer initialize met of zonder credentials
        ok = False
        if all([login, password, server]):
            ilogin = self._as_int_or_none(login)
            if ilogin is None:
                logger.error("MT5 login must be an integer; got: %r", login)
                return False
            # sommige builds accepteren creds direct in initialize()
            ok = mt5.initialize(login=ilogin, password=password, server=server)
            if not ok:
                logger.warning("MT5 initialize(login=...) failed: %s — trying initialize()+login()", mt5.last_error())
                ok = mt5.initialize()
                if ok:
                    ok = mt5.login(ilogin, password=password, server=server)
        else:
            ok = mt5.initialize()

        if not ok:
            logger.error("MT5 initialize/login failed: %s", mt5.last_error())
            return False

        # Symbol zichtbaar maken
        info = mt5.symbol_info(self.symbol)
        if info is None or not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error("MT5 symbol_select failed for %s: %s", self.symbol, mt5.last_error())
                mt5.shutdown()
                return False

        self._connected = True
        logger.info("feeds.mt5_feed: connected; symbol=%s timeframe=%s bars=%d", self.symbol, self.timeframe_str, self.bars)
        return True

    def fetch(self, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        if not self._connected:
            raise RuntimeError("MT5Feed not connected")

        if start is not None or end is not None:
            if start is None:
                raise ValueError("start required when end is provided")
            if end is None:
                end = pd.Timestamp.utcnow().tz_localize("UTC")
            start_dt = start.tz_convert("UTC").to_pydatetime()
            end_dt = end.tz_convert("UTC").to_pydatetime()
            rates = mt5.copy_rates_range(self.symbol, self.timeframe, start_dt, end_dt)
        else:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.bars)

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 returned no data for {self.symbol} ({self.timeframe_str}). Last error: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df = _to_utc_index(df)
        df = _ensure_ohlcv(df)

        logger.info("feeds.mt5_feed: fetched raw=%d bars (UTC indexed)", len(df))
        return df

    def disconnect(self):
        if self._connected:
            try:
                mt5.shutdown()
            finally:
                self._connected = False
                logger.info("feeds.mt5_feed: disconnected")
