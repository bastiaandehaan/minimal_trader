# feeds/mt5_feed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as e:  # pragma: no cover
    mt5 = None  # zodat unit-tests zonder MT5 niet crashen


@dataclass
class MT5Config:
    symbol: str = "GER40.cash"
    timeframe: str = "M1"  # M1, M5, M15, H1, ...
    bars: int = 10000
    tz: str = "UTC"


class MT5Feed:
    """Eenvoudige MT5 databrug → pandas DataFrame (UTC)."""

    def __init__(self, cfg: Optional[MT5Config] = None) -> None:
        self.cfg = cfg or MT5Config()
        self._connected = False

    def connect(self) -> bool:
        if mt5 is None:  # pragma: no cover
            raise RuntimeError("MetaTrader5 Python package is not available")
        self._connected = bool(mt5.initialize())
        return self._connected

    def disconnect(self) -> None:
        if mt5 and self._connected:  # pragma: no cover
            mt5.shutdown()
        self._connected = False

    # Mapping text → MT5 timeframe constant
    _TF_MAP = {
        "M1": getattr(mt5, "TIMEFRAME_M1", None) if mt5 else None,
        "M5": getattr(mt5, "TIMEFRAME_M5", None) if mt5 else None,
        "M15": getattr(mt5, "TIMEFRAME_M15", None) if mt5 else None,
        "H1": getattr(mt5, "TIMEFRAME_H1", None) if mt5 else None,
        "H4": getattr(mt5, "TIMEFRAME_H4", None) if mt5 else None,
        "D1": getattr(mt5, "TIMEFRAME_D1", None) if mt5 else None,
    }

    @staticmethod
    def _normalize_freq(freq: Optional[str]) -> Optional[str]:
        if not freq:
            return None
        # accepteer beide aliasen
        return freq.replace("T", "min")

    def load(self, resample: Optional[str] = None) -> pd.DataFrame:
        if mt5 is None:  # pragma: no cover
            raise RuntimeError("MetaTrader5 Python package is not available")

        if not self._connected:  # pragma: no cover
            raise RuntimeError("Call connect() before load().")

        tf_const = self._TF_MAP.get(self.cfg.timeframe.upper())
        if tf_const is None:  # pragma: no cover
            raise ValueError(f"Unsupported timeframe: {self.cfg.timeframe}")

        rates = mt5.copy_rates_from_pos(self.cfg.symbol, tf_const, 0, int(self.cfg.bars))
        if rates is None or len(rates) == 0:  # pragma: no cover
            raise RuntimeError("No rates returned from MT5")

        df = pd.DataFrame(rates)[["time", "open", "high", "low", "close", "tick_volume"]]
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={"tick_volume": "volume"}).sort_index()

        # Resample indien gevraagd
        freq = self._normalize_freq(resample)
        if freq:
            o = df["open"].resample(freq).first()
            h = df["high"].resample(freq).max()
            l = df["low"].resample(freq).min()
            c = df["close"].resample(freq).last()
            v = df["volume"].resample(freq).sum().fillna(0)
            df = pd.concat([o, h, l, c, v], axis=1).dropna(how="any")
            df.columns = ["open", "high", "low", "close", "volume"]

        return df[["open", "high", "low", "close", "volume"]]
