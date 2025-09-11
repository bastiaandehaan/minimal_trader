# feeds/csv_feed.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_freq(freq: str) -> str:
    """
    Normalize pandas resample aliases:
      - 'T'  -> 'min'  (e.g. '5T' -> '5min')
      - 'H'  -> 'h'    (e.g. '1H' -> '1h')
    Idempotent for already-correct inputs.
    """
    if not isinstance(freq, str):
        return freq
    original = freq
    norm = freq.replace("T", "min").replace("H", "h")
    if norm != original:
        logger.info("csv_feed: normalized resample freq '%s' -> '%s'", original, norm)
    return norm


class CSVFeed:
    """
    Minimal CSV OHLCV feed.
    Expects columns: time (optional), open, high, low, close[, volume]
    Index will be tz-aware UTC and sorted ascending.
    """

    def __init__(self, csv_path: str | Path, *, resample: Optional[str] = None):
        self.csv_path = Path(csv_path)
        self.resample = _normalize_freq(resample) if resample else None

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        df = pd.read_csv(self.csv_path)

        # normalize columns
        df.columns = [c.strip().lower() for c in df.columns]

        # time handling
        if "time" in df.columns:
            ts = pd.to_datetime(df["time"], utc=True)
            df = df.drop(columns=["time"])
        else:
            # if no time column, try to parse index
            if df.index.name and "time" in str(df.index.name).lower():
                ts = pd.to_datetime(df.index, utc=True)
            else:
                raise ValueError("CSV must include 'time' column or time-like index")

        # required columns
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # volume fallback
        if "volume" not in df.columns:
            df["volume"] = 0

        # build frame
        df.index = ts
        # clean & sort
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # resample if requested
        if self.resample:
            df = df.resample(self.resample).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            ).dropna()
            logger.info("feeds.csv_feed: resampled to %s: %d bars", self.resample, len(df))

        logger.info("feeds.csv_feed: Loaded %d bars from %s", len(df), self.csv_path.name)
        return df
