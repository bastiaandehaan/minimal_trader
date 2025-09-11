# feeds/csv_feed.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CSVFeed:
    def __init__(self, file_path: str):
        self.path = Path(file_path)

    def load(self, resample: Optional[str] = None) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        df = pd.read_csv(self.path, parse_dates=["time"])
        # verwacht schema: time, open, high, low, close, volume
        cols = ["time", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df = df[cols].drop_duplicates(subset=["time"]).set_index("time").sort_index()

        if resample:
            # naar OHLCV op gewenste resolutie
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            df = df.resample(resample).agg(agg).dropna()
            logger.info("feeds.csv_feed: resampled to %s: %d bars", resample, len(df))

        logger.info("feeds.csv_feed: Loaded %d bars from %s", len(df), self.path.name)
        return df
