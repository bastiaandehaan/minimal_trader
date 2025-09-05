from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


class DataFeed:
    def load(self, limit: int = 0) -> pd.DataFrame:
        raise NotImplementedError


@dataclass(frozen=True)
class CSVFeed(DataFeed):
    csv_path: str
    resample_to: str = None  # e.g., "1H", "5T", etc.

    def load(self, limit: int = 0) -> pd.DataFrame:
        p = Path(self.csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

        df = pd.read_csv(p)

        # Detect time column
        time_cols = ["time", "Time", "datetime", "Datetime", "date", "Date"]
        time_col = None
        for c in time_cols:
            if c in df.columns:
                time_col = c
                break

        if time_col is None:
            raise ValueError("No time column found in CSV")

        # Parse datetime and set index
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

        # Standardize columns
        df.columns = df.columns.str.lower()

        # Check required columns
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add volume if missing
        if "volume" not in df.columns:
            df["volume"] = 1000.0

        # Optional resampling (e.g., M1 to H1)
        if self.resample_to:
            df = self.resample_ohlc(df, self.resample_to)

        # Apply limit
        if limit and limit > 0:
            df = df.tail(limit)

        return df

    def resample_ohlc(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLC data to different timeframe"""
        return df.resample(freq).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum'}).dropna()