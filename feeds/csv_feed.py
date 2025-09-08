"""CSV data feed for backtesting."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CSVFeed:
    """Load OHLC data from CSV files."""

    def __init__(self, filepath: str, symbol: str = None):
        self.filepath = Path(filepath)
        self.symbol = symbol or self.filepath.stem
        self.data = None

    def load(self, resample: str = None) -> pd.DataFrame:
        """Load and prepare CSV data."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV not found: {self.filepath}")

        # Read CSV
        df = pd.read_csv(self.filepath)

        # Detect time column
        time_cols = ['time', 'Time', 'datetime', 'Datetime', 'date', 'Date']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            raise ValueError("No time column found in CSV")

        # Parse datetime and set index
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
        df = df.dropna(subset=[time_col])
        df.set_index(time_col, inplace=True)
        df.sort_index(inplace=True)

        # Standardize columns
        df.columns = df.columns.str.lower()

        # Check required columns
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Handle volume (use 0 if missing)
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Optional resampling
        if resample:
            df = self._resample_ohlc(df, resample)

        self.data = df
        logger.info(f"Loaded {len(df)} bars from {self.filepath.name}")
        return df

    def _resample_ohlc(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLC data to different timeframe."""
        return df.resample(freq).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum'}).dropna()

    def get_symbol_info(self) -> dict:
        """Return mock symbol info for CSV data."""
        return {'symbol': self.symbol, 'digits': 2, 'point': 0.01, 'tick_size': 0.01,
            'tick_value': 1.0, 'min_lot': 0.01, 'max_lot': 100.0, 'lot_step': 0.01,
            'spread': 2.0, 'stops_level': 0}