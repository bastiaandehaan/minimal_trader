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

    def load(self, limit: int = 0) -> pd.DataFrame:
        p = Path(self.csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

        df = pd.read_csv(p)
        # detect time column
        time_cols = ["time", "Time", "datetime", "Datetime", "date", "Date"]
        for c in time_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
                df = df.dropna(subset=[c]).set_index(c)
                break

        df.columns = df.columns.str.lower()

        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "volume" not in df.columns:
            df["volume"] = 1000.0

        df = df.sort_index()
        if limit and limit > 0:
            df = df.tail(limit)
        return df
