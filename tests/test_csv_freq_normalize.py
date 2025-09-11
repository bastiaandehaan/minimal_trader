# tests/test_csv_freq_normalize.py
import pandas as pd
from feeds.csv_feed import _normalize_freq

def test_normalize_freq_tokens():
    assert _normalize_freq("T") == "min"
    assert _normalize_freq("H") == "h"
    assert _normalize_freq("5T") == "5min"
    assert _normalize_freq("1H") == "1h"
    assert _normalize_freq("5min") == "5min"  # idempotent

def test_resample_accepts_normalized():
    idx = pd.date_range("2025-01-01", periods=10, freq="min", tz="UTC")
    df = pd.DataFrame(
        {"open": range(10), "high": range(10), "low": range(10),
         "close": range(10), "volume": 1},
        index=idx
    )
    out = df.resample(_normalize_freq("5T")).agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    )
    assert not out.empty
