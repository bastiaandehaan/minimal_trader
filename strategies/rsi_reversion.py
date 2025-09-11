# strategies/rsi_reversion.py
from __future__ import annotations

import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Classic Wilder RSI (EMA-based)."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


class RSIReversionStrategy:
    def __init__(self, params: Optional[Dict] = None):
        p = params or {}
        self.name = f"RSIRev_rsi{p.get('rsi_period', 14)}_atr{p.get('atr_period', 14)}"
        self.rsi_period = int(p.get("rsi_period", 14))
        self.oversold = float(p.get("oversold", 30.0))
        self.overbought = float(p.get("overbought", 70.0))
        self.atr_period = int(p.get("atr_period", 14))
        self.sl_multiplier = float(p.get("sl_multiplier", 1.5))
        self.tp_multiplier = float(p.get("tp_multiplier", 2.0))
        self.use_next_open = True  # engine erzorgt NEXT_OPEN
        self._signals: Dict[pd.Timestamp, Optional[str]] = {}

    def prepare(self, df: pd.DataFrame):
        """Bereken RSI en signaleer mean-reversion: RSI<oversold -> long, RSI>overbought -> short."""
        if df.empty:
            self._signals = {}
            return

        close = df["close"].astype(float)
        rsi = _rsi(close, self.rsi_period)

        # Signaal op huidige bar => entry op volgende bar (door engine)
        long_sig = rsi < self.oversold
        short_sig = rsi > self.overbought

        signals: Dict[pd.Timestamp, Optional[str]] = {}
        long_n = short_n = 0
        for ts, l, s in zip(df.index, long_sig, short_sig):
            if l and not s:
                signals[ts] = "long"
                long_n += 1
            elif s and not l:
                signals[ts] = "short"
                short_n += 1
            else:
                signals[ts] = None

        self._signals = signals
        logger.info("[%s] Prepared %d long and %d short signals", self.name, long_n, short_n)

    def generate_signal(self, ts: pd.Timestamp, row: pd.Series) -> Optional[str]:
        """Geef signaal voor deze timestamp; engine gebruikt 'm op de volgende bar-open."""
        return self._signals.get(ts, None)
