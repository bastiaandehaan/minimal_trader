# minimal_trader/strategies/rsi_reversion.py
"""
RSI Reversion Strategy – mean reversion met ATR-stops/targets.
- Geen look-ahead: signal gebruikt alleen data t/m bar i.
- Optioneel NEXT_OPEN entry: als er geen volgende bar is -> geen trade.
- Logging op DEBUG met kernmetadata (rsi/atr/entry/levels).
"""
from __future__ import annotations
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import logging

from strategies.abstract import AbstractStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class RSIReversionStrategy(AbstractStrategy):
    """
    Long bij RSI cross-up vanuit oversold; Short bij RSI cross-down vanuit overbought.
    Stops/targets op basis van ATR.
    """
    def __init__(self, params: dict | None = None):
        default_params = {
            'rsi_period': 14,
            'oversold': 30.0,
            'overbought': 70.0,
            'atr_period': 14,
            'sl_multiplier': 1.5,
            'tp_multiplier': 2.0,
            # Als True: entry prijs = next bar 'open' (indien beschikbaar)
            'use_next_open': True,
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    @property
    def name(self) -> str:
        p = self.params
        return f"RSIRev_rsi{p['rsi_period']}_atr{p['atr_period']}"

    @property
    def required_bars(self) -> int:
        # +1 omdat we mogelijk next-open gebruiken (maar engine opent nog steeds op huidige bar-timestamp)
        return max(self.params['rsi_period'], self.params['atr_period']) + 1

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Voegt kolommen toe: 'rsi', 'atr'
        OHLC schema vereist: ['open','high','low','close'].
        Index gesorteerd oplopend, geen dupes.
        """
        d = df.copy()

        # --- RSI (Wilder EWM) ---
        close = d['close']
        delta = close.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        # Wilder smoothing (alpha=1/period)
        period = int(self.params['rsi_period'])
        alpha = 1.0 / period
        roll_up = pd.Series(up, index=d.index).ewm(alpha=alpha, adjust=False).mean()
        roll_down = pd.Series(down, index=d.index).ewm(alpha=alpha, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        d['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        # --- ATR voor stops/targets (zelfde TR-methode als in repo) ---
        high_low = d['high'] - d['low']
        high_close = (d['high'] - d['close'].shift()).abs()
        low_close = (d['low'] - d['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        d['atr'] = tr.rolling(self.params['atr_period'], min_periods=self.params['atr_period']).mean()

        return d

    def _entry_price(self, df: pd.DataFrame, i: int) -> Optional[float]:
        """
        Respecteer 'NEXT_OPEN' zonder look-ahead in de signaaldetectie:
        - Signaal beslissen op bar i (zonder df.iloc[i+1] te raadplegen)
        - Entry prijs pas invullen uit bar (i+1).open als die bestaat.
        """
        if self.params.get('use_next_open', True):
            if i + 1 < len(df):
                return float(df.iloc[i + 1]['open'])
            return None
        # fallback: huidige close (compatibel met bestaande engine)
        return float(df.iloc[i]['close'])

    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[Signal, Dict]:
        """
        Signaalgeneratie op bar i met alleen data <= i.
        """
        if i < self.required_bars:
            return Signal(SignalType.NONE, strategy=self.name), {}

        cur = df.iloc[i]
        prev = df.iloc[i - 1]

        # Controles op geldige indicatoren
        if not (np.isfinite(cur.get('rsi', np.nan)) and np.isfinite(cur.get('atr', np.nan))):
            return Signal(SignalType.NONE, strategy=self.name), {}

        # Crossover logica
        rsi = float(cur['rsi'])
        rsi_prev = float(prev['rsi'])
        atr = float(cur['atr']) if np.isfinite(cur['atr']) else 0.0

        long_setup = (rsi_prev < self.params['oversold']) and (rsi > rsi_prev)
        short_setup = (rsi_prev > self.params['overbought']) and (rsi < rsi_prev)

        meta = {
            'close': float(cur['close']),
            'rsi': rsi,
            'atr': atr,
            'timestamp': cur.name,
        }

        if long_setup and atr > 0:
            entry = self._entry_price(df, i)
            if entry is None:  # geen volgende bar -> geen trade
                return Signal(SignalType.NONE, strategy=self.name), meta

            stop = entry - self.params['sl_multiplier'] * atr
            target = entry + self.params['tp_multiplier'] * atr

            logger.debug(
                f"[{self.name}] LONG signal @i={i} rsi={rsi:.2f} prev={rsi_prev:.2f} "
                f"entry={entry:.2f} SL={stop:.2f} TP={target:.2f}"
            )
            return Signal(
                type=SignalType.BUY,
                entry=entry,
                stop=stop,
                target=target,
                reason="RSI cross-up from oversold",
                strategy=self.name,
                timestamp=meta['timestamp']
            ), meta

        if short_setup and atr > 0:
            entry = self._entry_price(df, i)
            if entry is None:
                return Signal(SignalType.NONE, strategy=self.name), meta

            stop = entry + self.params['sl_multiplier'] * atr
            target = entry - self.params['tp_multiplier'] * atr

            logger.debug(
                f"[{self.name}] SHORT signal @i={i} rsi={rsi:.2f} prev={rsi_prev:.2f} "
                f"entry={entry:.2f} SL={stop:.2f} TP={target:.2f}"
            )
            return Signal(
                type=SignalType.SELL,
                entry=entry,
                stop=stop,
                target=target,
                reason="RSI cross-down from overbought",
                strategy=self.name,
                timestamp=meta['timestamp']
            ), meta

        return Signal(SignalType.NONE, strategy=self.name), meta

    # (optioneel) param-validatie hook
    def validate_params(self) -> bool:
        try:
            assert 2 <= int(self.params['rsi_period']) <= 200
            assert 2 <= int(self.params['atr_period']) <= 200
            assert 0.0 <= float(self.params['oversold']) < float(self.params['overbought']) <= 100.0
            assert float(self.params['sl_multiplier']) > 0
            assert float(self.params['tp_multiplier']) > 0
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Invalid params: {e}")
            return False
