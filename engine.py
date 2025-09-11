"""Multi-strategy trading engine with risk management and guard rails.

Dit engine-bestand ondersteunt meerdere strategieën en dwingt diverse
handelsregels af: NEXT_OPEN-invoer, risk-based sizing, trading-hours gate,
volatility floor (min ATR), cooldown tussen trades, en throttles (max trades
per bar/dag).  Het integreert de guard-functies uit utils.engine_guards.

Gebruik:
  1. Laad je OHLCV-data in een pandas DataFrame (UTC-index, kolommen
     ['open','high','low','close','volume']).
  2. Maak een MultiStrategyEngine-instance aan.
  3. Voeg strategieën toe met add_strategy().
  4. Roep run_backtest() aan met je DataFrame (en optioneel een GuardConfig).

"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.abstract import AbstractStrategy, SignalType
from utils.engine_guards import (
    GuardConfig,
    GuardState,
    apply_trading_hours,
    allow_entry_at_bar,
    register_entry,
    should_skip_low_vol,
    in_cooldown,
    register_exit,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track an open or closed trading position."""
    strategy: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    commission: float = 0.0


@dataclass
class EngineConfig:
    """Configuratie voor het engine."""
    initial_capital: float = 10_000.0
    risk_per_trade: float = 1.0  # % van kapitaal per trade
    max_positions: int = 3
    commission: float = 0.0002  # 2 bps
    slippage: float = 0.0001    # 1 bp
    time_exit_bars: int = 200
    allow_shorts: bool = True


class MultiStrategyEngine:
    """Voert meerdere strategieën uit met één centraal position-management."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.strategies: Dict[str, AbstractStrategy] = {}
        self.allocations: Dict[str, float] = {}
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity: float = self.config.initial_capital
        self.daily_start_equity: float = self.config.initial_capital
        self.max_equity: float = self.config.initial_capital

    def add_strategy(self, strategy: AbstractStrategy, allocation: float = 100.0) -> None:
        """Registreer een strategie en haar kapitaalallocatie."""
        if strategy.name in self.strategies:
            logger.warning("Strategy %s already exists", strategy.name)
            return
        self.strategies[strategy.name] = strategy
        self.allocations[strategy.name] = allocation
        logger.info("Added strategy %s with %.1f%% allocation", strategy.name, allocation)

    def calculate_position_size(self, capital: float, entry: float, stop: float) -> float:
        """Bepaal de positie-grootte op basis van risico."""
        risk_amount = capital * (self.config.risk_per_trade / 100.0)
        risk_points = abs(entry - stop)
        if risk_points <= 0:
            return 0.0
        size = risk_amount / risk_points
        return size

    def run_backtest(
        self,
        df: pd.DataFrame,
        guard_cfg: Optional[GuardConfig] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Voer de backtest uit over alle strategieën op de gegeven data."""
        if df.empty:
            logger.warning("Backtest aborted: no data provided")
            return {}

        # ATR berekenen als deze nog niet aanwezig is
        if "atr" not in df.columns:
            tr = pd.DataFrame({
                "high_low": df["high"] - df["low"],
                "high_close": (df["high"] - df["close"].shift().fillna(df["close"])).abs(),
                "low_close": (df["low"] - df["close"].shift().fillna(df["close"])).abs(),
            })
            true_range = tr.max(axis=1)
            df["atr"] = true_range.rolling(window=14, min_periods=14).mean()

        guard_cfg = guard_cfg or GuardConfig()
        guard_state = GuardState()

        # Alleen handelen binnen de trading hours (08:00–22:00 Europe/Brussels)
        df = apply_trading_hours(df, guard_cfg)
        logger.info("engine: NEXT_OPEN enforced for entries")

        # Reset interne staat
        self.positions.clear()
        self.closed_positions.clear()
        self.equity = self.config.initial_capital

        # Loop over elke bar
        for bar_i, (ts, row) in enumerate(df.iterrows()):
            # Eerst exits checken
            for pos in self.positions[:]:
                # Time exit
                if bar_i - pos.entry_bar >= self.config.time_exit_bars:
                    pos.exit_time = ts
                    pos.exit_price = row["open"]  # volgende bar open (benadering)
                    pos.exit_reason = "Time Exit"
                    pos.pnl = (pos.exit_price - pos.entry_price) * pos.size * (1 if pos.side == "long" else -1)
                    pos.pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
                    self.closed_positions.append(pos)
                    register_exit(pos.strategy, bar_i, guard_state)
                    self.positions.remove(pos)
                    self.equity += pos.pnl
                    continue
                # Stop loss / Take profit
                if pos.side == "long":
                    if row["low"] <= pos.stop_loss or row["high"] >= pos.take_profit:
                        pos.exit_time = ts
                        price = pos.stop_loss if row["low"] <= pos.stop_loss else pos.take_profit
                        pos.exit_price = price
                        pos.exit_reason = "Stop Loss" if row["low"] <= pos.stop_loss else "Take Profit"
                        pos.pnl = (pos.exit_price - pos.entry_price) * pos.size
                        pos.pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
                        self.closed_positions.append(pos)
                        register_exit(pos.strategy, bar_i, guard_state)
                        self.positions.remove(pos)
                        self.equity += pos.pnl
                else:  # short
                    if row["high"] >= pos.stop_loss or row["low"] <= pos.take_profit:
                        pos.exit_time = ts
                        price = pos.stop_loss if row["high"] >= pos.stop_loss else pos.take_profit
                        pos.exit_price = price
                        pos.exit_reason = "Stop Loss" if row["high"] >= pos.stop_loss else "Take Profit"
                        pos.pnl = (pos.entry_price - pos.exit_price) * pos.size
                        pos.pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
                        self.closed_positions.append(pos)
                        register_exit(pos.strategy, bar_i, guard_state)
                        self.positions.remove(pos)
                        self.equity += pos.pnl

            # Nieuwe entries evalueren
            for strategy_name, strategy in self.strategies.items():
                if len(self.positions) >= self.config.max_positions:
                    break

                # Guard: te lage ATR?
                atr_val = row["atr"]
                if should_skip_low_vol(atr_val, guard_cfg):
                    continue

                # Guard: cooldown per strategie
                if in_cooldown(strategy_name, bar_i, guard_cfg, guard_state):
                    continue

                # Guard: één trade per tijdstap en max trades per dag
                # Hier is guard_state toegevoegd als derde argument.
                if not allow_entry_at_bar(ts, guard_cfg, guard_state):
                    continue

                # Strategie-signaal ophalen
                try:
                    signal = strategy.generate_signal(row)
                except Exception as exc:
                    logger.error("Strategy %s failed to generate signal: %s", strategy_name, exc)
                    continue

                if signal is None or signal.type == SignalType.NONE:
                    continue

                # NEXT_OPEN: voer pas in op volgende bar
                if bar_i + 1 >= len(df.index):
                    continue
                next_open_ts = df.index[bar_i + 1]
                next_open_price = df.iloc[bar_i + 1]["open"]

                # Stop-loss / take-profit bepalen op basis van ATR-multipliers
                sl_mult = getattr(signal, "sl_multiplier", 2.0)
                tp_mult = getattr(signal, "tp_multiplier", 3.0)
                if signal.side == "long":
                    stop = next_open_price - sl_mult * atr_val
                    target = next_open_price + tp_mult * atr_val
                else:
                    stop = next_open_price + sl_mult * atr_val
                    target = next_open_price - tp_mult * atr_val

                # Risico-gebaseerde sizing
                size = self.calculate_position_size(self.equity, next_open_price, stop)
                if size <= 0:
                    continue

                # Positie aanmaken
                pos = Position(
                    strategy=strategy_name,
                    symbol=strategy.symbol,
                    side="long" if signal.side == "long" else "short",
                    entry_time=next_open_ts.to_pydatetime(),
                    entry_price=next_open_price,
                    size=size,
                    stop_loss=stop,
                    take_profit=target,
                    entry_bar=bar_i + 1,
                )
                self.positions.append(pos)
                register_entry(next_open_ts, guard_cfg, guard_state)

        # Overgebleven posities sluiten op de laatste close
        for pos in self.positions:
            last_row = df.iloc[-1]
            pos.exit_time = df.index[-1].to_pydatetime()
            pos.exit_price = last_row["close"]
            pos.exit_reason = "Final"
            pos.pnl = (pos.exit_price - pos.entry_price) * pos.size * (1 if pos.side == "long" else -1)
            pos.pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
            self.closed_positions.append(pos)
            self.equity += pos.pnl

        # Metrics opbouwen
        total_trades = len(self.closed_positions)
        longs = sum(1 for p in self.closed_positions if p.side == "long")
        shorts = total_trades - longs
        wins = sum(1 for p in self.closed_positions if p.pnl > 0)
        gross_win = sum(p.pnl for p in self.closed_positions if p.pnl > 0)
        gross_loss = -sum(p.pnl for p in self.closed_positions if p.pnl < 0)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        win_rate = wins / total_trades * 100.0 if total_trades > 0 else 0.0

        # Sharpe: simpele variant op basis van cumulatieve returns
        returns = []
        for p in self.closed_positions:
            returns.append(p.pnl / self.config.initial_capital)
        if returns and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max Drawdown
        equity_curve = np.cumsum([p.pnl for p in self.closed_positions]) + self.config.initial_capital
        if len(equity_curve) > 0:
            cummax = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - cummax) / cummax
            max_dd = -np.min(drawdowns) * 100.0
        else:
            max_dd = 0.0

        metrics = {
            "total_trades": total_trades,
            "total_longs": longs,
            "total_shorts": shorts,
            "total_return": (self.equity - self.config.initial_capital) / self.config.initial_capital * 100.0,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "final_capital": self.equity,
        }

        logger.info(
            "Backtest complete: trades=%d, return=%.2f%%, win_rate=%.2f%%, PF=%.2f, DD=%.2f%%",
            total_trades,
            metrics["total_return"],
            win_rate,
            profit_factor,
            max_dd,
        )

        return {
            "metrics": metrics,
            "positions": [p.__dict__ for p in self.closed_positions],
        }
