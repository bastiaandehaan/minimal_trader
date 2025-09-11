# engine.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from utils.engine_guards import (
    GuardConfig,
    GuardState,
    apply_trading_hours,
    should_skip_low_vol,
    in_cooldown,
    allow_entry_at_bar,
    register_entry,
    register_exit,
)

logger = logging.getLogger("engine")
logger.setLevel(logging.INFO)


@dataclass
class EngineConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 1.0  # in %
    max_positions: int = 3
    commission: float = 0.0002   # proportional fees (entry+exit)
    slippage: float = 0.0        # not applied for now
    time_exit_bars: int = 200
    allow_shorts: bool = True
    min_risk_pts: float = 0.5    # minimale SL-afstand in punten


@dataclass
class Position:
    strategy: str
    side: str                 # "long" | "short"
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0


class MultiStrategyEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.strategies: Dict[str, object] = {}      # name -> strategy instance
        self.allocations: Dict[str, float] = {}      # name -> %
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity: float = config.initial_capital

    # ------------------------------ API ---------------------------------

    def add_strategy(self, strategy, allocation: float):
        name = getattr(strategy, "name", strategy.__class__.__name__)
        strat_key = name
        if strat_key in self.strategies:
            raise ValueError(f"Strategy {strat_key} already added")
        self.strategies[strat_key] = strategy
        self.allocations[strat_key] = float(allocation)
        logger.info("engine: Added strategy %s with %.1f%% allocation", strat_key, allocation)

    def run_backtest(
        self,
        df: pd.DataFrame,
        guard_cfg: Optional[GuardConfig] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Backtest alle strategieën op de data. Entry: ALTIJD NEXT_OPEN."""
        if df.empty:
            logger.warning("Backtest aborted: no data provided")
            return self._empty_results()

        # ATR indien niet aanwezig
        if "atr" not in df.columns:
            tr = pd.DataFrame({
                "high_low": df["high"] - df["low"],
                "high_close": (df["high"] - df["close"].shift().fillna(df["close"])).abs(),
                "low_close": (df["low"] - df["close"].shift().fillna(df["close"])).abs(),
            })
            true_range = tr.max(axis=1)
            df["atr"] = true_range.rolling(window=14, min_periods=14).mean()

        # Trading hours filter
        guard_cfg = guard_cfg or GuardConfig()
        guard_state = GuardState()
        df = apply_trading_hours(df, guard_cfg)

        logger.info("engine: NEXT_OPEN enforced for entries")

        # Reset state
        self.positions.clear()
        self.closed_positions.clear()
        self.equity = self.config.initial_capital

        # Strategie precompute (prepare) en signal-store
        for strat_name, strat in self.strategies.items():
            if hasattr(strat, "prepare"):
                try:
                    strat.prepare(df)  # bereidt intern de signalen
                    logger.info("engine: Strategy %s prepared", strat_name)
                except Exception as e:
                    logger.exception("engine: Strategy %s prepare failed: %s", strat_name, e)

        # NEXT_OPEN staging: we gebruiken signal van bar i om op bar i+1 open te plaatsen
        prev_signals: Dict[str, Optional[str]] = {k: None for k in self.strategies}

        # Itereer over bars
        index = df.index
        for bar_i, ts in enumerate(index):
            row = df.loc[ts]

            # 1) Exits eerst
            self._process_exits(bar_i, ts, row, guard_state)

            # 2) Entries op basis van signalen van vorige bar (NEXT_OPEN)
            #    => entries op huidige bar-open prijs
            for strat_name, strat in self.strategies.items():
                signal_prev = prev_signals.get(strat_name)
                if signal_prev is None:
                    continue
                if len(self.positions) >= self.config.max_positions:
                    break

                # Guards
                atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else np.nan
                if should_skip_low_vol(atr_val, guard_cfg):
                    continue
                if in_cooldown(strat_name, bar_i, guard_cfg, guard_state):
                    continue
                if not allow_entry_at_bar(ts, guard_cfg, guard_state):
                    continue
                if (signal_prev == "short") and (not self.config.allow_shorts):
                    continue

                # Risico-gestuurde sizing
                entry_price = float(row["open"])
                sl, tp = self._compute_sl_tp(signal_prev, entry_price, atr_val, strat)
                if sl is None or tp is None:
                    continue

                risk_pts = abs(entry_price - sl)
                if risk_pts < self.config.min_risk_pts:
                    continue

                risk_cash = self.equity * (self.config.risk_per_trade / 100.0)
                size = max(risk_cash / max(risk_pts, 1e-9), 0.0)

                if size <= 0:
                    continue

                pos = Position(
                    strategy=strat_name,
                    side=signal_prev,
                    entry_bar=bar_i,
                    entry_time=ts,
                    entry_price=entry_price,
                    size=size,
                    stop_loss=sl,
                    take_profit=tp,
                )
                self.positions.append(pos)
                register_entry(strat_name, ts, bar_i, guard_state)

            # 3) Bepaal signalen van huidige bar voor volgende bar-entries
            next_signals: Dict[str, Optional[str]] = {}
            for strat_name, strat in self.strategies.items():
                sig = None
                try:
                    if hasattr(strat, "generate_signal"):
                        sig = strat.generate_signal(ts, row)
                except Exception as e:
                    logger.error(
                        "Strategy %s failed to generate signal: %s", strat_name, e
                    )
                next_signals[strat_name] = sig
            prev_signals = next_signals

        metrics = self._compute_metrics(df)
        return {
            "metrics": metrics,
            "positions": self._positions_to_frame(),
        }

    # ------------------------------ intern ---------------------------------

    def _process_exits(self, bar_i: int, ts: pd.Timestamp, row: pd.Series, guard_state: GuardState):
        # time exit, SL/TP
        for pos in self.positions[:]:
            # Time exit
            if bar_i - pos.entry_bar >= self.config.time_exit_bars:
                pos.exit_time = ts
                pos.exit_price = float(row["open"])  # exit op bar-open
                pos.exit_reason = "Time Exit"
                pos.pnl = (pos.exit_price - pos.entry_price) * pos.size * (1 if pos.side == "long" else -1)
                pos.pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
                self.closed_positions.append(pos)
                register_exit(pos.strategy, bar_i, guard_state)
                self.positions.remove(pos)
                self.equity += pos.pnl
                continue

            # Prijs exits
            if pos.side == "long":
                hit_sl = row["low"] <= pos.stop_loss
                hit_tp = row["high"] >= pos.take_profit
                if hit_sl or hit_tp:
                    pos.exit_time = ts
                    price = pos.stop_loss if hit_sl else pos.take_profit
                    pos.exit_price = float(price)
                    pos.exit_reason = "Stop Loss" if hit_sl else "Take Profit"
                    pos.pnl = (pos.exit_price - pos.entry_price) * pos.size
                    pos.pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
                    self.closed_positions.append(pos)
                    register_exit(pos.strategy, bar_i, guard_state)
                    self.positions.remove(pos)
                    self.equity += pos.pnl
            else:
                hit_sl = row["high"] >= pos.stop_loss
                hit_tp = row["low"] <= pos.take_profit
                if hit_sl or hit_tp:
                    pos.exit_time = ts
                    price = pos.stop_loss if hit_sl else pos.take_profit
                    pos.exit_price = float(price)
                    pos.exit_reason = "Stop Loss" if hit_sl else "Take Profit"
                    pos.pnl = (pos.entry_price - pos.exit_price) * pos.size
                    pos.pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
                    self.closed_positions.append(pos)
                    register_exit(pos.strategy, bar_i, guard_state)
                    self.positions.remove(pos)
                    self.equity += pos.pnl

    def _compute_sl_tp(self, side: str, entry: float, atr_val: float, strategy) -> Tuple[Optional[float], Optional[float]]:
        # haal params uit strategie als aanwezig
        sl_mult = getattr(strategy, "sl_multiplier", 1.5)
        tp_mult = getattr(strategy, "tp_multiplier", 2.0)
        atr = float(atr_val) if not (pd.isna(atr_val)) else np.nan
        if np.isnan(atr) or atr <= 0:
            return None, None
        sl_off = sl_mult * atr
        tp_off = tp_mult * atr
        if side == "long":
            return entry - sl_off, entry + tp_off
        else:
            return entry + sl_off, entry - tp_off

    def _positions_to_frame(self) -> pd.DataFrame:
        if not self.closed_positions:
            return pd.DataFrame(columns=[
                "strategy", "side", "entry_time", "entry_price",
                "exit_time", "exit_price", "pnl", "exit_reason"
            ])
        rows = []
        for p in self.closed_positions:
            rows.append({
                "strategy": p.strategy,
                "side": p.side,
                "entry_time": p.entry_time,
                "entry_price": p.entry_price,
                "exit_time": p.exit_time,
                "exit_price": p.exit_price,
                "pnl": p.pnl,
                "exit_reason": p.exit_reason,
            })
        return pd.DataFrame(rows)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        trades = len(self.closed_positions)
        longs = sum(1 for p in self.closed_positions if p.side == "long")
        shorts = trades - longs
        pnl_arr = np.array([p.pnl for p in self.closed_positions], dtype=float)
        gross_profit = pnl_arr[pnl_arr > 0].sum() if trades else 0.0
        gross_loss = -pnl_arr[pnl_arr < 0].sum() if trades else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        wins = (pnl_arr > 0).sum() if trades else 0
        win_rate = 100.0 * wins / trades if trades else 0.0

        # simple equity curve by cumulative PnL
        equity_curve = [self.config.initial_capital]
        cum = self.config.initial_capital
        max_peak = cum
        max_dd = 0.0
        for p in self.closed_positions:
            cum += p.pnl
            equity_curve.append(cum)
            max_peak = max(max_peak, cum)
            dd = (max_peak - cum) / max_peak * 100.0 if max_peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        final_capital = equity_curve[-1] if equity_curve else self.config.initial_capital
        total_return = 100.0 * (final_capital / self.config.initial_capital - 1.0)

        # Sharpe placeholder (0.0) – optioneel fijnslijpen met per-bar returns
        sharpe = 0.0

        logger.info(
            "engine: Backtest complete: trades=%d, return=%.2f%%, win_rate=%.2f%%, PF=%s, DD=%.2f%%",
            trades, total_return, win_rate,
            f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "inf",
            max_dd,
        )

        return {
            "total_trades": trades,
            "total_longs": longs,
            "total_shorts": shorts,
            "initial_capital": self.config.initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor if np.isfinite(profit_factor) else float("inf"),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

    @staticmethod
    def _empty_results() -> Dict[str, Dict[str, float]]:
        return {
            "metrics": {
                "total_trades": 0,
                "total_longs": 0,
                "total_shorts": 0,
                "initial_capital": 0.0,
                "final_capital": 0.0,
                "total_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": float("inf"),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            },
            "positions": pd.DataFrame(),
        }
