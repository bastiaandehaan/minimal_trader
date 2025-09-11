# engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config + Position
# ---------------------------------------------------------------------
@dataclass
class EngineConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 1.0            # in %
    max_positions: int = 3
    commission: float = 0.0002             # bps per trade leg
    slippage: float = 0.0001
    time_exit_bars: int = 200
    allow_shorts: bool = True


@dataclass
class Position:
    strategy: str
    symbol: str
    side: str                 # 'long'|'short'
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_bar: int

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------
class MultiStrategyEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.strategies: Dict[str, object] = {}
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity: float = config.initial_capital

    # ---- public API ---------------------------------------------------
    def add_strategy(self, strategy: object, allocation: float = 100.0) -> None:
        name = getattr(strategy, "name", None)
        if not name:
            # genereer herkenbare naam op basis van parameters
            params = getattr(strategy, "params", {})
            name = f"RSIRev_rsi{params.get('rsi_period','?')}_atr{params.get('atr_period','?')}"
            setattr(strategy, "name", name)
        setattr(strategy, "allocation", allocation)
        self.strategies[name] = strategy
        logger.info("engine: Added strategy %s with %.1f%% allocation", name, allocation)

    def run_backtest(self, df: pd.DataFrame, guard_cfg: Optional[GuardConfig] = None) -> Dict:
        if df.empty:
            logger.warning("Backtest aborted: no data")
            return self._empty_results()

        # Fill ATR if missing (14 SMA of True Range)
        if "atr" not in df.columns:
            tr = pd.DataFrame(
                {
                    "hl": df["high"] - df["low"],
                    "hc": (df["high"] - df["close"].shift().fillna(df["close"])).abs(),
                    "lc": (df["low"] - df["close"].shift().fillna(df["close"])).abs(),
                }
            )
            df["atr"] = tr.max(axis=1).rolling(window=14, min_periods=14).mean()

        guard_cfg = guard_cfg or GuardConfig()
        state = GuardState()

        # Trading hours filter
        df = apply_trading_hours(df, guard_cfg)

        # Pre-prepare strategies if they expose prepare()
        for s in self.strategies.values():
            if hasattr(s, "prepare") and callable(getattr(s, "prepare")):
                try:
                    s.prepare(df.copy())
                except Exception as e:
                    logger.error("engine: prepare() failed for %s: %s", getattr(s, "name", s), e)

        # Reset state
        self.positions.clear()
        self.closed_positions.clear()
        self.equity = self.config.initial_capital

        # Iterate bars
        for bar_i, (ts, row) in enumerate(df.iterrows()):
            # --- manage open positions (exits) ---
            for pos in self.positions[:]:
                # time exit
                if bar_i - pos.entry_bar >= self.config.time_exit_bars:
                    self._exit_position(pos, ts, row["open"], "Time Exit", bar_i, state)
                    continue

                if pos.side == "long":
                    if row["low"] <= pos.stop_loss or row["high"] >= pos.take_profit:
                        price = pos.stop_loss if row["low"] <= pos.stop_loss else pos.take_profit
                        reason = "Stop Loss" if row["low"] <= pos.stop_loss else "Take Profit"
                        self._exit_position(pos, ts, price, reason, bar_i, state)
                else:  # short
                    if row["high"] >= pos.stop_loss or row["low"] <= pos.take_profit:
                        price = pos.stop_loss if row["high"] >= pos.stop_loss else pos.take_profit
                        reason = "Stop Loss" if row["high"] >= pos.stop_loss else "Take Profit"
                        self._exit_position(pos, ts, price, reason, bar_i, state)

            # --- new entries ---
            for name, strat in self.strategies.items():
                if len(self.positions) >= self.config.max_positions:
                    break

                atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
                if should_skip_low_vol(atr_val, guard_cfg):
                    continue
                if in_cooldown(name, bar_i, guard_cfg, state):
                    continue
                if not allow_entry_at_bar(ts, guard_cfg, state):
                    continue

                side, sl, tp = self._get_signal(strat, ts, row, atr_val)
                if side is None:
                    continue
                if side == "short" and not self.config.allow_shorts:
                    continue

                entry_price = float(row["open"])  # "next open" benadering per bar
                size = self._position_size(entry_price, sl, side)

                pos = Position(
                    strategy=name,
                    symbol=getattr(strat, "symbol", "GER40.cash"),
                    side=side,
                    entry_time=ts,
                    entry_price=entry_price,
                    size=size,
                    stop_loss=sl,
                    take_profit=tp,
                    entry_bar=bar_i,
                )
                self.positions.append(pos)
                register_entry(ts, guard_cfg, state)

        results = self._build_results(df)
        logger.info(
            "engine: Backtest complete: trades=%d, return=%.2f%%, win_rate=%.2f%%, PF=%s, DD=%.2f%%",
            results["metrics"]["total_trades"],
            results["metrics"]["total_return"],
            results["metrics"]["win_rate"],
            ("inf" if np.isinf(results["metrics"]["profit_factor"]) else f"{results['metrics']['profit_factor']:.2f}"),
            results["metrics"]["max_drawdown"],
        )
        return results

    # ---- internals ----------------------------------------------------
    def _position_size(self, entry: float, stop: float, side: str) -> float:
        risk_cash = self.equity * (self.config.risk_per_trade / 100.0)
        per_unit_risk = abs((stop - entry) if side == "long" else (entry - stop))
        if per_unit_risk <= 0:
            per_unit_risk = max(0.01, entry * 0.0001)
        units = max(1.0, risk_cash / per_unit_risk)
        return float(units)

    def _exit_position(
        self,
        pos: Position,
        ts: pd.Timestamp,
        price: float,
        reason: str,
        bar_i: int,
        state: GuardState,
    ) -> None:
        pos.exit_time = ts
        pos.exit_price = float(price)
        pos.exit_reason = reason
        if pos.side == "long":
            pnl = (pos.exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - pos.exit_price) * pos.size
        # simple commission both legs
        pnl -= (pos.entry_price + pos.exit_price) * self.config.commission * pos.size
        pos.pnl = float(pnl)
        self.closed_positions.append(pos)
        self.positions.remove(pos)
        self.equity += pos.pnl
        register_exit(pos.strategy, bar_i, state)

    def _get_signal(
        self, strat: object, ts: pd.Timestamp, row: pd.Series, atr_val: float
    ) -> Tuple[Optional[str], float, float]:
        """
        Probeert meerdere methoden:
        - generate_signal(ts, row) of generate_signal(row)
        - generate(ts, row) / signal(ts, row) / get_signal(ts, row)
        - of lees precomputed kolommen op de strategie: .long_signal/.short_signal DataFrames
        Retourneert (side, sl, tp) of (None, 0, 0).
        """
        # 1) directe method calls
        for meth in ("generate_signal", "generate", "signal", "get_signal"):
            if hasattr(strat, meth):
                fn = getattr(strat, meth)
                try:
                    try:
                        out = fn(ts, row)           # signature (ts, row)
                    except TypeError:
                        out = fn(row)              # signature (row)
                except Exception as e:
                    logger.error("engine: %s.%s failed: %s", getattr(strat, "name", strat), meth, e)
                    out = None

                side, sl, tp = self._normalize_signal_output(out, row, atr_val, strat)
                if side:
                    return side, sl, tp

        # 2) precomputed flags op self (na prepare)
        for attr in ("signals", "signal_df"):
            if hasattr(strat, attr):
                sig_df = getattr(strat, attr)
                if isinstance(sig_df, pd.DataFrame) and ts in sig_df.index:
                    if bool(sig_df.loc[ts].get("long_signal", False)):
                        return self._default_bands("long", row, atr_val, strat)
                    if bool(sig_df.loc[ts].get("short_signal", False)):
                        return self._default_bands("short", row, atr_val, strat)

        # geen signaal
        return None, 0.0, 0.0

    def _normalize_signal_output(
        self, out, row: pd.Series, atr_val: float, strat: object
    ) -> Tuple[Optional[str], float, float]:
        """Accepteert verschillende outputvormen, geeft side/sl/tp terug."""
        if out is None:
            return None, 0.0, 0.0

        side: Optional[str] = None
        sl = tp = 0.0

        if isinstance(out, str):
            side = out if out in ("long", "short") else None
        elif isinstance(out, dict):
            side = out.get("side")
            sl = float(out.get("sl") or 0.0)
            tp = float(out.get("tp") or 0.0)

        if side not in ("long", "short"):
            return None, 0.0, 0.0

        # fallback SL/TP
        if sl == 0.0 or tp == 0.0:
            return self._default_bands(side, row, atr_val, strat)

        return side, sl, tp

    def _default_bands(self, side: str, row: pd.Series, atr_val: float, strat: object) -> Tuple[str, float, float]:
        params = getattr(strat, "params", {}) or {}
        sl_mult = float(params.get("sl_multiplier", 1.5))
        tp_mult = float(params.get("tp_multiplier", 2.0))
        atr = atr_val if atr_val and not np.isnan(atr_val) else float(row.get("atr", 0.0) or 0.0)
        atr = max(1e-6, atr)

        entry = float(row["open"])
        if side == "long":
            sl = entry - sl_mult * atr
            tp = entry + tp_mult * atr
        else:
            sl = entry + sl_mult * atr
            tp = entry - tp_mult * atr
        return side, float(sl), float(tp)

    def _build_results(self, df: pd.DataFrame) -> Dict:
        # equity curve: start + cumulatieve PnL van closed positions
        closed = self.closed_positions
        equity = [self.config.initial_capital]
        for p in closed:
            equity.append(equity[-1] + p.pnl)
        final_capital = equity[-1]
        total_pnl = final_capital - self.config.initial_capital

        wins = [p for p in closed if p.pnl > 0]
        losses = [p for p in closed if p.pnl < 0]
        win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
        gross_win = sum(p.pnl for p in wins) or 0.0
        gross_loss = -sum(p.pnl for p in losses) or 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

        # max drawdown uit equity
        eq = np.array(equity, dtype=float)
        if len(eq) > 1:
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / peak
            max_dd = float(np.max(dd) * 100.0)
        else:
            max_dd = 0.0

        # breakdown per strategie
        by_strat: Dict[str, Dict] = {}
        for p in closed:
            d = by_strat.setdefault(p.strategy, {"trades": 0, "longs": 0, "shorts": 0, "pnl": []})
            d["trades"] += 1
            d["longs"] += int(p.side == "long")
            d["shorts"] += int(p.side == "short")
            d["pnl"].append(p.pnl)

        strategies: Dict[str, Dict] = {}
        for name, d in by_strat.items():
            pnl_arr = np.array(d["pnl"], dtype=float)
            strategies[name] = {
                "trades": d["trades"],
                "longs": d["longs"],
                "shorts": d["shorts"],
                "win_rate": float((pnl_arr > 0).mean() * 100.0) if len(pnl_arr) else 0.0,
                "total_pnl": float(pnl_arr.sum()) if len(pnl_arr) else 0.0,
                "avg_pnl": float(pnl_arr.mean()) if len(pnl_arr) else 0.0,
                "best_trade": float(pnl_arr.max()) if len(pnl_arr) else 0.0,
                "worst_trade": float(pnl_arr.min()) if len(pnl_arr) else 0.0,
            }

        metrics = {
            "initial_capital": float(self.config.initial_capital),
            "final_capital": float(final_capital),
            "total_return": float((final_capital / self.config.initial_capital - 1.0) * 100.0),
            "total_trades": int(len(closed)),
            "total_longs": int(sum(p.side == "long" for p in closed)),
            "total_shorts": int(sum(p.side == "short" for p in closed)),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": 0.0,  # optioneel later
            "max_drawdown": float(max_dd),
        }

        return {"metrics": metrics, "strategies": strategies}

    def _empty_results(self) -> Dict:
        return {
            "metrics": {
                "initial_capital": float(self.config.initial_capital),
                "final_capital": float(self.config.initial_capital),
                "total_return": 0.0,
                "total_trades": 0,
                "total_longs": 0,
                "total_shorts": 0,
                "win_rate": 0.0,
                "profit_factor": float("inf"),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            },
            "strategies": {},
        }
