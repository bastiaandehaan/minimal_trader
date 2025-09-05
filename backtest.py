from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from strategy import Strategy, StrategyParams, SignalType


@dataclass(frozen=True)
class ExecConfig:
    initial_capital: float = 10_000.0
    risk_pct: float = 1.0           # % of equity
    commission: float = 0.0002      # 2 bps per side
    point_value: float = 1.0        # PnL per pt per 1.0 size
    time_exit_bars: int = 200       # fail-safe max bars in trade
    progress_every: int = 10_000    # progress log cadence


class Backtest:
    def __init__(self, strategy: Strategy, cfg: ExecConfig = ExecConfig()):
        self.strategy = strategy
        self.cfg = cfg

    def _position_size(self, equity: float, entry: float, stop: float) -> float:
        risk_amount = equity * (self.cfg.risk_pct / 100.0)
        risk_pts = abs(entry - stop)
        if risk_pts <= 0:
            return 0.0
        size = risk_amount / (risk_pts * self.cfg.point_value)
        return round(size, 2)

    def run(self, df: pd.DataFrame, logger=None) -> Dict:
        # Precompute indicators once (O(n))
        df = self.strategy.calculate_indicators(df)
        n = len(df)
        warmup = max(self.strategy.p.sma_period, self.strategy.p.atr_period)

        equity = self.cfg.initial_capital
        position: Optional[Dict] = None
        trades: List[Dict] = []

        for i in range(warmup, n):
            if logger and self.cfg.progress_every and i % self.cfg.progress_every == 0:
                logger.info(f"Backtesting... {i}/{n}")

            # Determine signal at bar i (no look-ahead)
            sig, meta = self.strategy.get_signal_at(df, i)
            bar = df.iloc[i]

            # 1) Check exits first
            if position is not None:
                exit_price = None
                exit_reason = None

                # hard SL/TP using bar's low/high
                if bar["low"] <= position["stop"]:
                    exit_price = position["stop"]
                    exit_reason = "Stop Loss"
                elif bar["high"] >= position["target"]:
                    exit_price = position["target"]
                    exit_reason = "Take Profit"
                elif sig.type == SignalType.SELL:
                    exit_price = float(bar["close"])
                    exit_reason = "Reversal Exit"
                elif (i - position["entry_index"]) > self.cfg.time_exit_bars:
                    exit_price = float(bar["close"])
                    exit_reason = "Time Exit"

                if exit_price is not None:
                    gross = (exit_price - position["entry"]) * position["size"] * self.cfg.point_value
                    fees = (position["entry"] * position["size"] * self.cfg.commission) \
                         + (exit_price * position["size"] * self.cfg.commission)
                    pnl = gross - fees
                    equity += pnl

                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": bar.name,
                        "side": "long",
                        "entry": position["entry"],
                        "exit": exit_price,
                        "size": position["size"],
                        "pnl": pnl,
                        "reason": exit_reason,
                    })
                    position = None

            # 2) Entries (only if flat)
            if position is None and sig.type == SignalType.BUY:
                entry = float(bar["close"])
                stop = float(sig.stop)
                target = float(sig.target)
                size = self._position_size(equity, entry, stop)
                if size > 0:
                    position = {
                        "entry_time": bar.name,
                        "entry_index": i,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "size": size,
                    }

        # Metrics
        trades_df = pd.DataFrame(trades)
        metrics = self._metrics(trades_df, equity)
        return {"metrics": metrics, "trades": trades_df}

    def _metrics(self, trades_df: pd.DataFrame, final_equity: float) -> Dict:
        ic = self.cfg.initial_capital
        if trades_df.empty:
            return {
                "initial_capital": ic,
                "final_capital": final_equity,
                "total_return_pct": ((final_equity / ic) - 1.0) * 100.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "message": "No trades",
            }

        wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        losses = trades_df.loc[trades_df["pnl"] <= 0, "pnl"]

        equity_curve = ic + trades_df["pnl"].cumsum()
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak.replace(0, np.nan)
        max_dd_pct = abs(dd.min()) * 100.0 if not dd.empty else 0.0

        ret = trades_df["pnl"] / ic * 100.0
        sharpe = 0.0
        if ret.std(ddof=0) > 0:
            days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).days or 1
            tpy = (252 / days) * len(trades_df) if days > 0 else len(trades_df)
            sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(max(tpy, 1))

        pf = 0.0
        if not losses.empty and losses.sum() != 0:
            pf = abs(wins.sum() / losses.sum()) if not wins.empty else 0.0

        return {
            "initial_capital": ic,
            "final_capital": final_equity,
            "total_return_pct": ((final_equity / ic) - 1.0) * 100.0,
            "num_trades": int(len(trades_df)),
            "win_rate": (len(wins) / len(trades_df)) * 100.0,
            "profit_factor": pf,
            "max_drawdown": float(max_dd_pct),
            "sharpe_ratio": float(sharpe),
        }
