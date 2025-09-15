# engine.py - REFACTORED VERSION
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from strategies.abstract import SignalType, Signal
from utils.engine_guards import (GuardConfig, GuardState, apply_trading_hours,
                                 should_skip_low_vol, in_cooldown, allow_entry_at_bar,
                                 register_entry, register_exit, )

logger = logging.getLogger("engine")


@dataclass
class EngineConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 1.0  # in %
    max_positions: int = 3
    commission: float = 0.0002  # proportional fees (entry+exit)
    slippage: float = 0.0  # slippage in proportion
    time_exit_bars: int = 200
    allow_shorts: bool = True
    min_risk_pts: float = 0.5  # minimum SL distance in points


@dataclass
class Position:
    strategy: str
    side: str  # "long" | "short"
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    metadata: Dict = field(default_factory=dict)


class MultiStrategyEngine:
    """Unified backtesting engine with single strategy interface."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.strategies: Dict[str, object] = {}  # name -> strategy instance
        self.allocations: Dict[str, float] = {}  # name -> %
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity: float = config.initial_capital

    def add_strategy(self, strategy, allocation: float):
        """Add strategy to engine."""
        name = strategy.name
        if name in self.strategies:
            raise ValueError(f"Strategy {name} already added")

        if not strategy.validate_params():
            raise ValueError(f"Invalid parameters for strategy {name}")

        self.strategies[name] = strategy
        self.allocations[name] = float(allocation)
        logger.info(f"Added strategy {name} with {allocation:.1f}% allocation")

    def run_backtest(self, df: pd.DataFrame,
            guard_cfg: Optional[GuardConfig] = None, ) -> Dict[str, object]:
        """Run backtest with NEXT_OPEN execution model."""
        if df.empty:
            logger.warning("Backtest aborted: no data provided")
            return self._empty_results()

        # Ensure ATR is available
        if "atr" not in df.columns:
            df = self._calculate_atr(df)

        # Apply trading hours filter
        guard_cfg = guard_cfg or GuardConfig()
        guard_state = GuardState()
        df = apply_trading_hours(df, guard_cfg)

        logger.info(f"Starting backtest with {len(df)} bars using NEXT_OPEN execution")

        # Reset state
        self.positions.clear()
        self.closed_positions.clear()
        self.equity = self.config.initial_capital

        # Reset strategies
        for strategy in self.strategies.values():
            strategy.reset()

        # Pre-calculate indicators for all strategies
        for strategy in self.strategies.values():
            df = strategy.calculate_indicators(df)

        # NEXT_OPEN execution: signals from bar i execute on bar i+1 open
        prev_signals: Dict[str, Optional[Signal]] = {name: None for name in
                                                     self.strategies}

        # Main backtest loop
        for bar_i, timestamp in enumerate(df.index):
            current_bar = df.loc[timestamp]

            # 1. Process exits first
            self._process_exits(bar_i, timestamp, current_bar, guard_state)

            # 2. Execute entries from previous bar signals (NEXT_OPEN)
            if bar_i > 0:  # Can't execute on first bar
                for strategy_name, signal in prev_signals.items():
                    if signal is None or signal.type == SignalType.NONE:
                        continue

                    if len(self.positions) >= self.config.max_positions:
                        break

                    # Apply guards
                    if not self._check_entry_guards(strategy_name, bar_i, timestamp,
                            current_bar, guard_cfg, guard_state, signal):
                        continue

                    # Execute entry
                    self._execute_entry(strategy_name, bar_i, timestamp, current_bar,
                                        signal, guard_state)

            # 3. Generate signals for next bar
            next_signals = {}
            for strategy_name, strategy in self.strategies.items():
                try:
                    if bar_i >= strategy.required_bars:
                        signal = strategy.get_signal(df, bar_i)
                        next_signals[strategy_name] = signal
                    else:
                        next_signals[strategy_name] = None
                except Exception as e:
                    logger.error(
                        f"Strategy {strategy_name} signal generation failed: {e}")
                    next_signals[strategy_name] = None

            prev_signals = next_signals

        # Calculate final metrics
        metrics = self._compute_metrics(df)

        logger.info(f"Backtest complete: {metrics['total_trades']} trades, "
                    f"{metrics['total_return']:.2f}% return, "
                    f"{metrics['win_rate']:.1f}% win rate")

        return {"metrics": metrics, "positions": self._positions_to_frame(), }

    def _check_entry_guards(self, strategy_name: str, bar_i: int,
            timestamp: pd.Timestamp, current_bar: pd.Series, guard_cfg: GuardConfig,
            guard_state: GuardState, signal: Signal) -> bool:
        """Check all entry guards."""
        # ATR guard
        atr_val = current_bar.get("atr", np.nan)
        if should_skip_low_vol(atr_val, guard_cfg):
            return False

        # Cooldown guard
        if in_cooldown(strategy_name, bar_i, guard_cfg, guard_state):
            return False

        # Trading hours and limits
        if not allow_entry_at_bar(timestamp, guard_cfg, guard_state):
            return False

        # Shorts allowed check
        if signal.type == SignalType.SELL and not self.config.allow_shorts:
            return False

        return True

    def _execute_entry(self, strategy_name: str, bar_i: int, timestamp: pd.Timestamp,
            current_bar: pd.Series, signal: Signal, guard_state: GuardState):
        """Execute entry order using NEXT_OPEN."""
        try:
            # Entry price is current bar's open (NEXT_OPEN execution)
            base_price = float(current_bar["open"])

            # Apply slippage
            if signal.type == SignalType.BUY:
                entry_price = base_price * (1 + self.config.slippage)
                side = "long"
            else:  # SELL
                entry_price = base_price * (1 - self.config.slippage)
                side = "short"

            # Use signal's stop/target or calculate defaults
            stop_loss = signal.stop
            take_profit = signal.target

            if stop_loss is None or take_profit is None:
                atr_val = current_bar.get("atr", np.nan)
                if pd.isna(atr_val):
                    logger.warning(
                        f"No stop/target in signal and no ATR available for {strategy_name}")
                    return

                # Default stop/target calculation
                if stop_loss is None:
                    if side == "long":
                        stop_loss = entry_price - (2.0 * atr_val)
                    else:
                        stop_loss = entry_price + (2.0 * atr_val)

                if take_profit is None:
                    if side == "long":
                        take_profit = entry_price + (1.5 * atr_val)
                    else:
                        take_profit = entry_price - (1.5 * atr_val)

            # Risk management: position sizing
            risk_pts = abs(entry_price - stop_loss)
            if risk_pts < self.config.min_risk_pts:
                logger.debug(
                    f"Risk too small for {strategy_name}: {risk_pts:.2f} < {self.config.min_risk_pts}")
                return

            allocation = self.allocations.get(strategy_name, 100.0)
            risk_capital = self.equity * (self.config.risk_per_trade / 100.0) * (
                        allocation / 100.0)
            position_size = risk_capital / risk_pts

            if position_size <= 0:
                logger.warning(
                    f"Invalid position size for {strategy_name}: {position_size}")
                return

            # Create position
            position = Position(strategy=strategy_name, side=side, entry_bar=bar_i,
                entry_time=timestamp, entry_price=entry_price, size=position_size,
                stop_loss=stop_loss, take_profit=take_profit,
                metadata=signal.metadata or {})

            self.positions.append(position)
            register_entry(strategy_name, timestamp, bar_i, guard_state)

            logger.debug(
                f"Entry: {strategy_name} {side} {position_size:.2f} @ {entry_price:.2f} "
                f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f})")

        except Exception as e:
            logger.error(f"Failed to execute entry for {strategy_name}: {e}")

    def _process_exits(self, bar_i: int, timestamp: pd.Timestamp,
                       current_bar: pd.Series, guard_state: GuardState):
        """Process all position exits."""
        for position in self.positions[:]:  # Copy list to allow modification
            # Time exit
            if bar_i - position.entry_bar >= self.config.time_exit_bars:
                self._exit_position(position, timestamp, current_bar["open"],
                                    "Time Exit", guard_state)
                continue

            # Price-based exits (stop loss / take profit)
            if position.side == "long":
                if current_bar["low"] <= position.stop_loss:
                    self._exit_position(position, timestamp, position.stop_loss,
                                        "Stop Loss", guard_state)
                elif position.take_profit and current_bar[
                    "high"] >= position.take_profit:
                    self._exit_position(position, timestamp, position.take_profit,
                                        "Take Profit", guard_state)
            else:  # short
                if current_bar["high"] >= position.stop_loss:
                    self._exit_position(position, timestamp, position.stop_loss,
                                        "Stop Loss", guard_state)
                elif position.take_profit and current_bar[
                    "low"] <= position.take_profit:
                    self._exit_position(position, timestamp, position.take_profit,
                                        "Take Profit", guard_state)

    def _exit_position(self, position: Position, exit_time: pd.Timestamp,
            base_exit_price: float, reason: str, guard_state: GuardState):
        """Exit a position with slippage."""
        # Apply slippage on exit (always disadvantageous)
        if position.side == "long":
            exit_price = base_exit_price * (1 - self.config.slippage)  # Sell lower
        else:
            exit_price = base_exit_price * (1 + self.config.slippage)  # Buy higher

        # Calculate PnL
        if position.side == "long":
            gross_pnl = (exit_price - position.entry_price) * position.size
        else:
            gross_pnl = (position.entry_price - exit_price) * position.size

        # Apply commission (on notional value)
        entry_commission = position.entry_price * position.size * self.config.commission
        exit_commission = exit_price * position.size * self.config.commission
        net_pnl = gross_pnl - entry_commission - exit_commission

        # Update position
        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_reason = reason
        position.pnl = net_pnl

        # Update state
        self.closed_positions.append(position)
        self.positions.remove(position)
        self.equity += net_pnl
        register_exit(position.strategy, position.entry_bar, guard_state)

        logger.debug(
            f"Exit: {position.strategy} {position.side} @ {exit_price:.2f} - {reason} - PnL: {net_pnl:.2f}")

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        df = df.copy()

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=period, min_periods=period).mean()

        return df

    def _positions_to_frame(self) -> pd.DataFrame:
        """Convert positions to DataFrame."""
        if not self.closed_positions:
            return pd.DataFrame()

        records = []
        for pos in self.closed_positions:
            records.append({"strategy": pos.strategy, "side": pos.side,
                "entry_time": pos.entry_time, "entry_price": pos.entry_price,
                "exit_time": pos.exit_time, "exit_price": pos.exit_price,
                "size": pos.size, "pnl": pos.pnl, "exit_reason": pos.exit_reason,
                "bars_held": pos.exit_time - pos.entry_time if pos.exit_time else None})

        return pd.DataFrame(records)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive backtest metrics."""
        if not self.closed_positions:
            return self._empty_results()["metrics"]

        trades = len(self.closed_positions)
        pnl_series = np.array([p.pnl for p in self.closed_positions])

        # Basic metrics
        total_return = (self.equity / self.config.initial_capital - 1.0) * 100.0
        wins = (pnl_series > 0).sum()
        win_rate = (wins / trades) * 100.0 if trades > 0 else 0.0

        # Profit factor
        gross_profit = pnl_series[pnl_series > 0].sum() if trades > 0 else 0.0
        gross_loss = -pnl_series[pnl_series < 0].sum() if trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Drawdown calculation
        equity_curve = np.cumsum(
            np.concatenate([[self.config.initial_capital], pnl_series]))
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdown_pct = ((rolling_max - equity_curve) / rolling_max * 100.0)
        max_drawdown = drawdown_pct.max() if len(drawdown_pct) > 0 else 0.0

        # Sharpe ratio (simplified)
        if trades > 1 and np.std(pnl_series) > 0:
            sharpe_ratio = np.mean(pnl_series) / np.std(pnl_series)
        else:
            sharpe_ratio = 0.0

        # Count by side
        longs = sum(1 for p in self.closed_positions if p.side == "long")
        shorts = trades - longs

        return {"total_trades": trades, "total_longs": longs, "total_shorts": shorts,
            "initial_capital": self.config.initial_capital,
            "final_capital": self.equity, "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor if np.isfinite(profit_factor) else float(
                "inf"), "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown,
            "gross_profit": gross_profit, "gross_loss": gross_loss,
            "avg_win": gross_profit / wins if wins > 0 else 0.0,
            "avg_loss": gross_loss / (trades - wins) if (trades - wins) > 0 else 0.0, }

    @staticmethod
    def _empty_results() -> Dict[str, object]:
        """Return empty results structure."""
        return {"metrics": {"total_trades": 0, "total_longs": 0, "total_shorts": 0,
            "initial_capital": 0.0, "final_capital": 0.0, "total_return": 0.0,
            "win_rate": 0.0, "profit_factor": float("inf"), "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, }, "positions": pd.DataFrame(), }