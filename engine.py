"""Multi-strategy execution engine."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from strategies.abstract import AbstractStrategy, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track individual position."""
    strategy: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_bar: int = 0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    commission: float = 0.0


@dataclass
class EngineConfig:
    """Engine configuration."""
    initial_capital: float = 10000.0
    risk_per_trade: float = 1.0  # % of capital
    max_positions: int = 3
    commission: float = 0.0002  # 2 bps
    slippage: float = 0.0001  # 1 bp
    time_exit_bars: int = 200


class MultiStrategyEngine:
    """Execute multiple strategies with position management."""

    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self.strategies: Dict[str, AbstractStrategy] = {}
        self.allocations: Dict[str, float] = {}
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity = self.config.initial_capital

    def add_strategy(self, strategy: AbstractStrategy, allocation: float = 100.0):
        """Add strategy with capital allocation percentage."""
        if strategy.name in self.strategies:
            logger.warning(f"Strategy {strategy.name} already exists")
            return

        self.strategies[strategy.name] = strategy
        self.allocations[strategy.name] = allocation
        logger.info(f"Added strategy {strategy.name} with {allocation}% allocation")

    def calculate_position_size(self, capital: float, entry: float, stop: float) -> float:
        """Calculate position size based on risk."""
        risk_amount = capital * (self.config.risk_per_trade / 100)
        risk_points = abs(entry - stop)

        if risk_points <= 0:
            return 0.0

        size = risk_amount / risk_points
        return round(size, 2)

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run all strategies on historical data."""
        if not self.strategies:
            raise ValueError("No strategies added")

        # Prepare data for each strategy
        strategy_data = {}
        for name, strategy in self.strategies.items():
            df = strategy.calculate_indicators(data.copy())
            strategy_data[name] = df
            logger.info(f"Prepared data for {name}: {len(df)} bars")

        # Main backtest loop
        max_bars = len(data)

        for i in range(max_bars):
            current_bar = data.iloc[i]

            # Check exits first
            self._check_exits(i, current_bar)

            # Check for new signals from each strategy
            for strat_name, strategy in self.strategies.items():
                if i < strategy.required_bars:
                    continue

                # Check if we can open more positions
                active_positions = len([p for p in self.positions if p.strategy == strat_name])
                if active_positions >= self.config.max_positions:
                    continue

                # Get signal
                df = strategy_data[strat_name]
                signal, meta = strategy.get_signal(df, i)

                if signal.type == SignalType.BUY:
                    # Calculate capital for this strategy
                    strategy_capital = self.equity * (self.allocations[strat_name] / 100)

                    # Calculate position size
                    size = self.calculate_position_size(
                        strategy_capital,
                        signal.entry,
                        signal.stop
                    )

                    if size > 0:
                        position = Position(
                            strategy=strat_name,
                            symbol=data.index.name or "SYMBOL",
                            side='long',
                            entry_time=current_bar.name,
                            entry_price=signal.entry,
                            size=size,
                            stop_loss=signal.stop,
                            take_profit=signal.target,
                            entry_bar=i
                        )

                        # Apply commission
                        position.commission = position.entry_price * position.size * self.config.commission

                        self.positions.append(position)
                        logger.debug(f"{strat_name} BUY at {signal.entry:.2f}, size={size}")

        # Close any remaining positions
        for position in self.positions:
            self._close_position(position, data.iloc[-1]['close'], "End of data")

        # Generate results
        return self._generate_results()

    def _check_exits(self, bar_index: int, current_bar):
        """Check and execute exits for all positions."""
        positions_to_close = []

        for position in self.positions:
            exit_price = None
            exit_reason = None

            # Check stop loss
            if current_bar['low'] <= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = "Stop Loss"

            # Check take profit
            elif current_bar['high'] >= position.take_profit:
                exit_price = position.take_profit
                exit_reason = "Take Profit"

            # Check time exit
            elif (bar_index - position.entry_bar) >= self.config.time_exit_bars:
                exit_price = current_bar['close']
                exit_reason = "Time Exit"

            if exit_price:
                position.exit_time = current_bar.name
                position.exit_price = exit_price
                position.exit_reason = exit_reason
                positions_to_close.append(position)

        # Close positions
        for position in positions_to_close:
            self._close_position(position, position.exit_price, position.exit_reason)

    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close position and calculate PnL."""
        position.exit_price = exit_price
        position.exit_reason = reason

        # Calculate PnL
        gross_pnl = (exit_price - position.entry_price) * position.size
        exit_commission = exit_price * position.size * self.config.commission
        position.pnl = gross_pnl - position.commission - exit_commission

        # Update equity
        self.equity += position.pnl

        # Move to closed positions
        self.positions.remove(position)
        self.closed_positions.append(position)

        logger.debug(f"Closed {position.strategy} position: {reason} at {exit_price:.2f}, PnL={position.pnl:.2f}")

    def _generate_results(self) -> Dict:
        """Generate backtest results and metrics."""
        if not self.closed_positions:
            return {
                'metrics': {
                    'total_trades': 0,
                    'initial_capital': self.config.initial_capital,
                    'final_capital': self.equity,
                    'total_return': 0.0,
                    'message': 'No trades executed'
                },
                'trades': pd.DataFrame()
            }

        # Create trades DataFrame
        trades_data = []
        for pos in self.closed_positions:
            trades_data.append({
                'strategy': pos.strategy,
                'symbol': pos.symbol,
                'side': pos.side,
                'entry_time': pos.entry_time,
                'exit_time': pos.exit_time,
                'entry_price': pos.entry_price,
                'exit_price': pos.exit_price,
                'size': pos.size,
                'pnl': pos.pnl,
                'exit_reason': pos.exit_reason
            })

        trades_df = pd.DataFrame(trades_data)

        # Calculate metrics per strategy
        strategy_metrics = {}
        for strat_name in self.strategies.keys():
            strat_trades = trades_df[trades_df['strategy'] == strat_name]
            if len(strat_trades) > 0:
                wins = strat_trades[strat_trades['pnl'] > 0]
                strategy_metrics[strat_name] = {
                    'trades': len(strat_trades),
                    'wins': len(wins),
                    'win_rate': len(wins) / len(strat_trades) * 100,
                    'total_pnl': strat_trades['pnl'].sum(),
                    'avg_pnl': strat_trades['pnl'].mean()
                }

        # Overall metrics
        total_return = ((self.equity / self.config.initial_capital) - 1) * 100
        wins = trades_df[trades_df['pnl'] > 0]

        metrics = {
            'total_trades': len(trades_df),
            'initial_capital': self.config.initial_capital,
            'final_capital': self.equity,
            'total_return': total_return,
            'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'strategy_metrics': strategy_metrics,
            'sharpe_ratio': self._calculate_sharpe(trades_df),
            'max_drawdown': self._calculate_max_drawdown(trades_df)
        }

        return {
            'metrics': metrics,
            'trades': trades_df
        }

    def _calculate_sharpe(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio."""
        if len(trades_df) < 2:
            return 0.0

        returns = trades_df['pnl'] / self.config.initial_capital * 100
        if returns.std() == 0:
            return 0.0

        # Annualize (rough estimate)
        days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        if days > 0:
            trades_per_year = (252 / days) * len(trades_df)
            return (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
        return 0.0

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage."""
        if trades_df.empty:
            return 0.0

        equity_curve = self.config.initial_capital + trades_df['pnl'].cumsum()
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        return abs(dd.min()) * 100 if not dd.empty else 0.0