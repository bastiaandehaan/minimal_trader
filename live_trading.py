# live_trading.py - Complete Live Trading System
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from threading import Thread, Event
import pandas as pd

from engine import EngineConfig
from strategies.abstract import AbstractStrategy, Signal, SignalType
from feeds.mt5_feed import MT5Feed
from utils.engine_guards import GuardConfig, GuardState

logger = logging.getLogger("live_trading")


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class LiveOrder:
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    order_type: str  # "market", "limit", "stop"
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_size: float = 0.0
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LivePosition:
    id: str
    symbol: str
    strategy: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiveTradingEngine:
    """Real-time trading engine with MT5 integration."""

    def __init__(self, engine_config: EngineConfig, guard_config: GuardConfig,
            symbol: str, timeframe: str = "M1"):
        self.config = engine_config
        self.guard_config = guard_config
        self.guard_state = GuardState()
        self.symbol = symbol
        self.timeframe = timeframe

        # Trading state
        self.strategies: Dict[str, AbstractStrategy] = {}
        self.allocations: Dict[str, float] = {}
        self.positions: Dict[str, LivePosition] = {}
        self.orders: Dict[str, LiveOrder] = {}
        self.equity = engine_config.initial_capital
        self.is_running = False
        self.stop_event = Event()

        # Data management
        self.data_feed: Optional[MT5Feed] = None
        self.market_data: pd.DataFrame = pd.DataFrame()
        self.last_update = datetime.utcnow()

        # Callbacks
        self.on_signal_callbacks: List[Callable] = []
        self.on_order_callbacks: List[Callable] = []
        self.on_position_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []

        # Risk management
        self.daily_pnl = 0.0
        self.max_daily_loss = engine_config.initial_capital * 0.05  # 5% daily stop
        self.max_positions_per_strategy = 1

        logger.info(f"LiveTradingEngine initialized for {symbol} on {timeframe}")

    def add_strategy(self, strategy: AbstractStrategy, allocation: float):
        """Add strategy to live trading."""
        if not strategy.validate_params():
            raise ValueError(f"Invalid parameters for strategy {strategy.name}")

        self.strategies[strategy.name] = strategy
        self.allocations[strategy.name] = allocation
        logger.info(f"Added strategy {strategy.name} with {allocation}% allocation")

    def add_callback(self, event_type: str, callback: Callable):
        """Add event callback."""
        callbacks_map = {'signal': self.on_signal_callbacks,
            'order': self.on_order_callbacks, 'position': self.on_position_callbacks,
            'error': self.on_error_callbacks}

        if event_type in callbacks_map:
            callbacks_map[event_type].append(callback)
            logger.info(f"Added {event_type} callback")

    async def start(self):
        """Start live trading."""
        if self.is_running:
            logger.warning("Live trading already running")
            return

        try:
            # Initialize data feed
            self.data_feed = MT5Feed(self.symbol, self.timeframe, bars=1000)
            if not self.data_feed.connect():
                raise ConnectionError("Failed to connect to MT5")

            # Load initial data
            await self._update_market_data()

            # Validate strategies with current data
            for strategy in self.strategies.values():
                if len(self.market_data) < strategy.required_bars:
                    raise ValueError(
                        f"Insufficient data for {strategy.name}: need {strategy.required_bars}, have {len(self.market_data)}")

            self.is_running = True
            logger.info("Live trading started")

            # Start main trading loop
            await self._run_trading_loop()

        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
            await self._notify_error(e)
            await self.stop()

    async def stop(self):
        """Stop live trading gracefully."""
        if not self.is_running:
            return

        logger.info("Stopping live trading...")
        self.is_running = False
        self.stop_event.set()

        # Close all open positions (optional - comment out for manual management)
        # await self._close_all_positions("System shutdown")

        # Disconnect data feed
        if self.data_feed:
            self.data_feed.disconnect()

        logger.info("Live trading stopped")

    async def _run_trading_loop(self):
        """Main trading loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Update market data
                await self._update_market_data()

                # Check daily risk limits
                if self._check_daily_limits():
                    logger.warning("Daily limits exceeded, stopping trading")
                    break

                # Process strategies
                await self._process_strategies()

                # Update positions
                await self._update_positions()

                # Clean up filled orders
                self._cleanup_orders()

                # Sleep until next update (based on timeframe)
                await asyncio.sleep(self._get_update_interval())

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await self._notify_error(e)
                await asyncio.sleep(5)  # Wait before retrying

    async def _update_market_data(self):
        """Update market data from MT5."""
        try:
            # Fetch latest bars
            new_data = self.data_feed.fetch()

            if new_data.empty:
                logger.warning("No new market data received")
                return

            # Merge with existing data
            if self.market_data.empty:
                self.market_data = new_data
            else:
                # Append new bars and keep last 1000 bars
                self.market_data = pd.concat(
                    [self.market_data, new_data]).drop_duplicates().tail(1000)

            # Calculate indicators for all strategies
            for strategy in self.strategies.values():
                self.market_data = strategy.calculate_indicators(self.market_data)

            self.last_update = datetime.utcnow()
            logger.debug(f"Market data updated: {len(self.market_data)} bars")

        except Exception as e:
            logger.error(f"Failed to update market data: {e}")
            raise

    async def _process_strategies(self):
        """Process all strategies for signals."""
        if self.market_data.empty:
            return

        current_bar_idx = len(self.market_data) - 1

        for strategy_name, strategy in self.strategies.items():
            try:
                # Check if strategy can generate signals
                if current_bar_idx < strategy.required_bars:
                    continue

                # Check position limits
                strategy_positions = [p for p in self.positions.values() if
                                      p.strategy == strategy_name and p.status == PositionStatus.OPEN]
                if len(strategy_positions) >= self.max_positions_per_strategy:
                    continue

                # Generate signal
                signal = strategy.get_signal(self.market_data, current_bar_idx)

                if signal.type != SignalType.NONE:
                    await self._handle_signal(strategy_name, signal)

            except Exception as e:
                logger.error(f"Error processing strategy {strategy_name}: {e}")
                await self._notify_error(e)

    async def _handle_signal(self, strategy_name: str, signal: Signal):
        """Handle trading signal."""
        try:
            logger.info(
                f"Signal from {strategy_name}: {signal.type.name} - {signal.reason}")

            # Notify callbacks
            for callback in self.on_signal_callbacks:
                try:
                    await callback(strategy_name, signal)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")

            # Calculate position size
            allocation = self.allocations.get(strategy_name, 0.0)
            risk_capital = self.equity * (self.config.risk_per_trade / 100.0) * (
                        allocation / 100.0)

            if not signal.stop or signal.entry is None:
                logger.warning(f"Signal missing entry or stop price: {signal}")
                return

            risk_per_unit = abs(signal.entry - signal.stop)
            position_size = risk_capital / max(risk_per_unit, 1e-6)

            if position_size <= 0:
                logger.warning(f"Invalid position size: {position_size}")
                return

            # Create and execute order
            await self._create_market_order(strategy_name=strategy_name,
                side="buy" if signal.type == SignalType.BUY else "sell",
                size=position_size, stop_loss=signal.stop, take_profit=signal.target,
                signal_metadata=signal.metadata)

        except Exception as e:
            logger.error(f"Error handling signal: {e}")
            await self._notify_error(e)

    async def _create_market_order(self, strategy_name: str, side: str, size: float,
            stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
            signal_metadata: Dict = None):
        """Create and execute market order."""
        try:
            import MetaTrader5 as mt5

            # Get symbol info for proper lot sizing
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                raise Exception(f"Symbol {self.symbol} not found")

            # Round volume to symbol's volume step
            volume_step = symbol_info.volume_step
            volume = round(size / volume_step) * volume_step
            volume = max(volume, symbol_info.volume_min)
            volume = min(volume, symbol_info.volume_max)

            if volume <= 0:
                logger.warning(
                    f"Invalid volume calculated: {volume} (original: {size})")
                return

            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

            request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
                "volume": volume, "type": order_type, "deviation": 20,
                "magic": hash(strategy_name) % 100000,  # Strategy identifier
                "comment": f"{strategy_name}_{datetime.utcnow().strftime('%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC, }

            logger.info(
                f"Sending order: {side} {volume} {self.symbol} for {strategy_name}")

            # Execute order
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                raise Exception(f"order_send failed: {error}")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Order failed: {result.retcode} - {result.comment}")

            logger.info(
                f"Order successful: {result.order} - {result.volume} @ {result.price}")

            # Create order record
            order_id = str(result.order)
            order = LiveOrder(id=order_id, symbol=self.symbol, side=side, size=volume,
                order_type="market", status=OrderStatus.FILLED,
                filled_at=datetime.utcnow(), filled_price=result.price,
                filled_size=result.volume, strategy=strategy_name,
                metadata=signal_metadata or {})
            self.orders[order_id] = order

            # Create position
            position_id = f"{strategy_name}_{order_id}"
            position = LivePosition(id=position_id, symbol=self.symbol,
                strategy=strategy_name, side="long" if side == "buy" else "short",
                size=result.volume, entry_price=result.price,
                entry_time=datetime.utcnow(), stop_loss=stop_loss,
                take_profit=take_profit, current_price=result.price,
                metadata=signal_metadata or {})
            self.positions[position_id] = position

            logger.info(
                f"Position opened: {position_id} - {side} {result.volume} @ {result.price}")

            # Update equity (rough estimate)
            self.equity -= (result.volume * result.price * self.config.commission)

            # Notify callbacks
            for callback in self.on_position_callbacks:
                try:
                    await callback("opened", position)
                except Exception as e:
                    logger.error(f"Position callback error: {e}")

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            await self._notify_error(e)

    async def _set_stop_loss(self, position_id: str, stop_price: float):
        """Set stop loss for position."""
        # Implementation depends on your broker's API
        # This is a placeholder for MT5 stop loss orders
        pass

    async def _set_take_profit(self, position_id: str, target_price: float):
        """Set take profit for position."""
        # Implementation depends on your broker's API
        # This is a placeholder for MT5 take profit orders
        pass

    async def _update_positions(self):
        """Update all open positions."""
        if self.market_data.empty:
            return

        current_price = float(self.market_data['close'].iloc[-1])

        for position_id, position in list(self.positions.items()):
            if position.status != PositionStatus.OPEN:
                continue

            # Update current price and unrealized PnL
            position.current_price = current_price

            if position.side == "long":
                position.unrealized_pnl = (
                                                      current_price - position.entry_price) * position.size
            else:  # short
                position.unrealized_pnl = (
                                                      position.entry_price - current_price) * position.size

            # Check stop loss and take profit
            should_close = False
            exit_reason = ""

            if position.side == "long":
                if position.stop_loss and current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif position.take_profit and current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            else:  # short
                if position.stop_loss and current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif position.take_profit and current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"

            if should_close:
                await self._close_position(position_id, exit_reason)

    async def _close_position(self, position_id: str, reason: str):
        """Close a position."""
        position = self.positions.get(position_id)
        if not position or position.status != PositionStatus.OPEN:
            return

        try:
            import MetaTrader5 as mt5

            # Prepare close order
            order_type = mt5.ORDER_TYPE_SELL if position.side == "long" else mt5.ORDER_TYPE_BUY

            request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
                "volume": position.size, "type": order_type, "deviation": 20,
                "magic": hash(position.strategy) % 100000,
                "comment": f"Close_{reason}_{datetime.utcnow().strftime('%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC, }

            # Execute close order
            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close position {position_id}: {result}")
                return

            # Update position
            position.status = PositionStatus.CLOSED
            position.exit_price = result.price
            position.exit_time = datetime.utcnow()
            position.realized_pnl = position.unrealized_pnl

            # Update equity
            self.equity += position.realized_pnl
            self.daily_pnl += position.realized_pnl

            logger.info(
                f"Position closed: {position_id} - {reason} - PnL: {position.realized_pnl:.2f}")

            # Notify callbacks
            for callback in self.on_position_callbacks:
                try:
                    await callback("closed", position)
                except Exception as e:
                    logger.error(f"Position callback error: {e}")

        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            await self._notify_error(e)

    async def _close_all_positions(self, reason: str):
        """Close all open positions."""
        open_positions = [p for p in self.positions.values() if
                          p.status == PositionStatus.OPEN]

        logger.info(f"Closing {len(open_positions)} open positions: {reason}")

        for position in open_positions:
            await self._close_position(position.id, reason)

    def _cleanup_orders(self):
        """Clean up old filled orders."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        orders_to_remove = [order_id for order_id, order in self.orders.items() if
            order.status == OrderStatus.FILLED and order.filled_at and order.filled_at < cutoff_time]

        for order_id in orders_to_remove:
            del self.orders[order_id]

    def _check_daily_limits(self) -> bool:
        """Check if daily risk limits are exceeded."""
        if self.daily_pnl <= -self.max_daily_loss:
            logger.error(
                f"Daily loss limit exceeded: {self.daily_pnl:.2f} <= {-self.max_daily_loss:.2f}")
            return True

        open_positions_count = len(
            [p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        if open_positions_count >= self.config.max_positions:
            logger.warning(f"Maximum positions reached: {open_positions_count}")
            return True

        return False

    def _get_update_interval(self) -> float:
        """Get update interval based on timeframe."""
        intervals = {'M1': 60,  # 1 minute
            'M5': 300,  # 5 minutes
            'M15': 900,  # 15 minutes
            'M30': 1800,  # 30 minutes
            'H1': 3600,  # 1 hour
            'H4': 14400,  # 4 hours
            'D1': 86400  # 1 day
        }
        return intervals.get(self.timeframe, 60)

    async def _notify_error(self, error: Exception):
        """Notify error callbacks."""
        for callback in self.on_error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        open_positions = [p for p in self.positions.values() if
                          p.status == PositionStatus.OPEN]
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)

        return {'is_running': self.is_running, 'equity': self.equity,
            'daily_pnl': self.daily_pnl, 'open_positions': len(open_positions),
            'total_unrealized_pnl': total_unrealized, 'last_update': self.last_update,
            'market_data_bars': len(self.market_data),
            'strategies': list(self.strategies.keys()), 'symbol': self.symbol,
            'timeframe': self.timeframe}

    def get_positions(self) -> List[LivePosition]:
        """Get all positions."""
        return list(self.positions.values())

    def get_open_positions(self) -> List[LivePosition]:
        """Get only open positions."""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def reset_daily_pnl(self):
        """Reset daily PnL (call at start of new trading day)."""
        self.daily_pnl = 0.0
        logger.info("Daily PnL reset")


class LiveTradingManager:
    """High-level manager for live trading operations."""

    def __init__(self, config):
        """Initialize with config dict or config file path."""
        if isinstance(config, str):
            # Config file path provided
            self.config_path = config
            self.config = self._load_config_from_file()
        elif isinstance(config, dict):
            # Config dict provided directly
            self.config_path = None
            self.config = config
        else:
            raise ValueError("Config must be dict or file path string")

        self.engine: Optional[LiveTradingEngine] = None

    def _load_config_from_file(self) -> Dict:
        """Load configuration from file."""
        import yaml
        from pathlib import Path

        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with config_file.open('r') as f:
            return yaml.safe_load(f)

    def create_engine(self) -> LiveTradingEngine:
        """Create live trading engine from config."""
        from engine import EngineConfig
        from utils.engine_guards import GuardConfig

        # Engine config
        engine_cfg = self.config.get('engine', {})
        engine_config = EngineConfig(
            initial_capital=float(engine_cfg.get('initial_capital', 10000.0)),
            risk_per_trade=float(engine_cfg.get('risk_per_trade', 1.0)),
            max_positions=int(engine_cfg.get('max_positions', 3)),
            commission=float(engine_cfg.get('commission', 0.0002)),
            slippage=float(engine_cfg.get('slippage', 0.0)),
            time_exit_bars=int(engine_cfg.get('time_exit_bars', 200)),
            allow_shorts=bool(engine_cfg.get('allow_shorts', True)),
            min_risk_pts=float(engine_cfg.get('min_risk_pts', 0.5)))

        # Guard config
        guard_cfg = self.config.get('guards', {})
        guard_config = GuardConfig(
            trading_hours_tz=guard_cfg.get('trading_hours_tz', 'Europe/Brussels'),
            trading_hours_start=guard_cfg.get('trading_hours_start', '08:00'),
            trading_hours_end=guard_cfg.get('trading_hours_end', '22:00'),
            min_atr_pts=float(guard_cfg.get('min_atr_pts', 20.0)),
            cooldown_bars=int(guard_cfg.get('cooldown_bars', 3)),
            max_trades_per_day=int(guard_cfg.get('max_trades_per_day', 10)),
            one_trade_per_timestamp=bool(
                guard_cfg.get('one_trade_per_timestamp', True)))

        # Trading config
        data_cfg = self.config.get('data', {})
        symbol = data_cfg.get('symbol', 'GER40.cash')
        timeframe = data_cfg.get('timeframe', 'M1')

        # Create engine
        self.engine = LiveTradingEngine(engine_config=engine_config,
            guard_config=guard_config, symbol=symbol, timeframe=timeframe)

        # Add strategies
        self._add_strategies_from_config()

        return self.engine

    def _add_strategies_from_config(self):
        """Add strategies to engine from config."""
        if not self.engine:
            raise RuntimeError("Engine not created")

        strategies_cfg = self.config.get('strategies', {})

        # RSI Reversion Strategy
        rsi_cfg = strategies_cfg.get('rsi_reversion', {})
        if rsi_cfg.get('enabled', False):
            from strategies.rsi_reversion import RSIReversionStrategy

            params = rsi_cfg.get('params', {})
            allocation = float(rsi_cfg.get('allocation', 100.0))

            strategy = RSIReversionStrategy(params)
            self.engine.add_strategy(strategy, allocation)

        # Add other strategies here as needed

    async def start_live_trading(self):
        """Start live trading with monitoring."""
        if not self.engine:
            self.create_engine()

        # Add monitoring callbacks
        self.engine.add_callback('signal', self._on_signal)
        self.engine.add_callback('position', self._on_position)
        self.engine.add_callback('error', self._on_error)

        try:
            await self.engine.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Live trading error: {e}")
        finally:
            await self.engine.stop()

    async def _on_signal(self, strategy_name: str, signal: Signal):
        """Handle signal events."""
        logger.info(f"Signal: {strategy_name} -> {signal.type.name} @ {signal.entry}")

    async def _on_position(self, event_type: str, position: LivePosition):
        """Handle position events."""
        if event_type == "opened":
            logger.info(
                f"Position opened: {position.id} - {position.side} {position.size} @ {position.entry_price}")
        elif event_type == "closed":
            logger.info(
                f"Position closed: {position.id} - PnL: {position.realized_pnl:.2f}")

    async def _on_error(self, error: Exception):
        """Handle error events."""
        logger.error(
            f"Trading error: {error}")  # Could send notifications, alerts, etc.


# Utility functions for running live trading

def run_live_trading(config_path: str):
    """Run live trading from command line."""
    manager = LiveTradingManager(config_path)

    try:
        asyncio.run(manager.start_live_trading())
    except KeyboardInterrupt:
        print("\nLive trading stopped by user")
    except Exception as e:
        print(f"Live trading failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python live_trading.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_live_trading(config_path)