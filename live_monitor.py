# live_monitor.py - Live Trading Monitoring Tool
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from live_trading import LiveTradingManager, LivePosition, Signal
from strategies.abstract import SignalType

logger = logging.getLogger("live_monitor")


class LiveTradingMonitor:
    """Enhanced monitoring for live trading with detailed logging."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.manager = LiveTradingManager(config_path)
        self.signals_log: List[Dict] = []
        self.positions_log: List[Dict] = []
        self.errors_log: List[Dict] = []

        # Setup enhanced logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup detailed logging for monitoring."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # File handler for live trading logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"live_trading_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        file_handler.setFormatter(formatter)

        # Add to all relevant loggers
        for logger_name in ['live_trading', 'feeds.mt5_feed', 'strategies']:
            log = logging.getLogger(logger_name)
            log.addHandler(file_handler)
            log.setLevel(logging.DEBUG)

        logger.info(f"Detailed logging enabled: {log_file}")

    async def start_monitoring(self):
        """Start live trading with enhanced monitoring."""
        print("🚀 Starting Live Trading Monitor")
        print("=" * 60)

        # Create engine
        engine = self.manager.create_engine()

        # Add monitoring callbacks
        engine.add_callback('signal', self._on_signal)
        engine.add_callback('position', self._on_position)
        engine.add_callback('error', self._on_error)

        # Start status monitoring task
        status_task = asyncio.create_task(self._monitor_status(engine))

        try:
            # Start live trading
            await self.manager.start_live_trading()
        except KeyboardInterrupt:
            print("\n🛑 Stopping live trading...")
        except Exception as e:
            print(f"❌ Live trading error: {e}")
            logger.error(f"Live trading failed: {e}")
        finally:
            status_task.cancel()
            await self._save_session_logs()

    async def _monitor_status(self, engine):
        """Monitor and display live trading status."""
        try:
            while True:
                status = engine.get_status()
                self._print_status(status)
                await asyncio.sleep(30)  # Update every 30 seconds
        except asyncio.CancelledError:
            pass

    def _print_status(self, status: Dict):
        """Print current trading status."""
        print(f"\n📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 40)
        print(f"Running: {'✅' if status['is_running'] else '❌'}")
        print(f"Equity: ${status['equity']:,.2f}")
        print(f"Daily PnL: ${status['daily_pnl']:+,.2f}")
        print(f"Open Positions: {status['open_positions']}")
        print(f"Unrealized PnL: ${status['total_unrealized_pnl']:+,.2f}")
        print(f"Data Bars: {status['market_data_bars']}")
        print(f"Last Update: {status['last_update'].strftime('%H:%M:%S')}")

    async def _on_signal(self, strategy_name: str, signal: Signal):
        """Handle signal events with detailed logging."""
        signal_data = {'timestamp': datetime.utcnow().isoformat(),
            'strategy': strategy_name, 'type': signal.type.name, 'entry': signal.entry,
            'stop': signal.stop, 'target': signal.target, 'reason': signal.reason,
            'confidence': signal.confidence, 'metadata': signal.metadata}

        self.signals_log.append(signal_data)

        # Console output
        print(f"\n🎯 SIGNAL: {strategy_name}")
        print(f"   Type: {signal.type.name}")
        print(f"   Entry: {signal.entry:.2f}")
        print(f"   Stop: {signal.stop:.2f}")
        print(f"   Target: {signal.target:.2f}")
        print(f"   Reason: {signal.reason}")
        print(f"   Confidence: {signal.confidence:.2f}")

        logger.info(
            f"Signal: {strategy_name} -> {signal.type.name} @ {signal.entry} ({signal.reason})")

    async def _on_position(self, event_type: str, position: LivePosition):
        """Handle position events with detailed logging."""
        position_data = {'timestamp': datetime.utcnow().isoformat(),
            'event': event_type, 'position_id': position.id,
            'strategy': position.strategy, 'side': position.side, 'size': position.size,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl if event_type == 'closed' else None,
            'metadata': position.metadata}

        self.positions_log.append(position_data)

        # Console output
        if event_type == "opened":
            print(f"\n📈 POSITION OPENED")
            print(f"   ID: {position.id}")
            print(f"   Strategy: {position.strategy}")
            print(f"   Side: {position.side.upper()}")
            print(f"   Size: {position.size:.2f}")
            print(f"   Entry: {position.entry_price:.2f}")
            print(f"   Stop: {position.stop_loss:.2f}")
            print(f"   Target: {position.take_profit:.2f}")

        elif event_type == "closed":
            print(f"\n📉 POSITION CLOSED")
            print(f"   ID: {position.id}")
            print(f"   Strategy: {position.strategy}")
            print(f"   Side: {position.side.upper()}")
            print(f"   Entry: {position.entry_price:.2f}")
            print(f"   Exit: {position.exit_price:.2f}")
            print(f"   PnL: ${position.realized_pnl:+,.2f}")

        logger.info(f"Position {event_type}: {position.id} - {position.strategy}")

    async def _on_error(self, error: Exception):
        """Handle error events with detailed logging."""
        error_data = {'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__, 'error_message': str(error), }

        self.errors_log.append(error_data)

        # Console output
        print(f"\n❌ ERROR: {type(error).__name__}")
        print(f"   Message: {str(error)}")

        logger.error(f"Trading error: {error}", exc_info=True)

    async def _save_session_logs(self):
        """Save session logs to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)

            # Save signals log
            if self.signals_log:
                signals_file = logs_dir / f"signals_{timestamp}.json"
                with signals_file.open('w') as f:
                    json.dump(self.signals_log, f, indent=2)
                print(f"📄 Signals log saved: {signals_file}")

            # Save positions log
            if self.positions_log:
                positions_file = logs_dir / f"positions_{timestamp}.json"
                with positions_file.open('w') as f:
                    json.dump(self.positions_log, f, indent=2)
                print(f"📄 Positions log saved: {positions_file}")

                # Also save as CSV for analysis
                positions_df = pd.DataFrame(self.positions_log)
                csv_file = logs_dir / f"positions_{timestamp}.csv"
                positions_df.to_csv(csv_file, index=False)
                print(f"📊 Positions CSV saved: {csv_file}")

            # Save errors log
            if self.errors_log:
                errors_file = logs_dir / f"errors_{timestamp}.json"
                with errors_file.open('w') as f:
                    json.dump(self.errors_log, f, indent=2)
                print(f"📄 Errors log saved: {errors_file}")

        except Exception as e:
            logger.error(f"Failed to save session logs: {e}")

    def print_session_summary(self):
        """Print summary of trading session."""
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)

        print(f"Signals Generated: {len(self.signals_log)}")

        if self.signals_log:
            signal_types = {}
            for signal in self.signals_log:
                signal_types[signal['type']] = signal_types.get(signal['type'], 0) + 1

            for signal_type, count in signal_types.items():
                print(f"  - {signal_type}: {count}")

        print(f"Position Events: {len(self.positions_log)}")

        opened_positions = [p for p in self.positions_log if p['event'] == 'opened']
        closed_positions = [p for p in self.positions_log if p['event'] == 'closed']

        print(f"  - Opened: {len(opened_positions)}")
        print(f"  - Closed: {len(closed_positions)}")

        if closed_positions:
            total_pnl = sum(
                p['realized_pnl'] for p in closed_positions if p['realized_pnl'])
            print(f"  - Total Realized PnL: ${total_pnl:+,.2f}")

        print(f"Errors: {len(self.errors_log)}")

        if self.errors_log:
            error_types = {}
            for error in self.errors_log:
                error_types[error['error_type']] = error_types.get(error['error_type'],
                                                                   0) + 1

            for error_type, count in error_types.items():
                print(f"  - {error_type}: {count}")


def main():
    """Main entry point for live trading monitor."""
    if len(sys.argv) != 2:
        print("Usage: python live_monitor.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    monitor = LiveTradingMonitor(config_path)

    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        monitor.print_session_summary()


if __name__ == "__main__":
    main()