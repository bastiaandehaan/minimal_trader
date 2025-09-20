# main.py - FINAL VERSION WITH DAX TREND CONTINUATION STRATEGY
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from engine import MultiStrategyEngine, EngineConfig
from feeds.csv_feed import CSVFeed
from utils.engine_guards import GuardConfig

# Setup logging
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_file.open('r') as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty or invalid config file: {config_path}")

    return config


def create_engine_config(config: Dict) -> EngineConfig:
    """Create EngineConfig from configuration."""
    engine_cfg = config.get('engine', {})

    return EngineConfig(
        initial_capital=float(engine_cfg.get('initial_capital', 10_000.0)),
        risk_per_trade=float(engine_cfg.get('risk_per_trade', 1.0)),
        max_positions=int(engine_cfg.get('max_positions', 3)),
        commission=float(engine_cfg.get('commission', 0.0002)),
        slippage=float(engine_cfg.get('slippage', 0.0)),
        time_exit_bars=int(engine_cfg.get('time_exit_bars', 200)),
        allow_shorts=bool(engine_cfg.get('allow_shorts', True)),
        min_risk_pts=float(engine_cfg.get('min_risk_pts', 0.5)))


def create_guard_config(config: Dict) -> GuardConfig:
    """Create GuardConfig from configuration."""
    guard_cfg = config.get('guards', {})

    return GuardConfig(
        trading_hours_tz=guard_cfg.get('trading_hours_tz', 'Europe/Brussels'),
        trading_hours_start=guard_cfg.get('trading_hours_start', '08:00'),
        trading_hours_end=guard_cfg.get('trading_hours_end', '22:00'),
        min_atr_pts=float(guard_cfg.get('min_atr_pts', 20.0)),
        cooldown_bars=int(guard_cfg.get('cooldown_bars', 3)),
        max_trades_per_day=int(guard_cfg.get('max_trades_per_day', 10)),
        one_trade_per_timestamp=bool(guard_cfg.get('one_trade_per_timestamp', True)))


def create_strategies_from_config(config: Dict) -> List[Tuple[object, float]]:
    """Create strategy instances from configuration."""
    strategies = []
    strategies_cfg = config.get('strategies', {})

    # DAX Trend Continuation Strategy
    dax_trend_cfg = strategies_cfg.get('dax_trend_continuation', {})
    if dax_trend_cfg.get('enabled', False):
        from strategies.dax_sr_breakout import DAXTrendContinuationStrategy

        params = dax_trend_cfg.get('params', {})
        allocation = float(dax_trend_cfg.get('allocation', 100.0))

        strategy = DAXTrendContinuationStrategy(params)
        strategies.append((strategy, allocation))
        logger.info(f"Loaded DAX Trend Continuation strategy with {allocation}% allocation")

    # RSI Reversion Strategy
    rsi_cfg = strategies_cfg.get('rsi_reversion', {})
    if rsi_cfg.get('enabled', False):
        from strategies.rsi_reversion import RSIReversionStrategy

        params = rsi_cfg.get('params', {})
        allocation = float(rsi_cfg.get('allocation', 100.0))

        strategy = RSIReversionStrategy(params)
        strategies.append((strategy, allocation))
        logger.info(f"Loaded RSI strategy with {allocation}% allocation")

    # Simple Test Strategy
    simple_cfg = strategies_cfg.get('simple_test', {})
    if simple_cfg.get('enabled', False):
        from strategies.simple_test import SimpleTestStrategy

        params = simple_cfg.get('params', {})
        allocation = float(simple_cfg.get('allocation', 100.0))

        strategy = SimpleTestStrategy(params)
        strategies.append((strategy, allocation))
        logger.info(f"Loaded SimpleTest strategy with {allocation}% allocation")

    if not strategies:
        raise ValueError("No strategies enabled in configuration")

    return strategies


def print_backtest_results(results: Dict):
    """Print formatted backtest results."""
    metrics = results['metrics']
    positions = results['positions']

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    # Performance metrics
    print(f"Total Trades:      {metrics['total_trades']:,}")
    print(f"  - Long trades:   {metrics['total_longs']:,}")
    print(f"  - Short trades:  {metrics['total_shorts']:,}")
    print(f"")
    print(f"Initial Capital:   ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital:     ${metrics['final_capital']:,.2f}")
    print(f"Total Return:      {metrics['total_return']:+.2f}%")
    print(f"")
    print(f"Win Rate:          {metrics['win_rate']:.1f}%")

    pf = metrics['profit_factor']
    pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
    print(f"Profit Factor:     {pf_str}")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:+.2f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']:.2f}%")

    if metrics['total_trades'] > 0:
        print(f"")
        print(f"Gross Profit:      ${metrics['gross_profit']:,.2f}")
        print(f"Gross Loss:        ${metrics['gross_loss']:,.2f}")
        print(f"Average Win:       ${metrics['avg_win']:,.2f}")
        print(f"Average Loss:      ${metrics['avg_loss']:,.2f}")

    # Strategy breakdown
    if not positions.empty:
        print(f"\n" + "-" * 40)
        print("STRATEGY BREAKDOWN")
        print("-" * 40)

        strategy_stats = positions.groupby('strategy').agg(
            {'pnl': ['count', 'sum', 'mean'],
                'side': lambda x: (x == 'long').sum()}).round(2)

        strategy_stats.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Long_Count']
        strategy_stats['Short_Count'] = strategy_stats['Trades'] - strategy_stats[
            'Long_Count']
        strategy_stats['Win_Rate'] = positions.groupby('strategy')['pnl'].apply(
            lambda x: (x > 0).mean() * 100).round(1)

        for strategy_name, row in strategy_stats.iterrows():
            print(f"\n{strategy_name}:")
            print(
                f"  Trades: {int(row['Trades'])} (L: {int(row['Long_Count'])}, S: {int(row['Short_Count'])})")
            print(f"  Total PnL: ${row['Total_PnL']:+,.2f}")
            print(f"  Avg PnL: ${row['Avg_PnL']:+,.2f}")
            print(f"  Win Rate: {row['Win_Rate']:.1f}%")

    print("\n" + "=" * 60)

    # Save detailed results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if not positions.empty:
        positions.to_csv(output_dir / "trades.csv", index=False)
        print(f"Detailed trades saved to: {output_dir / 'trades.csv'}")


def run_backtest_csv(csv_path: str, config: Dict):
    """Run backtest using CSV data."""
    logger.info(f"Starting CSV backtest with {csv_path}")

    # Load data
    data_cfg = config.get('data', {})
    resample = data_cfg.get('resample', '5min')

    feed = CSVFeed(csv_path)
    df = feed.load(resample=resample)

    if df.empty:
        raise ValueError(f"No data loaded from {csv_path}")

    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Create engine
    engine_config = create_engine_config(config)
    engine = MultiStrategyEngine(engine_config)

    # Add strategies
    strategies = create_strategies_from_config(config)
    for strategy, allocation in strategies:
        engine.add_strategy(strategy, allocation)

    # Create guards
    guard_config = create_guard_config(config)

    # Run backtest
    logger.info("Running backtest...")
    results = engine.run_backtest(df, guard_config)

    # Print results
    print_backtest_results(results)


def run_backtest_mt5(config: Dict):
    """Run backtest using MT5 data."""
    try:
        from feeds.mt5_feed import MT5Feed
    except ImportError:
        logger.error("MT5 not available. Install MetaTrader5 package.")
        return

    logger.info("Starting MT5 backtest")

    # MT5 configuration
    mt5_cfg = config.get('mt5', {})
    symbol = mt5_cfg.get('symbol', 'GER40.cash')
    timeframe = mt5_cfg.get('timeframe', 'M1')
    bars = int(mt5_cfg.get('bars', 10_000))

    # Connect and fetch data
    feed = MT5Feed(symbol=symbol, timeframe=timeframe, bars=bars)

    if not feed.connect(login=mt5_cfg.get('login'), password=mt5_cfg.get('password'),
            server=mt5_cfg.get('server')):
        logger.error("Failed to connect to MT5")
        return

    try:
        df = feed.fetch()
        logger.info(f"Fetched {len(df)} bars from MT5")

        # Resample if needed
        data_cfg = config.get('data', {})
        resample = data_cfg.get('resample')
        if resample:
            df = df.resample(resample).agg(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                    'volume': 'sum'}).dropna()
            logger.info(f"Resampled to {resample}: {len(df)} bars")

        # Create engine
        engine_config = create_engine_config(config)
        engine = MultiStrategyEngine(engine_config)

        # Add strategies
        strategies = create_strategies_from_config(config)
        for strategy, allocation in strategies:
            engine.add_strategy(strategy, allocation)

        # Create guards
        guard_config = create_guard_config(config)

        # Run backtest
        logger.info("Running backtest...")
        results = engine.run_backtest(df, guard_config)

        # Print results
        print_backtest_results(results)

    finally:
        feed.disconnect()


async def run_live_trading(config: Dict, dry_run: bool = False):
    """Run live trading."""
    try:
        from live_trading import LiveTradingManager
    except ImportError:
        logger.error("Live trading module not available")
        return

    if dry_run:
        logger.info("Starting PAPER TRADING (dry-run mode)")
        config['live'] = config.get('live', {})
        config['live']['dry_run'] = True
    else:
        logger.info("Starting LIVE TRADING")

    # Create manager with config dict
    manager = LiveTradingManager(config)

    try:
        await manager.start_live_trading()
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    except Exception as e:
        logger.error(f"Live trading error: {e}")
        raise


def test_mt5_connection(config: Dict):
    """Test MT5 connection."""
    try:
        from feeds.mt5_feed import MT5Feed
    except ImportError:
        print("❌ MT5 package not available. Install with: pip install MetaTrader5")
        return

    mt5_cfg = config.get('mt5', {})
    symbol = mt5_cfg.get('symbol', 'GER40.cash')

    print(f"Testing MT5 connection for {symbol}...")

    feed = MT5Feed(symbol=symbol, timeframe='M1', bars=10)

    if feed.connect(login=mt5_cfg.get('login'), password=mt5_cfg.get('password'),
            server=mt5_cfg.get('server')):
        try:
            df = feed.fetch()
            print(f"✅ MT5 connection successful! Fetched {len(df)} bars.")
            print(f"Latest price: {df['close'].iloc[-1]:.2f}")
            print(f"Time range: {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            print(f"❌ Data fetch failed: {e}")
        finally:
            feed.disconnect()
    else:
        print("❌ MT5 connection failed")


def create_sample_config():
    """Create a sample configuration file."""
    config = {
        'engine': {
            'initial_capital': 10000.0,
            'risk_per_trade': 1.5,
            'max_positions': 2,
            'commission': 0.0002,
            'slippage': 0.0001,
            'time_exit_bars': 50,
            'allow_shorts': True,
            'min_risk_pts': 20.0
        },
        'data': {
            'symbol': 'GER40.cash',
            'timeframe': 'H4',
            'resample': None
        },
        'mt5': {
            'symbol': 'GER40.cash',
            'timeframe': 'H4',
            'bars': 2000,
            # 'login': 123456,
            # 'password': 'your_password',
            # 'server': 'FTMO-Demo'
        },
        'strategies': {
            'dax_trend_continuation': {
                'enabled': True,
                'allocation': 100.0,
                'params': {
                    'ema_fast': 8,
                    'ema_slow': 21,
                    'atr_period': 14,
                    'sl_multiplier': 2.5,
                    'tp_multiplier': 1.8,
                    'min_trend_strength': 25.0,
                    'pullback_lookback': 3
                }
            },
            'rsi_reversion': {
                'enabled': False,
                'allocation': 100.0,
                'params': {
                    'rsi_period': 14,
                    'oversold': 35.0,
                    'overbought': 65.0,
                    'atr_period': 14,
                    'sl_multiplier': 2.0,
                    'tp_multiplier': 1.5
                }
            },
            'simple_test': {
                'enabled': False,
                'allocation': 100.0,
                'params': {
                    'entry_bar': 100,
                    'trade_interval': 500,
                    'max_trades': 5
                }
            }
        },
        'guards': {
            'trading_hours_tz': 'Europe/Berlin',
            'trading_hours_start': '08:00',
            'trading_hours_end': '22:00',
            'min_atr_pts': 30.0,
            'cooldown_bars': 2,
            'max_trades_per_day': 2,
            'one_trade_per_timestamp': True
        }
    }

    config_file = Path('config_dax_4h_sample.yaml')
    with config_file.open('w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Sample DAX 4H configuration created: {config_file}")
    print("Edit this file with your MT5 settings before running.")


def build_argument_parser():
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Minimal Trader - DAX 4H Trend Continuation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  # Create sample config for DAX 4H
  python main.py config --sample

  # Test MT5 connection
  python main.py test --config config_dax_4h.yaml

  # Run CSV backtest
  python main.py backtest --csv data/GER40_4H.csv --config config_dax_4h.yaml

  # Run MT5 backtest for DAX 4H
  python main.py backtest --mt5 --config config_dax_4h.yaml

  # Run live trading (paper trading)
  python main.py live --config config_dax_4h.yaml --dry-run

  # Run live trading with monitoring
  python main.py live --config config_dax_4h.yaml --monitor

  # Run REAL live trading (BE CAREFUL!)
  python main.py live --config config_dax_4h.yaml
        """)

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--sample', action='store_true',
                               help='Create sample DAX 4H config file')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test connections')
    test_parser.add_argument('--config', required=True, help='Configuration file path')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--config', required=True,
                                 help='Configuration file path')
    backtest_group = backtest_parser.add_mutually_exclusive_group(required=True)
    backtest_group.add_argument('--csv', help='CSV file path for backtesting')
    backtest_group.add_argument('--mt5', action='store_true',
                                help='Use MT5 data for backtesting')

    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--config', required=True, help='Configuration file path')
    live_parser.add_argument('--dry-run', action='store_true',
                             help='Paper trading mode - simulate trades without real execution')
    live_parser.add_argument('--monitor', action='store_true',
                             help='Use enhanced monitoring mode with detailed logging')

    return parser


def main():
    """Main entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'config':
            if args.sample:
                create_sample_config()
            else:
                parser.error("Specify --sample to create sample config")

        elif args.command == 'test':
            config = load_config(args.config)
            test_mt5_connection(config)

        elif args.command == 'backtest':
            config = load_config(args.config)

            if args.csv:
                run_backtest_csv(args.csv, config)
            elif args.mt5:
                run_backtest_mt5(config)

        elif args.command == 'live':
            config = load_config(args.config)

            if args.monitor:
                # Use enhanced monitoring
                try:
                    from live_monitor import LiveTradingMonitor
                    print("🔍 Starting Enhanced Live Trading Monitor")
                    monitor = LiveTradingMonitor(args.config)
                    asyncio.run(monitor.start_monitoring())
                except ImportError:
                    print(
                        "❌ Enhanced monitoring not available, falling back to basic live trading")
                    asyncio.run(run_live_trading(config, dry_run=args.dry_run))
            else:
                # Basic live trading
                if args.dry_run:
                    print("🧪 PAPER TRADING MODE: No real trades will be executed")
                asyncio.run(run_live_trading(config, dry_run=args.dry_run))

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()