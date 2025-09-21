# main.py - Minimal Trader with DAX SR Breakout Strategy
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
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG voor maximale details
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    logger.debug(f"Attempting to load config from: {config_path}")
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_file.open('r') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError(f"Empty or invalid config file: {config_path}")
        logger.debug(f"Full loaded config: {config}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise

def create_engine_config(config: Dict) -> EngineConfig:
    """Create EngineConfig from configuration."""
    engine_cfg = config.get('engine', {})
    logger.debug(f"Engine config: {engine_cfg}")
    return EngineConfig(
        initial_capital=float(engine_cfg.get('initial_capital', 10_000.0)),
        risk_per_trade=float(engine_cfg.get('risk_per_trade', 1.0)),
        max_positions=int(engine_cfg.get('max_positions', 3)),
        commission=float(engine_cfg.get('commission', 0.0002)),
        slippage=float(engine_cfg.get('slippage', 0.0)),
        time_exit_bars=int(engine_cfg.get('time_exit_bars', 200)),
        allow_shorts=bool(engine_cfg.get('allow_shorts', True)),
        min_risk_pts=float(engine_cfg.get('min_risk_pts', 0.5))
    )

def create_guard_config(config: Dict) -> GuardConfig:
    """Create GuardConfig from configuration."""
    guard_cfg = config.get('guards', {})
    logger.debug(f"Guard config: {guard_cfg}")
    return GuardConfig(
        trading_hours_tz=guard_cfg.get('trading_hours_tz', 'Europe/Brussels'),
        trading_hours_start=guard_cfg.get('trading_hours_start', '08:00'),
        trading_hours_end=guard_cfg.get('trading_hours_end', '22:00'),
        min_atr_pts=float(guard_cfg.get('min_atr_pts', 20.0)),
        cooldown_bars=int(guard_cfg.get('cooldown_bars', 3)),
        max_trades_per_day=int(guard_cfg.get('max_trades_per_day', 10)),
        one_trade_per_timestamp=bool(guard_cfg.get('one_trade_per_timestamp', True))
    )

def create_strategies_from_config(config: Dict) -> List[Tuple[object, float]]:
    """Create strategy instances from configuration."""
    strategies = []
    strategies_cfg = config.get('strategies', {})
    logger.debug(f"Strategies section from config: {strategies_cfg}")

    # DAX SR Breakout Strategy
    dax_sr_cfg = strategies_cfg.get('dax_sr_breakout', {})
    logger.debug(f"Checking 'dax_sr_breakout': enabled={dax_sr_cfg.get('enabled', False)}")
    if dax_sr_cfg.get('enabled', False):
        from strategies.dax_sr_breakout import DAXSRBreakoutStrategy
        params = dax_sr_cfg.get('params', {})
        allocation = float(dax_sr_cfg.get('allocation', 100.0))
        try:
            strategy = DAXSRBreakoutStrategy(params)
            strategies.append((strategy, allocation))
            logger.info(f"Loaded DAX SR Breakout strategy with {allocation}% allocation")
        except Exception as e:
            logger.error(f"Failed to load DAX SR Breakout strategy: {e}")
            raise
    else:
        logger.debug("'dax_sr_breakout' not enabled or missing")

    # RSI Reversion Strategy
    rsi_cfg = strategies_cfg.get('rsi_reversion', {})
    logger.debug(f"Checking 'rsi_reversion': enabled={rsi_cfg.get('enabled', False)}")
    if rsi_cfg.get('enabled', False):
        from strategies.rsi_reversion import RSIReversionStrategy
        params = rsi_cfg.get('params', {})
        allocation = float(rsi_cfg.get('allocation', 100.0))
        strategy = RSIReversionStrategy(params)
        strategies.append((strategy, allocation))
        logger.info(f"Loaded RSI strategy with {allocation}% allocation")
    else:
        logger.debug("'rsi_reversion' not enabled or missing")

    # Simple Test Strategy
    simple_cfg = strategies_cfg.get('simple_test', {})
    logger.debug(f"Checking 'simple_test': enabled={simple_cfg.get('enabled', False)}")
    if simple_cfg.get('enabled', False):
        from strategies.simple_test import SimpleTestStrategy
        params = simple_cfg.get('params', {})
        allocation = float(simple_cfg.get('allocation', 100.0))
        strategy = SimpleTestStrategy(params)
        strategies.append((strategy, allocation))
        logger.info(f"Loaded SimpleTest strategy with {allocation}% allocation")
    else:
        logger.debug("'simple_test' not enabled or missing")

    if not strategies:
        logger.error("No strategies enabled in configuration - check keys and 'enabled: true'")
    else:
        logger.info(f"Loaded {len(strategies)} strategies")

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
            'timeframe': 'M1',
            'resample': None
        },
        'mt5': {
            'symbol': 'GER40.cash',
            'timeframe': 'M1',
            'bars': 10000,
            # 'login': 123456,
            # 'password': 'your_password',
            # 'server': 'FTMO-Demo'
        },
        'strategies': {
            'dax_sr_breakout': {
                'enabled': True,
                'allocation': 100.0,
                'params': {
                    'lookback_period': 100,
                    'sr_threshold': 0.001,
                    'atr_period': 14,
                    'sl_multiplier': 2.0,
                    'tp_multiplier': 3.0,
                    'ema_fast_period': 8,
                    'ema_slow_period': 21,
                    'min_trend_strength': 30.0,
                    'time_exit_bars': 200
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

    config_file = Path('configs/config_dax_sr_sample.yaml')
    with config_file.open('w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Sample DAX SR configuration created: {config_file}")
    print("Edit this file with your MT5 settings before running.")

def test_mt5_connection(config: Dict):
    """Test MT5 connection."""
    try:
        from feeds.mt5_feed import MT5Feed
    except ImportError:
        logger.error("MT5 not available. Install MetaTrader5 package.")
        return

    mt5_cfg = config.get('mt5', {})
    symbol = mt5_cfg.get('symbol', 'GER40.cash')
    timeframe = mt5_cfg.get('timeframe', 'M1')
    bars = int(mt5_cfg.get('bars', 10_000))

    feed = MT5Feed(symbol=symbol, timeframe=timeframe, bars=bars)
    if feed.connect(login=mt5_cfg.get('login'), password=mt5_cfg.get('password'),
                   server=mt5_cfg.get('server')):
        logger.info("MT5 connection successful")
        feed.disconnect()
    else:
        logger.error("MT5 connection failed")

def run_backtest_csv(csv_path: str, config: Dict):
    """Run backtest using CSV data."""
    try:
        feed = CSVFeed(csv_path)
        df = feed.fetch()
        logger.info(f"Fetched {len(df)} bars from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    data_cfg = config.get('data', {})
    resample = data_cfg.get('resample')
    if resample:
        df = df.resample(resample).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
             'volume': 'sum'}).dropna()
        logger.info(f"Resampled to {resample}: {len(df)} bars")

    engine_config = create_engine_config(config)
    engine = MultiStrategyEngine(engine_config)

    strategies = create_strategies_from_config(config)
    for strategy, allocation in strategies:
        engine.add_strategy(strategy, allocation)
        logger.info(f"Added strategy: {strategy.__class__.__name__}, allocation={allocation}")

    guard_config = create_guard_config(config)
    results = engine.run_backtest(df, guard_config)
    print_backtest_results(results)

async def run_live_trading(config: Dict, dry_run: bool = False):
    """Run live trading."""
    try:
        from feeds.mt5_feed import MT5Feed
    except ImportError:
        logger.error("MT5 not available. Install MetaTrader5 package.")
        return

    mt5_cfg = config.get('mt5', {})
    symbol = mt5_cfg.get('symbol', 'GER40.cash')
    timeframe = mt5_cfg.get('timeframe', 'M1')
    bars = int(mt5_cfg.get('bars', 10_000))

    feed = MT5Feed(symbol=symbol, timeframe=timeframe, bars=bars)
    if not feed.connect(login=mt5_cfg.get('login'), password=mt5_cfg.get('password'),
                       server=mt5_cfg.get('server')):
        logger.error("Failed to connect to MT5")
        return

    try:
        from live_trading import LiveTrader
        trader = LiveTrader(config, feed, dry_run=dry_run)
        await trader.run()
    finally:
        feed.disconnect()

def run_backtest_mt5(config: Dict):
    """Run backtest using MT5 data."""
    try:
        from feeds.mt5_feed import MT5Feed
    except ImportError:
        logger.error("MT5 not available. Install MetaTrader5 package.")
        return

    logger.info("Starting MT5 backtest")

    mt5_cfg = config.get('mt5', {})
    logger.debug(f"MT5 config: {mt5_cfg}")
    symbol = mt5_cfg.get('symbol', 'GER40.cash')
    timeframe = mt5_cfg.get('timeframe', 'M1')
    bars = int(mt5_cfg.get('bars', 10_000))

    feed = MT5Feed(symbol=symbol, timeframe=timeframe, bars=bars)
    logger.debug(f"MT5Feed initialized: symbol={symbol}, timeframe={timeframe}, bars={bars}")

    if not feed.connect(login=mt5_cfg.get('login'), password=mt5_cfg.get('password'),
                       server=mt5_cfg.get('server')):
        logger.error("Failed to connect to MT5")
        return

    try:
        df = feed.fetch()
        logger.info(f"Fetched {len(df)} bars from MT5")
        logger.debug(f"Dataframe head: {df.head().to_dict()}")

        data_cfg = config.get('data', {})
        resample = data_cfg.get('resample')
        logger.debug(f"Resample setting: {resample}")
        if resample:
            df = df.resample(resample).agg(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                 'volume': 'sum'}).dropna()
            logger.info(f"Resampled to {resample}: {len(df)} bars")

        engine_config = create_engine_config(config)
        logger.debug(f"Engine config: {engine_config}")
        engine = MultiStrategyEngine(engine_config)

        strategies = create_strategies_from_config(config)
        logger.debug(f"Loaded strategies: {[s[0].__class__.__name__ for s in strategies]}")
        for strategy, allocation in strategies:
            engine.add_strategy(strategy, allocation)
            logger.info(f"Added strategy: {strategy.__class__.__name__}, allocation={allocation}")

        guard_config = create_guard_config(config)
        logger.debug(f"Guard config: {guard_config}")

        logger.info("Running backtest...")
        results = engine.run_backtest(df, guard_config)
        logger.debug(f"Backtest results: {results}")
        print_backtest_results(results)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise
    finally:
        feed.disconnect()

def build_argument_parser():
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Minimal Trader - DAX SR Breakout Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  # Create sample config for DAX SR
  python main.py config --sample

  # Test MT5 connection
  python main.py test --config configs/config_dax_sr.yaml

  # Run CSV backtest
  python main.py backtest --csv data/GER40_M1.csv --config configs/config_dax_sr.yaml

  # Run MT5 backtest for DAX SR
  python main.py backtest --mt5 --config configs/config_dax_sr.yaml

  # Run live trading (paper trading)
  python main.py live --config configs/config_dax_sr.yaml --dry-run

  # Run live trading with monitoring
  python main.py live --config configs/config_dax_sr.yaml --monitor

  # Run REAL live trading (BE CAREFUL!)
  python main.py live --config configs/config_dax_sr.yaml
        """)

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--sample', action='store_true',
                               help='Create sample DAX SR config file')

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
                try:
                    from live_monitor import LiveTradingMonitor
                    print("🔍 Starting Enhanced Live Trading Monitor")
                    monitor = LiveTradingMonitor(args.config)
                    asyncio.run(monitor.start_monitoring())
                except ImportError:
                    print("❌ Enhanced monitoring not available, falling back to basic live trading")
                    asyncio.run(run_live_trading(config, dry_run=args.dry_run))
            else:
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