#!/usr/bin/env python3
"""Main entry point for multi-strategy trading system."""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd

# Import strategies

from strategies.rsi_reversion import RSIReversionStrategy  # NEW!

# Import feeds
from feeds.mt5_feed import MT5Feed
from feeds.csv_feed import CSVFeed

# Import engine
from engine import MultiStrategyEngine, EngineConfig


def setup_logging(config: dict):
    """Configure logging."""
    Path("logs").mkdir(exist_ok=True)

    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    logging.basicConfig(level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.FileHandler(log_config.get('file', 'logs/trader.log')),
            logging.StreamHandler()])
    return logging.getLogger('main')


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_strategies(config: dict) -> list:
    """Create strategy instances from config."""
    strategies = []
    strat_config = config.get('strategies', {})





    # RSI Reversion Strategy (NEW!)
    if strat_config.get('rsi_reversion', {}).get('enabled', False):
        rsi_params = strat_config['rsi_reversion'].get('params', {})
        allocation = strat_config['rsi_reversion'].get('allocation', 100.0)
        strategies.append((RSIReversionStrategy(rsi_params), allocation))

    return strategies


def run_backtest_mt5(config: dict, logger):
    """Run backtest using MT5 data."""
    # Setup MT5 feed
    data_config = config.get('data', {})
    mt5_config = config.get('mt5', {})

    feed = MT5Feed(symbol=data_config.get('symbol', 'GER40.cash'),
        timeframe=data_config.get('timeframe', 'H1'))

    # Connect to MT5
    if not feed.connect(login=mt5_config.get('login'),
            password=mt5_config.get('password'), server=mt5_config.get('server')):
        logger.error("Failed to connect to MT5")
        return

    try:
        # Get historical data
        bars = data_config.get('bars', 5000)
        df = feed.get_historical(bars=bars)

        if df.empty:
            logger.error("No data received from MT5")
            return

        # Optional resampling
        if data_config.get('resample'):
            df = df.resample(data_config['resample']).agg(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                    'volume': 'sum'}).dropna()
            logger.info(f"Resampled to {data_config['resample']}: {len(df)} bars")

        # Run backtest
        run_backtest_on_data(df, config, logger)

    finally:
        feed.disconnect()


def run_backtest_csv(csv_path: str, config: dict, logger):
    """Run backtest using CSV data."""
    # Load CSV
    feed = CSVFeed(csv_path)

    data_config = config.get('data', {})
    df = feed.load(resample=data_config.get('resample'))

    if df.empty:
        logger.error("No data loaded from CSV")
        return

    # Run backtest
    run_backtest_on_data(df, config, logger)


def run_backtest_on_data(df: pd.DataFrame, config: dict, logger):
    """Run backtest on prepared data."""
    # Setup engine
    engine_config = config.get('engine', {})
    engine = MultiStrategyEngine(
        EngineConfig(initial_capital=engine_config.get('initial_capital', 10000),
            risk_per_trade=engine_config.get('risk_per_trade', 1.0),
            max_positions=engine_config.get('max_positions', 3),
            commission=engine_config.get('commission', 0.0002),
            slippage=engine_config.get('slippage', 0.0001),
            time_exit_bars=engine_config.get('time_exit_bars', 200),
            allow_shorts=engine_config.get('allow_shorts', True)  # NEW!
        ))

    # Add strategies
    strategies = create_strategies(config)
    if not strategies:
        logger.error("No strategies enabled in config")
        return

    for strategy, allocation in strategies:
        engine.add_strategy(strategy, allocation)

    # Run backtest
    logger.info(
        f"Running backtest on {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    results = engine.run_backtest(df)

    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    metrics = results['metrics']
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"  - Longs:       {metrics.get('total_longs', 0)}")
    print(f"  - Shorts:      {metrics.get('total_shorts', 0)}")
    print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital:   ${metrics['final_capital']:,.2f}")
    print(f"Total Return:    {metrics['total_return']:.2f}%")
    print(f"Win Rate:        {metrics['win_rate']:.2f}%")
    print(f"Profit Factor:   {metrics.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")

    # Strategy breakdown
    if 'strategy_metrics' in metrics:
        print("\n" + "-" * 40)
        print("STRATEGY BREAKDOWN")
        print("-" * 40)
        for strat_name, strat_metrics in metrics['strategy_metrics'].items():
            print(f"\n{strat_name}:")
            print(f"  Trades:      {strat_metrics['trades']}")
            print(f"    - Longs:   {strat_metrics.get('longs', 0)}")
            print(f"    - Shorts:  {strat_metrics.get('shorts', 0)}")
            print(f"  Win Rate:    {strat_metrics['win_rate']:.2f}%")
            print(f"  Total PnL:   ${strat_metrics['total_pnl']:.2f}")
            print(f"  Avg PnL:     ${strat_metrics['avg_pnl']:.2f}")
            print(f"  Best Trade:  ${strat_metrics.get('best_trade', 0):.2f}")
            print(f"  Worst Trade: ${strat_metrics.get('worst_trade', 0):.2f}")

    # Save trades
    trades_df = results['trades']
    if not trades_df.empty:
        Path("output").mkdir(exist_ok=True)
        output_file = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\nTrades saved to: {output_file}")

        # Show sample trades
        print("\n" + "-" * 40)
        print("SAMPLE TRADES (First 5)")
        print("-" * 40)
        print(trades_df[['strategy', 'side', 'entry_price', 'exit_price', 'pnl',
                         'exit_reason']].head())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Multi-Strategy Trading System')
    parser.add_argument('mode', choices=['backtest', 'live', 'test'],
                        help='Operating mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file')
    parser.add_argument('--csv', type=str,
                        help='CSV file for backtest (optional, uses MT5 if not provided)')
    parser.add_argument('--symbol', type=str, help='Override symbol from config')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config)

    # Override symbol if provided
    if args.symbol:
        config['data']['symbol'] = args.symbol

    # Execute mode
    if args.mode == 'backtest':
        if args.csv:
            logger.info(f"Running backtest on CSV: {args.csv}")
            run_backtest_csv(args.csv, config, logger)
        else:
            logger.info("Running backtest with MT5 data")
            run_backtest_mt5(config, logger)

    elif args.mode == 'test':
        # Test MT5 connection
        logger.info("Testing MT5 connection...")
        feed = MT5Feed()
        if feed.connect():
            info = feed.get_symbol_info()
            print(f"Connected! Symbol info: {info}")
            tick = feed.get_latest_tick()
            print(f"Latest tick: {tick}")
            feed.disconnect()
        else:
            print("Connection failed")

    elif args.mode == 'live':
        logger.warning("Live trading not yet implemented")
        sys.exit(1)


if __name__ == '__main__':
    main()