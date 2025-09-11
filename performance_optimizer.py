#!/usr/bin/env python3
"""
Performance Test Runner & Optimizer for Minimal Trader
Tests multiple configurations and finds optimal parameters
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
from datetime import datetime
import itertools

# Assume we're in the minimal_trader directory
# Import your existing components
import sys

sys.path.append('.')

from engine import MultiStrategyEngine, EngineConfig
from strategies.rsi_reversion import RSIReversionStrategy
from feeds.csv_feed import CSVFeed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Optimize strategy parameters for maximum performance."""

    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.results = []

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for testing."""
        if not self.data_file.exists():
            # Create synthetic data if file doesn't exist
            logger.warning(
                f"Data file {self.data_file} not found, creating synthetic data")
            return self._create_synthetic_data()

        try:
            feed = CSVFeed(str(self.data_file))
            df = feed.load()
            logger.info(f"Loaded {len(df)} bars from {self.data_file.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._create_synthetic_data()

    def _create_synthetic_data(self, n_bars: int = 5000) -> pd.DataFrame:
        """Create realistic synthetic GER40 data for testing."""
        np.random.seed(42)  # Reproducible results

        # Start from realistic GER40 level
        base_price = 21600
        dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5T')

        # Generate price series with realistic characteristics
        # 1. Random walk with mean reversion
        returns = np.random.normal(0, 0.0008, n_bars)  # ~0.08% volatility per 5min

        # Add mean reversion component
        prices = [base_price]
        for i in range(1, n_bars):
            # Mean reversion force
            distance_from_mean = (prices[-1] - base_price) / base_price
            mean_reversion = -0.1 * distance_from_mean  # Pull back to mean

            # Combined return
            total_return = returns[i] + mean_reversion
            new_price = prices[-1] * (1 + total_return)
            prices.append(new_price)

        close = np.array(prices)

        # Generate OHLC
        open_prices = np.concatenate([[close[0]], close[:-1]])

        # Add realistic intrabar movement
        volatility = np.abs(
            np.random.normal(0, 3, n_bars))  # Points of intrabar movement
        high = np.maximum(open_prices, close) + volatility
        low = np.minimum(open_prices, close) - volatility

        # Volume with some correlation to volatility
        base_volume = 2000
        volume_noise = np.random.uniform(0.5, 2.0, n_bars)
        volume_vol_correlation = 1 + volatility / 10  # Higher volume on higher volatility
        volume = (base_volume * volume_noise * volume_vol_correlation).astype(int)

        df = pd.DataFrame(
            {'open': open_prices, 'high': high, 'low': low, 'close': close,
                'volume': volume}, index=dates)

        logger.info(
            f"Created synthetic data: {len(df)} bars, price range {close.min():.1f}-{close.max():.1f}")
        return df

    def test_configuration(self, config: dict, data: pd.DataFrame) -> dict:
        """Test a single configuration and return performance metrics."""
        try:
            # Setup engine
            engine_config = config.get('engine', {})
            engine = MultiStrategyEngine(EngineConfig(
                initial_capital=engine_config.get('initial_capital', 10000),
                risk_per_trade=engine_config.get('risk_per_trade', 1.0),
                max_positions=engine_config.get('max_positions', 3),
                commission=engine_config.get('commission', 0.0002),
                slippage=engine_config.get('slippage', 0.0001),
                time_exit_bars=engine_config.get('time_exit_bars', 200),
                allow_shorts=engine_config.get('allow_shorts', True)))

            # Add RSI strategy
            strategies_config = config.get('strategies', {})
            if strategies_config.get('rsi_reversion', {}).get('enabled', False):
                params = strategies_config['rsi_reversion'].get('params', {})
                allocation = strategies_config['rsi_reversion'].get('allocation', 100.0)
                strategy = RSIReversionStrategy(params)
                engine.add_strategy(strategy, allocation)

            # Run backtest
            results = engine.run_backtest(data)

            if results['metrics']['total_trades'] == 0:
                return {'error': 'No trades generated'}

            # Extract key metrics
            metrics = results['metrics']
            return {'total_trades': metrics['total_trades'],
                'total_longs': metrics.get('total_longs', 0),
                'total_shorts': metrics.get('total_shorts', 0),
                'total_return': metrics['total_return'],
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'profit_factor': metrics.get('profit_factor', 0),
                'final_capital': metrics['final_capital'], 'config': config}

        except Exception as e:
            logger.error(f"Error testing configuration: {e}")
            return {'error': str(e)}

    def optimize_rsi_parameters(self, data: pd.DataFrame) -> dict:
        """Run parameter optimization for RSI strategy."""
        logger.info("Starting RSI parameter optimization...")

        # Parameter ranges to test
        param_ranges = {'rsi_period': [10, 12, 14, 16, 18, 21],
            'oversold': [20, 25, 30, 35], 'overbought': [65, 70, 75, 80],
            'atr_period': [10, 14, 20], 'sl_multiplier': [1.0, 1.2, 1.5, 2.0],
            'tp_multiplier': [2.0, 2.5, 3.0, 4.0],
            'risk_per_trade': [0.5, 0.8, 1.0, 1.5]}

        best_result = None
        best_score = -999
        total_combinations = np.prod([len(v) for v in param_ranges.values()])

        logger.info(f"Testing {total_combinations} parameter combinations...")

        tested = 0
        for combo in itertools.product(*param_ranges.values()):
            tested += 1
            if tested % 50 == 0:
                logger.info(
                    f"Progress: {tested}/{total_combinations} ({tested / total_combinations * 100:.1f}%)")

            # Create configuration
            rsi_period, oversold, overbought, atr_period, sl_mult, tp_mult, risk = combo

            # Skip invalid combinations
            if oversold >= overbought:
                continue
            if tp_mult <= sl_mult:  # R:R ratio should be positive
                continue

            config = {'engine': {'initial_capital': 10000, 'risk_per_trade': risk,
                'max_positions': 3, 'commission': 0.0002, 'time_exit_bars': 120,
                'allow_shorts': True}, 'strategies': {
                'rsi_reversion': {'enabled': True, 'allocation': 100.0,
                    'params': {'rsi_period': rsi_period, 'oversold': oversold,
                        'overbought': overbought, 'atr_period': atr_period,
                        'sl_multiplier': sl_mult, 'tp_multiplier': tp_mult,
                        'use_next_open': False}}}}

            # Test configuration
            result = self.test_configuration(config, data)

            if 'error' in result:
                continue

            # Calculate composite score (you can adjust weights)
            score = (result['total_return'] * 0.3 +  # 30% weight on returns
                     result['sharpe_ratio'] * 20 * 0.25 +  # 25% weight on Sharpe
                     (100 - result['max_drawdown']) * 0.2 +  # 20% weight on drawdown
                     result['win_rate'] * 0.15 +  # 15% weight on win rate
                     min(result['profit_factor'], 5) * 2 * 0.1
            # 10% weight on profit factor (capped)
            )

            result['composite_score'] = score
            self.results.append(result)

            if score > best_score:
                best_score = score
                best_result = result
                logger.info(
                    f"New best score: {score:.2f} - Return: {result['total_return']:.1f}%, "
                    f"Sharpe: {result['sharpe_ratio']:.2f}, DD: {result['max_drawdown']:.1f}%")

        logger.info(f"Optimization complete! Tested {tested} combinations.")
        return best_result

    def run_full_optimization(self) -> dict:
        """Run complete optimization suite."""
        logger.info("=" * 60)
        logger.info("MINIMAL TRADER PERFORMANCE OPTIMIZATION")
        logger.info("=" * 60)

        # Load data
        data = self.load_data()

        # Optimize RSI parameters
        best_result = self.optimize_rsi_parameters(data)

        if best_result:
            logger.info("\n" + "=" * 60)
            logger.info("OPTIMIZATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Best Score: {best_result['composite_score']:.2f}")
            logger.info(f"Total Return: {best_result['total_return']:.2f}%")
            logger.info(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {best_result['max_drawdown']:.2f}%")
            logger.info(f"Win Rate: {best_result['win_rate']:.2f}%")
            logger.info(f"Profit Factor: {best_result['profit_factor']:.2f}")
            logger.info(f"Total Trades: {best_result['total_trades']}")
            logger.info(
                f"Longs/Shorts: {best_result['total_longs']}/{best_result['total_shorts']}")

            # Show optimal parameters
            optimal_params = best_result['config']['strategies']['rsi_reversion'][
                'params']
            logger.info("\nOptimal Parameters:")
            for key, value in optimal_params.items():
                logger.info(f"  {key}: {value}")

            # Save results
            self._save_results(best_result)

        return best_result

    def _save_results(self, best_result: dict):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save optimal config
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        optimal_config = best_result['config']
        config_file = output_dir / f"optimal_config_{timestamp}.yaml"

        with open(config_file, 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False, indent=2)

        logger.info(f"Optimal configuration saved to: {config_file}")

        # Save all results
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('composite_score', ascending=False)
        results_file = output_dir / f"optimization_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        logger.info(f"All results saved to: {results_file}")
        logger.info(f"Top 10 configurations by score:")
        print(results_df[['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
                          'composite_score']].head(10))


def main():
    """Run the optimization."""
    # Try to use real data first, fall back to synthetic
    data_files = ["GER40.cash_minutes.csv", "data/samples/GER40.cash_M1.sample.csv",
        "synthetic_data.csv"  # Will be created if others don't exist
    ]

    data_file = None
    for file in data_files:
        if Path(file).exists():
            data_file = file
            break

    if data_file is None:
        data_file = data_files[0]  # Will trigger synthetic data creation

    optimizer = PerformanceOptimizer(data_file)
    best_result = optimizer.run_full_optimization()

    if best_result:
        print("\n" + "=" * 60)
        print("🎯 OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print(f"💰 Best Total Return: {best_result['total_return']:.2f}%")
        print(f"📈 Best Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {best_result['max_drawdown']:.2f}%")
        print(f"🎯 Win Rate: {best_result['win_rate']:.2f}%")
        print(f"⚡ Profit Factor: {best_result['profit_factor']:.2f}")
        print("\nUse the generated optimal_config_*.yaml file for best performance!")
    else:
        print("❌ Optimization failed - check logs for details")


if __name__ == "__main__":
    main()