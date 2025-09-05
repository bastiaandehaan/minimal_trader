#!/usr/bin/env python3
"""
Minimal Viable Trading System
One file to rule them all - strategie, backtest, live trading
"""
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleBreakoutTrader:
    """Dead simple breakout strategy - no BS, just works"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.symbol = self.config['trading']['symbol']
        self.risk_pct = self.config['trading']['risk_per_trade']
        self.sma_period = self.config['strategy']['sma_period']
        self.atr_period = self.config['strategy']['atr_period']

        logger.info(f"Initialized trader for {self.symbol}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA, ATR and Volume metrics"""
        df = df.copy()

        # SMA
        df['sma'] = df['close'].rolling(self.sma_period).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(self.atr_period).mean()

        # Volume
        df['volume_avg'] = df['volume'].rolling(self.sma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']

        return df

    def get_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Dict]:
        """
        Returns signal and metadata
        Simple rules:
        - BUY: price crosses above SMA + volume confirmation
        - SELL: price crosses below SMA
        - None: no clear signal
        """
        df = self.calculate_indicators(df)

        if len(df) < self.sma_period:
            return None, {}

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Detect crossovers
        cross_above = (prev['close'] <= prev['sma']) and (last['close'] > last['sma'])
        cross_below = (prev['close'] >= prev['sma']) and (last['close'] < last['sma'])

        # Volume confirmation
        volume_confirmed = last['volume_ratio'] > self.config['strategy']['volume_threshold']

        metadata = {
            'close': last['close'],
            'sma': last['sma'],
            'atr': last['atr'],
            'volume_ratio': last['volume_ratio'],
            'timestamp': last.name if hasattr(last, 'name') else None
        }

        if cross_above and volume_confirmed:
            metadata['reason'] = f"Bullish cross + volume {last['volume_ratio']:.1f}x"
            return 'BUY', metadata
        elif cross_below:
            metadata['reason'] = "Bearish cross"
            return 'SELL', metadata

        return None, metadata

    def calculate_position_size(self, equity: float, entry: float, stop: float) -> float:
        """Position sizing based on risk management"""
        risk_amount = equity * (self.risk_pct / 100)
        risk_points = abs(entry - stop)

        if risk_points == 0:
            return 0

        # For forex/indices: size in lots
        # Adjust multiplier based on your broker
        point_value = 1.0  # EUR per point for GER40
        size = risk_amount / (risk_points * point_value)

        return round(size, 2)

class Backtester:
    """Simple backtesting engine - no look-ahead, realistic fills"""

    def __init__(self, trader: SimpleBreakoutTrader):
        self.trader = trader
        self.config = trader.config

    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest and return results"""
        initial_capital = self.config['trading']['initial_capital']
        equity = initial_capital

        trades = []
        position = None

        df = self.trader.calculate_indicators(df)

        for i in range(self.trader.sma_period, len(df)):
            window = df.iloc[:i+1]
            signal, meta = self.trader.get_signal(window)

            current_bar = df.iloc[i]

            # Check exits first (before new signals)
            if position:
                exit_price = None
                exit_reason = None

                # Check stop loss
                if current_bar['low'] <= position['stop']:
                    exit_price = position['stop']
                    exit_reason = 'Stop Loss'
                # Check take profit
                elif current_bar['high'] >= position['target']:
                    exit_price = position['target']
                    exit_reason = 'Take Profit'
                # Check signal reversal
                elif signal == 'SELL':
                    exit_price = current_bar['close']
                    exit_reason = 'Signal Reversal'

                if exit_price:
                    # Calculate PnL
                    pnl = (exit_price - position['entry']) * position['size']
                    pnl -= (position['entry'] * position['size'] * self.config['backtest']['commission'] * 2)

                    equity += pnl

                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_bar.name,
                        'side': position['side'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'return_pct': (pnl / equity) * 100,
                        'reason': exit_reason
                    })

                    position = None
                    logger.debug(f"Exit {exit_reason} at {exit_price:.2f}, PnL: {pnl:.2f}")

            # New signals (only if no position)
            if signal and not position:
                if signal == 'BUY':
                    entry = current_bar['close']
                    stop = entry - current_bar['atr'] * self.config['strategy']['sl_multiplier']
                    target = entry + current_bar['atr'] * self.config['strategy']['tp_multiplier']

                    size = self.trader.calculate_position_size(equity, entry, stop)

                    if size > 0:
                        position = {
                            'entry_time': current_bar.name,
                            'side': 'long',
                            'entry': entry,
                            'stop': stop,
                            'target': target,
                            'size': size
                        }
                        logger.debug(f"BUY at {entry:.2f}, stop: {stop:.2f}, target: {target:.2f}")

        # Calculate metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        if not trades_df.empty:
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] <= 0]

            metrics = {
                'initial_capital': initial_capital,
                'final_capital': equity,
                'total_return_pct': ((equity / initial_capital) - 1) * 100,
                'num_trades': len(trades_df),
                'num_wins': len(wins),
                'num_losses': len(losses),
                'win_rate': (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
                'avg_win': wins['pnl'].mean() if not wins.empty else 0,
                'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
                'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(trades_df),
                'sharpe_ratio': self._calculate_sharpe(trades_df)
            }
        else:
            metrics = {
                'initial_capital': initial_capital,
                'final_capital': equity,
                'total_return_pct': 0,
                'num_trades': 0,
                'message': 'No trades generated'
            }

        return {'metrics': metrics, 'trades': trades_df}

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage"""
        if trades_df.empty:
            return 0

        cumulative = trades_df['pnl'].cumsum()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / (running_max + self.config['trading']['initial_capital'])
        return abs(drawdown.min()) * 100 if not drawdown.empty else 0

    def _calculate_sharpe(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio (simplified)"""
        if trades_df.empty or len(trades_df) < 2:
            return 0

        returns = trades_df['return_pct']
        if returns.std() == 0:
            return 0

        # Annualize based on average trades per year (estimate)
        days_in_sample = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        if days_in_sample > 0:
            trades_per_year = (252 / days_in_sample) * len(trades_df)
            return (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
        return 0

class LiveTrader:
    """Live trading interface - can be paper or real"""

    def __init__(self, trader: SimpleBreakoutTrader):
        self.trader = trader
        self.config = trader.config
        self.position = None
        self.equity = self.config['trading']['initial_capital']

    def fetch_latest_data(self, lookback_bars: int = 100) -> pd.DataFrame:
        """Fetch latest price data"""
        # For now, use CSV data for testing
        # Replace with MT5 or broker API
        csv_files = list(Path('data').glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("No CSV files in data/ directory")

        df = pd.read_csv(csv_files[0], parse_dates=['time'])
        df.set_index('time', inplace=True)

        # Get last N bars
        return df.tail(lookback_bars)

    def check_for_signals(self):
        """Check current market for signals"""
        try:
            df = self.fetch_latest_data()
            signal, meta = self.trader.get_signal(df)

            if signal:
                logger.info(f"SIGNAL: {signal} at {meta.get('timestamp', 'now')}")
                logger.info(f"Reason: {meta.get('reason', 'N/A')}")
                logger.info(f"Price: {meta.get('close', 0):.2f}, SMA: {meta.get('sma', 0):.2f}")

                if self.config['live']['mode'] == 'live':
                    self.place_order(signal, meta)
                else:
                    logger.info("PAPER TRADE - no real order placed")

        except Exception as e:
            logger.error(f"Error checking signals: {e}")

    def place_order(self, signal: str, meta: Dict):
        """Place actual trade order"""
        # TODO: Implement MT5 order placement
        logger.warning("Live order placement not yet implemented")

    def run_forever(self):
        """Main loop for live trading"""
        logger.info(f"Starting live trader in {self.config['live']['mode']} mode")
        logger.info(f"Checking every {self.config['live']['check_interval']} seconds")

        while True:
            try:
                self.check_for_signals()
                time.sleep(self.config['live']['check_interval'])
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait a minute before retry

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare data"""
    df = pd.read_csv(csv_path)

    # Handle different column names
    time_cols = ['time', 'Time', 'datetime', 'Datetime', 'date', 'Date']
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Required columns
    required = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must have columns: {required}")

    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 1000  # Dummy volume

    return df.sort_index()

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Minimal Viable Trading System')
    parser.add_argument('mode', choices=['backtest', 'live', 'signal'],
                      help='Operating mode')
    parser.add_argument('--csv', type=str, help='CSV file path for backtesting')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Configuration file')

    args = parser.parse_args()

    # Create directories
    Path('logs').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)

    # Initialize trader
    trader = SimpleBreakoutTrader(args.config)

    if args.mode == 'backtest':
        if not args.csv:
            csv_files = list(Path('data').glob('*.csv'))
            if not csv_files:
                logger.error("No CSV files found. Use --csv or add files to data/")
                sys.exit(1)
            args.csv = str(csv_files[0])
            logger.info(f"Using {args.csv}")

        # Load data
        df = load_data(args.csv)
        logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

        # Run backtest
        backtester = Backtester(trader)
        results = backtester.run(df)

        # Display results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{key:20s}: {value:,.2f}")
            else:
                print(f"{key:20s}: {value}")

        # Save trades
        if not results['trades'].empty:
            output_file = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results['trades'].to_csv(output_file, index=False)
            print(f"\nTrades saved to: {output_file}")

    elif args.mode == 'live':
        live = LiveTrader(trader)
        live.run_forever()

    elif args.mode == 'signal':
        # One-time signal check
        live = LiveTrader(trader)
        live.check_for_signals()

if __name__ == '__main__':
    main()
