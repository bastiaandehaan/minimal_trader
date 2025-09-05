# System Architecture - Minimal Trader v2

## Components
- `strategy.py`: Pure signal generation (SMA cross + ATR SL/TP, volume confirm)
- `backtest.py`: Execution engine (O(n), SL/TP, reversal/timeout exit, metrics)
- `data_feed.py`: Data access layer (CSVFeed, later LiveFeed/MT5)
- `main.py`: CLI orchestration (backtest, signal)
- `tests/`: unit + integration tests

## Data Flow
CLI → Config → CSVFeed.load() → Strategy.calculate_indicators() → loop i: Strategy.get_signal_at(i) → Backtest execution → Trades & Metrics.
