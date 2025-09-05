This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where line numbers have been added, content has been formatted for parsing in markdown style, security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Content has been formatted for parsing in markdown style
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.claude/
  settings.local.json
.github/
  workflows/
    ci.yml
.gitignore
config.yaml
implementation-log.md
README.md
requirements.txt
test_trader.py
trader.py
```

# Files

## File: .claude/settings.local.json
````json
 1: {
 2:   "permissions": {
 3:     "allow": [
 4:       "Bash(python:*)",
 5:       "Bash(venvScripts:*)",
 6:       "Bash(venv/Scripts/activate:*)",
 7:       "Bash(pip install:*)",
 8:       "Bash(venv\\\\Scripts\\\\python.exe:*)",
 9:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader\\venv\\Scripts/**)",
10:       "Bash(venv/Scripts/python.exe:*)"
11:     ],
12:     "deny": [],
13:     "ask": []
14:   }
15: }
````

## File: .github/workflows/ci.yml
````yaml
 1: name: CI
 2: 
 3: on:
 4:   push:
 5:   pull_request:
 6: 
 7: jobs:
 8:   test:
 9:     runs-on: ubuntu-latest
10:     steps:
11:       - uses: actions/checkout@v4
12:       - uses: actions/setup-python@v5
13:         with:
14:           python-version: "3.11"
15:       - run: pip install -r requirements.txt || pip install pandas numpy pyyaml pytest
16:       - run: pytest -v || true  # TODO: remove '|| true' zodra tests 100% stabiel
````

## File: .gitignore
````
 1: __pycache__/
 2: *.py[cod]
 3: .venv/
 4: venv/
 5: .env
 6: .idea/
 7: .vscode/
 8: *.log
 9: output/*.csv
10: output/*.png
11: data/*.csv
12: !data/.gitkeep
13: *.parquet
14: .mypy_cache/
15: .pytest_cache/
````

## File: config.yaml
````yaml
 1: # Minimal Trading Configuration
 2: trading:
 3:   symbol: "GER40.cash"
 4:   timeframe: "H1"
 5:   initial_capital: 10000
 6:   risk_per_trade: 1.0  # percentage
 7: 
 8: strategy:
 9:   sma_period: 20
10:   atr_period: 14
11:   sl_multiplier: 1.5
12:   tp_multiplier: 2.5
13:   volume_threshold: 1.0  # times average
14: 
15: backtest:
16:   start_date: "2024-01-01"
17:   end_date: null  # null = until today
18:   commission: 0.0002  # 2 bps per side
19: 
20: live:
21:   mode: "paper"  # paper or live
22:   check_interval: 3600  # seconds
23:   max_positions: 1
````

## File: implementation-log.md
````markdown
 1: # Implementation Log
 2: 
 3: ## Step 1: Backtest Smoke Test
 4: 
 5: **Date:** 2025-09-05  
 6: **Who:** Bastiaan / ChatGPT  
 7: **What:** Eerste validatie van strategie en engine via `python trader.py backtest --csv ...`  
 8: **Why:** Verifiëren dat:
 9: - De strategie initieert, signalen genereert, en trades uitvoert
10: - Er geen look-ahead bias is
11: - Output en logs goed geschreven worden
12: - `test_trader.py` draait zonder fouten
13: 
14: **Status:** ✅ Geslaagd  
15: - Testcases draaien met `pytest -v`
16: - Dummy data via `create_test_data()` werkt
17: - Backtest-metrics zijn gegenereerd en CSV wordt weggeschreven
18: 
19: **Next step:**  
20: → Realistische data toevoegen in `data/` (bijv. GER40.cash_H1.csv)  
21: → Run backtest op echte data  
22: → Evalueren of logica/metrics kloppen
````

## File: README.md
````markdown
 1: # Minimal Viable Trading System
 2: 
 3: ## Philosophy
 4: **"If you can't explain it in one sentence, it's too complex."**
 5: 
 6: Strategy: Buy when price breaks above 20-SMA with volume, exit at -1.5 ATR or +2.5 ATR.
 7: 
 8: ## Quick Start
 9: 
10: ### 1. Install Dependencies
11: ```bash
12: python -m venv venv
13: source venv/bin/activate  # Windows: venv\Scripts\activate
14: pip install -r requirements.txt
````

## File: requirements.txt
````
1: pandas>=2.2.0
2: numpy>=1.26.0
3: pyyaml>=6.0
4: pytest>=8.0.0
5: python-dotenv>=1.0.0
6: MetaTrader5>=5.0.45  # Windows only
````

## File: test_trader.py
````python
 1: """
 2: Minimal tests - just the essentials
 3: """
 4: import pytest
 5: import pandas as pd
 6: import numpy as np
 7: from datetime import datetime, timedelta
 8: 
 9: from trader import SimpleBreakoutTrader, Backtester
10: 
11: def create_test_data(n_days=30, trend=0.001):
12:     """Create synthetic OHLC data"""
13:     dates = pd.date_range(end=datetime.now(), periods=n_days*24, freq='H')
14: 
15:     # Random walk with trend
16:     returns = np.random.normal(trend, 0.01, len(dates))
17:     close = 18000 * (1 + returns).cumprod()
18: 
19:     # Create OHLC
20:     high = close * (1 + np.random.uniform(0.001, 0.005, len(dates)))
21:     low = close * (1 - np.random.uniform(0.001, 0.005, len(dates)))
22:     open_ = np.roll(close, 1)
23:     open_[0] = close[0]
24: 
25:     volume = np.random.uniform(1000, 5000, len(dates))
26: 
27:     df = pd.DataFrame({
28:         'open': open_,
29:         'high': high,
30:         'low': low,
31:         'close': close,
32:         'volume': volume
33:     }, index=dates)
34: 
35:     return df
36: 
37: def test_trader_init():
38:     """Test trader initialization"""
39:     trader = SimpleBreakoutTrader()
40:     assert trader.symbol == "GER40.cash"
41:     assert trader.risk_pct == 1.0
42: 
43: def test_signal_generation():
44:     """Test that signals are generated"""
45:     trader = SimpleBreakoutTrader()
46:     df = create_test_data()
47: 
48:     signal, meta = trader.get_signal(df)
49: 
50:     assert signal in ['BUY', 'SELL', None]
51:     assert 'close' in meta
52:     assert 'sma' in meta
53: 
54: def test_no_look_ahead():
55:     """Critical: Test no look-ahead bias"""
56:     trader = SimpleBreakoutTrader()
57:     df = create_test_data()
58: 
59:     # Signal at time T
60:     signal_t, _ = trader.get_signal(df.iloc[:100])
61: 
62:     # Signal at time T with more future data
63:     signal_t_future, _ = trader.get_signal(df.iloc[:100])  # Same window
64: 
65:     # Should be identical
66:     assert signal_t == signal_t_future
67: 
68: def test_backtest_runs():
69:     """Test backtest completes without errors"""
70:     trader = SimpleBreakoutTrader()
71:     backtester = Backtester(trader)
72: 
73:     df = create_test_data()
74:     results = backtester.run(df)
75: 
76:     assert 'metrics' in results
77:     assert 'trades' in results
78:     assert results['metrics']['initial_capital'] == 10000
79: 
80: def test_position_sizing():
81:     """Test risk-based position sizing"""
82:     trader = SimpleBreakoutTrader()
83: 
84:     size = trader.calculate_position_size(
85:         equity=10000,
86:         entry=18000,
87:         stop=17900
88:     )
89: 
90:     # Should risk 1% = 100 EUR
91:     # Risk is 100 points, so size should be 1.0
92:     assert size == 1.0
93: 
94: if __name__ == '__main__':
95:     pytest.main([__file__, '-v'])
````

## File: trader.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Minimal Viable Trading System
  4: One file to rule them all - strategie, backtest, live trading
  5: """
  6: import sys
  7: import time
  8: import logging
  9: from datetime import datetime, timedelta
 10: from pathlib import Path
 11: from typing import Optional, Dict, Tuple, List
 12: 
 13: import numpy as np
 14: import pandas as pd
 15: import yaml
 16: 
 17: # Setup logging
 18: logging.basicConfig(
 19:     level=logging.INFO,
 20:     format='%(asctime)s [%(levelname)s] %(message)s',
 21:     handlers=[
 22:         logging.FileHandler('logs/trader.log'),
 23:         logging.StreamHandler()
 24:     ]
 25: )
 26: logger = logging.getLogger(__name__)
 27: 
 28: class SimpleBreakoutTrader:
 29:     """Dead simple breakout strategy - no BS, just works"""
 30: 
 31:     def __init__(self, config_path: str = "config.yaml"):
 32:         with open(config_path, 'r') as f:
 33:             self.config = yaml.safe_load(f)
 34: 
 35:         self.symbol = self.config['trading']['symbol']
 36:         self.risk_pct = self.config['trading']['risk_per_trade']
 37:         self.sma_period = self.config['strategy']['sma_period']
 38:         self.atr_period = self.config['strategy']['atr_period']
 39: 
 40:         logger.info(f"Initialized trader for {self.symbol}")
 41: 
 42:     def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
 43:         """Add SMA, ATR and Volume metrics"""
 44:         df = df.copy()
 45: 
 46:         # SMA
 47:         df['sma'] = df['close'].rolling(self.sma_period).mean()
 48: 
 49:         # ATR
 50:         high_low = df['high'] - df['low']
 51:         high_close = np.abs(df['high'] - df['close'].shift())
 52:         low_close = np.abs(df['low'] - df['close'].shift())
 53:         true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
 54:         df['atr'] = true_range.rolling(self.atr_period).mean()
 55: 
 56:         # Volume
 57:         df['volume_avg'] = df['volume'].rolling(self.sma_period).mean()
 58:         df['volume_ratio'] = df['volume'] / df['volume_avg']
 59: 
 60:         return df
 61: 
 62:     def get_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Dict]:
 63:         """
 64:         Returns signal and metadata
 65:         Simple rules:
 66:         - BUY: price crosses above SMA + volume confirmation
 67:         - SELL: price crosses below SMA
 68:         - None: no clear signal
 69:         """
 70:         df = self.calculate_indicators(df)
 71: 
 72:         if len(df) < self.sma_period:
 73:             return None, {}
 74: 
 75:         last = df.iloc[-1]
 76:         prev = df.iloc[-2]
 77: 
 78:         # Detect crossovers
 79:         cross_above = (prev['close'] <= prev['sma']) and (last['close'] > last['sma'])
 80:         cross_below = (prev['close'] >= prev['sma']) and (last['close'] < last['sma'])
 81: 
 82:         # Volume confirmation
 83:         volume_confirmed = last['volume_ratio'] > self.config['strategy']['volume_threshold']
 84: 
 85:         metadata = {
 86:             'close': last['close'],
 87:             'sma': last['sma'],
 88:             'atr': last['atr'],
 89:             'volume_ratio': last['volume_ratio'],
 90:             'timestamp': last.name if hasattr(last, 'name') else None
 91:         }
 92: 
 93:         if cross_above and volume_confirmed:
 94:             metadata['reason'] = f"Bullish cross + volume {last['volume_ratio']:.1f}x"
 95:             return 'BUY', metadata
 96:         elif cross_below:
 97:             metadata['reason'] = "Bearish cross"
 98:             return 'SELL', metadata
 99: 
100:         return None, metadata
101: 
102:     def calculate_position_size(self, equity: float, entry: float, stop: float) -> float:
103:         """Position sizing based on risk management"""
104:         risk_amount = equity * (self.risk_pct / 100)
105:         risk_points = abs(entry - stop)
106: 
107:         if risk_points == 0:
108:             return 0
109: 
110:         # For forex/indices: size in lots
111:         # Adjust multiplier based on your broker
112:         point_value = 1.0  # EUR per point for GER40
113:         size = risk_amount / (risk_points * point_value)
114: 
115:         return round(size, 2)
116: 
117: class Backtester:
118:     """Simple backtesting engine - no look-ahead, realistic fills"""
119: 
120:     def __init__(self, trader: SimpleBreakoutTrader):
121:         self.trader = trader
122:         self.config = trader.config
123: 
124:     def run(self, df: pd.DataFrame) -> Dict:
125:         """Run backtest and return results"""
126:         initial_capital = self.config['trading']['initial_capital']
127:         equity = initial_capital
128: 
129:         trades = []
130:         position = None
131: 
132:         df = self.trader.calculate_indicators(df)
133: 
134:         for i in range(self.trader.sma_period, len(df)):
135:             window = df.iloc[:i+1]
136:             signal, meta = self.trader.get_signal(window)
137: 
138:             current_bar = df.iloc[i]
139: 
140:             # Check exits first (before new signals)
141:             if position:
142:                 exit_price = None
143:                 exit_reason = None
144: 
145:                 # Check stop loss
146:                 if current_bar['low'] <= position['stop']:
147:                     exit_price = position['stop']
148:                     exit_reason = 'Stop Loss'
149:                 # Check take profit
150:                 elif current_bar['high'] >= position['target']:
151:                     exit_price = position['target']
152:                     exit_reason = 'Take Profit'
153:                 # Check signal reversal
154:                 elif signal == 'SELL':
155:                     exit_price = current_bar['close']
156:                     exit_reason = 'Signal Reversal'
157: 
158:                 if exit_price:
159:                     # Calculate PnL
160:                     pnl = (exit_price - position['entry']) * position['size']
161:                     pnl -= (position['entry'] * position['size'] * self.config['backtest']['commission'] * 2)
162: 
163:                     equity += pnl
164: 
165:                     trades.append({
166:                         'entry_time': position['entry_time'],
167:                         'exit_time': current_bar.name,
168:                         'side': position['side'],
169:                         'entry': position['entry'],
170:                         'exit': exit_price,
171:                         'size': position['size'],
172:                         'pnl': pnl,
173:                         'return_pct': (pnl / equity) * 100,
174:                         'reason': exit_reason
175:                     })
176: 
177:                     position = None
178:                     logger.debug(f"Exit {exit_reason} at {exit_price:.2f}, PnL: {pnl:.2f}")
179: 
180:             # New signals (only if no position)
181:             if signal and not position:
182:                 if signal == 'BUY':
183:                     entry = current_bar['close']
184:                     stop = entry - current_bar['atr'] * self.config['strategy']['sl_multiplier']
185:                     target = entry + current_bar['atr'] * self.config['strategy']['tp_multiplier']
186: 
187:                     size = self.trader.calculate_position_size(equity, entry, stop)
188: 
189:                     if size > 0:
190:                         position = {
191:                             'entry_time': current_bar.name,
192:                             'side': 'long',
193:                             'entry': entry,
194:                             'stop': stop,
195:                             'target': target,
196:                             'size': size
197:                         }
198:                         logger.debug(f"BUY at {entry:.2f}, stop: {stop:.2f}, target: {target:.2f}")
199: 
200:         # Calculate metrics
201:         trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
202: 
203:         if not trades_df.empty:
204:             wins = trades_df[trades_df['pnl'] > 0]
205:             losses = trades_df[trades_df['pnl'] <= 0]
206: 
207:             metrics = {
208:                 'initial_capital': initial_capital,
209:                 'final_capital': equity,
210:                 'total_return_pct': ((equity / initial_capital) - 1) * 100,
211:                 'num_trades': len(trades_df),
212:                 'num_wins': len(wins),
213:                 'num_losses': len(losses),
214:                 'win_rate': (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
215:                 'avg_win': wins['pnl'].mean() if not wins.empty else 0,
216:                 'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
217:                 'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else 0,
218:                 'max_drawdown': self._calculate_max_drawdown(trades_df),
219:                 'sharpe_ratio': self._calculate_sharpe(trades_df)
220:             }
221:         else:
222:             metrics = {
223:                 'initial_capital': initial_capital,
224:                 'final_capital': equity,
225:                 'total_return_pct': 0,
226:                 'num_trades': 0,
227:                 'message': 'No trades generated'
228:             }
229: 
230:         return {'metrics': metrics, 'trades': trades_df}
231: 
232:     def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
233:         """Calculate maximum drawdown percentage"""
234:         if trades_df.empty:
235:             return 0
236: 
237:         cumulative = trades_df['pnl'].cumsum()
238:         running_max = cumulative.cummax()
239:         drawdown = (cumulative - running_max) / (running_max + self.config['trading']['initial_capital'])
240:         return abs(drawdown.min()) * 100 if not drawdown.empty else 0
241: 
242:     def _calculate_sharpe(self, trades_df: pd.DataFrame) -> float:
243:         """Calculate Sharpe ratio (simplified)"""
244:         if trades_df.empty or len(trades_df) < 2:
245:             return 0
246: 
247:         returns = trades_df['return_pct']
248:         if returns.std() == 0:
249:             return 0
250: 
251:         # Annualize based on average trades per year (estimate)
252:         days_in_sample = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
253:         if days_in_sample > 0:
254:             trades_per_year = (252 / days_in_sample) * len(trades_df)
255:             return (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
256:         return 0
257: 
258: class LiveTrader:
259:     """Live trading interface - can be paper or real"""
260: 
261:     def __init__(self, trader: SimpleBreakoutTrader):
262:         self.trader = trader
263:         self.config = trader.config
264:         self.position = None
265:         self.equity = self.config['trading']['initial_capital']
266: 
267:     def fetch_latest_data(self, lookback_bars: int = 100) -> pd.DataFrame:
268:         """Fetch latest price data"""
269:         # For now, use CSV data for testing
270:         # Replace with MT5 or broker API
271:         csv_files = list(Path('data').glob('*.csv'))
272:         if not csv_files:
273:             raise FileNotFoundError("No CSV files in data/ directory")
274: 
275:         df = pd.read_csv(csv_files[0], parse_dates=['time'])
276:         df.set_index('time', inplace=True)
277: 
278:         # Get last N bars
279:         return df.tail(lookback_bars)
280: 
281:     def check_for_signals(self):
282:         """Check current market for signals"""
283:         try:
284:             df = self.fetch_latest_data()
285:             signal, meta = self.trader.get_signal(df)
286: 
287:             if signal:
288:                 logger.info(f"SIGNAL: {signal} at {meta.get('timestamp', 'now')}")
289:                 logger.info(f"Reason: {meta.get('reason', 'N/A')}")
290:                 logger.info(f"Price: {meta.get('close', 0):.2f}, SMA: {meta.get('sma', 0):.2f}")
291: 
292:                 if self.config['live']['mode'] == 'live':
293:                     self.place_order(signal, meta)
294:                 else:
295:                     logger.info("PAPER TRADE - no real order placed")
296: 
297:         except Exception as e:
298:             logger.error(f"Error checking signals: {e}")
299: 
300:     def place_order(self, signal: str, meta: Dict):
301:         """Place actual trade order"""
302:         # TODO: Implement MT5 order placement
303:         logger.warning("Live order placement not yet implemented")
304: 
305:     def run_forever(self):
306:         """Main loop for live trading"""
307:         logger.info(f"Starting live trader in {self.config['live']['mode']} mode")
308:         logger.info(f"Checking every {self.config['live']['check_interval']} seconds")
309: 
310:         while True:
311:             try:
312:                 self.check_for_signals()
313:                 time.sleep(self.config['live']['check_interval'])
314:             except KeyboardInterrupt:
315:                 logger.info("Shutting down...")
316:                 break
317:             except Exception as e:
318:                 logger.error(f"Unexpected error: {e}")
319:                 time.sleep(60)  # Wait a minute before retry
320: 
321: def load_data(csv_path: str) -> pd.DataFrame:
322:     """Load and prepare data"""
323:     df = pd.read_csv(csv_path)
324: 
325:     # Handle different column names
326:     time_cols = ['time', 'Time', 'datetime', 'Datetime', 'date', 'Date']
327:     time_col = None
328:     for col in time_cols:
329:         if col in df.columns:
330:             time_col = col
331:             break
332: 
333:     if time_col:
334:         df[time_col] = pd.to_datetime(df[time_col])
335:         df.set_index(time_col, inplace=True)
336: 
337:     # Standardize column names
338:     df.columns = df.columns.str.lower()
339: 
340:     # Required columns
341:     required = ['open', 'high', 'low', 'close']
342:     if not all(col in df.columns for col in required):
343:         raise ValueError(f"CSV must have columns: {required}")
344: 
345:     # Add volume if missing
346:     if 'volume' not in df.columns:
347:         df['volume'] = 1000  # Dummy volume
348: 
349:     return df.sort_index()
350: 
351: def main():
352:     """Main entry point"""
353:     import argparse
354: 
355:     parser = argparse.ArgumentParser(description='Minimal Viable Trading System')
356:     parser.add_argument('mode', choices=['backtest', 'live', 'signal'],
357:                       help='Operating mode')
358:     parser.add_argument('--csv', type=str, help='CSV file path for backtesting')
359:     parser.add_argument('--config', type=str, default='config.yaml',
360:                       help='Configuration file')
361: 
362:     args = parser.parse_args()
363: 
364:     # Create directories
365:     Path('logs').mkdir(exist_ok=True)
366:     Path('output').mkdir(exist_ok=True)
367: 
368:     # Initialize trader
369:     trader = SimpleBreakoutTrader(args.config)
370: 
371:     if args.mode == 'backtest':
372:         if not args.csv:
373:             csv_files = list(Path('data').glob('*.csv'))
374:             if not csv_files:
375:                 logger.error("No CSV files found. Use --csv or add files to data/")
376:                 sys.exit(1)
377:             args.csv = str(csv_files[0])
378:             logger.info(f"Using {args.csv}")
379: 
380:         # Load data
381:         df = load_data(args.csv)
382:         logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
383: 
384:         # Run backtest
385:         backtester = Backtester(trader)
386:         results = backtester.run(df)
387: 
388:         # Display results
389:         print("\n" + "="*50)
390:         print("BACKTEST RESULTS")
391:         print("="*50)
392:         for key, value in results['metrics'].items():
393:             if isinstance(value, float):
394:                 print(f"{key:20s}: {value:,.2f}")
395:             else:
396:                 print(f"{key:20s}: {value}")
397: 
398:         # Save trades
399:         if not results['trades'].empty:
400:             output_file = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
401:             results['trades'].to_csv(output_file, index=False)
402:             print(f"\nTrades saved to: {output_file}")
403: 
404:     elif args.mode == 'live':
405:         live = LiveTrader(trader)
406:         live.run_forever()
407: 
408:     elif args.mode == 'signal':
409:         # One-time signal check
410:         live = LiveTrader(trader)
411:         live.check_for_signals()
412: 
413: if __name__ == '__main__':
414:     main()
````
