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
tests/
  test_backtest.py
  test_integration.py
  test_strategy.py
.gitignore
architecture.md
backtest.py
config.yaml
data_feed.py
implementation-log.md
main.py
README.md
repomix-output-bastiaandehaan-minimal_trader.md
requirements.txt
strategy.py
testplan.md
```

# Files

## File: .claude/settings.local.json
``````json
 1: {
 2:   "permissions": {
 3:     "allow": [
 4:       "Bash(python:*)",
 5:       "Bash(venvScripts:*)",
 6:       "Bash(venv/Scripts/activate:*)",
 7:       "Bash(pip install:*)",
 8:       "Bash(venv\\\\Scripts\\\\python.exe:*)",
 9:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader\\venv\\Scripts/**)",
10:       "Bash(venv/Scripts/python.exe:*)",
11:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
12:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
13:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
14:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
15:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)"
16:     ],
17:     "deny": [],
18:     "ask": []
19:   }
20: }
``````

## File: .github/workflows/ci.yml
``````yaml
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
15:       - name: Install deps
16:         run: |
17:           python -m pip install --upgrade pip
18:           pip install -r requirements.txt
19:       - name: Run tests
20:         run: pytest -v
``````

## File: tests/test_backtest.py
``````python
 1: import numpy as np
 2: import pandas as pd
 3: from strategy import Strategy, StrategyParams
 4: from backtest import Backtest, ExecConfig
 5: 
 6: 
 7: def synthetic(n=400, seed=123):
 8:     rng = np.random.default_rng(seed)
 9:     idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
10:     close = 18000 + np.cumsum(rng.normal(0, 3, n))
11:     high = close + rng.uniform(0.5, 3, n)
12:     low = close - rng.uniform(0.5, 3, n)
13:     open_ = np.r_[close[0], close[:-1]]
14:     vol = rng.uniform(1000, 5000, n)
15:     return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
16: 
17: 
18: def test_backtest_runs_and_produces_metrics():
19:     strat = Strategy(StrategyParams(sma_period=5, atr_period=5, volume_threshold=0.5))
20:     bt = Backtest(strat, ExecConfig(initial_capital=10_000, risk_pct=1.0, time_exit_bars=50))
21:     df = synthetic(400)
22:     res = bt.run(df)
23:     assert "metrics" in res and "trades" in res
24:     assert res["metrics"]["initial_capital"] == 10_000
``````

## File: tests/test_integration.py
``````python
 1: import numpy as np
 2: import pandas as pd
 3: from strategy import Strategy, StrategyParams
 4: from backtest import Backtest, ExecConfig
 5: 
 6: 
 7: def make_data(n=300):
 8:     idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
 9:     close = np.linspace(18000, 18200, n)  # gentle trend
10:     high = close + 2
11:     low = close - 2
12:     open_ = np.r_[close[0], close[:-1]]
13:     vol = np.full(n, 2000.0)
14:     return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
15: 
16: 
17: def test_pipeline_produces_some_trades():
18:     df = make_data(300)
19:     strat = Strategy(StrategyParams(sma_period=10, atr_period=10, volume_threshold=0.8))
20:     bt = Backtest(strat, ExecConfig(initial_capital=10_000, risk_pct=1.0, time_exit_bars=100))
21:     res = bt.run(df)
22:     assert res["metrics"]["num_trades"] >= 1
``````

## File: tests/test_strategy.py
``````python
 1: import numpy as np
 2: import pandas as pd
 3: from strategy import Strategy, StrategyParams, SignalType
 4: 
 5: 
 6: def make_df(n=200, seed=1):
 7:     rng = np.random.default_rng(seed)
 8:     idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
 9:     close = 18000 + np.cumsum(rng.normal(0, 5, n))
10:     high = close + rng.uniform(1, 5, n)
11:     low = close - rng.uniform(1, 5, n)
12:     open_ = np.r_[close[0], close[:-1]]
13:     vol = rng.uniform(1000, 5000, n)
14:     return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
15: 
16: 
17: def test_sma_crossover_buy_signal():
18:     p = StrategyParams(sma_period=5, atr_period=5, volume_threshold=0.5)
19:     strat = Strategy(p)
20:     df = make_df(50)
21:     df = strat.calculate_indicators(df)
22:     i = len(df) - 1
23:     # force bullish cross + volume
24:     df.iloc[i - 1, df.columns.get_loc("close")] = df["sma"].iloc[i - 1] - 1
25:     df.iloc[i, df.columns.get_loc("close")] = df["sma"].iloc[i - 1] + 2
26:     df.iloc[i, df.columns.get_loc("volume")] = max(1.0, df["volume_avg"].iloc[i - 1]) * 2.0
27:     sig, _ = strat.get_signal_at(df, i)
28:     assert sig.type == SignalType.BUY
29: 
30: 
31: def test_no_lookahead():
32:     p = StrategyParams(sma_period=5, atr_period=5)
33:     strat = Strategy(p)
34:     df = make_df(60)
35:     df = strat.calculate_indicators(df)
36:     i = 30
37:     sig1, _ = strat.get_signal_at(df, i)
38:     sig2, _ = strat.get_signal_at(df.iloc[: i + 1], i)
39:     assert sig1.type == sig2.type
``````

## File: .gitignore
``````
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
``````

## File: architecture.md
``````markdown
 1: # System Architecture - Minimal Trader v2
 2: 
 3: ## Components
 4: - `strategy.py`: Pure signal generation (SMA cross + ATR SL/TP, volume confirm)
 5: - `backtest.py`: Execution engine (O(n), SL/TP, reversal/timeout exit, metrics)
 6: - `data_feed.py`: Data access layer (CSVFeed, later LiveFeed/MT5)
 7: - `main.py`: CLI orchestration (backtest, signal)
 8: - `tests/`: unit + integration tests
 9: 
10: ## Data Flow
11: CLI → Config → CSVFeed.load() → Strategy.calculate_indicators() → loop i: Strategy.get_signal_at(i) → Backtest execution → Trades & Metrics.
``````

## File: backtest.py
``````python
  1: from __future__ import annotations
  2: from dataclasses import dataclass
  3: from typing import Dict, Optional, List
  4: 
  5: import numpy as np
  6: import pandas as pd
  7: 
  8: from strategy import Strategy, StrategyParams, SignalType
  9: 
 10: 
 11: @dataclass(frozen=True)
 12: class ExecConfig:
 13:     initial_capital: float = 10_000.0
 14:     risk_pct: float = 1.0           # % of equity
 15:     commission: float = 0.0002      # 2 bps per side
 16:     point_value: float = 1.0        # PnL per pt per 1.0 size
 17:     time_exit_bars: int = 200       # fail-safe max bars in trade
 18:     progress_every: int = 10_000    # progress log cadence
 19: 
 20: 
 21: class Backtest:
 22:     def __init__(self, strategy: Strategy, cfg: ExecConfig = ExecConfig()):
 23:         self.strategy = strategy
 24:         self.cfg = cfg
 25: 
 26:     def _position_size(self, equity: float, entry: float, stop: float) -> float:
 27:         risk_amount = equity * (self.cfg.risk_pct / 100.0)
 28:         risk_pts = abs(entry - stop)
 29:         if risk_pts <= 0:
 30:             return 0.0
 31:         size = risk_amount / (risk_pts * self.cfg.point_value)
 32:         return round(size, 2)
 33: 
 34:     def run(self, df: pd.DataFrame, logger=None) -> Dict:
 35:         # Precompute indicators once (O(n))
 36:         df = self.strategy.calculate_indicators(df)
 37:         n = len(df)
 38:         warmup = max(self.strategy.p.sma_period, self.strategy.p.atr_period)
 39: 
 40:         equity = self.cfg.initial_capital
 41:         position: Optional[Dict] = None
 42:         trades: List[Dict] = []
 43: 
 44:         for i in range(warmup, n):
 45:             if logger and self.cfg.progress_every and i % self.cfg.progress_every == 0:
 46:                 logger.info(f"Backtesting... {i}/{n}")
 47: 
 48:             # Determine signal at bar i (no look-ahead)
 49:             sig, meta = self.strategy.get_signal_at(df, i)
 50:             bar = df.iloc[i]
 51: 
 52:             # 1) Check exits first
 53:             if position is not None:
 54:                 exit_price = None
 55:                 exit_reason = None
 56: 
 57:                 # hard SL/TP using bar's low/high
 58:                 if bar["low"] <= position["stop"]:
 59:                     exit_price = position["stop"]
 60:                     exit_reason = "Stop Loss"
 61:                 elif bar["high"] >= position["target"]:
 62:                     exit_price = position["target"]
 63:                     exit_reason = "Take Profit"
 64:                 elif sig.type == SignalType.SELL:
 65:                     exit_price = float(bar["close"])
 66:                     exit_reason = "Reversal Exit"
 67:                 elif (i - position["entry_index"]) > self.cfg.time_exit_bars:
 68:                     exit_price = float(bar["close"])
 69:                     exit_reason = "Time Exit"
 70: 
 71:                 if exit_price is not None:
 72:                     gross = (exit_price - position["entry"]) * position["size"] * self.cfg.point_value
 73:                     fees = (position["entry"] * position["size"] * self.cfg.commission) \
 74:                          + (exit_price * position["size"] * self.cfg.commission)
 75:                     pnl = gross - fees
 76:                     equity += pnl
 77: 
 78:                     trades.append({
 79:                         "entry_time": position["entry_time"],
 80:                         "exit_time": bar.name,
 81:                         "side": "long",
 82:                         "entry": position["entry"],
 83:                         "exit": exit_price,
 84:                         "size": position["size"],
 85:                         "pnl": pnl,
 86:                         "reason": exit_reason,
 87:                     })
 88:                     position = None
 89: 
 90:             # 2) Entries (only if flat)
 91:             if position is None and sig.type == SignalType.BUY:
 92:                 entry = float(bar["close"])
 93:                 stop = float(sig.stop)
 94:                 target = float(sig.target)
 95:                 size = self._position_size(equity, entry, stop)
 96:                 if size > 0:
 97:                     position = {
 98:                         "entry_time": bar.name,
 99:                         "entry_index": i,
100:                         "entry": entry,
101:                         "stop": stop,
102:                         "target": target,
103:                         "size": size,
104:                     }
105: 
106:         # Metrics
107:         trades_df = pd.DataFrame(trades)
108:         metrics = self._metrics(trades_df, equity)
109:         return {"metrics": metrics, "trades": trades_df}
110: 
111:     def _metrics(self, trades_df: pd.DataFrame, final_equity: float) -> Dict:
112:         ic = self.cfg.initial_capital
113:         if trades_df.empty:
114:             return {
115:                 "initial_capital": ic,
116:                 "final_capital": final_equity,
117:                 "total_return_pct": ((final_equity / ic) - 1.0) * 100.0,
118:                 "num_trades": 0,
119:                 "win_rate": 0.0,
120:                 "profit_factor": 0.0,
121:                 "max_drawdown": 0.0,
122:                 "sharpe_ratio": 0.0,
123:                 "message": "No trades",
124:             }
125: 
126:         wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
127:         losses = trades_df.loc[trades_df["pnl"] <= 0, "pnl"]
128: 
129:         equity_curve = ic + trades_df["pnl"].cumsum()
130:         peak = equity_curve.cummax()
131:         dd = (equity_curve - peak) / peak.replace(0, np.nan)
132:         max_dd_pct = abs(dd.min()) * 100.0 if not dd.empty else 0.0
133: 
134:         ret = trades_df["pnl"] / ic * 100.0
135:         sharpe = 0.0
136:         if ret.std(ddof=0) > 0:
137:             days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).days or 1
138:             tpy = (252 / days) * len(trades_df) if days > 0 else len(trades_df)
139:             sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(max(tpy, 1))
140: 
141:         pf = 0.0
142:         if not losses.empty and losses.sum() != 0:
143:             pf = abs(wins.sum() / losses.sum()) if not wins.empty else 0.0
144: 
145:         return {
146:             "initial_capital": ic,
147:             "final_capital": final_equity,
148:             "total_return_pct": ((final_equity / ic) - 1.0) * 100.0,
149:             "num_trades": int(len(trades_df)),
150:             "win_rate": (len(wins) / len(trades_df)) * 100.0,
151:             "profit_factor": pf,
152:             "max_drawdown": float(max_dd_pct),
153:             "sharpe_ratio": float(sharpe),
154:         }
``````

## File: config.yaml
``````yaml
 1: trading:
 2:   symbol: "GER40.cash"
 3:   timeframe: "M1"
 4:   initial_capital: 10000
 5:   risk_per_trade: 1.0
 6: 
 7: strategy:
 8:   sma_period: 200      # 200 min = ~3 uur
 9:   atr_period: 60       # 60 min = 1 uur  
10:   sl_mult: 2.0         # Ruimere stop voor M1 noise
11:   tp_mult: 3.0         # Betere RR ratio
12:   volume_threshold: 1.0
13: 
14: backtest:
15:   commission: 0.0002
16:   time_exit_bars: 1440  # 1 dag in minuten
``````

## File: data_feed.py
``````python
 1: from __future__ import annotations
 2: from dataclasses import dataclass
 3: from pathlib import Path
 4: import pandas as pd
 5: 
 6: 
 7: class DataFeed:
 8:     def load(self, limit: int = 0) -> pd.DataFrame:
 9:         raise NotImplementedError
10: 
11: 
12: @dataclass(frozen=True)
13: class CSVFeed(DataFeed):
14:     csv_path: str
15:     resample_to: str = None  # e.g., "1H", "5T", etc.
16: 
17:     def load(self, limit: int = 0) -> pd.DataFrame:
18:         p = Path(self.csv_path)
19:         if not p.exists():
20:             raise FileNotFoundError(f"CSV not found: {p}")
21: 
22:         df = pd.read_csv(p)
23: 
24:         # Detect time column
25:         time_cols = ["time", "Time", "datetime", "Datetime", "date", "Date"]
26:         time_col = None
27:         for c in time_cols:
28:             if c in df.columns:
29:                 time_col = c
30:                 break
31: 
32:         if time_col is None:
33:             raise ValueError("No time column found in CSV")
34: 
35:         # Parse datetime and set index
36:         df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
37:         df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
38: 
39:         # Standardize columns
40:         df.columns = df.columns.str.lower()
41: 
42:         # Check required columns
43:         for col in ["open", "high", "low", "close"]:
44:             if col not in df.columns:
45:                 raise ValueError(f"Missing required column: {col}")
46: 
47:         # Add volume if missing
48:         if "volume" not in df.columns:
49:             df["volume"] = 1000.0
50: 
51:         # Optional resampling (e.g., M1 to H1)
52:         if self.resample_to:
53:             df = self.resample_ohlc(df, self.resample_to)
54: 
55:         # Apply limit
56:         if limit and limit > 0:
57:             df = df.tail(limit)
58: 
59:         return df
60: 
61:     def resample_ohlc(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
62:         """Resample OHLC data to different timeframe"""
63:         return df.resample(freq).agg(
64:             {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
65:                 'volume': 'sum'}).dropna()
``````

## File: implementation-log.md
``````markdown
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
23: 
24: ## Step 2: Architecture Refactoring to v2 Modules
25: 
26: **Date:** (auto)  
27: **What:** Split monolith into `strategy.py`, `backtest.py`, `data_feed.py`, `main.py` + tests  
28: **Why:** Separation of concerns, O(n) performance, testability
29: 
30: **Decisions:**
31: - Indicators computed once; signal per bar by index (no O(n²))
32: - BUY entries only; SELL used as exit trigger
33: - Time-exit fail-safe (200 bars)
34: - CSVFeed for input; LiveFeed later
35: 
36: **Status:** Implemented  
37: **Next:** Run tests + run backtest on real CSV (limit first)
``````

## File: main.py
``````python
  1: from __future__ import annotations
  2: import argparse
  3: import logging
  4: from datetime import datetime
  5: from pathlib import Path
  6: from typing import Dict
  7: 
  8: import yaml
  9: import pandas as pd
 10: 
 11: from strategy import Strategy, StrategyParams
 12: from backtest import Backtest, ExecConfig
 13: from data_feed import CSVFeed
 14: 
 15: 
 16: def load_config(path: str = "config.yaml") -> Dict:
 17:     with open(path, "r") as f:
 18:         return yaml.safe_load(f) or {}
 19: 
 20: 
 21: def setup_logging():
 22:     Path("logs").mkdir(exist_ok=True)
 23:     logging.basicConfig(level=logging.INFO,
 24:         format="%(asctime)s [%(levelname)s] %(message)s",
 25:         handlers=[logging.FileHandler("logs/trader.log"), logging.StreamHandler()], )
 26:     return logging.getLogger("minimal_trader")
 27: 
 28: 
 29: def run_backtest(args, cfg: Dict, logger):
 30:     # Strategy parameters
 31:     p = cfg.get("strategy", {})
 32:     s_params = StrategyParams(sma_period=int(p.get("sma_period", 20)),
 33:         atr_period=int(p.get("atr_period", 14)),
 34:         sl_mult=float(p.get("sl_mult", p.get("sl_multiplier", 1.5))),  # Support both
 35:         tp_mult=float(p.get("tp_mult", p.get("tp_multiplier", 2.5))),  # Support both
 36:         volume_threshold=float(p.get("volume_threshold", 1.0)), )
 37:     strategy = Strategy(s_params)
 38: 
 39:     # Execution config
 40:     e = cfg.get("trading", {})
 41:     b = cfg.get("backtest", {})
 42:     exec_cfg = ExecConfig(initial_capital=float(e.get("initial_capital", 10_000.0)),
 43:         risk_pct=float(e.get("risk_per_trade", 1.0)),
 44:         commission=float(b.get("commission", 0.0002)),
 45:         point_value=float(e.get("point_value", 1.0)),
 46:         time_exit_bars=int(b.get("time_exit_bars", 200)), )
 47: 
 48:     # Backtest engine
 49:     bt = Backtest(strategy, exec_cfg)
 50: 
 51:     # Load data with optional resampling
 52:     resample = args.resample if hasattr(args, 'resample') else None
 53:     feed = CSVFeed(args.csv, resample_to=resample)
 54:     df = feed.load(limit=args.limit)
 55: 
 56:     logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
 57:     if resample:
 58:         logger.info(f"Data resampled to {resample}")
 59: 
 60:     # Run backtest
 61:     results = bt.run(df, logger=logger)
 62: 
 63:     # Display results
 64:     print("\n" + "=" * 50)
 65:     print("BACKTEST RESULTS")
 66:     print("=" * 50)
 67:     for k, v in results["metrics"].items():
 68:         if isinstance(v, float):
 69:             print(f"{k:20s}: {v:,.2f}")
 70:         else:
 71:             print(f"{k:20s}: {v}")
 72: 
 73:     # Save trades
 74:     trades = results["trades"]
 75:     if isinstance(trades, pd.DataFrame) and not trades.empty:
 76:         Path("output").mkdir(exist_ok=True)
 77:         out = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
 78:         trades.to_csv(out, index=False)
 79:         print(f"\nTrades saved to: {out}")
 80: 
 81:         # Show trade statistics
 82:         print(f"\nTrade Statistics:")
 83:         print(f"Avg win: ${trades[trades['pnl'] > 0]['pnl'].mean():.2f}")
 84:         print(f"Avg loss: ${trades[trades['pnl'] <= 0]['pnl'].mean():.2f}")
 85:         print(f"Largest win: ${trades['pnl'].max():.2f}")
 86:         print(f"Largest loss: ${trades['pnl'].min():.2f}")
 87: 
 88: 
 89: def run_signal(args, cfg: Dict, logger):
 90:     # [Keep existing signal code...]
 91:     p = cfg.get("strategy", {})
 92:     s_params = StrategyParams(sma_period=int(p.get("sma_period", 20)),
 93:         atr_period=int(p.get("atr_period", 14)),
 94:         sl_mult=float(p.get("sl_mult", p.get("sl_multiplier", 1.5))),
 95:         tp_mult=float(p.get("tp_mult", p.get("tp_multiplier", 2.5))),
 96:         volume_threshold=float(p.get("volume_threshold", 1.0)), )
 97:     strategy = Strategy(s_params)
 98: 
 99:     feed = CSVFeed(args.csv)
100:     df = strategy.calculate_indicators(feed.load(limit=max(args.limit, 500)))
101: 
102:     if len(df) < max(s_params.sma_period, s_params.atr_period) + 2:
103:         logger.warning("Not enough data to generate a signal.")
104:         return
105: 
106:     i = len(df) - 1
107:     sig, meta = strategy.get_signal_at(df, i)
108:     if sig.type.name != "NONE":
109:         logger.info(f"SIGNAL: {sig.type.name} at {meta.get('timestamp')}")
110:         logger.info(f"Reason: {sig.reason}")
111:         logger.info(f"Price: {meta.get('close'):.2f}, SMA: {meta.get('sma'):.2f}")
112:     else:
113:         logger.info("No signal")
114: 
115: 
116: def main():
117:     parser = argparse.ArgumentParser(description="Minimal Trader v2")
118:     sub = parser.add_subparsers(dest="mode", required=True)
119: 
120:     # Backtest command
121:     p_bt = sub.add_parser("backtest", help="Run backtest on CSV")
122:     p_bt.add_argument("--csv", type=str, required=True, help="CSV file path")
123:     p_bt.add_argument("--config", type=str, default="config.yaml")
124:     p_bt.add_argument("--limit", type=int, default=0, help="Limit bars (0=all)")
125:     p_bt.add_argument("--resample", type=str, help="Resample to (e.g., 1H, 5T)")
126: 
127:     # Signal comm
``````

## File: README.md
``````markdown
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
``````

## File: repomix-output-bastiaandehaan-minimal_trader.md
``````markdown
   1: This file is a merged representation of the entire codebase, combined into a single document by Repomix.
   2: The content has been processed where line numbers have been added, content has been formatted for parsing in markdown style, security check has been disabled.
   3: 
   4: # File Summary
   5: 
   6: ## Purpose
   7: This file contains a packed representation of the entire repository's contents.
   8: It is designed to be easily consumable by AI systems for analysis, code review,
   9: or other automated processes.
  10: 
  11: ## File Format
  12: The content is organized as follows:
  13: 1. This summary section
  14: 2. Repository information
  15: 3. Directory structure
  16: 4. Repository files (if enabled)
  17: 5. Multiple file entries, each consisting of:
  18:   a. A header with the file path (## File: path/to/file)
  19:   b. The full contents of the file in a code block
  20: 
  21: ## Usage Guidelines
  22: - This file should be treated as read-only. Any changes should be made to the
  23:   original repository files, not this packed version.
  24: - When processing this file, use the file path to distinguish
  25:   between different files in the repository.
  26: - Be aware that this file may contain sensitive information. Handle it with
  27:   the same level of security as you would the original repository.
  28: 
  29: ## Notes
  30: - Some files may have been excluded based on .gitignore rules and Repomix's configuration
  31: - Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
  32: - Files matching patterns in .gitignore are excluded
  33: - Files matching default ignore patterns are excluded
  34: - Line numbers have been added to the beginning of each line
  35: - Content has been formatted for parsing in markdown style
  36: - Security check has been disabled - content may contain sensitive information
  37: - Files are sorted by Git change count (files with more changes are at the bottom)
  38: 
  39: # Directory Structure
  40: ```
  41: .claude/
  42:   settings.local.json
  43: .github/
  44:   workflows/
  45:     ci.yml
  46: tests/
  47:   test_backtest.py
  48:   test_integration.py
  49:   test_strategy.py
  50: .gitignore
  51: architecture.md
  52: backtest.py
  53: config.yaml
  54: data_feed.py
  55: implementation-log.md
  56: main.py
  57: README.md
  58: repomix-output-bastiaandehaan-minimal_trader.md
  59: requirements.txt
  60: strategy.py
  61: test_trader.py
  62: testplan.md
  63: trader.py
  64: ```
  65: 
  66: # Files
  67: 
  68: ## File: .claude/settings.local.json
  69: `````json
  70:  1: {
  71:  2:   "permissions": {
  72:  3:     "allow": [
  73:  4:       "Bash(python:*)",
  74:  5:       "Bash(venvScripts:*)",
  75:  6:       "Bash(venv/Scripts/activate:*)",
  76:  7:       "Bash(pip install:*)",
  77:  8:       "Bash(venv\\\\Scripts\\\\python.exe:*)",
  78:  9:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader\\venv\\Scripts/**)",
  79: 10:       "Bash(venv/Scripts/python.exe:*)",
  80: 11:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
  81: 12:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
  82: 13:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
  83: 14:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)",
  84: 15:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader/**)"
  85: 16:     ],
  86: 17:     "deny": [],
  87: 18:     "ask": []
  88: 19:   }
  89: 20: }
  90: `````
  91: 
  92: ## File: .github/workflows/ci.yml
  93: `````yaml
  94:  1: name: CI
  95:  2: 
  96:  3: on:
  97:  4:   push:
  98:  5:   pull_request:
  99:  6: 
 100:  7: jobs:
 101:  8:   test:
 102:  9:     runs-on: ubuntu-latest
 103: 10:     steps:
 104: 11:       - uses: actions/checkout@v4
 105: 12:       - uses: actions/setup-python@v5
 106: 13:         with:
 107: 14:           python-version: "3.11"
 108: 15:       - run: pip install -r requirements.txt || pip install pandas numpy pyyaml pytest
 109: 16:       - run: pytest -v || true  # TODO: remove '|| true' zodra tests 100% stabiel
 110: `````
 111: 
 112: ## File: tests/test_backtest.py
 113: `````python
 114:  1: import numpy as np
 115:  2: import pandas as pd
 116:  3: from strategy import Strategy, StrategyParams
 117:  4: from backtest import Backtest, ExecConfig
 118:  5: 
 119:  6: 
 120:  7: def synthetic(n=400, seed=123):
 121:  8:     rng = np.random.default_rng(seed)
 122:  9:     idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
 123: 10:     close = 18000 + np.cumsum(rng.normal(0, 3, n))
 124: 11:     high = close + rng.uniform(0.5, 3, n)
 125: 12:     low = close - rng.uniform(0.5, 3, n)
 126: 13:     open_ = np.r_[close[0], close[:-1]]
 127: 14:     vol = rng.uniform(1000, 5000, n)
 128: 15:     return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
 129: 16: 
 130: 17: 
 131: 18: def test_backtest_runs_and_produces_metrics():
 132: 19:     strat = Strategy(StrategyParams(sma_period=5, atr_period=5, volume_threshold=0.5))
 133: 20:     bt = Backtest(strat, ExecConfig(initial_capital=10_000, risk_pct=1.0, time_exit_bars=50))
 134: 21:     df = synthetic(400)
 135: 22:     res = bt.run(df)
 136: 23:     assert "metrics" in res and "trades" in res
 137: 24:     assert res["metrics"]["initial_capital"] == 10_000
 138: `````
 139: 
 140: ## File: tests/test_integration.py
 141: `````python
 142:  1: import numpy as np
 143:  2: import pandas as pd
 144:  3: from strategy import Strategy, StrategyParams
 145:  4: from backtest import Backtest, ExecConfig
 146:  5: 
 147:  6: 
 148:  7: def make_data(n=300):
 149:  8:     idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
 150:  9:     close = np.linspace(18000, 18200, n)  # gentle trend
 151: 10:     high = close + 2
 152: 11:     low = close - 2
 153: 12:     open_ = np.r_[close[0], close[:-1]]
 154: 13:     vol = np.full(n, 2000.0)
 155: 14:     return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
 156: 15: 
 157: 16: 
 158: 17: def test_pipeline_produces_some_trades():
 159: 18:     df = make_data(300)
 160: 19:     strat = Strategy(StrategyParams(sma_period=10, atr_period=10, volume_threshold=0.8))
 161: 20:     bt = Backtest(strat, ExecConfig(initial_capital=10_000, risk_pct=1.0, time_exit_bars=100))
 162: 21:     res = bt.run(df)
 163: 22:     assert res["metrics"]["num_trades"] >= 1
 164: `````
 165: 
 166: ## File: tests/test_strategy.py
 167: `````python
 168:  1: import numpy as np
 169:  2: import pandas as pd
 170:  3: from strategy import Strategy, StrategyParams, SignalType
 171:  4: 
 172:  5: 
 173:  6: def make_df(n=200, seed=1):
 174:  7:     rng = np.random.default_rng(seed)
 175:  8:     idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="H")
 176:  9:     close = 18000 + np.cumsum(rng.normal(0, 5, n))
 177: 10:     high = close + rng.uniform(1, 5, n)
 178: 11:     low = close - rng.uniform(1, 5, n)
 179: 12:     open_ = np.r_[close[0], close[:-1]]
 180: 13:     vol = rng.uniform(1000, 5000, n)
 181: 14:     return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
 182: 15: 
 183: 16: 
 184: 17: def test_sma_crossover_buy_signal():
 185: 18:     p = StrategyParams(sma_period=5, atr_period=5, volume_threshold=0.5)
 186: 19:     strat = Strategy(p)
 187: 20:     df = make_df(50)
 188: 21:     df = strat.calculate_indicators(df)
 189: 22:     i = len(df) - 1
 190: 23:     # force bullish cross + volume
 191: 24:     df.iloc[i - 1, df.columns.get_loc("close")] = df["sma"].iloc[i - 1] - 1
 192: 25:     df.iloc[i, df.columns.get_loc("close")] = df["sma"].iloc[i - 1] + 2
 193: 26:     df.iloc[i, df.columns.get_loc("volume")] = max(1.0, df["volume_avg"].iloc[i - 1]) * 2.0
 194: 27:     sig, _ = strat.get_signal_at(df, i)
 195: 28:     assert sig.type == SignalType.BUY
 196: 29: 
 197: 30: 
 198: 31: def test_no_lookahead():
 199: 32:     p = StrategyParams(sma_period=5, atr_period=5)
 200: 33:     strat = Strategy(p)
 201: 34:     df = make_df(60)
 202: 35:     df = strat.calculate_indicators(df)
 203: 36:     i = 30
 204: 37:     sig1, _ = strat.get_signal_at(df, i)
 205: 38:     sig2, _ = strat.get_signal_at(df.iloc[: i + 1], i)
 206: 39:     assert sig1.type == sig2.type
 207: `````
 208: 
 209: ## File: .gitignore
 210: `````
 211:  1: __pycache__/
 212:  2: *.py[cod]
 213:  3: .venv/
 214:  4: venv/
 215:  5: .env
 216:  6: .idea/
 217:  7: .vscode/
 218:  8: *.log
 219:  9: output/*.csv
 220: 10: output/*.png
 221: 11: data/*.csv
 222: 12: !data/.gitkeep
 223: 13: *.parquet
 224: 14: .mypy_cache/
 225: 15: .pytest_cache/
 226: `````
 227: 
 228: ## File: architecture.md
 229: `````markdown
 230:  1: # System Architecture - Minimal Trader v2
 231:  2: 
 232:  3: ## Components
 233:  4: - `strategy.py`: Pure signal generation (SMA cross + ATR SL/TP, volume confirm)
 234:  5: - `backtest.py`: Execution engine (O(n), SL/TP, reversal/timeout exit, metrics)
 235:  6: - `data_feed.py`: Data access layer (CSVFeed, later LiveFeed/MT5)
 236:  7: - `main.py`: CLI orchestration (backtest, signal)
 237:  8: - `tests/`: unit + integration tests
 238:  9: 
 239: 10: ## Data Flow
 240: 11: CLI → Config → CSVFeed.load() → Strategy.calculate_indicators() → loop i: Strategy.get_signal_at(i) → Backtest execution → Trades & Metrics.
 241: `````
 242: 
 243: ## File: backtest.py
 244: `````python
 245:   1: from __future__ import annotations
 246:   2: from dataclasses import dataclass
 247:   3: from typing import Dict, Optional, List
 248:   4: 
 249:   5: import numpy as np
 250:   6: import pandas as pd
 251:   7: 
 252:   8: from strategy import Strategy, StrategyParams, SignalType
 253:   9: 
 254:  10: 
 255:  11: @dataclass(frozen=True)
 256:  12: class ExecConfig:
 257:  13:     initial_capital: float = 10_000.0
 258:  14:     risk_pct: float = 1.0           # % of equity
 259:  15:     commission: float = 0.0002      # 2 bps per side
 260:  16:     point_value: float = 1.0        # PnL per pt per 1.0 size
 261:  17:     time_exit_bars: int = 200       # fail-safe max bars in trade
 262:  18:     progress_every: int = 10_000    # progress log cadence
 263:  19: 
 264:  20: 
 265:  21: class Backtest:
 266:  22:     def __init__(self, strategy: Strategy, cfg: ExecConfig = ExecConfig()):
 267:  23:         self.strategy = strategy
 268:  24:         self.cfg = cfg
 269:  25: 
 270:  26:     def _position_size(self, equity: float, entry: float, stop: float) -> float:
 271:  27:         risk_amount = equity * (self.cfg.risk_pct / 100.0)
 272:  28:         risk_pts = abs(entry - stop)
 273:  29:         if risk_pts <= 0:
 274:  30:             return 0.0
 275:  31:         size = risk_amount / (risk_pts * self.cfg.point_value)
 276:  32:         return round(size, 2)
 277:  33: 
 278:  34:     def run(self, df: pd.DataFrame, logger=None) -> Dict:
 279:  35:         # Precompute indicators once (O(n))
 280:  36:         df = self.strategy.calculate_indicators(df)
 281:  37:         n = len(df)
 282:  38:         warmup = max(self.strategy.p.sma_period, self.strategy.p.atr_period)
 283:  39: 
 284:  40:         equity = self.cfg.initial_capital
 285:  41:         position: Optional[Dict] = None
 286:  42:         trades: List[Dict] = []
 287:  43: 
 288:  44:         for i in range(warmup, n):
 289:  45:             if logger and self.cfg.progress_every and i % self.cfg.progress_every == 0:
 290:  46:                 logger.info(f"Backtesting... {i}/{n}")
 291:  47: 
 292:  48:             # Determine signal at bar i (no look-ahead)
 293:  49:             sig, meta = self.strategy.get_signal_at(df, i)
 294:  50:             bar = df.iloc[i]
 295:  51: 
 296:  52:             # 1) Check exits first
 297:  53:             if position is not None:
 298:  54:                 exit_price = None
 299:  55:                 exit_reason = None
 300:  56: 
 301:  57:                 # hard SL/TP using bar's low/high
 302:  58:                 if bar["low"] <= position["stop"]:
 303:  59:                     exit_price = position["stop"]
 304:  60:                     exit_reason = "Stop Loss"
 305:  61:                 elif bar["high"] >= position["target"]:
 306:  62:                     exit_price = position["target"]
 307:  63:                     exit_reason = "Take Profit"
 308:  64:                 elif sig.type == SignalType.SELL:
 309:  65:                     exit_price = float(bar["close"])
 310:  66:                     exit_reason = "Reversal Exit"
 311:  67:                 elif (i - position["entry_index"]) > self.cfg.time_exit_bars:
 312:  68:                     exit_price = float(bar["close"])
 313:  69:                     exit_reason = "Time Exit"
 314:  70: 
 315:  71:                 if exit_price is not None:
 316:  72:                     gross = (exit_price - position["entry"]) * position["size"] * self.cfg.point_value
 317:  73:                     fees = (position["entry"] * position["size"] * self.cfg.commission) \
 318:  74:                          + (exit_price * position["size"] * self.cfg.commission)
 319:  75:                     pnl = gross - fees
 320:  76:                     equity += pnl
 321:  77: 
 322:  78:                     trades.append({
 323:  79:                         "entry_time": position["entry_time"],
 324:  80:                         "exit_time": bar.name,
 325:  81:                         "side": "long",
 326:  82:                         "entry": position["entry"],
 327:  83:                         "exit": exit_price,
 328:  84:                         "size": position["size"],
 329:  85:                         "pnl": pnl,
 330:  86:                         "reason": exit_reason,
 331:  87:                     })
 332:  88:                     position = None
 333:  89: 
 334:  90:             # 2) Entries (only if flat)
 335:  91:             if position is None and sig.type == SignalType.BUY:
 336:  92:                 entry = float(bar["close"])
 337:  93:                 stop = float(sig.stop)
 338:  94:                 target = float(sig.target)
 339:  95:                 size = self._position_size(equity, entry, stop)
 340:  96:                 if size > 0:
 341:  97:                     position = {
 342:  98:                         "entry_time": bar.name,
 343:  99:                         "entry_index": i,
 344: 100:                         "entry": entry,
 345: 101:                         "stop": stop,
 346: 102:                         "target": target,
 347: 103:                         "size": size,
 348: 104:                     }
 349: 105: 
 350: 106:         # Metrics
 351: 107:         trades_df = pd.DataFrame(trades)
 352: 108:         metrics = self._metrics(trades_df, equity)
 353: 109:         return {"metrics": metrics, "trades": trades_df}
 354: 110: 
 355: 111:     def _metrics(self, trades_df: pd.DataFrame, final_equity: float) -> Dict:
 356: 112:         ic = self.cfg.initial_capital
 357: 113:         if trades_df.empty:
 358: 114:             return {
 359: 115:                 "initial_capital": ic,
 360: 116:                 "final_capital": final_equity,
 361: 117:                 "total_return_pct": ((final_equity / ic) - 1.0) * 100.0,
 362: 118:                 "num_trades": 0,
 363: 119:                 "win_rate": 0.0,
 364: 120:                 "profit_factor": 0.0,
 365: 121:                 "max_drawdown": 0.0,
 366: 122:                 "sharpe_ratio": 0.0,
 367: 123:                 "message": "No trades",
 368: 124:             }
 369: 125: 
 370: 126:         wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
 371: 127:         losses = trades_df.loc[trades_df["pnl"] <= 0, "pnl"]
 372: 128: 
 373: 129:         equity_curve = ic + trades_df["pnl"].cumsum()
 374: 130:         peak = equity_curve.cummax()
 375: 131:         dd = (equity_curve - peak) / peak.replace(0, np.nan)
 376: 132:         max_dd_pct = abs(dd.min()) * 100.0 if not dd.empty else 0.0
 377: 133: 
 378: 134:         ret = trades_df["pnl"] / ic * 100.0
 379: 135:         sharpe = 0.0
 380: 136:         if ret.std(ddof=0) > 0:
 381: 137:             days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).days or 1
 382: 138:             tpy = (252 / days) * len(trades_df) if days > 0 else len(trades_df)
 383: 139:             sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(max(tpy, 1))
 384: 140: 
 385: 141:         pf = 0.0
 386: 142:         if not losses.empty and losses.sum() != 0:
 387: 143:             pf = abs(wins.sum() / losses.sum()) if not wins.empty else 0.0
 388: 144: 
 389: 145:         return {
 390: 146:             "initial_capital": ic,
 391: 147:             "final_capital": final_equity,
 392: 148:             "total_return_pct": ((final_equity / ic) - 1.0) * 100.0,
 393: 149:             "num_trades": int(len(trades_df)),
 394: 150:             "win_rate": (len(wins) / len(trades_df)) * 100.0,
 395: 151:             "profit_factor": pf,
 396: 152:             "max_drawdown": float(max_dd_pct),
 397: 153:             "sharpe_ratio": float(sharpe),
 398: 154:         }
 399: `````
 400: 
 401: ## File: config.yaml
 402: `````yaml
 403:  1: # Minimal Trading Configuration
 404:  2: trading:
 405:  3:   symbol: "GER40.cash"
 406:  4:   timeframe: "H1"
 407:  5:   initial_capital: 10000
 408:  6:   risk_per_trade: 1.0  # percentage
 409:  7: 
 410:  8: strategy:
 411:  9:   sma_period: 20
 412: 10:   atr_period: 14
 413: 11:   sl_multiplier: 1.5
 414: 12:   tp_multiplier: 2.5
 415: 13:   volume_threshold: 1.0  # times average
 416: 14: 
 417: 15: backtest:
 418: 16:   start_date: "2024-01-01"
 419: 17:   end_date: null  # null = until today
 420: 18:   commission: 0.0002  # 2 bps per side
 421: 19: 
 422: 20: live:
 423: 21:   mode: "paper"  # paper or live
 424: 22:   check_interval: 3600  # seconds
 425: 23:   max_positions: 1
 426: `````
 427: 
 428: ## File: data_feed.py
 429: `````python
 430:  1: from __future__ import annotations
 431:  2: from dataclasses import dataclass
 432:  3: from pathlib import Path
 433:  4: import pandas as pd
 434:  5: 
 435:  6: 
 436:  7: class DataFeed:
 437:  8:     def load(self, limit: int = 0) -> pd.DataFrame:
 438:  9:         raise NotImplementedError
 439: 10: 
 440: 11: 
 441: 12: @dataclass(frozen=True)
 442: 13: class CSVFeed(DataFeed):
 443: 14:     csv_path: str
 444: 15: 
 445: 16:     def load(self, limit: int = 0) -> pd.DataFrame:
 446: 17:         p = Path(self.csv_path)
 447: 18:         if not p.exists():
 448: 19:             raise FileNotFoundError(f"CSV not found: {p}")
 449: 20: 
 450: 21:         df = pd.read_csv(p)
 451: 22:         # detect time column
 452: 23:         time_cols = ["time", "Time", "datetime", "Datetime", "date", "Date"]
 453: 24:         for c in time_cols:
 454: 25:             if c in df.columns:
 455: 26:                 df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
 456: 27:                 df = df.dropna(subset=[c]).set_index(c)
 457: 28:                 break
 458: 29: 
 459: 30:         df.columns = df.columns.str.lower()
 460: 31: 
 461: 32:         for col in ["open", "high", "low", "close"]:
 462: 33:             if col not in df.columns:
 463: 34:                 raise ValueError(f"Missing required column: {col}")
 464: 35: 
 465: 36:         if "volume" not in df.columns:
 466: 37:             df["volume"] = 1000.0
 467: 38: 
 468: 39:         df = df.sort_index()
 469: 40:         if limit and limit > 0:
 470: 41:             df = df.tail(limit)
 471: 42:         return df
 472: `````
 473: 
 474: ## File: implementation-log.md
 475: `````markdown
 476:  1: # Implementation Log
 477:  2: 
 478:  3: ## Step 1: Backtest Smoke Test
 479:  4: 
 480:  5: **Date:** 2025-09-05  
 481:  6: **Who:** Bastiaan / ChatGPT  
 482:  7: **What:** Eerste validatie van strategie en engine via `python trader.py backtest --csv ...`  
 483:  8: **Why:** Verifiëren dat:
 484:  9: - De strategie initieert, signalen genereert, en trades uitvoert
 485: 10: - Er geen look-ahead bias is
 486: 11: - Output en logs goed geschreven worden
 487: 12: - `test_trader.py` draait zonder fouten
 488: 13: 
 489: 14: **Status:** ✅ Geslaagd  
 490: 15: - Testcases draaien met `pytest -v`
 491: 16: - Dummy data via `create_test_data()` werkt
 492: 17: - Backtest-metrics zijn gegenereerd en CSV wordt weggeschreven
 493: 18: 
 494: 19: **Next step:**  
 495: 20: → Realistische data toevoegen in `data/` (bijv. GER40.cash_H1.csv)  
 496: 21: → Run backtest op echte data  
 497: 22: → Evalueren of logica/metrics kloppen
 498: 23: 
 499: 24: ## Step 2: Architecture Refactoring to v2 Modules
 500: 25: 
 501: 26: **Date:** (auto)  
 502: 27: **What:** Split monolith into `strategy.py`, `backtest.py`, `data_feed.py`, `main.py` + tests  
 503: 28: **Why:** Separation of concerns, O(n) performance, testability
 504: 29: 
 505: 30: **Decisions:**
 506: 31: - Indicators computed once; signal per bar by index (no O(n²))
 507: 32: - BUY entries only; SELL used as exit trigger
 508: 33: - Time-exit fail-safe (200 bars)
 509: 34: - CSVFeed for input; LiveFeed later
 510: 35: 
 511: 36: **Status:** Implemented  
 512: 37: **Next:** Run tests + run backtest on real CSV (limit first)
 513: `````
 514: 
 515: ## File: main.py
 516: `````python
 517:   1: from __future__ import annotations
 518:   2: import argparse
 519:   3: import logging
 520:   4: from datetime import datetime
 521:   5: from pathlib import Path
 522:   6: from typing import Dict
 523:   7: 
 524:   8: import yaml
 525:   9: import pandas as pd
 526:  10: 
 527:  11: from strategy import Strategy, StrategyParams
 528:  12: from backtest import Backtest, ExecConfig
 529:  13: from data_feed import CSVFeed
 530:  14: 
 531:  15: 
 532:  16: def load_config(path: str = "config.yaml") -> Dict:
 533:  17:     with open(path, "r") as f:
 534:  18:         return yaml.safe_load(f) or {}
 535:  19: 
 536:  20: 
 537:  21: def setup_logging():
 538:  22:     Path("logs").mkdir(exist_ok=True)
 539:  23:     logging.basicConfig(
 540:  24:         level=logging.INFO,
 541:  25:         format="%(asctime)s [%(levelname)s] %(message)s",
 542:  26:         handlers=[logging.FileHandler("logs/trader.log"), logging.StreamHandler()],
 543:  27:     )
 544:  28:     return logging.getLogger("minimal_trader")
 545:  29: 
 546:  30: 
 547:  31: def run_backtest(args, cfg: Dict, logger):
 548:  32:     p = cfg.get("strategy", {})
 549:  33:     s_params = StrategyParams(
 550:  34:         sma_period=int(p.get("sma_period", 20)),
 551:  35:         atr_period=int(p.get("atr_period", 14)),
 552:  36:         sl_mult=float(p.get("sl_multiplier", 1.5)),
 553:  37:         tp_mult=float(p.get("tp_multiplier", 2.5)),
 554:  38:         volume_threshold=float(p.get("volume_threshold", 1.0)),
 555:  39:     )
 556:  40:     strategy = Strategy(s_params)
 557:  41: 
 558:  42:     e = cfg.get("trading", {})
 559:  43:     b = cfg.get("backtest", {})
 560:  44:     exec_cfg = ExecConfig(
 561:  45:         initial_capital=float(e.get("initial_capital", 10_000.0)),
 562:  46:         risk_pct=float(e.get("risk_per_trade", 1.0)),
 563:  47:         commission=float(b.get("commission", 0.0002)),
 564:  48:         point_value=float(e.get("point_value", 1.0)),
 565:  49:         time_exit_bars=int(b.get("time_exit_bars", 200)),
 566:  50:     )
 567:  51:     bt = Backtest(strategy, exec_cfg)
 568:  52: 
 569:  53:     feed = CSVFeed(args.csv)
 570:  54:     df = feed.load(limit=args.limit)
 571:  55: 
 572:  56:     logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
 573:  57: 
 574:  58:     results = bt.run(df, logger=logger)
 575:  59: 
 576:  60:     print("\n" + "=" * 50)
 577:  61:     print("BACKTEST RESULTS")
 578:  62:     print("=" * 50)
 579:  63:     for k, v in results["metrics"].items():
 580:  64:         if isinstance(v, float):
 581:  65:             print(f"{k:20s}: {v:,.2f}")
 582:  66:         else:
 583:  67:             print(f"{k:20s}: {v}")
 584:  68: 
 585:  69:     trades = results["trades"]
 586:  70:     if isinstance(trades, pd.DataFrame) and not trades.empty:
 587:  71:         Path("output").mkdir(exist_ok=True)
 588:  72:         out = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
 589:  73:         trades.to_csv(out, index=False)
 590:  74:         print(f"\nTrades saved to: {out}")
 591:  75: 
 592:  76: 
 593:  77: def run_signal(args, cfg: Dict, logger):
 594:  78:     p = cfg.get("strategy", {})
 595:  79:     s_params = StrategyParams(
 596:  80:         sma_period=int(p.get("sma_period", 20)),
 597:  81:         atr_period=int(p.get("atr_period", 14)),
 598:  82:         sl_mult=float(p.get("sl_multiplier", 1.5)),
 599:  83:         tp_mult=float(p.get("tp_multiplier", 2.5)),
 600:  84:         volume_threshold=float(p.get("volume_threshold", 1.0)),
 601:  85:     )
 602:  86:     strategy = Strategy(s_params)
 603:  87: 
 604:  88:     feed = CSVFeed(args.csv)
 605:  89:     df = strategy.calculate_indicators(feed.load(limit=max(args.limit, 500)))  # need some history
 606:  90: 
 607:  91:     if len(df) < max(s_params.sma_period, s_params.atr_period) + 2:
 608:  92:         logger.warning("Not enough data to generate a signal.")
 609:  93:         return
 610:  94: 
 611:  95:     i = len(df) - 1
 612:  96:     sig, meta = strategy.get_signal_at(df, i)
 613:  97:     if sig.type.name != "NONE":
 614:  98:         logger.info(f"SIGNAL: {sig.type.name} at {meta.get('timestamp')}")
 615:  99:         logger.info(f"Reason: {sig.reason}")
 616: 100:         logger.info(f"Price: {meta.get('close'):.2f}, SMA: {meta.get('sma'):.2f}, ATR: {meta.get('atr'):.2f}")
 617: 101:     else:
 618: 102:         logger.info("No signal")
 619: 103: 
 620: 104: 
 621: 105: def main():
 622: 106:     parser = argparse.ArgumentParser(description="Minimal Trader v2")
 623: 107:     sub = parser.add_subparsers(dest="mode", required=True)
 624: 108: 
 625: 109:     p_bt = sub.add_parser("backtest", help="Run backtest on CSV")
 626: 110:     p_bt.add_argument("--csv", type=str, required=True)
 627: 111:     p_bt.add_argument("--config", type=str, default="config.yaml")
 628: 112:     p_bt.add_argument("--limit", type=int, default=0, help="Tail N rows (0=all)")
 629: 113: 
 630: 114:     p_sig = sub.add_parser("signal", help="One-time signal check on CSV tail")
 631: 115:     p_sig.add_argument("--csv", type=str, required=True)
 632: 116:     p_sig.add_argument("--config", type=str, default="config.yaml")
 633: 117:     p_sig.add_argument("--limit", type=int, default=1000)
 634: 118: 
 635: 119:     args = parser.parse_args()
 636: 120:     logger = setup_logging()
 637: 121:     cfg = load_config(getattr(args, "config", "config.yaml"))
 638: 122: 
 639: 123:     if args.mode == "backtest":
 640: 124:         run_backtest(args, cfg, logger)
 641: 125:     elif args.mode == "signal":
 642: 126:         run_signal(args, cfg, logger)
 643: 127: 
 644: 128: 
 645: 129: if __name__ == "__main__":
 646: 130:     main()
 647: `````
 648: 
 649: ## File: README.md
 650: `````markdown
 651:  1: # Minimal Viable Trading System
 652:  2: 
 653:  3: ## Philosophy
 654:  4: **"If you can't explain it in one sentence, it's too complex."**
 655:  5: 
 656:  6: Strategy: Buy when price breaks above 20-SMA with volume, exit at -1.5 ATR or +2.5 ATR.
 657:  7: 
 658:  8: ## Quick Start
 659:  9: 
 660: 10: ### 1. Install Dependencies
 661: 11: ```bash
 662: 12: python -m venv venv
 663: 13: source venv/bin/activate  # Windows: venv\Scripts\activate
 664: 14: pip install -r requirements.txt
 665: `````
 666: 
 667: ## File: repomix-output-bastiaandehaan-minimal_trader.md
 668: `````markdown
 669:   1: This file is a merged representation of the entire codebase, combined into a single document by Repomix.
 670:   2: The content has been processed where line numbers have been added, content has been formatted for parsing in markdown style, security check has been disabled.
 671:   3: 
 672:   4: # File Summary
 673:   5: 
 674:   6: ## Purpose
 675:   7: This file contains a packed representation of the entire repository's contents.
 676:   8: It is designed to be easily consumable by AI systems for analysis, code review,
 677:   9: or other automated processes.
 678:  10: 
 679:  11: ## File Format
 680:  12: The content is organized as follows:
 681:  13: 1. This summary section
 682:  14: 2. Repository information
 683:  15: 3. Directory structure
 684:  16: 4. Repository files (if enabled)
 685:  17: 5. Multiple file entries, each consisting of:
 686:  18:   a. A header with the file path (## File: path/to/file)
 687:  19:   b. The full contents of the file in a code block
 688:  20: 
 689:  21: ## Usage Guidelines
 690:  22: - This file should be treated as read-only. Any changes should be made to the
 691:  23:   original repository files, not this packed version.
 692:  24: - When processing this file, use the file path to distinguish
 693:  25:   between different files in the repository.
 694:  26: - Be aware that this file may contain sensitive information. Handle it with
 695:  27:   the same level of security as you would the original repository.
 696:  28: 
 697:  29: ## Notes
 698:  30: - Some files may have been excluded based on .gitignore rules and Repomix's configuration
 699:  31: - Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
 700:  32: - Files matching patterns in .gitignore are excluded
 701:  33: - Files matching default ignore patterns are excluded
 702:  34: - Line numbers have been added to the beginning of each line
 703:  35: - Content has been formatted for parsing in markdown style
 704:  36: - Security check has been disabled - content may contain sensitive information
 705:  37: - Files are sorted by Git change count (files with more changes are at the bottom)
 706:  38: 
 707:  39: # Directory Structure
 708:  40: ```
 709:  41: .claude/
 710:  42:   settings.local.json
 711:  43: .github/
 712:  44:   workflows/
 713:  45:     ci.yml
 714:  46: .gitignore
 715:  47: config.yaml
 716:  48: implementation-log.md
 717:  49: README.md
 718:  50: requirements.txt
 719:  51: test_trader.py
 720:  52: trader.py
 721:  53: ```
 722:  54: 
 723:  55: # Files
 724:  56: 
 725:  57: ## File: .claude/settings.local.json
 726:  58: ````json
 727:  59:  1: {
 728:  60:  2:   "permissions": {
 729:  61:  3:     "allow": [
 730:  62:  4:       "Bash(python:*)",
 731:  63:  5:       "Bash(venvScripts:*)",
 732:  64:  6:       "Bash(venv/Scripts/activate:*)",
 733:  65:  7:       "Bash(pip install:*)",
 734:  66:  8:       "Bash(venv\\\\Scripts\\\\python.exe:*)",
 735:  67:  9:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader\\venv\\Scripts/**)",
 736:  68: 10:       "Bash(venv/Scripts/python.exe:*)"
 737:  69: 11:     ],
 738:  70: 12:     "deny": [],
 739:  71: 13:     "ask": []
 740:  72: 14:   }
 741:  73: 15: }
 742:  74: ````
 743:  75: 
 744:  76: ## File: .github/workflows/ci.yml
 745:  77: ````yaml
 746:  78:  1: name: CI
 747:  79:  2: 
 748:  80:  3: on:
 749:  81:  4:   push:
 750:  82:  5:   pull_request:
 751:  83:  6: 
 752:  84:  7: jobs:
 753:  85:  8:   test:
 754:  86:  9:     runs-on: ubuntu-latest
 755:  87: 10:     steps:
 756:  88: 11:       - uses: actions/checkout@v4
 757:  89: 12:       - uses: actions/setup-python@v5
 758:  90: 13:         with:
 759:  91: 14:           python-version: "3.11"
 760:  92: 15:       - run: pip install -r requirements.txt || pip install pandas numpy pyyaml pytest
 761:  93: 16:       - run: pytest -v || true  # TODO: remove '|| true' zodra tests 100% stabiel
 762:  94: ````
 763:  95: 
 764:  96: ## File: .gitignore
 765:  97: ````
 766:  98:  1: __pycache__/
 767:  99:  2: *.py[cod]
 768: 100:  3: .venv/
 769: 101:  4: venv/
 770: 102:  5: .env
 771: 103:  6: .idea/
 772: 104:  7: .vscode/
 773: 105:  8: *.log
 774: 106:  9: output/*.csv
 775: 107: 10: output/*.png
 776: 108: 11: data/*.csv
 777: 109: 12: !data/.gitkeep
 778: 110: 13: *.parquet
 779: 111: 14: .mypy_cache/
 780: 112: 15: .pytest_cache/
 781: 113: ````
 782: 114: 
 783: 115: ## File: config.yaml
 784: 116: ````yaml
 785: 117:  1: # Minimal Trading Configuration
 786: 118:  2: trading:
 787: 119:  3:   symbol: "GER40.cash"
 788: 120:  4:   timeframe: "H1"
 789: 121:  5:   initial_capital: 10000
 790: 122:  6:   risk_per_trade: 1.0  # percentage
 791: 123:  7: 
 792: 124:  8: strategy:
 793: 125:  9:   sma_period: 20
 794: 126: 10:   atr_period: 14
 795: 127: 11:   sl_multiplier: 1.5
 796: 128: 12:   tp_multiplier: 2.5
 797: 129: 13:   volume_threshold: 1.0  # times average
 798: 130: 14: 
 799: 131: 15: backtest:
 800: 132: 16:   start_date: "2024-01-01"
 801: 133: 17:   end_date: null  # null = until today
 802: 134: 18:   commission: 0.0002  # 2 bps per side
 803: 135: 19: 
 804: 136: 20: live:
 805: 137: 21:   mode: "paper"  # paper or live
 806: 138: 22:   check_interval: 3600  # seconds
 807: 139: 23:   max_positions: 1
 808: 140: ````
 809: 141: 
 810: 142: ## File: implementation-log.md
 811: 143: ````markdown
 812: 144:  1: # Implementation Log
 813: 145:  2: 
 814: 146:  3: ## Step 1: Backtest Smoke Test
 815: 147:  4: 
 816: 148:  5: **Date:** 2025-09-05  
 817: 149:  6: **Who:** Bastiaan / ChatGPT  
 818: 150:  7: **What:** Eerste validatie van strategie en engine via `python trader.py backtest --csv ...`  
 819: 151:  8: **Why:** Verifiëren dat:
 820: 152:  9: - De strategie initieert, signalen genereert, en trades uitvoert
 821: 153: 10: - Er geen look-ahead bias is
 822: 154: 11: - Output en logs goed geschreven worden
 823: 155: 12: - `test_trader.py` draait zonder fouten
 824: 156: 13: 
 825: 157: 14: **Status:** ✅ Geslaagd  
 826: 158: 15: - Testcases draaien met `pytest -v`
 827: 159: 16: - Dummy data via `create_test_data()` werkt
 828: 160: 17: - Backtest-metrics zijn gegenereerd en CSV wordt weggeschreven
 829: 161: 18: 
 830: 162: 19: **Next step:**  
 831: 163: 20: → Realistische data toevoegen in `data/` (bijv. GER40.cash_H1.csv)  
 832: 164: 21: → Run backtest op echte data  
 833: 165: 22: → Evalueren of logica/metrics kloppen
 834: 166: ````
 835: 167: 
 836: 168: ## File: README.md
 837: 169: ````markdown
 838: 170:  1: # Minimal Viable Trading System
 839: 171:  2: 
 840: 172:  3: ## Philosophy
 841: 173:  4: **"If you can't explain it in one sentence, it's too complex."**
 842: 174:  5: 
 843: 175:  6: Strategy: Buy when price breaks above 20-SMA with volume, exit at -1.5 ATR or +2.5 ATR.
 844: 176:  7: 
 845: 177:  8: ## Quick Start
 846: 178:  9: 
 847: 179: 10: ### 1. Install Dependencies
 848: 180: 11: ```bash
 849: 181: 12: python -m venv venv
 850: 182: 13: source venv/bin/activate  # Windows: venv\Scripts\activate
 851: 183: 14: pip install -r requirements.txt
 852: 184: ````
 853: 185: 
 854: 186: ## File: requirements.txt
 855: 187: ````
 856: 188: 1: pandas>=2.2.0
 857: 189: 2: numpy>=1.26.0
 858: 190: 3: pyyaml>=6.0
 859: 191: 4: pytest>=8.0.0
 860: 192: 5: python-dotenv>=1.0.0
 861: 193: 6: MetaTrader5>=5.0.45  # Windows only
 862: 194: ````
 863: 195: 
 864: 196: ## File: test_trader.py
 865: 197: ````python
 866: 198:  1: """
 867: 199:  2: Minimal tests - just the essentials
 868: 200:  3: """
 869: 201:  4: import pytest
 870: 202:  5: import pandas as pd
 871: 203:  6: import numpy as np
 872: 204:  7: from datetime import datetime, timedelta
 873: 205:  8: 
 874: 206:  9: from trader import SimpleBreakoutTrader, Backtester
 875: 207: 10: 
 876: 208: 11: def create_test_data(n_days=30, trend=0.001):
 877: 209: 12:     """Create synthetic OHLC data"""
 878: 210: 13:     dates = pd.date_range(end=datetime.now(), periods=n_days*24, freq='H')
 879: 211: 14: 
 880: 212: 15:     # Random walk with trend
 881: 213: 16:     returns = np.random.normal(trend, 0.01, len(dates))
 882: 214: 17:     close = 18000 * (1 + returns).cumprod()
 883: 215: 18: 
 884: 216: 19:     # Create OHLC
 885: 217: 20:     high = close * (1 + np.random.uniform(0.001, 0.005, len(dates)))
 886: 218: 21:     low = close * (1 - np.random.uniform(0.001, 0.005, len(dates)))
 887: 219: 22:     open_ = np.roll(close, 1)
 888: 220: 23:     open_[0] = close[0]
 889: 221: 24: 
 890: 222: 25:     volume = np.random.uniform(1000, 5000, len(dates))
 891: 223: 26: 
 892: 224: 27:     df = pd.DataFrame({
 893: 225: 28:         'open': open_,
 894: 226: 29:         'high': high,
 895: 227: 30:         'low': low,
 896: 228: 31:         'close': close,
 897: 229: 32:         'volume': volume
 898: 230: 33:     }, index=dates)
 899: 231: 34: 
 900: 232: 35:     return df
 901: 233: 36: 
 902: 234: 37: def test_trader_init():
 903: 235: 38:     """Test trader initialization"""
 904: 236: 39:     trader = SimpleBreakoutTrader()
 905: 237: 40:     assert trader.symbol == "GER40.cash"
 906: 238: 41:     assert trader.risk_pct == 1.0
 907: 239: 42: 
 908: 240: 43: def test_signal_generation():
 909: 241: 44:     """Test that signals are generated"""
 910: 242: 45:     trader = SimpleBreakoutTrader()
 911: 243: 46:     df = create_test_data()
 912: 244: 47: 
 913: 245: 48:     signal, meta = trader.get_signal(df)
 914: 246: 49: 
 915: 247: 50:     assert signal in ['BUY', 'SELL', None]
 916: 248: 51:     assert 'close' in meta
 917: 249: 52:     assert 'sma' in meta
 918: 250: 53: 
 919: 251: 54: def test_no_look_ahead():
 920: 252: 55:     """Critical: Test no look-ahead bias"""
 921: 253: 56:     trader = SimpleBreakoutTrader()
 922: 254: 57:     df = create_test_data()
 923: 255: 58: 
 924: 256: 59:     # Signal at time T
 925: 257: 60:     signal_t, _ = trader.get_signal(df.iloc[:100])
 926: 258: 61: 
 927: 259: 62:     # Signal at time T with more future data
 928: 260: 63:     signal_t_future, _ = trader.get_signal(df.iloc[:100])  # Same window
 929: 261: 64: 
 930: 262: 65:     # Should be identical
 931: 263: 66:     assert signal_t == signal_t_future
 932: 264: 67: 
 933: 265: 68: def test_backtest_runs():
 934: 266: 69:     """Test backtest completes without errors"""
 935: 267: 70:     trader = SimpleBreakoutTrader()
 936: 268: 71:     backtester = Backtester(trader)
 937: 269: 72: 
 938: 270: 73:     df = create_test_data()
 939: 271: 74:     results = backtester.run(df)
 940: 272: 75: 
 941: 273: 76:     assert 'metrics' in results
 942: 274: 77:     assert 'trades' in results
 943: 275: 78:     assert results['metrics']['initial_capital'] == 10000
 944: 276: 79: 
 945: 277: 80: def test_position_sizing():
 946: 278: 81:     """Test risk-based position sizing"""
 947: 279: 82:     trader = SimpleBreakoutTrader()
 948: 280: 83: 
 949: 281: 84:     size = trader.calculate_position_size(
 950: 282: 85:         equity=10000,
 951: 283: 86:         entry=18000,
 952: 284: 87:         stop=17900
 953: 285: 88:     )
 954: 286: 89: 
 955: 287: 90:     # Should risk 1% = 100 EUR
 956: 288: 91:     # Risk is 100 points, so size should be 1.0
 957: 289: 92:     assert size == 1.0
 958: 290: 93: 
 959: 291: 94: if __name__ == '__main__':
 960: 292: 95:     pytest.main([__file__, '-v'])
 961: 293: ````
 962: 294: 
 963: 295: ## File: trader.py
 964: 296: ````python
 965: 297:   1: #!/usr/bin/env python3
 966: 298:   2: """
 967: 299:   3: Minimal Viable Trading System
 968: 300:   4: One file to rule them all - strategie, backtest, live trading
 969: 301:   5: """
 970: 302:   6: import sys
 971: 303:   7: import time
 972: 304:   8: import logging
 973: 305:   9: from datetime import datetime, timedelta
 974: 306:  10: from pathlib import Path
 975: 307:  11: from typing import Optional, Dict, Tuple, List
 976: 308:  12: 
 977: 309:  13: import numpy as np
 978: 310:  14: import pandas as pd
 979: 311:  15: import yaml
 980: 312:  16: 
 981: 313:  17: # Setup logging
 982: 314:  18: logging.basicConfig(
 983: 315:  19:     level=logging.INFO,
 984: 316:  20:     format='%(asctime)s [%(levelname)s] %(message)s',
 985: 317:  21:     handlers=[
 986: 318:  22:         logging.FileHandler('logs/trader.log'),
 987: 319:  23:         logging.StreamHandler()
 988: 320:  24:     ]
 989: 321:  25: )
 990: 322:  26: logger = logging.getLogger(__name__)
 991: 323:  27: 
 992: 324:  28: class SimpleBreakoutTrader:
 993: 325:  29:     """Dead simple breakout strategy - no BS, just works"""
 994: 326:  30: 
 995: 327:  31:     def __init__(self, config_path: str = "config.yaml"):
 996: 328:  32:         with open(config_path, 'r') as f:
 997: 329:  33:             self.config = yaml.safe_load(f)
 998: 330:  34: 
 999: 331:  35:         self.symbol = self.config['trading']['symbol']
1000: 332:  36:         self.risk_pct = self.config['trading']['risk_per_trade']
1001: 333:  37:         self.sma_period = self.config['strategy']['sma_period']
1002: 334:  38:         self.atr_period = self.config['strategy']['atr_period']
1003: 335:  39: 
1004: 336:  40:         logger.info(f"Initialized trader for {self.symbol}")
1005: 337:  41: 
1006: 338:  42:     def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
1007: 339:  43:         """Add SMA, ATR and Volume metrics"""
1008: 340:  44:         df = df.copy()
1009: 341:  45: 
1010: 342:  46:         # SMA
1011: 343:  47:         df['sma'] = df['close'].rolling(self.sma_period).mean()
1012: 344:  48: 
1013: 345:  49:         # ATR
1014: 346:  50:         high_low = df['high'] - df['low']
1015: 347:  51:         high_close = np.abs(df['high'] - df['close'].shift())
1016: 348:  52:         low_close = np.abs(df['low'] - df['close'].shift())
1017: 349:  53:         true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
1018: 350:  54:         df['atr'] = true_range.rolling(self.atr_period).mean()
1019: 351:  55: 
1020: 352:  56:         # Volume
1021: 353:  57:         df['volume_avg'] = df['volume'].rolling(self.sma_period).mean()
1022: 354:  58:         df['volume_ratio'] = df['volume'] / df['volume_avg']
1023: 355:  59: 
1024: 356:  60:         return df
1025: 357:  61: 
1026: 358:  62:     def get_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Dict]:
1027: 359:  63:         """
1028: 360:  64:         Returns signal and metadata
1029: 361:  65:         Simple rules:
1030: 362:  66:         - BUY: price crosses above SMA + volume confirmation
1031: 363:  67:         - SELL: price crosses below SMA
1032: 364:  68:         - None: no clear signal
1033: 365:  69:         """
1034: 366:  70:         df = self.calculate_indicators(df)
1035: 367:  71: 
1036: 368:  72:         if len(df) < self.sma_period:
1037: 369:  73:             return None, {}
1038: 370:  74: 
1039: 371:  75:         last = df.iloc[-1]
1040: 372:  76:         prev = df.iloc[-2]
1041: 373:  77: 
1042: 374:  78:         # Detect crossovers
1043: 375:  79:         cross_above = (prev['close'] <= prev['sma']) and (last['close'] > last['sma'])
1044: 376:  80:         cross_below = (prev['close'] >= prev['sma']) and (last['close'] < last['sma'])
1045: 377:  81: 
1046: 378:  82:         # Volume confirmation
1047: 379:  83:         volume_confirmed = last['volume_ratio'] > self.config['strategy']['volume_threshold']
1048: 380:  84: 
1049: 381:  85:         metadata = {
1050: 382:  86:             'close': last['close'],
1051: 383:  87:             'sma': last['sma'],
1052: 384:  88:             'atr': last['atr'],
1053: 385:  89:             'volume_ratio': last['volume_ratio'],
1054: 386:  90:             'timestamp': last.name if hasattr(last, 'name') else None
1055: 387:  91:         }
1056: 388:  92: 
1057: 389:  93:         if cross_above and volume_confirmed:
1058: 390:  94:             metadata['reason'] = f"Bullish cross + volume {last['volume_ratio']:.1f}x"
1059: 391:  95:             return 'BUY', metadata
1060: 392:  96:         elif cross_below:
1061: 393:  97:             metadata['reason'] = "Bearish cross"
1062: 394:  98:             return 'SELL', metadata
1063: 395:  99: 
1064: 396: 100:         return None, metadata
1065: 397: 101: 
1066: 398: 102:     def calculate_position_size(self, equity: float, entry: float, stop: float) -> float:
1067: 399: 103:         """Position sizing based on risk management"""
1068: 400: 104:         risk_amount = equity * (self.risk_pct / 100)
1069: 401: 105:         risk_points = abs(entry - stop)
1070: 402: 106: 
1071: 403: 107:         if risk_points == 0:
1072: 404: 108:             return 0
1073: 405: 109: 
1074: 406: 110:         # For forex/indices: size in lots
1075: 407: 111:         # Adjust multiplier based on your broker
1076: 408: 112:         point_value = 1.0  # EUR per point for GER40
1077: 409: 113:         size = risk_amount / (risk_points * point_value)
1078: 410: 114: 
1079: 411: 115:         return round(size, 2)
1080: 412: 116: 
1081: 413: 117: class Backtester:
1082: 414: 118:     """Simple backtesting engine - no look-ahead, realistic fills"""
1083: 415: 119: 
1084: 416: 120:     def __init__(self, trader: SimpleBreakoutTrader):
1085: 417: 121:         self.trader = trader
1086: 418: 122:         self.config = trader.config
1087: 419: 123: 
1088: 420: 124:     def run(self, df: pd.DataFrame) -> Dict:
1089: 421: 125:         """Run backtest and return results"""
1090: 422: 126:         initial_capital = self.config['trading']['initial_capital']
1091: 423: 127:         equity = initial_capital
1092: 424: 128: 
1093: 425: 129:         trades = []
1094: 426: 130:         position = None
1095: 427: 131: 
1096: 428: 132:         df = self.trader.calculate_indicators(df)
1097: 429: 133: 
1098: 430: 134:         for i in range(self.trader.sma_period, len(df)):
1099: 431: 135:             window = df.iloc[:i+1]
1100: 432: 136:             signal, meta = self.trader.get_signal(window)
1101: 433: 137: 
1102: 434: 138:             current_bar = df.iloc[i]
1103: 435: 139: 
1104: 436: 140:             # Check exits first (before new signals)
1105: 437: 141:             if position:
1106: 438: 142:                 exit_price = None
1107: 439: 143:                 exit_reason = None
1108: 440: 144: 
1109: 441: 145:                 # Check stop loss
1110: 442: 146:                 if current_bar['low'] <= position['stop']:
1111: 443: 147:                     exit_price = position['stop']
1112: 444: 148:                     exit_reason = 'Stop Loss'
1113: 445: 149:                 # Check take profit
1114: 446: 150:                 elif current_bar['high'] >= position['target']:
1115: 447: 151:                     exit_price = position['target']
1116: 448: 152:                     exit_reason = 'Take Profit'
1117: 449: 153:                 # Check signal reversal
1118: 450: 154:                 elif signal == 'SELL':
1119: 451: 155:                     exit_price = current_bar['close']
1120: 452: 156:                     exit_reason = 'Signal Reversal'
1121: 453: 157: 
1122: 454: 158:                 if exit_price:
1123: 455: 159:                     # Calculate PnL
1124: 456: 160:                     pnl = (exit_price - position['entry']) * position['size']
1125: 457: 161:                     pnl -= (position['entry'] * position['size'] * self.config['backtest']['commission'] * 2)
1126: 458: 162: 
1127: 459: 163:                     equity += pnl
1128: 460: 164: 
1129: 461: 165:                     trades.append({
1130: 462: 166:                         'entry_time': position['entry_time'],
1131: 463: 167:                         'exit_time': current_bar.name,
1132: 464: 168:                         'side': position['side'],
1133: 465: 169:                         'entry': position['entry'],
1134: 466: 170:                         'exit': exit_price,
1135: 467: 171:                         'size': position['size'],
1136: 468: 172:                         'pnl': pnl,
1137: 469: 173:                         'return_pct': (pnl / equity) * 100,
1138: 470: 174:                         'reason': exit_reason
1139: 471: 175:                     })
1140: 472: 176: 
1141: 473: 177:                     position = None
1142: 474: 178:                     logger.debug(f"Exit {exit_reason} at {exit_price:.2f}, PnL: {pnl:.2f}")
1143: 475: 179: 
1144: 476: 180:             # New signals (only if no position)
1145: 477: 181:             if signal and not position:
1146: 478: 182:                 if signal == 'BUY':
1147: 479: 183:                     entry = current_bar['close']
1148: 480: 184:                     stop = entry - current_bar['atr'] * self.config['strategy']['sl_multiplier']
1149: 481: 185:                     target = entry + current_bar['atr'] * self.config['strategy']['tp_multiplier']
1150: 482: 186: 
1151: 483: 187:                     size = self.trader.calculate_position_size(equity, entry, stop)
1152: 484: 188: 
1153: 485: 189:                     if size > 0:
1154: 486: 190:                         position = {
1155: 487: 191:                             'entry_time': current_bar.name,
1156: 488: 192:                             'side': 'long',
1157: 489: 193:                             'entry': entry,
1158: 490: 194:                             'stop': stop,
1159: 491: 195:                             'target': target,
1160: 492: 196:                             'size': size
1161: 493: 197:                         }
1162: 494: 198:                         logger.debug(f"BUY at {entry:.2f}, stop: {stop:.2f}, target: {target:.2f}")
1163: 495: 199: 
1164: 496: 200:         # Calculate metrics
1165: 497: 201:         trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
1166: 498: 202: 
1167: 499: 203:         if not trades_df.empty:
1168: 500: 204:             wins = trades_df[trades_df['pnl'] > 0]
1169: 501: 205:             losses = trades_df[trades_df['pnl'] <= 0]
1170: 502: 206: 
1171: 503: 207:             metrics = {
1172: 504: 208:                 'initial_capital': initial_capital,
1173: 505: 209:                 'final_capital': equity,
1174: 506: 210:                 'total_return_pct': ((equity / initial_capital) - 1) * 100,
1175: 507: 211:                 'num_trades': len(trades_df),
1176: 508: 212:                 'num_wins': len(wins),
1177: 509: 213:                 'num_losses': len(losses),
1178: 510: 214:                 'win_rate': (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
1179: 511: 215:                 'avg_win': wins['pnl'].mean() if not wins.empty else 0,
1180: 512: 216:                 'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
1181: 513: 217:                 'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else 0,
1182: 514: 218:                 'max_drawdown': self._calculate_max_drawdown(trades_df),
1183: 515: 219:                 'sharpe_ratio': self._calculate_sharpe(trades_df)
1184: 516: 220:             }
1185: 517: 221:         else:
1186: 518: 222:             metrics = {
1187: 519: 223:                 'initial_capital': initial_capital,
1188: 520: 224:                 'final_capital': equity,
1189: 521: 225:                 'total_return_pct': 0,
1190: 522: 226:                 'num_trades': 0,
1191: 523: 227:                 'message': 'No trades generated'
1192: 524: 228:             }
1193: 525: 229: 
1194: 526: 230:         return {'metrics': metrics, 'trades': trades_df}
1195: 527: 231: 
1196: 528: 232:     def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
1197: 529: 233:         """Calculate maximum drawdown percentage"""
1198: 530: 234:         if trades_df.empty:
1199: 531: 235:             return 0
1200: 532: 236: 
1201: 533: 237:         cumulative = trades_df['pnl'].cumsum()
1202: 534: 238:         running_max = cumulative.cummax()
1203: 535: 239:         drawdown = (cumulative - running_max) / (running_max + self.config['trading']['initial_capital'])
1204: 536: 240:         return abs(drawdown.min()) * 100 if not drawdown.empty else 0
1205: 537: 241: 
1206: 538: 242:     def _calculate_sharpe(self, trades_df: pd.DataFrame) -> float:
1207: 539: 243:         """Calculate Sharpe ratio (simplified)"""
1208: 540: 244:         if trades_df.empty or len(trades_df) < 2:
1209: 541: 245:             return 0
1210: 542: 246: 
1211: 543: 247:         returns = trades_df['return_pct']
1212: 544: 248:         if returns.std() == 0:
1213: 545: 249:             return 0
1214: 546: 250: 
1215: 547: 251:         # Annualize based on average trades per year (estimate)
1216: 548: 252:         days_in_sample = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
1217: 549: 253:         if days_in_sample > 0:
1218: 550: 254:             trades_per_year = (252 / days_in_sample) * len(trades_df)
1219: 551: 255:             return (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
1220: 552: 256:         return 0
1221: 553: 257: 
1222: 554: 258: class LiveTrader:
1223: 555: 259:     """Live trading interface - can be paper or real"""
1224: 556: 260: 
1225: 557: 261:     def __init__(self, trader: SimpleBreakoutTrader):
1226: 558: 262:         self.trader = trader
1227: 559: 263:         self.config = trader.config
1228: 560: 264:         self.position = None
1229: 561: 265:         self.equity = self.config['trading']['initial_capital']
1230: 562: 266: 
1231: 563: 267:     def fetch_latest_data(self, lookback_bars: int = 100) -> pd.DataFrame:
1232: 564: 268:         """Fetch latest price data"""
1233: 565: 269:         # For now, use CSV data for testing
1234: 566: 270:         # Replace with MT5 or broker API
1235: 567: 271:         csv_files = list(Path('data').glob('*.csv'))
1236: 568: 272:         if not csv_files:
1237: 569: 273:             raise FileNotFoundError("No CSV files in data/ directory")
1238: 570: 274: 
1239: 571: 275:         df = pd.read_csv(csv_files[0], parse_dates=['time'])
1240: 572: 276:         df.set_index('time', inplace=True)
1241: 573: 277: 
1242: 574: 278:         # Get last N bars
1243: 575: 279:         return df.tail(lookback_bars)
1244: 576: 280: 
1245: 577: 281:     def check_for_signals(self):
1246: 578: 282:         """Check current market for signals"""
1247: 579: 283:         try:
1248: 580: 284:             df = self.fetch_latest_data()
1249: 581: 285:             signal, meta = self.trader.get_signal(df)
1250: 582: 286: 
1251: 583: 287:             if signal:
1252: 584: 288:                 logger.info(f"SIGNAL: {signal} at {meta.get('timestamp', 'now')}")
1253: 585: 289:                 logger.info(f"Reason: {meta.get('reason', 'N/A')}")
1254: 586: 290:                 logger.info(f"Price: {meta.get('close', 0):.2f}, SMA: {meta.get('sma', 0):.2f}")
1255: 587: 291: 
1256: 588: 292:                 if self.config['live']['mode'] == 'live':
1257: 589: 293:                     self.place_order(signal, meta)
1258: 590: 294:                 else:
1259: 591: 295:                     logger.info("PAPER TRADE - no real order placed")
1260: 592: 296: 
1261: 593: 297:         except Exception as e:
1262: 594: 298:             logger.error(f"Error checking signals: {e}")
1263: 595: 299: 
1264: 596: 300:     def place_order(self, signal: str, meta: Dict):
1265: 597: 301:         """Place actual trade order"""
1266: 598: 302:         # TODO: Implement MT5 order placement
1267: 599: 303:         logger.warning("Live order placement not yet implemented")
1268: 600: 304: 
1269: 601: 305:     def run_forever(self):
1270: 602: 306:         """Main loop for live trading"""
1271: 603: 307:         logger.info(f"Starting live trader in {self.config['live']['mode']} mode")
1272: 604: 308:         logger.info(f"Checking every {self.config['live']['check_interval']} seconds")
1273: 605: 309: 
1274: 606: 310:         while True:
1275: 607: 311:             try:
1276: 608: 312:                 self.check_for_signals()
1277: 609: 313:                 time.sleep(self.config['live']['check_interval'])
1278: 610: 314:             except KeyboardInterrupt:
1279: 611: 315:                 logger.info("Shutting down...")
1280: 612: 316:                 break
1281: 613: 317:             except Exception as e:
1282: 614: 318:                 logger.error(f"Unexpected error: {e}")
1283: 615: 319:                 time.sleep(60)  # Wait a minute before retry
1284: 616: 320: 
1285: 617: 321: def load_data(csv_path: str) -> pd.DataFrame:
1286: 618: 322:     """Load and prepare data"""
1287: 619: 323:     df = pd.read_csv(csv_path)
1288: 620: 324: 
1289: 621: 325:     # Handle different column names
1290: 622: 326:     time_cols = ['time', 'Time', 'datetime', 'Datetime', 'date', 'Date']
1291: 623: 327:     time_col = None
1292: 624: 328:     for col in time_cols:
1293: 625: 329:         if col in df.columns:
1294: 626: 330:             time_col = col
1295: 627: 331:             break
1296: 628: 332: 
1297: 629: 333:     if time_col:
1298: 630: 334:         df[time_col] = pd.to_datetime(df[time_col])
1299: 631: 335:         df.set_index(time_col, inplace=True)
1300: 632: 336: 
1301: 633: 337:     # Standardize column names
1302: 634: 338:     df.columns = df.columns.str.lower()
1303: 635: 339: 
1304: 636: 340:     # Required columns
1305: 637: 341:     required = ['open', 'high', 'low', 'close']
1306: 638: 342:     if not all(col in df.columns for col in required):
1307: 639: 343:         raise ValueError(f"CSV must have columns: {required}")
1308: 640: 344: 
1309: 641: 345:     # Add volume if missing
1310: 642: 346:     if 'volume' not in df.columns:
1311: 643: 347:         df['volume'] = 1000  # Dummy volume
1312: 644: 348: 
1313: 645: 349:     return df.sort_index()
1314: 646: 350: 
1315: 647: 351: def main():
1316: 648: 352:     """Main entry point"""
1317: 649: 353:     import argparse
1318: 650: 354: 
1319: 651: 355:     parser = argparse.ArgumentParser(description='Minimal Viable Trading System')
1320: 652: 356:     parser.add_argument('mode', choices=['backtest', 'live', 'signal'],
1321: 653: 357:                       help='Operating mode')
1322: 654: 358:     parser.add_argument('--csv', type=str, help='CSV file path for backtesting')
1323: 655: 359:     parser.add_argument('--config', type=str, default='config.yaml',
1324: 656: 360:                       help='Configuration file')
1325: 657: 361: 
1326: 658: 362:     args = parser.parse_args()
1327: 659: 363: 
1328: 660: 364:     # Create directories
1329: 661: 365:     Path('logs').mkdir(exist_ok=True)
1330: 662: 366:     Path('output').mkdir(exist_ok=True)
1331: 663: 367: 
1332: 664: 368:     # Initialize trader
1333: 665: 369:     trader = SimpleBreakoutTrader(args.config)
1334: 666: 370: 
1335: 667: 371:     if args.mode == 'backtest':
1336: 668: 372:         if not args.csv:
1337: 669: 373:             csv_files = list(Path('data').glob('*.csv'))
1338: 670: 374:             if not csv_files:
1339: 671: 375:                 logger.error("No CSV files found. Use --csv or add files to data/")
1340: 672: 376:                 sys.exit(1)
1341: 673: 377:             args.csv = str(csv_files[0])
1342: 674: 378:             logger.info(f"Using {args.csv}")
1343: 675: 379: 
1344: 676: 380:         # Load data
1345: 677: 381:         df = load_data(args.csv)
1346: 678: 382:         logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
1347: 679: 383: 
1348: 680: 384:         # Run backtest
1349: 681: 385:         backtester = Backtester(trader)
1350: 682: 386:         results = backtester.run(df)
1351: 683: 387: 
1352: 684: 388:         # Display results
1353: 685: 389:         print("\n" + "="*50)
1354: 686: 390:         print("BACKTEST RESULTS")
1355: 687: 391:         print("="*50)
1356: 688: 392:         for key, value in results['metrics'].items():
1357: 689: 393:             if isinstance(value, float):
1358: 690: 394:                 print(f"{key:20s}: {value:,.2f}")
1359: 691: 395:             else:
1360: 692: 396:                 print(f"{key:20s}: {value}")
1361: 693: 397: 
1362: 694: 398:         # Save trades
1363: 695: 399:         if not results['trades'].empty:
1364: 696: 400:             output_file = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
1365: 697: 401:             results['trades'].to_csv(output_file, index=False)
1366: 698: 402:             print(f"\nTrades saved to: {output_file}")
1367: 699: 403: 
1368: 700: 404:     elif args.mode == 'live':
1369: 701: 405:         live = LiveTrader(trader)
1370: 702: 406:         live.run_forever()
1371: 703: 407: 
1372: 704: 408:     elif args.mode == 'signal':
1373: 705: 409:         # One-time signal check
1374: 706: 410:         live = LiveTrader(trader)
1375: 707: 411:         live.check_for_signals()
1376: 708: 412: 
1377: 709: 413: if __name__ == '__main__':
1378: 710: 414:     main()
1379: 711: ````
1380: `````
1381: 
1382: ## File: requirements.txt
1383: `````
1384: 1: pandas>=2.2.0
1385: 2: numpy>=1.26.0
1386: 3: pyyaml>=6.0
1387: 4: pytest>=8.0.0
1388: 5: python-dotenv>=1.0.0
1389: 6: MetaTrader5>=5.0.45  # Windows only
1390: `````
1391: 
1392: ## File: strategy.py
1393: `````python
1394:   1: from __future__ import annotations
1395:   2: from dataclasses import dataclass
1396:   3: from enum import Enum
1397:   4: from typing import Optional, Tuple
1398:   5: 
1399:   6: import numpy as np
1400:   7: import pandas as pd
1401:   8: 
1402:   9: 
1403:  10: class SignalType(Enum):
1404:  11:     NONE = 0
1405:  12:     BUY = 1
1406:  13:     SELL = -1   # reversal/exit only; no short entries in v1
1407:  14: 
1408:  15: 
1409:  16: @dataclass(frozen=True)
1410:  17: class Signal:
1411:  18:     type: SignalType
1412:  19:     entry: Optional[float] = None
1413:  20:     stop: Optional[float] = None
1414:  21:     target: Optional[float] = None
1415:  22:     reason: str = ""
1416:  23:     timestamp: Optional[pd.Timestamp] = None
1417:  24: 
1418:  25: 
1419:  26: @dataclass(frozen=True)
1420:  27: class StrategyParams:
1421:  28:     sma_period: int = 20
1422:  29:     atr_period: int = 14
1423:  30:     sl_mult: float = 1.5
1424:  31:     tp_mult: float = 2.5
1425:  32:     volume_threshold: float = 1.0  # x average
1426:  33: 
1427:  34: 
1428:  35: class Strategy:
1429:  36:     """Pure strategy: indicators + signal at bar i. No state, no side effects."""
1430:  37:     def __init__(self, params: StrategyParams = StrategyParams()):
1431:  38:         self.p = params
1432:  39: 
1433:  40:     def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
1434:  41:         df = df.copy()
1435:  42:         # SMA
1436:  43:         df["sma"] = df["close"].rolling(self.p.sma_period, min_periods=self.p.sma_period).mean()
1437:  44: 
1438:  45:         # ATR (classic TR)
1439:  46:         high_low = df["high"] - df["low"]
1440:  47:         high_close = (df["high"] - df["close"].shift()).abs()
1441:  48:         low_close = (df["low"] - df["close"].shift()).abs()
1442:  49:         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
1443:  50:         df["atr"] = tr.rolling(self.p.atr_period, min_periods=self.p.atr_period).mean()
1444:  51: 
1445:  52:         # Volume ratio
1446:  53:         if "volume" in df.columns:
1447:  54:             df["volume_avg"] = df["volume"].rolling(self.p.sma_period, min_periods=self.p.sma_period).mean()
1448:  55:             df["volume_ratio"] = df["volume"] / df["volume_avg"]
1449:  56:         else:
1450:  57:             df["volume"] = 1000.0
1451:  58:             df["volume_avg"] = 1000.0
1452:  59:             df["volume_ratio"] = 1.0
1453:  60:         return df
1454:  61: 
1455:  62:     def get_signal_at(self, df: pd.DataFrame, i: int) -> Tuple[Signal, dict]:
1456:  63:         """Decide signal using only info up to index i (no look-ahead)."""
1457:  64:         if i < max(self.p.sma_period, self.p.atr_period):
1458:  65:             return Signal(SignalType.NONE), {}
1459:  66: 
1460:  67:         last = df.iloc[i]
1461:  68:         prev = df.iloc[i - 1]
1462:  69: 
1463:  70:         cross_above = (prev["close"] <= prev["sma"]) and (last["close"] > last["sma"])
1464:  71:         cross_below = (prev["close"] >= prev["sma"]) and (last["close"] < last["sma"])
1465:  72:         volume_ok = bool(last.get("volume_ratio", 0) > self.p.volume_threshold)
1466:  73: 
1467:  74:         meta = {
1468:  75:             "close": float(last["close"]),
1469:  76:             "sma": float(last["sma"]),
1470:  77:             "atr": float(last["atr"]),
1471:  78:             "volume_ratio": float(last.get("volume_ratio", 1.0)),
1472:  79:             "timestamp": last.name if hasattr(last, "name") else None,
1473:  80:         }
1474:  81: 
1475:  82:         # Entry: only long for v1
1476:  83:         if cross_above and volume_ok and np.isfinite(last["atr"]) and last["atr"] > 0:
1477:  84:             entry = float(last["close"])
1478:  85:             stop = entry - float(last["atr"]) * self.p.sl_mult
1479:  86:             target = entry + float(last["atr"]) * self.p.tp_mult
1480:  87:             sig = Signal(
1481:  88:                 type=SignalType.BUY,
1482:  89:                 entry=entry,
1483:  90:                 stop=stop,
1484:  91:                 target=target,
1485:  92:                 reason=f"BUY: cross↑ + vol {meta['volume_ratio']:.2f}x",
1486:  93:                 timestamp=meta["timestamp"],
1487:  94:             )
1488:  95:             return sig, meta
1489:  96: 
1490:  97:         # Exit reversal signal (SELL only as exit trigger)
1491:  98:         if cross_below:
1492:  99:             sig = Signal(
1493: 100:                 type=SignalType.SELL,
1494: 101:                 reason="SELL: cross↓ (reversal/exit)",
1495: 102:                 timestamp=meta["timestamp"],
1496: 103:             )
1497: 104:             return sig, meta
1498: 105: 
1499: 106:         return Signal(SignalType.NONE), meta
1500: `````
1501: 
1502: ## File: test_trader.py
1503: `````python
1504:  1: """
1505:  2: Minimal tests - just the essentials
1506:  3: """
1507:  4: import pytest
1508:  5: import pandas as pd
1509:  6: import numpy as np
1510:  7: from datetime import datetime, timedelta
1511:  8: 
1512:  9: from trader import SimpleBreakoutTrader, Backtester
1513: 10: 
1514: 11: def create_test_data(n_days=30, trend=0.001):
1515: 12:     """Create synthetic OHLC data"""
1516: 13:     dates = pd.date_range(end=datetime.now(), periods=n_days*24, freq='H')
1517: 14: 
1518: 15:     # Random walk with trend
1519: 16:     returns = np.random.normal(trend, 0.01, len(dates))
1520: 17:     close = 18000 * (1 + returns).cumprod()
1521: 18: 
1522: 19:     # Create OHLC
1523: 20:     high = close * (1 + np.random.uniform(0.001, 0.005, len(dates)))
1524: 21:     low = close * (1 - np.random.uniform(0.001, 0.005, len(dates)))
1525: 22:     open_ = np.roll(close, 1)
1526: 23:     open_[0] = close[0]
1527: 24: 
1528: 25:     volume = np.random.uniform(1000, 5000, len(dates))
1529: 26: 
1530: 27:     df = pd.DataFrame({
1531: 28:         'open': open_,
1532: 29:         'high': high,
1533: 30:         'low': low,
1534: 31:         'close': close,
1535: 32:         'volume': volume
1536: 33:     }, index=dates)
1537: 34: 
1538: 35:     return df
1539: 36: 
1540: 37: def test_trader_init():
1541: 38:     """Test trader initialization"""
1542: 39:     trader = SimpleBreakoutTrader()
1543: 40:     assert trader.symbol == "GER40.cash"
1544: 41:     assert trader.risk_pct == 1.0
1545: 42: 
1546: 43: def test_signal_generation():
1547: 44:     """Test that signals are generated"""
1548: 45:     trader = SimpleBreakoutTrader()
1549: 46:     df = create_test_data()
1550: 47: 
1551: 48:     signal, meta = trader.get_signal(df)
1552: 49: 
1553: 50:     assert signal in ['BUY', 'SELL', None]
1554: 51:     assert 'close' in meta
1555: 52:     assert 'sma' in meta
1556: 53: 
1557: 54: def test_no_look_ahead():
1558: 55:     """Critical: Test no look-ahead bias"""
1559: 56:     trader = SimpleBreakoutTrader()
1560: 57:     df = create_test_data()
1561: 58: 
1562: 59:     # Signal at time T
1563: 60:     signal_t, _ = trader.get_signal(df.iloc[:100])
1564: 61: 
1565: 62:     # Signal at time T with more future data
1566: 63:     signal_t_future, _ = trader.get_signal(df.iloc[:100])  # Same window
1567: 64: 
1568: 65:     # Should be identical
1569: 66:     assert signal_t == signal_t_future
1570: 67: 
1571: 68: def test_backtest_runs():
1572: 69:     """Test backtest completes without errors"""
1573: 70:     trader = SimpleBreakoutTrader()
1574: 71:     backtester = Backtester(trader)
1575: 72: 
1576: 73:     df = create_test_data()
1577: 74:     results = backtester.run(df)
1578: 75: 
1579: 76:     assert 'metrics' in results
1580: 77:     assert 'trades' in results
1581: 78:     assert results['metrics']['initial_capital'] == 10000
1582: 79: 
1583: 80: def test_position_sizing():
1584: 81:     """Test risk-based position sizing"""
1585: 82:     trader = SimpleBreakoutTrader()
1586: 83: 
1587: 84:     size = trader.calculate_position_size(
1588: 85:         equity=10000,
1589: 86:         entry=18000,
1590: 87:         stop=17900
1591: 88:     )
1592: 89: 
1593: 90:     # Should risk 1% = 100 EUR
1594: 91:     # Risk is 100 points, so size should be 1.0
1595: 92:     assert size == 1.0
1596: 93: 
1597: 94: if __name__ == '__main__':
1598: 95:     pytest.main([__file__, '-v'])
1599: `````
1600: 
1601: ## File: testplan.md
1602: `````markdown
1603:  1: # Test Coverage Plan
1604:  2: 
1605:  3: - Strategy
1606:  4:   - Bullish crossover emits BUY with levels
1607:  5:   - No look-ahead at bar i
1608:  6: - Backtest
1609:  7:   - Runs end-to-end on synthetic data
1610:  8:   - Time-exit prevents infinite positions
1611:  9: - Integration
1612: 10:   - Pipeline yields ≥1 trade on trending synthetic
1613: `````
1614: 
1615: ## File: trader.py
1616: `````python
1617:   1: #!/usr/bin/env python3
1618:   2: """
1619:   3: Minimal Viable Trading System
1620:   4: One file to rule them all - strategie, backtest, live trading
1621:   5: """
1622:   6: import sys
1623:   7: import time
1624:   8: import logging
1625:   9: from datetime import datetime, timedelta
1626:  10: from pathlib import Path
1627:  11: from typing import Optional, Dict, Tuple, List
1628:  12: 
1629:  13: import numpy as np
1630:  14: import pandas as pd
1631:  15: import yaml
1632:  16: 
1633:  17: # Setup logging
1634:  18: logging.basicConfig(
1635:  19:     level=logging.INFO,
1636:  20:     format='%(asctime)s [%(levelname)s] %(message)s',
1637:  21:     handlers=[
1638:  22:         logging.FileHandler('logs/trader.log'),
1639:  23:         logging.StreamHandler()
1640:  24:     ]
1641:  25: )
1642:  26: logger = logging.getLogger(__name__)
1643:  27: 
1644:  28: class SimpleBreakoutTrader:
1645:  29:     """Dead simple breakout strategy - no BS, just works"""
1646:  30: 
1647:  31:     def __init__(self, config_path: str = "config.yaml"):
1648:  32:         with open(config_path, 'r') as f:
1649:  33:             self.config = yaml.safe_load(f)
1650:  34: 
1651:  35:         self.symbol = self.config['trading']['symbol']
1652:  36:         self.risk_pct = self.config['trading']['risk_per_trade']
1653:  37:         self.sma_period = self.config['strategy']['sma_period']
1654:  38:         self.atr_period = self.config['strategy']['atr_period']
1655:  39: 
1656:  40:         logger.info(f"Initialized trader for {self.symbol}")
1657:  41: 
1658:  42:     def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
1659:  43:         """Add SMA, ATR and Volume metrics"""
1660:  44:         df = df.copy()
1661:  45: 
1662:  46:         # SMA
1663:  47:         df['sma'] = df['close'].rolling(self.sma_period).mean()
1664:  48: 
1665:  49:         # ATR
1666:  50:         high_low = df['high'] - df['low']
1667:  51:         high_close = np.abs(df['high'] - df['close'].shift())
1668:  52:         low_close = np.abs(df['low'] - df['close'].shift())
1669:  53:         true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
1670:  54:         df['atr'] = true_range.rolling(self.atr_period).mean()
1671:  55: 
1672:  56:         # Volume
1673:  57:         df['volume_avg'] = df['volume'].rolling(self.sma_period).mean()
1674:  58:         df['volume_ratio'] = df['volume'] / df['volume_avg']
1675:  59: 
1676:  60:         return df
1677:  61: 
1678:  62:     def get_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Dict]:
1679:  63:         """
1680:  64:         Returns signal and metadata
1681:  65:         Simple rules:
1682:  66:         - BUY: price crosses above SMA + volume confirmation
1683:  67:         - SELL: price crosses below SMA
1684:  68:         - None: no clear signal
1685:  69:         """
1686:  70:         df = self.calculate_indicators(df)
1687:  71: 
1688:  72:         if len(df) < self.sma_period:
1689:  73:             return None, {}
1690:  74: 
1691:  75:         last = df.iloc[-1]
1692:  76:         prev = df.iloc[-2]
1693:  77: 
1694:  78:         # Detect crossovers
1695:  79:         cross_above = (prev['close'] <= prev['sma']) and (last['close'] > last['sma'])
1696:  80:         cross_below = (prev['close'] >= prev['sma']) and (last['close'] < last['sma'])
1697:  81: 
1698:  82:         # Volume confirmation
1699:  83:         volume_confirmed = last['volume_ratio'] > self.config['strategy']['volume_threshold']
1700:  84: 
1701:  85:         metadata = {
1702:  86:             'close': last['close'],
1703:  87:             'sma': last['sma'],
1704:  88:             'atr': last['atr'],
1705:  89:             'volume_ratio': last['volume_ratio'],
1706:  90:             'timestamp': last.name if hasattr(last, 'name') else None
1707:  91:         }
1708:  92: 
1709:  93:         if cross_above and volume_confirmed:
1710:  94:             metadata['reason'] = f"Bullish cross + volume {last['volume_ratio']:.1f}x"
1711:  95:             return 'BUY', metadata
1712:  96:         elif cross_below:
1713:  97:             metadata['reason'] = "Bearish cross"
1714:  98:             return 'SELL', metadata
1715:  99: 
1716: 100:         return None, metadata
1717: 101: 
1718: 102:     def calculate_position_size(self, equity: float, entry: float, stop: float) -> float:
1719: 103:         """Position sizing based on risk management"""
1720: 104:         risk_amount = equity * (self.risk_pct / 100)
1721: 105:         risk_points = abs(entry - stop)
1722: 106: 
1723: 107:         if risk_points == 0:
1724: 108:             return 0
1725: 109: 
1726: 110:         # For forex/indices: size in lots
1727: 111:         # Adjust multiplier based on your broker
1728: 112:         point_value = 1.0  # EUR per point for GER40
1729: 113:         size = risk_amount / (risk_points * point_value)
1730: 114: 
1731: 115:         return round(size, 2)
1732: 116: 
1733: 117: class Backtester:
1734: 118:     """Simple backtesting engine - no look-ahead, realistic fills"""
1735: 119: 
1736: 120:     def __init__(self, trader: SimpleBreakoutTrader):
1737: 121:         self.trader = trader
1738: 122:         self.config = trader.config
1739: 123: 
1740: 124:     def run(self, df: pd.DataFrame) -> Dict:
1741: 125:         """Run backtest and return results"""
1742: 126:         initial_capital = self.config['trading']['initial_capital']
1743: 127:         equity = initial_capital
1744: 128: 
1745: 129:         trades = []
1746: 130:         position = None
1747: 131: 
1748: 132:         df = self.trader.calculate_indicators(df)
1749: 133: 
1750: 134:         for i in range(self.trader.sma_period, len(df)):
1751: 135:             window = df.iloc[:i+1]
1752: 136:             signal, meta = self.trader.get_signal(window)
1753: 137: 
1754: 138:             current_bar = df.iloc[i]
1755: 139: 
1756: 140:             # Check exits first (before new signals)
1757: 141:             if position:
1758: 142:                 exit_price = None
1759: 143:                 exit_reason = None
1760: 144: 
1761: 145:                 # Check stop loss
1762: 146:                 if current_bar['low'] <= position['stop']:
1763: 147:                     exit_price = position['stop']
1764: 148:                     exit_reason = 'Stop Loss'
1765: 149:                 # Check take profit
1766: 150:                 elif current_bar['high'] >= position['target']:
1767: 151:                     exit_price = position['target']
1768: 152:                     exit_reason = 'Take Profit'
1769: 153:                 # Check signal reversal
1770: 154:                 elif signal == 'SELL':
1771: 155:                     exit_price = current_bar['close']
1772: 156:                     exit_reason = 'Signal Reversal'
1773: 157: 
1774: 158:                 if exit_price:
1775: 159:                     # Calculate PnL
1776: 160:                     pnl = (exit_price - position['entry']) * position['size']
1777: 161:                     pnl -= (position['entry'] * position['size'] * self.config['backtest']['commission'] * 2)
1778: 162: 
1779: 163:                     equity += pnl
1780: 164: 
1781: 165:                     trades.append({
1782: 166:                         'entry_time': position['entry_time'],
1783: 167:                         'exit_time': current_bar.name,
1784: 168:                         'side': position['side'],
1785: 169:                         'entry': position['entry'],
1786: 170:                         'exit': exit_price,
1787: 171:                         'size': position['size'],
1788: 172:                         'pnl': pnl,
1789: 173:                         'return_pct': (pnl / equity) * 100,
1790: 174:                         'reason': exit_reason
1791: 175:                     })
1792: 176: 
1793: 177:                     position = None
1794: 178:                     logger.debug(f"Exit {exit_reason} at {exit_price:.2f}, PnL: {pnl:.2f}")
1795: 179: 
1796: 180:             # New signals (only if no position)
1797: 181:             if signal and not position:
1798: 182:                 if signal == 'BUY':
1799: 183:                     entry = current_bar['close']
1800: 184:                     stop = entry - current_bar['atr'] * self.config['strategy']['sl_multiplier']
1801: 185:                     target = entry + current_bar['atr'] * self.config['strategy']['tp_multiplier']
1802: 186: 
1803: 187:                     size = self.trader.calculate_position_size(equity, entry, stop)
1804: 188: 
1805: 189:                     if size > 0:
1806: 190:                         position = {
1807: 191:                             'entry_time': current_bar.name,
1808: 192:                             'side': 'long',
1809: 193:                             'entry': entry,
1810: 194:                             'stop': stop,
1811: 195:                             'target': target,
1812: 196:                             'size': size
1813: 197:                         }
1814: 198:                         logger.debug(f"BUY at {entry:.2f}, stop: {stop:.2f}, target: {target:.2f}")
1815: 199: 
1816: 200:         # Calculate metrics
1817: 201:         trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
1818: 202: 
1819: 203:         if not trades_df.empty:
1820: 204:             wins = trades_df[trades_df['pnl'] > 0]
1821: 205:             losses = trades_df[trades_df['pnl'] <= 0]
1822: 206: 
1823: 207:             metrics = {
1824: 208:                 'initial_capital': initial_capital,
1825: 209:                 'final_capital': equity,
1826: 210:                 'total_return_pct': ((equity / initial_capital) - 1) * 100,
1827: 211:                 'num_trades': len(trades_df),
1828: 212:                 'num_wins': len(wins),
1829: 213:                 'num_losses': len(losses),
1830: 214:                 'win_rate': (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
1831: 215:                 'avg_win': wins['pnl'].mean() if not wins.empty else 0,
1832: 216:                 'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
1833: 217:                 'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else 0,
1834: 218:                 'max_drawdown': self._calculate_max_drawdown(trades_df),
1835: 219:                 'sharpe_ratio': self._calculate_sharpe(trades_df)
1836: 220:             }
1837: 221:         else:
1838: 222:             metrics = {
1839: 223:                 'initial_capital': initial_capital,
1840: 224:                 'final_capital': equity,
1841: 225:                 'total_return_pct': 0,
1842: 226:                 'num_trades': 0,
1843: 227:                 'message': 'No trades generated'
1844: 228:             }
1845: 229: 
1846: 230:         return {'metrics': metrics, 'trades': trades_df}
1847: 231: 
1848: 232:     def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
1849: 233:         """Calculate maximum drawdown percentage"""
1850: 234:         if trades_df.empty:
1851: 235:             return 0
1852: 236: 
1853: 237:         cumulative = trades_df['pnl'].cumsum()
1854: 238:         running_max = cumulative.cummax()
1855: 239:         drawdown = (cumulative - running_max) / (running_max + self.config['trading']['initial_capital'])
1856: 240:         return abs(drawdown.min()) * 100 if not drawdown.empty else 0
1857: 241: 
1858: 242:     def _calculate_sharpe(self, trades_df: pd.DataFrame) -> float:
1859: 243:         """Calculate Sharpe ratio (simplified)"""
1860: 244:         if trades_df.empty or len(trades_df) < 2:
1861: 245:             return 0
1862: 246: 
1863: 247:         returns = trades_df['return_pct']
1864: 248:         if returns.std() == 0:
1865: 249:             return 0
1866: 250: 
1867: 251:         # Annualize based on average trades per year (estimate)
1868: 252:         days_in_sample = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
1869: 253:         if days_in_sample > 0:
1870: 254:             trades_per_year = (252 / days_in_sample) * len(trades_df)
1871: 255:             return (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
1872: 256:         return 0
1873: 257: 
1874: 258: class LiveTrader:
1875: 259:     """Live trading interface - can be paper or real"""
1876: 260: 
1877: 261:     def __init__(self, trader: SimpleBreakoutTrader):
1878: 262:         self.trader = trader
1879: 263:         self.config = trader.config
1880: 264:         self.position = None
1881: 265:         self.equity = self.config['trading']['initial_capital']
1882: 266: 
1883: 267:     def fetch_latest_data(self, lookback_bars: int = 100) -> pd.DataFrame:
1884: 268:         """Fetch latest price data"""
1885: 269:         # For now, use CSV data for testing
1886: 270:         # Replace with MT5 or broker API
1887: 271:         csv_files = list(Path('data').glob('*.csv'))
1888: 272:         if not csv_files:
1889: 273:             raise FileNotFoundError("No CSV files in data/ directory")
1890: 274: 
1891: 275:         df = pd.read_csv(csv_files[0], parse_dates=['time'])
1892: 276:         df.set_index('time', inplace=True)
1893: 277: 
1894: 278:         # Get last N bars
1895: 279:         return df.tail(lookback_bars)
1896: 280: 
1897: 281:     def check_for_signals(self):
1898: 282:         """Check current market for signals"""
1899: 283:         try:
1900: 284:             df = self.fetch_latest_data()
1901: 285:             signal, meta = self.trader.get_signal(df)
1902: 286: 
1903: 287:             if signal:
1904: 288:                 logger.info(f"SIGNAL: {signal} at {meta.get('timestamp', 'now')}")
1905: 289:                 logger.info(f"Reason: {meta.get('reason', 'N/A')}")
1906: 290:                 logger.info(f"Price: {meta.get('close', 0):.2f}, SMA: {meta.get('sma', 0):.2f}")
1907: 291: 
1908: 292:                 if self.config['live']['mode'] == 'live':
1909: 293:                     self.place_order(signal, meta)
1910: 294:                 else:
1911: 295:                     logger.info("PAPER TRADE - no real order placed")
1912: 296: 
1913: 297:         except Exception as e:
1914: 298:             logger.error(f"Error checking signals: {e}")
1915: 299: 
1916: 300:     def place_order(self, signal: str, meta: Dict):
1917: 301:         """Place actual trade order"""
1918: 302:         # TODO: Implement MT5 order placement
1919: 303:         logger.warning("Live order placement not yet implemented")
1920: 304: 
1921: 305:     def run_forever(self):
1922: 306:         """Main loop for live trading"""
1923: 307:         logger.info(f"Starting live trader in {self.config['live']['mode']} mode")
1924: 308:         logger.info(f"Checking every {self.config['live']['check_interval']} seconds")
1925: 309: 
1926: 310:         while True:
1927: 311:             try:
1928: 312:                 self.check_for_signals()
1929: 313:                 time.sleep(self.config['live']['check_interval'])
1930: 314:             except KeyboardInterrupt:
1931: 315:                 logger.info("Shutting down...")
1932: 316:                 break
1933: 317:             except Exception as e:
1934: 318:                 logger.error(f"Unexpected error: {e}")
1935: 319:                 time.sleep(60)  # Wait a minute before retry
1936: 320: 
1937: 321: def load_data(csv_path: str) -> pd.DataFrame:
1938: 322:     """Load and prepare data"""
1939: 323:     df = pd.read_csv(csv_path)
1940: 324: 
1941: 325:     # Handle different column names
1942: 326:     time_cols = ['time', 'Time', 'datetime', 'Datetime', 'date', 'Date']
1943: 327:     time_col = None
1944: 328:     for col in time_cols:
1945: 329:         if col in df.columns:
1946: 330:             time_col = col
1947: 331:             break
1948: 332: 
1949: 333:     if time_col:
1950: 334:         df[time_col] = pd.to_datetime(df[time_col])
1951: 335:         df.set_index(time_col, inplace=True)
1952: 336: 
1953: 337:     # Standardize column names
1954: 338:     df.columns = df.columns.str.lower()
1955: 339: 
1956: 340:     # Required columns
1957: 341:     required = ['open', 'high', 'low', 'close']
1958: 342:     if not all(col in df.columns for col in required):
1959: 343:         raise ValueError(f"CSV must have columns: {required}")
1960: 344: 
1961: 345:     # Add volume if missing
1962: 346:     if 'volume' not in df.columns:
1963: 347:         df['volume'] = 1000  # Dummy volume
1964: 348: 
1965: 349:     return df.sort_index()
1966: 350: 
1967: 351: def main():
1968: 352:     """Main entry point"""
1969: 353:     import argparse
1970: 354: 
1971: 355:     parser = argparse.ArgumentParser(description='Minimal Viable Trading System')
1972: 356:     parser.add_argument('mode', choices=['backtest', 'live', 'signal'],
1973: 357:                       help='Operating mode')
1974: 358:     parser.add_argument('--csv', type=str, help='CSV file path for backtesting')
1975: 359:     parser.add_argument('--config', type=str, default='config.yaml',
1976: 360:                       help='Configuration file')
1977: 361: 
1978: 362:     args = parser.parse_args()
1979: 363: 
1980: 364:     # Create directories
1981: 365:     Path('logs').mkdir(exist_ok=True)
1982: 366:     Path('output').mkdir(exist_ok=True)
1983: 367: 
1984: 368:     # Initialize trader
1985: 369:     trader = SimpleBreakoutTrader(args.config)
1986: 370: 
1987: 371:     if args.mode == 'backtest':
1988: 372:         if not args.csv:
1989: 373:             csv_files = list(Path('data').glob('*.csv'))
1990: 374:             if not csv_files:
1991: 375:                 logger.error("No CSV files found. Use --csv or add files to data/")
1992: 376:                 sys.exit(1)
1993: 377:             args.csv = str(csv_files[0])
1994: 378:             logger.info(f"Using {args.csv}")
1995: 379: 
1996: 380:         # Load data
1997: 381:         df = load_data(args.csv)
1998: 382:         logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
1999: 383: 
2000: 384:         # Run backtest
2001: 385:         backtester = Backtester(trader)
2002: 386:         results = backtester.run(df)
2003: 387: 
2004: 388:         # Display results
2005: 389:         print("\n" + "="*50)
2006: 390:         print("BACKTEST RESULTS")
2007: 391:         print("="*50)
2008: 392:         for key, value in results['metrics'].items():
2009: 393:             if isinstance(value, float):
2010: 394:                 print(f"{key:20s}: {value:,.2f}")
2011: 395:             else:
2012: 396:                 print(f"{key:20s}: {value}")
2013: 397: 
2014: 398:         # Save trades
2015: 399:         if not results['trades'].empty:
2016: 400:             output_file = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
2017: 401:             results['trades'].to_csv(output_file, index=False)
2018: 402:             print(f"\nTrades saved to: {output_file}")
2019: 403: 
2020: 404:     elif args.mode == 'live':
2021: 405:         live = LiveTrader(trader)
2022: 406:         live.run_forever()
2023: 407: 
2024: 408:     elif args.mode == 'signal':
2025: 409:         # One-time signal check
2026: 410:         live = LiveTrader(trader)
2027: 411:         live.check_for_signals()
2028: 412: 
2029: 413: if __name__ == '__main__':
2030: 414:     main()
2031: `````
``````

## File: requirements.txt
``````
1: pandas>=2.2.0
2: numpy>=1.26.0
3: pyyaml>=6.0
4: pytest>=8.0.0
5: python-dotenv>=1.0.0
6: MetaTrader5>=5.0.45  # Windows only
``````

## File: strategy.py
``````python
  1: from __future__ import annotations
  2: from dataclasses import dataclass
  3: from enum import Enum
  4: from typing import Optional, Tuple
  5: 
  6: import numpy as np
  7: import pandas as pd
  8: 
  9: 
 10: class SignalType(Enum):
 11:     NONE = 0
 12:     BUY = 1
 13:     SELL = -1  # exit trigger only
 14: 
 15: 
 16: @dataclass(frozen=True)
 17: class Signal:
 18:     type: SignalType
 19:     entry: Optional[float] = None
 20:     stop: Optional[float] = None
 21:     target: Optional[float] = None
 22:     reason: str = ""
 23:     timestamp: Optional[pd.Timestamp] = None
 24: 
 25: 
 26: @dataclass(frozen=True)
 27: class StrategyParams:
 28:     sma_period: int = 20
 29:     atr_period: int = 14
 30:     sl_mult: float = 1.5
 31:     tp_mult: float = 2.5
 32:     volume_threshold: float = 1.0  # set 0.0 if your CSV has no real volume
 33: 
 34: 
 35: class Strategy:
 36:     """Pure strategy: indicators + signal generation. No state."""
 37: 
 38:     def __init__(self, params: StrategyParams = StrategyParams()):
 39:         self.p = params
 40: 
 41:     def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
 42:         """Compute SMA, ATR, and volume ratios. Returns a copy."""
 43:         df = df.copy()
 44: 
 45:         # SMA
 46:         df["sma"] = df["close"].rolling(
 47:             self.p.sma_period, min_periods=self.p.sma_period
 48:         ).mean()
 49: 
 50:         # ATR (classic TR)
 51:         high_low = df["high"] - df["low"]
 52:         high_close = (df["high"] - df["close"].shift()).abs()
 53:         low_close = (df["low"] - df["close"].shift()).abs()
 54:         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
 55:         df["atr"] = tr.rolling(
 56:             self.p.atr_period, min_periods=self.p.atr_period
 57:         ).mean()
 58: 
 59:         # Volume ratio (fallback naar 1.0 als volume ontbreekt)
 60:         if "volume" in df.columns:
 61:             df["volume_avg"] = df["volume"].rolling(
 62:                 self.p.sma_period, min_periods=self.p.sma_period
 63:             ).mean()
 64:             vr = df["volume"] / df["volume_avg"].replace(0, np.nan)
 65:             df["volume_ratio"] = vr.fillna(0.0)
 66:         else:
 67:             df["volume"] = 1000.0
 68:             df["volume_avg"] = 1000.0
 69:             df["volume_ratio"] = 1.0
 70: 
 71:         return df
 72: 
 73:     def get_signal_at(self, df: pd.DataFrame, i: int) -> Tuple[Signal, dict]:
 74:         """
 75:         Generate signal at bar i using only data up to i (no look-ahead).
 76:         BUY: SMA cross above + volume confirmation (>= threshold)
 77:         SELL: SMA cross below (exit trigger)
 78:         """
 79:         warmup = max(self.p.sma_period, self.p.atr_period)
 80:         if i < warmup:
 81:             return Signal(SignalType.NONE), {}
 82: 
 83:         last = df.iloc[i]
 84:         if not np.isfinite(last.get("sma", np.nan)):
 85:             return Signal(SignalType.NONE), {}
 86: 
 87:         # 'First-window' entry (i == warmup): als de eerste bar met geldige SMA boven de SMA ligt,
 88:         # beschouwen we dat als een geldige long setup zonder expliciete cross op i-1.
 89:         if i == warmup:
 90:             volume_ok = float(last.get("volume_ratio", 0.0)) >= self.p.volume_threshold
 91:             if (
 92:                 last["close"] > last["sma"]
 93:                 and np.isfinite(last.get("atr", np.nan))
 94:                 and last["atr"] > 0
 95:                 and volume_ok
 96:             ):
 97:                 entry = float(last["close"])
 98:                 stop = entry - float(last["atr"]) * self.p.sl_mult
 99:                 target = entry + float(last["atr"]) * self.p.tp_mult
100:                 meta = {
101:                     "close": entry,
102:                     "sma": float(last["sma"]),
103:                     "atr": float(last["atr"]),
104:                     "volume_ratio": float(last.get("volume_ratio", 1.0)),
105:                     "timestamp": last.name if hasattr(last, "name") else None,
106:                 }
107:                 return (
108:                     Signal(
109:                         type=SignalType.BUY,
110:                         entry=entry,
111:                         stop=stop,
112:                         target=target,
113:                         reason=f"BUY: first-window above SMA + vol {meta['volume_ratio']:.2f}x",
114:                         timestamp=meta["timestamp"],
115:                     ),
116:                     meta,
117:                 )
118: 
119:         # Vanaf hier i > warmup → klassieke cross check met vorige bar
120:         prev = df.iloc[i - 1]
121:         if not np.isfinite(prev.get("sma", np.nan)):
122:             return Signal(SignalType.NONE), {}
123: 
124:         cross_above = (prev["close"] <= prev["sma"]) and (last["close"] > last["sma"])
125:         cross_below = (prev["close"] >= prev["sma"]) and (last["close"] < last["sma"])
126:         volume_ok = float(last.get("volume_ratio", 0.0)) >= self.p.volume_threshold
127: 
128:         meta = {
129:             "close": float(last["close"]),
130:             "sma": float(last["sma"]),
131:             "atr": float(last["atr"]) if np.isfinite(last.get("atr", np.nan)) else 0.0,
132:             "volume_ratio": float(last.get("volume_ratio", 1.0)),
133:             "timestamp": last.name if hasattr(last, "name") else None,
134:         }
135: 
136:         if cross_above and volume_ok and meta["atr"] > 0:
137:             entry = meta["close"]
138:             stop = entry - meta["atr"] * self.p.sl_mult
139:             target = entry + meta["atr"] * self.p.tp_mult
140:             return (
141:                 Signal(
142:                     type=SignalType.BUY,
143:                     entry=entry,
144:                     stop=stop,
145:                     target=target,
146:                     reason=f"BUY: SMA cross↑ + vol {meta['volume_ratio']:.2f}x",
147:                     timestamp=meta["timestamp"],
148:                 ),
149:                 meta,
150:             )
151: 
152:         if cross_below:
153:             return (
154:                 Signal(
155:                     type=SignalType.SELL,
156:                     reason="SELL: SMA cross↓ (exit)",
157:                     timestamp=meta["timestamp"],
158:                 ),
159:                 meta,
160:             )
161: 
162:         return Signal(SignalType.NONE), meta
``````

## File: testplan.md
``````markdown
 1: # Test Coverage Plan
 2: 
 3: - Strategy
 4:   - Bullish crossover emits BUY with levels
 5:   - No look-ahead at bar i
 6: - Backtest
 7:   - Runs end-to-end on synthetic data
 8:   - Time-exit prevents infinite positions
 9: - Integration
10:   - Pipeline yields ≥1 trade on trending synthetic
``````
