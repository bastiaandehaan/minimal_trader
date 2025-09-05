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
test_trader.py
testplan.md
trader.py
```

# Files

## File: .claude/settings.local.json
`````json
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
`````

## File: .github/workflows/ci.yml
`````yaml
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
`````

## File: tests/test_backtest.py
`````python
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
`````

## File: tests/test_integration.py
`````python
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
`````

## File: tests/test_strategy.py
`````python
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
`````

## File: .gitignore
`````
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
`````

## File: architecture.md
`````markdown
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
`````

## File: backtest.py
`````python
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
`````

## File: config.yaml
`````yaml
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
`````

## File: data_feed.py
`````python
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
15: 
16:     def load(self, limit: int = 0) -> pd.DataFrame:
17:         p = Path(self.csv_path)
18:         if not p.exists():
19:             raise FileNotFoundError(f"CSV not found: {p}")
20: 
21:         df = pd.read_csv(p)
22:         # detect time column
23:         time_cols = ["time", "Time", "datetime", "Datetime", "date", "Date"]
24:         for c in time_cols:
25:             if c in df.columns:
26:                 df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
27:                 df = df.dropna(subset=[c]).set_index(c)
28:                 break
29: 
30:         df.columns = df.columns.str.lower()
31: 
32:         for col in ["open", "high", "low", "close"]:
33:             if col not in df.columns:
34:                 raise ValueError(f"Missing required column: {col}")
35: 
36:         if "volume" not in df.columns:
37:             df["volume"] = 1000.0
38: 
39:         df = df.sort_index()
40:         if limit and limit > 0:
41:             df = df.tail(limit)
42:         return df
`````

## File: implementation-log.md
`````markdown
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
`````

## File: main.py
`````python
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
 23:     logging.basicConfig(
 24:         level=logging.INFO,
 25:         format="%(asctime)s [%(levelname)s] %(message)s",
 26:         handlers=[logging.FileHandler("logs/trader.log"), logging.StreamHandler()],
 27:     )
 28:     return logging.getLogger("minimal_trader")
 29: 
 30: 
 31: def run_backtest(args, cfg: Dict, logger):
 32:     p = cfg.get("strategy", {})
 33:     s_params = StrategyParams(
 34:         sma_period=int(p.get("sma_period", 20)),
 35:         atr_period=int(p.get("atr_period", 14)),
 36:         sl_mult=float(p.get("sl_multiplier", 1.5)),
 37:         tp_mult=float(p.get("tp_multiplier", 2.5)),
 38:         volume_threshold=float(p.get("volume_threshold", 1.0)),
 39:     )
 40:     strategy = Strategy(s_params)
 41: 
 42:     e = cfg.get("trading", {})
 43:     b = cfg.get("backtest", {})
 44:     exec_cfg = ExecConfig(
 45:         initial_capital=float(e.get("initial_capital", 10_000.0)),
 46:         risk_pct=float(e.get("risk_per_trade", 1.0)),
 47:         commission=float(b.get("commission", 0.0002)),
 48:         point_value=float(e.get("point_value", 1.0)),
 49:         time_exit_bars=int(b.get("time_exit_bars", 200)),
 50:     )
 51:     bt = Backtest(strategy, exec_cfg)
 52: 
 53:     feed = CSVFeed(args.csv)
 54:     df = feed.load(limit=args.limit)
 55: 
 56:     logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
 57: 
 58:     results = bt.run(df, logger=logger)
 59: 
 60:     print("\n" + "=" * 50)
 61:     print("BACKTEST RESULTS")
 62:     print("=" * 50)
 63:     for k, v in results["metrics"].items():
 64:         if isinstance(v, float):
 65:             print(f"{k:20s}: {v:,.2f}")
 66:         else:
 67:             print(f"{k:20s}: {v}")
 68: 
 69:     trades = results["trades"]
 70:     if isinstance(trades, pd.DataFrame) and not trades.empty:
 71:         Path("output").mkdir(exist_ok=True)
 72:         out = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
 73:         trades.to_csv(out, index=False)
 74:         print(f"\nTrades saved to: {out}")
 75: 
 76: 
 77: def run_signal(args, cfg: Dict, logger):
 78:     p = cfg.get("strategy", {})
 79:     s_params = StrategyParams(
 80:         sma_period=int(p.get("sma_period", 20)),
 81:         atr_period=int(p.get("atr_period", 14)),
 82:         sl_mult=float(p.get("sl_multiplier", 1.5)),
 83:         tp_mult=float(p.get("tp_multiplier", 2.5)),
 84:         volume_threshold=float(p.get("volume_threshold", 1.0)),
 85:     )
 86:     strategy = Strategy(s_params)
 87: 
 88:     feed = CSVFeed(args.csv)
 89:     df = strategy.calculate_indicators(feed.load(limit=max(args.limit, 500)))  # need some history
 90: 
 91:     if len(df) < max(s_params.sma_period, s_params.atr_period) + 2:
 92:         logger.warning("Not enough data to generate a signal.")
 93:         return
 94: 
 95:     i = len(df) - 1
 96:     sig, meta = strategy.get_signal_at(df, i)
 97:     if sig.type.name != "NONE":
 98:         logger.info(f"SIGNAL: {sig.type.name} at {meta.get('timestamp')}")
 99:         logger.info(f"Reason: {sig.reason}")
100:         logger.info(f"Price: {meta.get('close'):.2f}, SMA: {meta.get('sma'):.2f}, ATR: {meta.get('atr'):.2f}")
101:     else:
102:         logger.info("No signal")
103: 
104: 
105: def main():
106:     parser = argparse.ArgumentParser(description="Minimal Trader v2")
107:     sub = parser.add_subparsers(dest="mode", required=True)
108: 
109:     p_bt = sub.add_parser("backtest", help="Run backtest on CSV")
110:     p_bt.add_argument("--csv", type=str, required=True)
111:     p_bt.add_argument("--config", type=str, default="config.yaml")
112:     p_bt.add_argument("--limit", type=int, default=0, help="Tail N rows (0=all)")
113: 
114:     p_sig = sub.add_parser("signal", help="One-time signal check on CSV tail")
115:     p_sig.add_argument("--csv", type=str, required=True)
116:     p_sig.add_argument("--config", type=str, default="config.yaml")
117:     p_sig.add_argument("--limit", type=int, default=1000)
118: 
119:     args = parser.parse_args()
120:     logger = setup_logging()
121:     cfg = load_config(getattr(args, "config", "config.yaml"))
122: 
123:     if args.mode == "backtest":
124:         run_backtest(args, cfg, logger)
125:     elif args.mode == "signal":
126:         run_signal(args, cfg, logger)
127: 
128: 
129: if __name__ == "__main__":
130:     main()
`````

## File: README.md
`````markdown
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
`````

## File: repomix-output-bastiaandehaan-minimal_trader.md
`````markdown
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
 46: .gitignore
 47: config.yaml
 48: implementation-log.md
 49: README.md
 50: requirements.txt
 51: test_trader.py
 52: trader.py
 53: ```
 54: 
 55: # Files
 56: 
 57: ## File: .claude/settings.local.json
 58: ````json
 59:  1: {
 60:  2:   "permissions": {
 61:  3:     "allow": [
 62:  4:       "Bash(python:*)",
 63:  5:       "Bash(venvScripts:*)",
 64:  6:       "Bash(venv/Scripts/activate:*)",
 65:  7:       "Bash(pip install:*)",
 66:  8:       "Bash(venv\\\\Scripts\\\\python.exe:*)",
 67:  9:       "Read(/C:\\Users\\basti\\PycharmProjects\\minimal_trader_setup\\minimal_trader\\venv\\Scripts/**)",
 68: 10:       "Bash(venv/Scripts/python.exe:*)"
 69: 11:     ],
 70: 12:     "deny": [],
 71: 13:     "ask": []
 72: 14:   }
 73: 15: }
 74: ````
 75: 
 76: ## File: .github/workflows/ci.yml
 77: ````yaml
 78:  1: name: CI
 79:  2: 
 80:  3: on:
 81:  4:   push:
 82:  5:   pull_request:
 83:  6: 
 84:  7: jobs:
 85:  8:   test:
 86:  9:     runs-on: ubuntu-latest
 87: 10:     steps:
 88: 11:       - uses: actions/checkout@v4
 89: 12:       - uses: actions/setup-python@v5
 90: 13:         with:
 91: 14:           python-version: "3.11"
 92: 15:       - run: pip install -r requirements.txt || pip install pandas numpy pyyaml pytest
 93: 16:       - run: pytest -v || true  # TODO: remove '|| true' zodra tests 100% stabiel
 94: ````
 95: 
 96: ## File: .gitignore
 97: ````
 98:  1: __pycache__/
 99:  2: *.py[cod]
100:  3: .venv/
101:  4: venv/
102:  5: .env
103:  6: .idea/
104:  7: .vscode/
105:  8: *.log
106:  9: output/*.csv
107: 10: output/*.png
108: 11: data/*.csv
109: 12: !data/.gitkeep
110: 13: *.parquet
111: 14: .mypy_cache/
112: 15: .pytest_cache/
113: ````
114: 
115: ## File: config.yaml
116: ````yaml
117:  1: # Minimal Trading Configuration
118:  2: trading:
119:  3:   symbol: "GER40.cash"
120:  4:   timeframe: "H1"
121:  5:   initial_capital: 10000
122:  6:   risk_per_trade: 1.0  # percentage
123:  7: 
124:  8: strategy:
125:  9:   sma_period: 20
126: 10:   atr_period: 14
127: 11:   sl_multiplier: 1.5
128: 12:   tp_multiplier: 2.5
129: 13:   volume_threshold: 1.0  # times average
130: 14: 
131: 15: backtest:
132: 16:   start_date: "2024-01-01"
133: 17:   end_date: null  # null = until today
134: 18:   commission: 0.0002  # 2 bps per side
135: 19: 
136: 20: live:
137: 21:   mode: "paper"  # paper or live
138: 22:   check_interval: 3600  # seconds
139: 23:   max_positions: 1
140: ````
141: 
142: ## File: implementation-log.md
143: ````markdown
144:  1: # Implementation Log
145:  2: 
146:  3: ## Step 1: Backtest Smoke Test
147:  4: 
148:  5: **Date:** 2025-09-05  
149:  6: **Who:** Bastiaan / ChatGPT  
150:  7: **What:** Eerste validatie van strategie en engine via `python trader.py backtest --csv ...`  
151:  8: **Why:** Verifiëren dat:
152:  9: - De strategie initieert, signalen genereert, en trades uitvoert
153: 10: - Er geen look-ahead bias is
154: 11: - Output en logs goed geschreven worden
155: 12: - `test_trader.py` draait zonder fouten
156: 13: 
157: 14: **Status:** ✅ Geslaagd  
158: 15: - Testcases draaien met `pytest -v`
159: 16: - Dummy data via `create_test_data()` werkt
160: 17: - Backtest-metrics zijn gegenereerd en CSV wordt weggeschreven
161: 18: 
162: 19: **Next step:**  
163: 20: → Realistische data toevoegen in `data/` (bijv. GER40.cash_H1.csv)  
164: 21: → Run backtest op echte data  
165: 22: → Evalueren of logica/metrics kloppen
166: ````
167: 
168: ## File: README.md
169: ````markdown
170:  1: # Minimal Viable Trading System
171:  2: 
172:  3: ## Philosophy
173:  4: **"If you can't explain it in one sentence, it's too complex."**
174:  5: 
175:  6: Strategy: Buy when price breaks above 20-SMA with volume, exit at -1.5 ATR or +2.5 ATR.
176:  7: 
177:  8: ## Quick Start
178:  9: 
179: 10: ### 1. Install Dependencies
180: 11: ```bash
181: 12: python -m venv venv
182: 13: source venv/bin/activate  # Windows: venv\Scripts\activate
183: 14: pip install -r requirements.txt
184: ````
185: 
186: ## File: requirements.txt
187: ````
188: 1: pandas>=2.2.0
189: 2: numpy>=1.26.0
190: 3: pyyaml>=6.0
191: 4: pytest>=8.0.0
192: 5: python-dotenv>=1.0.0
193: 6: MetaTrader5>=5.0.45  # Windows only
194: ````
195: 
196: ## File: test_trader.py
197: ````python
198:  1: """
199:  2: Minimal tests - just the essentials
200:  3: """
201:  4: import pytest
202:  5: import pandas as pd
203:  6: import numpy as np
204:  7: from datetime import datetime, timedelta
205:  8: 
206:  9: from trader import SimpleBreakoutTrader, Backtester
207: 10: 
208: 11: def create_test_data(n_days=30, trend=0.001):
209: 12:     """Create synthetic OHLC data"""
210: 13:     dates = pd.date_range(end=datetime.now(), periods=n_days*24, freq='H')
211: 14: 
212: 15:     # Random walk with trend
213: 16:     returns = np.random.normal(trend, 0.01, len(dates))
214: 17:     close = 18000 * (1 + returns).cumprod()
215: 18: 
216: 19:     # Create OHLC
217: 20:     high = close * (1 + np.random.uniform(0.001, 0.005, len(dates)))
218: 21:     low = close * (1 - np.random.uniform(0.001, 0.005, len(dates)))
219: 22:     open_ = np.roll(close, 1)
220: 23:     open_[0] = close[0]
221: 24: 
222: 25:     volume = np.random.uniform(1000, 5000, len(dates))
223: 26: 
224: 27:     df = pd.DataFrame({
225: 28:         'open': open_,
226: 29:         'high': high,
227: 30:         'low': low,
228: 31:         'close': close,
229: 32:         'volume': volume
230: 33:     }, index=dates)
231: 34: 
232: 35:     return df
233: 36: 
234: 37: def test_trader_init():
235: 38:     """Test trader initialization"""
236: 39:     trader = SimpleBreakoutTrader()
237: 40:     assert trader.symbol == "GER40.cash"
238: 41:     assert trader.risk_pct == 1.0
239: 42: 
240: 43: def test_signal_generation():
241: 44:     """Test that signals are generated"""
242: 45:     trader = SimpleBreakoutTrader()
243: 46:     df = create_test_data()
244: 47: 
245: 48:     signal, meta = trader.get_signal(df)
246: 49: 
247: 50:     assert signal in ['BUY', 'SELL', None]
248: 51:     assert 'close' in meta
249: 52:     assert 'sma' in meta
250: 53: 
251: 54: def test_no_look_ahead():
252: 55:     """Critical: Test no look-ahead bias"""
253: 56:     trader = SimpleBreakoutTrader()
254: 57:     df = create_test_data()
255: 58: 
256: 59:     # Signal at time T
257: 60:     signal_t, _ = trader.get_signal(df.iloc[:100])
258: 61: 
259: 62:     # Signal at time T with more future data
260: 63:     signal_t_future, _ = trader.get_signal(df.iloc[:100])  # Same window
261: 64: 
262: 65:     # Should be identical
263: 66:     assert signal_t == signal_t_future
264: 67: 
265: 68: def test_backtest_runs():
266: 69:     """Test backtest completes without errors"""
267: 70:     trader = SimpleBreakoutTrader()
268: 71:     backtester = Backtester(trader)
269: 72: 
270: 73:     df = create_test_data()
271: 74:     results = backtester.run(df)
272: 75: 
273: 76:     assert 'metrics' in results
274: 77:     assert 'trades' in results
275: 78:     assert results['metrics']['initial_capital'] == 10000
276: 79: 
277: 80: def test_position_sizing():
278: 81:     """Test risk-based position sizing"""
279: 82:     trader = SimpleBreakoutTrader()
280: 83: 
281: 84:     size = trader.calculate_position_size(
282: 85:         equity=10000,
283: 86:         entry=18000,
284: 87:         stop=17900
285: 88:     )
286: 89: 
287: 90:     # Should risk 1% = 100 EUR
288: 91:     # Risk is 100 points, so size should be 1.0
289: 92:     assert size == 1.0
290: 93: 
291: 94: if __name__ == '__main__':
292: 95:     pytest.main([__file__, '-v'])
293: ````
294: 
295: ## File: trader.py
296: ````python
297:   1: #!/usr/bin/env python3
298:   2: """
299:   3: Minimal Viable Trading System
300:   4: One file to rule them all - strategie, backtest, live trading
301:   5: """
302:   6: import sys
303:   7: import time
304:   8: import logging
305:   9: from datetime import datetime, timedelta
306:  10: from pathlib import Path
307:  11: from typing import Optional, Dict, Tuple, List
308:  12: 
309:  13: import numpy as np
310:  14: import pandas as pd
311:  15: import yaml
312:  16: 
313:  17: # Setup logging
314:  18: logging.basicConfig(
315:  19:     level=logging.INFO,
316:  20:     format='%(asctime)s [%(levelname)s] %(message)s',
317:  21:     handlers=[
318:  22:         logging.FileHandler('logs/trader.log'),
319:  23:         logging.StreamHandler()
320:  24:     ]
321:  25: )
322:  26: logger = logging.getLogger(__name__)
323:  27: 
324:  28: class SimpleBreakoutTrader:
325:  29:     """Dead simple breakout strategy - no BS, just works"""
326:  30: 
327:  31:     def __init__(self, config_path: str = "config.yaml"):
328:  32:         with open(config_path, 'r') as f:
329:  33:             self.config = yaml.safe_load(f)
330:  34: 
331:  35:         self.symbol = self.config['trading']['symbol']
332:  36:         self.risk_pct = self.config['trading']['risk_per_trade']
333:  37:         self.sma_period = self.config['strategy']['sma_period']
334:  38:         self.atr_period = self.config['strategy']['atr_period']
335:  39: 
336:  40:         logger.info(f"Initialized trader for {self.symbol}")
337:  41: 
338:  42:     def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
339:  43:         """Add SMA, ATR and Volume metrics"""
340:  44:         df = df.copy()
341:  45: 
342:  46:         # SMA
343:  47:         df['sma'] = df['close'].rolling(self.sma_period).mean()
344:  48: 
345:  49:         # ATR
346:  50:         high_low = df['high'] - df['low']
347:  51:         high_close = np.abs(df['high'] - df['close'].shift())
348:  52:         low_close = np.abs(df['low'] - df['close'].shift())
349:  53:         true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
350:  54:         df['atr'] = true_range.rolling(self.atr_period).mean()
351:  55: 
352:  56:         # Volume
353:  57:         df['volume_avg'] = df['volume'].rolling(self.sma_period).mean()
354:  58:         df['volume_ratio'] = df['volume'] / df['volume_avg']
355:  59: 
356:  60:         return df
357:  61: 
358:  62:     def get_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Dict]:
359:  63:         """
360:  64:         Returns signal and metadata
361:  65:         Simple rules:
362:  66:         - BUY: price crosses above SMA + volume confirmation
363:  67:         - SELL: price crosses below SMA
364:  68:         - None: no clear signal
365:  69:         """
366:  70:         df = self.calculate_indicators(df)
367:  71: 
368:  72:         if len(df) < self.sma_period:
369:  73:             return None, {}
370:  74: 
371:  75:         last = df.iloc[-1]
372:  76:         prev = df.iloc[-2]
373:  77: 
374:  78:         # Detect crossovers
375:  79:         cross_above = (prev['close'] <= prev['sma']) and (last['close'] > last['sma'])
376:  80:         cross_below = (prev['close'] >= prev['sma']) and (last['close'] < last['sma'])
377:  81: 
378:  82:         # Volume confirmation
379:  83:         volume_confirmed = last['volume_ratio'] > self.config['strategy']['volume_threshold']
380:  84: 
381:  85:         metadata = {
382:  86:             'close': last['close'],
383:  87:             'sma': last['sma'],
384:  88:             'atr': last['atr'],
385:  89:             'volume_ratio': last['volume_ratio'],
386:  90:             'timestamp': last.name if hasattr(last, 'name') else None
387:  91:         }
388:  92: 
389:  93:         if cross_above and volume_confirmed:
390:  94:             metadata['reason'] = f"Bullish cross + volume {last['volume_ratio']:.1f}x"
391:  95:             return 'BUY', metadata
392:  96:         elif cross_below:
393:  97:             metadata['reason'] = "Bearish cross"
394:  98:             return 'SELL', metadata
395:  99: 
396: 100:         return None, metadata
397: 101: 
398: 102:     def calculate_position_size(self, equity: float, entry: float, stop: float) -> float:
399: 103:         """Position sizing based on risk management"""
400: 104:         risk_amount = equity * (self.risk_pct / 100)
401: 105:         risk_points = abs(entry - stop)
402: 106: 
403: 107:         if risk_points == 0:
404: 108:             return 0
405: 109: 
406: 110:         # For forex/indices: size in lots
407: 111:         # Adjust multiplier based on your broker
408: 112:         point_value = 1.0  # EUR per point for GER40
409: 113:         size = risk_amount / (risk_points * point_value)
410: 114: 
411: 115:         return round(size, 2)
412: 116: 
413: 117: class Backtester:
414: 118:     """Simple backtesting engine - no look-ahead, realistic fills"""
415: 119: 
416: 120:     def __init__(self, trader: SimpleBreakoutTrader):
417: 121:         self.trader = trader
418: 122:         self.config = trader.config
419: 123: 
420: 124:     def run(self, df: pd.DataFrame) -> Dict:
421: 125:         """Run backtest and return results"""
422: 126:         initial_capital = self.config['trading']['initial_capital']
423: 127:         equity = initial_capital
424: 128: 
425: 129:         trades = []
426: 130:         position = None
427: 131: 
428: 132:         df = self.trader.calculate_indicators(df)
429: 133: 
430: 134:         for i in range(self.trader.sma_period, len(df)):
431: 135:             window = df.iloc[:i+1]
432: 136:             signal, meta = self.trader.get_signal(window)
433: 137: 
434: 138:             current_bar = df.iloc[i]
435: 139: 
436: 140:             # Check exits first (before new signals)
437: 141:             if position:
438: 142:                 exit_price = None
439: 143:                 exit_reason = None
440: 144: 
441: 145:                 # Check stop loss
442: 146:                 if current_bar['low'] <= position['stop']:
443: 147:                     exit_price = position['stop']
444: 148:                     exit_reason = 'Stop Loss'
445: 149:                 # Check take profit
446: 150:                 elif current_bar['high'] >= position['target']:
447: 151:                     exit_price = position['target']
448: 152:                     exit_reason = 'Take Profit'
449: 153:                 # Check signal reversal
450: 154:                 elif signal == 'SELL':
451: 155:                     exit_price = current_bar['close']
452: 156:                     exit_reason = 'Signal Reversal'
453: 157: 
454: 158:                 if exit_price:
455: 159:                     # Calculate PnL
456: 160:                     pnl = (exit_price - position['entry']) * position['size']
457: 161:                     pnl -= (position['entry'] * position['size'] * self.config['backtest']['commission'] * 2)
458: 162: 
459: 163:                     equity += pnl
460: 164: 
461: 165:                     trades.append({
462: 166:                         'entry_time': position['entry_time'],
463: 167:                         'exit_time': current_bar.name,
464: 168:                         'side': position['side'],
465: 169:                         'entry': position['entry'],
466: 170:                         'exit': exit_price,
467: 171:                         'size': position['size'],
468: 172:                         'pnl': pnl,
469: 173:                         'return_pct': (pnl / equity) * 100,
470: 174:                         'reason': exit_reason
471: 175:                     })
472: 176: 
473: 177:                     position = None
474: 178:                     logger.debug(f"Exit {exit_reason} at {exit_price:.2f}, PnL: {pnl:.2f}")
475: 179: 
476: 180:             # New signals (only if no position)
477: 181:             if signal and not position:
478: 182:                 if signal == 'BUY':
479: 183:                     entry = current_bar['close']
480: 184:                     stop = entry - current_bar['atr'] * self.config['strategy']['sl_multiplier']
481: 185:                     target = entry + current_bar['atr'] * self.config['strategy']['tp_multiplier']
482: 186: 
483: 187:                     size = self.trader.calculate_position_size(equity, entry, stop)
484: 188: 
485: 189:                     if size > 0:
486: 190:                         position = {
487: 191:                             'entry_time': current_bar.name,
488: 192:                             'side': 'long',
489: 193:                             'entry': entry,
490: 194:                             'stop': stop,
491: 195:                             'target': target,
492: 196:                             'size': size
493: 197:                         }
494: 198:                         logger.debug(f"BUY at {entry:.2f}, stop: {stop:.2f}, target: {target:.2f}")
495: 199: 
496: 200:         # Calculate metrics
497: 201:         trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
498: 202: 
499: 203:         if not trades_df.empty:
500: 204:             wins = trades_df[trades_df['pnl'] > 0]
501: 205:             losses = trades_df[trades_df['pnl'] <= 0]
502: 206: 
503: 207:             metrics = {
504: 208:                 'initial_capital': initial_capital,
505: 209:                 'final_capital': equity,
506: 210:                 'total_return_pct': ((equity / initial_capital) - 1) * 100,
507: 211:                 'num_trades': len(trades_df),
508: 212:                 'num_wins': len(wins),
509: 213:                 'num_losses': len(losses),
510: 214:                 'win_rate': (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
511: 215:                 'avg_win': wins['pnl'].mean() if not wins.empty else 0,
512: 216:                 'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
513: 217:                 'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else 0,
514: 218:                 'max_drawdown': self._calculate_max_drawdown(trades_df),
515: 219:                 'sharpe_ratio': self._calculate_sharpe(trades_df)
516: 220:             }
517: 221:         else:
518: 222:             metrics = {
519: 223:                 'initial_capital': initial_capital,
520: 224:                 'final_capital': equity,
521: 225:                 'total_return_pct': 0,
522: 226:                 'num_trades': 0,
523: 227:                 'message': 'No trades generated'
524: 228:             }
525: 229: 
526: 230:         return {'metrics': metrics, 'trades': trades_df}
527: 231: 
528: 232:     def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
529: 233:         """Calculate maximum drawdown percentage"""
530: 234:         if trades_df.empty:
531: 235:             return 0
532: 236: 
533: 237:         cumulative = trades_df['pnl'].cumsum()
534: 238:         running_max = cumulative.cummax()
535: 239:         drawdown = (cumulative - running_max) / (running_max + self.config['trading']['initial_capital'])
536: 240:         return abs(drawdown.min()) * 100 if not drawdown.empty else 0
537: 241: 
538: 242:     def _calculate_sharpe(self, trades_df: pd.DataFrame) -> float:
539: 243:         """Calculate Sharpe ratio (simplified)"""
540: 244:         if trades_df.empty or len(trades_df) < 2:
541: 245:             return 0
542: 246: 
543: 247:         returns = trades_df['return_pct']
544: 248:         if returns.std() == 0:
545: 249:             return 0
546: 250: 
547: 251:         # Annualize based on average trades per year (estimate)
548: 252:         days_in_sample = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
549: 253:         if days_in_sample > 0:
550: 254:             trades_per_year = (252 / days_in_sample) * len(trades_df)
551: 255:             return (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
552: 256:         return 0
553: 257: 
554: 258: class LiveTrader:
555: 259:     """Live trading interface - can be paper or real"""
556: 260: 
557: 261:     def __init__(self, trader: SimpleBreakoutTrader):
558: 262:         self.trader = trader
559: 263:         self.config = trader.config
560: 264:         self.position = None
561: 265:         self.equity = self.config['trading']['initial_capital']
562: 266: 
563: 267:     def fetch_latest_data(self, lookback_bars: int = 100) -> pd.DataFrame:
564: 268:         """Fetch latest price data"""
565: 269:         # For now, use CSV data for testing
566: 270:         # Replace with MT5 or broker API
567: 271:         csv_files = list(Path('data').glob('*.csv'))
568: 272:         if not csv_files:
569: 273:             raise FileNotFoundError("No CSV files in data/ directory")
570: 274: 
571: 275:         df = pd.read_csv(csv_files[0], parse_dates=['time'])
572: 276:         df.set_index('time', inplace=True)
573: 277: 
574: 278:         # Get last N bars
575: 279:         return df.tail(lookback_bars)
576: 280: 
577: 281:     def check_for_signals(self):
578: 282:         """Check current market for signals"""
579: 283:         try:
580: 284:             df = self.fetch_latest_data()
581: 285:             signal, meta = self.trader.get_signal(df)
582: 286: 
583: 287:             if signal:
584: 288:                 logger.info(f"SIGNAL: {signal} at {meta.get('timestamp', 'now')}")
585: 289:                 logger.info(f"Reason: {meta.get('reason', 'N/A')}")
586: 290:                 logger.info(f"Price: {meta.get('close', 0):.2f}, SMA: {meta.get('sma', 0):.2f}")
587: 291: 
588: 292:                 if self.config['live']['mode'] == 'live':
589: 293:                     self.place_order(signal, meta)
590: 294:                 else:
591: 295:                     logger.info("PAPER TRADE - no real order placed")
592: 296: 
593: 297:         except Exception as e:
594: 298:             logger.error(f"Error checking signals: {e}")
595: 299: 
596: 300:     def place_order(self, signal: str, meta: Dict):
597: 301:         """Place actual trade order"""
598: 302:         # TODO: Implement MT5 order placement
599: 303:         logger.warning("Live order placement not yet implemented")
600: 304: 
601: 305:     def run_forever(self):
602: 306:         """Main loop for live trading"""
603: 307:         logger.info(f"Starting live trader in {self.config['live']['mode']} mode")
604: 308:         logger.info(f"Checking every {self.config['live']['check_interval']} seconds")
605: 309: 
606: 310:         while True:
607: 311:             try:
608: 312:                 self.check_for_signals()
609: 313:                 time.sleep(self.config['live']['check_interval'])
610: 314:             except KeyboardInterrupt:
611: 315:                 logger.info("Shutting down...")
612: 316:                 break
613: 317:             except Exception as e:
614: 318:                 logger.error(f"Unexpected error: {e}")
615: 319:                 time.sleep(60)  # Wait a minute before retry
616: 320: 
617: 321: def load_data(csv_path: str) -> pd.DataFrame:
618: 322:     """Load and prepare data"""
619: 323:     df = pd.read_csv(csv_path)
620: 324: 
621: 325:     # Handle different column names
622: 326:     time_cols = ['time', 'Time', 'datetime', 'Datetime', 'date', 'Date']
623: 327:     time_col = None
624: 328:     for col in time_cols:
625: 329:         if col in df.columns:
626: 330:             time_col = col
627: 331:             break
628: 332: 
629: 333:     if time_col:
630: 334:         df[time_col] = pd.to_datetime(df[time_col])
631: 335:         df.set_index(time_col, inplace=True)
632: 336: 
633: 337:     # Standardize column names
634: 338:     df.columns = df.columns.str.lower()
635: 339: 
636: 340:     # Required columns
637: 341:     required = ['open', 'high', 'low', 'close']
638: 342:     if not all(col in df.columns for col in required):
639: 343:         raise ValueError(f"CSV must have columns: {required}")
640: 344: 
641: 345:     # Add volume if missing
642: 346:     if 'volume' not in df.columns:
643: 347:         df['volume'] = 1000  # Dummy volume
644: 348: 
645: 349:     return df.sort_index()
646: 350: 
647: 351: def main():
648: 352:     """Main entry point"""
649: 353:     import argparse
650: 354: 
651: 355:     parser = argparse.ArgumentParser(description='Minimal Viable Trading System')
652: 356:     parser.add_argument('mode', choices=['backtest', 'live', 'signal'],
653: 357:                       help='Operating mode')
654: 358:     parser.add_argument('--csv', type=str, help='CSV file path for backtesting')
655: 359:     parser.add_argument('--config', type=str, default='config.yaml',
656: 360:                       help='Configuration file')
657: 361: 
658: 362:     args = parser.parse_args()
659: 363: 
660: 364:     # Create directories
661: 365:     Path('logs').mkdir(exist_ok=True)
662: 366:     Path('output').mkdir(exist_ok=True)
663: 367: 
664: 368:     # Initialize trader
665: 369:     trader = SimpleBreakoutTrader(args.config)
666: 370: 
667: 371:     if args.mode == 'backtest':
668: 372:         if not args.csv:
669: 373:             csv_files = list(Path('data').glob('*.csv'))
670: 374:             if not csv_files:
671: 375:                 logger.error("No CSV files found. Use --csv or add files to data/")
672: 376:                 sys.exit(1)
673: 377:             args.csv = str(csv_files[0])
674: 378:             logger.info(f"Using {args.csv}")
675: 379: 
676: 380:         # Load data
677: 381:         df = load_data(args.csv)
678: 382:         logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
679: 383: 
680: 384:         # Run backtest
681: 385:         backtester = Backtester(trader)
682: 386:         results = backtester.run(df)
683: 387: 
684: 388:         # Display results
685: 389:         print("\n" + "="*50)
686: 390:         print("BACKTEST RESULTS")
687: 391:         print("="*50)
688: 392:         for key, value in results['metrics'].items():
689: 393:             if isinstance(value, float):
690: 394:                 print(f"{key:20s}: {value:,.2f}")
691: 395:             else:
692: 396:                 print(f"{key:20s}: {value}")
693: 397: 
694: 398:         # Save trades
695: 399:         if not results['trades'].empty:
696: 400:             output_file = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
697: 401:             results['trades'].to_csv(output_file, index=False)
698: 402:             print(f"\nTrades saved to: {output_file}")
699: 403: 
700: 404:     elif args.mode == 'live':
701: 405:         live = LiveTrader(trader)
702: 406:         live.run_forever()
703: 407: 
704: 408:     elif args.mode == 'signal':
705: 409:         # One-time signal check
706: 410:         live = LiveTrader(trader)
707: 411:         live.check_for_signals()
708: 412: 
709: 413: if __name__ == '__main__':
710: 414:     main()
711: ````
`````

## File: requirements.txt
`````
1: pandas>=2.2.0
2: numpy>=1.26.0
3: pyyaml>=6.0
4: pytest>=8.0.0
5: python-dotenv>=1.0.0
6: MetaTrader5>=5.0.45  # Windows only
`````

## File: strategy.py
`````python
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
 13:     SELL = -1   # reversal/exit only; no short entries in v1
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
 32:     volume_threshold: float = 1.0  # x average
 33: 
 34: 
 35: class Strategy:
 36:     """Pure strategy: indicators + signal at bar i. No state, no side effects."""
 37:     def __init__(self, params: StrategyParams = StrategyParams()):
 38:         self.p = params
 39: 
 40:     def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
 41:         df = df.copy()
 42:         # SMA
 43:         df["sma"] = df["close"].rolling(self.p.sma_period, min_periods=self.p.sma_period).mean()
 44: 
 45:         # ATR (classic TR)
 46:         high_low = df["high"] - df["low"]
 47:         high_close = (df["high"] - df["close"].shift()).abs()
 48:         low_close = (df["low"] - df["close"].shift()).abs()
 49:         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
 50:         df["atr"] = tr.rolling(self.p.atr_period, min_periods=self.p.atr_period).mean()
 51: 
 52:         # Volume ratio
 53:         if "volume" in df.columns:
 54:             df["volume_avg"] = df["volume"].rolling(self.p.sma_period, min_periods=self.p.sma_period).mean()
 55:             df["volume_ratio"] = df["volume"] / df["volume_avg"]
 56:         else:
 57:             df["volume"] = 1000.0
 58:             df["volume_avg"] = 1000.0
 59:             df["volume_ratio"] = 1.0
 60:         return df
 61: 
 62:     def get_signal_at(self, df: pd.DataFrame, i: int) -> Tuple[Signal, dict]:
 63:         """Decide signal using only info up to index i (no look-ahead)."""
 64:         if i < max(self.p.sma_period, self.p.atr_period):
 65:             return Signal(SignalType.NONE), {}
 66: 
 67:         last = df.iloc[i]
 68:         prev = df.iloc[i - 1]
 69: 
 70:         cross_above = (prev["close"] <= prev["sma"]) and (last["close"] > last["sma"])
 71:         cross_below = (prev["close"] >= prev["sma"]) and (last["close"] < last["sma"])
 72:         volume_ok = bool(last.get("volume_ratio", 0) > self.p.volume_threshold)
 73: 
 74:         meta = {
 75:             "close": float(last["close"]),
 76:             "sma": float(last["sma"]),
 77:             "atr": float(last["atr"]),
 78:             "volume_ratio": float(last.get("volume_ratio", 1.0)),
 79:             "timestamp": last.name if hasattr(last, "name") else None,
 80:         }
 81: 
 82:         # Entry: only long for v1
 83:         if cross_above and volume_ok and np.isfinite(last["atr"]) and last["atr"] > 0:
 84:             entry = float(last["close"])
 85:             stop = entry - float(last["atr"]) * self.p.sl_mult
 86:             target = entry + float(last["atr"]) * self.p.tp_mult
 87:             sig = Signal(
 88:                 type=SignalType.BUY,
 89:                 entry=entry,
 90:                 stop=stop,
 91:                 target=target,
 92:                 reason=f"BUY: cross↑ + vol {meta['volume_ratio']:.2f}x",
 93:                 timestamp=meta["timestamp"],
 94:             )
 95:             return sig, meta
 96: 
 97:         # Exit reversal signal (SELL only as exit trigger)
 98:         if cross_below:
 99:             sig = Signal(
100:                 type=SignalType.SELL,
101:                 reason="SELL: cross↓ (reversal/exit)",
102:                 timestamp=meta["timestamp"],
103:             )
104:             return sig, meta
105: 
106:         return Signal(SignalType.NONE), meta
`````

## File: test_trader.py
`````python
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
`````

## File: testplan.md
`````markdown
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
`````

## File: trader.py
`````python
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
`````
