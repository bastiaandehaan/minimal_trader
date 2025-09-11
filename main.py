# main.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from engine import MultiStrategyEngine, EngineConfig
from strategies.rsi_reversion import RSIReversionStrategy
from feeds.csv_feed import CSVFeed
from feeds.mt5_feed import MT5Feed, MT5Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_engine_from_config(cfg: Dict) -> MultiStrategyEngine:
    e = cfg.get("engine") or {}
    engine = MultiStrategyEngine(
        EngineConfig(
            initial_capital=float(e.get("initial_capital", 10_000)),
            risk_per_trade=float(e.get("risk_per_trade", 1.0)),
            max_positions=int(e.get("max_positions", 3)),
            commission=float(e.get("commission", 0.0002)),
            slippage=float(e.get("slippage", 0.0001)),
            time_exit_bars=int(e.get("time_exit_bars", 200)),
            allow_shorts=bool(e.get("allow_shorts", True)),
        )
    )
    s = (cfg.get("strategies") or {}).get("rsi_reversion", {})
    if s.get("enabled", True):
        engine.add_strategy(RSIReversionStrategy(s.get("params", {})), float(s.get("allocation", 100.0)))
    return engine


def run_backtest_csv(csv_path: str, cfg: Dict) -> None:
    data_cfg = cfg.get("data") or {}
    resample = data_cfg.get("resample")  # e.g. "5min" / "5T"
    feed = CSVFeed(csv_path, resample=resample)
    df = feed.load()

    logger.info("feeds.csv_feed: Loaded %d bars from %s", len(df), Path(csv_path).name)

    engine = _build_engine_from_config(cfg)
    logger.info("main: Running backtest on %d bars from %s to %s", len(df), df.index.min(), df.index.max())
    results = engine.run_backtest(df)
    _print_results(results)


def run_backtest_mt5(cfg: Dict) -> None:  # pragma: no cover
    data_cfg = cfg.get("data") or {}
    mt5_cfg = cfg.get("mt5") or {}
    resample = data_cfg.get("resample")

    feed = MT5Feed(MT5Config(symbol=mt5_cfg.get("symbol", "GER40.cash"),
                             timeframe=mt5_cfg.get("timeframe", "M1"),
                             bars=int(mt5_cfg.get("bars", 10_000))))
    feed.connect()
    df = feed.load(resample=resample)
    logger.info("feeds.mt5_feed: Loaded %d bars from MT5 %s", len(df), feed.cfg.symbol)

    engine = _build_engine_from_config(cfg)
    logger.info("main: Running backtest on %d bars from %s to %s", len(df), df.index.min(), df.index.max())
    results = engine.run_backtest(df)
    _print_results(results)
    feed.disconnect()


def _print_results(results: Dict) -> None:
    m = results.get("metrics", {})
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades:    {m.get('total_trades', 0)}")
    print(f"  - Longs:       {m.get('total_longs', 0)}")
    print(f"  - Shorts:      {m.get('total_shorts', 0)}")
    print(f"Initial Capital: ${m.get('initial_capital', 0.0):.2f}")
    print(f"Final Capital:   ${m.get('final_capital', 0.0):.2f}")
    print(f"Total Return:    {m.get('total_return', 0.0):.2f}%")
    print(f"Win Rate:        {m.get('win_rate', 0.0):.2f}%")
    pf = m.get("profit_factor", 0.0)
    print(f"Profit Factor:   {'inf' if pd.isna(pf) or pd.isinf(pf) else f'{pf:.2f}'}")
    print(f"Sharpe Ratio:    {m.get('sharpe_ratio', 0.0):.2f}")
    print(f"Max Drawdown:    {m.get('max_drawdown', 0.0):.2f}%")

    print("\n----------------------------------------")
    print("STRATEGY BREAKDOWN")
    print("----------------------------------------")
    for name, sm in (results.get("strategies") or {}).items():
        print(f"\n{name}:")
        print(f"  Trades:      {sm.get('trades', 0)}")
        print(f"    - Longs:   {sm.get('longs', 0)}")
        print(f"    - Shorts:  {sm.get('shorts', 0)}")
        print(f"  Win Rate:    {sm.get('win_rate', 0.0):.2f}%")
        print(f"  Total PnL:   ${sm.get('total_pnl', 0.0):.2f}")
        print(f"  Avg PnL:     ${sm.get('avg_pnl', 0.0):.2f}")
        print(f"  Best Trade:  ${sm.get('best_trade', 0.0):.2f}")
        print(f"  Worst Trade: ${sm.get('worst_trade', 0.0):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="run backtest")
    bt.add_argument("--csv", type=str, help="Path to CSV")
    bt.add_argument("--config", type=str, help="YAML config", default=None)

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.csv:
        logger.info("main: Running backtest on CSV: %s", args.csv)
        run_backtest_csv(args.csv, cfg)
    else:
        logger.info("main: Running backtest with MT5 data")
        run_backtest_mt5(cfg)


if __name__ == "__main__":
    main()
