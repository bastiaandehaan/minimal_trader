from __future__ import annotations
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml
import pandas as pd

from strategy import Strategy, StrategyParams
from backtest import Backtest, ExecConfig
from data_feed import CSVFeed


def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs/trader.log"), logging.StreamHandler()],
    )
    return logging.getLogger("minimal_trader")


def run_backtest(args, cfg: Dict, logger):
    print(">>> [main] run_backtest start", flush=True)
    p = cfg.get("strategy", {})
    s_params = StrategyParams(
        sma_period=int(p.get("sma_period", 20)),
        atr_period=int(p.get("atr_period", 14)),
        sl_mult=float(p.get("sl_mult", p.get("sl_multiplier", 1.5))),
        tp_mult=float(p.get("tp_mult", p.get("tp_multiplier", 2.5))),
        volume_threshold=float(p.get("volume_threshold", 1.0)),
    )
    strategy = Strategy(s_params)

    e = cfg.get("trading", {})
    b = cfg.get("backtest", {})
    exec_cfg = ExecConfig(
        initial_capital=float(e.get("initial_capital", 10_000.0)),
        risk_pct=float(e.get("risk_per_trade", 1.0)),
        commission=float(b.get("commission", 0.0002)),
        point_value=float(e.get("point_value", 1.0)),
        time_exit_bars=int(b.get("time_exit_bars", 200)),
    )
    bt = Backtest(strategy, exec_cfg)

    feed = CSVFeed(args.csv, resample_to=(args.resample if hasattr(args, "resample") else None))
    df = feed.load(limit=args.limit)

    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    if getattr(args, "resample", None):
        logger.info(f"Data resampled to {args.resample}")

    results = bt.run(df, logger=logger)

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for k, v in results["metrics"].items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:,.2f}")
        else:
            print(f"{k:20s}: {v}")

    trades = results["trades"]
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        Path("output").mkdir(exist_ok=True)
        out = f"output/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades.to_csv(out, index=False)
        print(f"\nTrades saved to: {out}")

    print(">>> [main] run_backtest done", flush=True)


def run_signal(args, cfg: Dict, logger):
    print(">>> [main] run_signal start", flush=True)
    p = cfg.get("strategy", {})
    s_params = StrategyParams(
        sma_period=int(p.get("sma_period", 20)),
        atr_period=int(p.get("atr_period", 14)),
        sl_mult=float(p.get("sl_mult", p.get("sl_multiplier", 1.5))),
        tp_mult=float(p.get("tp_mult", p.get("tp_multiplier", 2.5))),
        volume_threshold=float(p.get("volume_threshold", 1.0)),
    )
    strategy = Strategy(s_params)

    feed = CSVFeed(args.csv)
    df = strategy.calculate_indicators(feed.load(limit=max(args.limit, 500)))

    if len(df) < max(s_params.sma_period, s_params.atr_period) + 2:
        logger.warning("Not enough data to generate a signal.")
        return

    i = len(df) - 1
    sig, meta = strategy.get_signal_at(df, i)
    if sig.type.name != "NONE":
        logger.info(f"SIGNAL: {sig.type.name} at {meta.get('timestamp')}")
        logger.info(f"Reason: {sig.reason}")
        logger.info(f"Price: {meta.get('close'):.2f}, SMA: {meta.get('sma'):.2f}")
    else:
        logger.info("No signal")
    print(">>> [main] run_signal done", flush=True)


def main():
    print(">>> [main] argparse setup", flush=True)
    parser = argparse.ArgumentParser(description="Minimal Trader v2")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_bt = sub.add_parser("backtest", help="Run backtest on CSV")
    p_bt.add_argument("--csv", type=str, required=True)
    p_bt.add_argument("--config", type=str, default="config.yaml")
    p_bt.add_argument("--limit", type=int, default=0)
    p_bt.add_argument("--resample", type=str, help="Resample to (e.g., 1H, 5T)")

    p_sig = sub.add_parser("signal", help="One-time signal on CSV tail")
    p_sig.add_argument("--csv", type=str, required=True)
    p_sig.add_argument("--config", type=str, default="config.yaml")
    p_sig.add_argument("--limit", type=int, default=1000)

    args = parser.parse_args()
    logger = setup_logging()
    cfg = load_config(getattr(args, "config", "config.yaml"))

    print(f">>> [main] mode={args.mode}", flush=True)

    if args.mode == "backtest":
        run_backtest(args, cfg, logger)
    elif args.mode == "signal":
        run_signal(args, cfg, logger)


if __name__ == "__main__":
    import sys
    print(">>> [main] __main__ start argv=", sys.argv, flush=True)
    main()
    print(">>> [main] __main__ end", flush=True)
