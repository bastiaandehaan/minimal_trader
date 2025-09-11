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
from utils.engine_guards import GuardConfig

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


# ----------------- helpers -----------------

def _load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def _build_engine_from_config(cfg: Dict) -> MultiStrategyEngine:
    ecfg = cfg.get("engine") or {}
    eng = MultiStrategyEngine(EngineConfig(
        initial_capital=float(ecfg.get("initial_capital", 10_000.0)),
        risk_per_trade=float(ecfg.get("risk_per_trade", 1.0)),
        max_positions=int(ecfg.get("max_positions", 3)),
        commission=float(ecfg.get("commission", 0.0002)),
        slippage=float(ecfg.get("slippage", 0.0)),
        time_exit_bars=int(ecfg.get("time_exit_bars", 200)),
        allow_shorts=bool(ecfg.get("allow_shorts", True)),
        min_risk_pts=float(ecfg.get("min_risk_pts", 0.5)),
    ))
    return eng


def _build_strategy_from_config(cfg: Dict) -> RSIReversionStrategy:
    scfg = ((cfg.get("strategies") or {}).get("rsi_reversion") or {})
    params = scfg.get("params") or {}
    strat = RSIReversionStrategy(params)
    allocation = float(scfg.get("allocation", 100.0))
    return strat, allocation


def _build_guards_from_config(cfg: Dict) -> GuardConfig:
    g = cfg.get("guards") or {}
    return GuardConfig(
        trading_hours_tz=g.get("trading_hours_tz", "Europe/Brussels"),
        trading_hours_start=g.get("trading_hours_start", "08:00"),
        trading_hours_end=g.get("trading_hours_end", "22:00"),
        min_atr_pts=float(g.get("min_atr_pts", 20.0)),
        cooldown_bars=int(g.get("cooldown_bars", 3)),
        max_trades_per_day=int(g.get("max_trades_per_day", 10)),
        one_trade_per_timestamp=bool(g.get("one_trade_per_timestamp", True)),
    )


def _print_results(results: Dict):
    m = results["metrics"]
    print("\n============================================================")
    print("BACKTEST RESULTS")
    print("============================================================")
    print(f"Total Trades:    {m['total_trades']}")
    print(f"  - Longs:       {m['total_longs']}")
    print(f"  - Shorts:      {m['total_shorts']}")
    print(f"Initial Capital: ${m['initial_capital']:.2f}")
    print(f"Final Capital:   ${m['final_capital']:.2f}")
    print(f"Total Return:    {m['total_return']:.2f}%")
    print(f"Win Rate:        {m['win_rate']:.2f}%")
    pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
    print(f"Profit Factor:   {pf}")
    print(f"Sharpe Ratio:    {m['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {m['max_drawdown']:.2f}%")

    trades = results["positions"]
    if not trades.empty:
        by_strat = trades.groupby("strategy")["pnl"].agg(["count", "sum"]).reset_index()
        print("\n----------------------------------------")
        print("STRATEGY BREAKDOWN")
        print("----------------------------------------\n")
        for _, r in by_strat.iterrows():
            print(f"{r['strategy']}: Trades={int(r['count'])}, Total PnL=${r['sum']:.2f}")

        print("\nTrades saved to: output/trades.csv")


def run_backtest_csv(csv_path: str, cfg: Dict):
    data_cfg = cfg.get("data") or {}
    resample = data_cfg.get("resample", "5min")

    feed = CSVFeed(csv_path)
    df = feed.load(resample=resample)
    logger.info("feeds.csv_feed: Loaded %d bars from %s", len(df), Path(csv_path).name)

    engine = _build_engine_from_config(cfg)
    strat, alloc = _build_strategy_from_config(cfg)
    engine.add_strategy(strat, alloc)

    guard_cfg = _build_guards_from_config(cfg)
    logger.info("main: Running backtest on %d bars from %s to %s", len(df), df.index[0], df.index[-1])
    results = engine.run_backtest(df, guard_cfg)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    results["positions"].to_csv(out_dir / "trades.csv", index=False)

    _print_results(results)


def run_backtest_mt5(cfg: Dict):
    # Lazy import om corrupte/afwezige MT5 installatie niet te laten crashen als je CSV gebruikt
    try:
        from feeds.mt5_feed import MT5Feed
    except Exception:
        logger.error("MT5 feed not available")
        return

    mt5_cfg = (cfg.get("mt5") or {})
    symbol = mt5_cfg.get("symbol", "GER40.cash")
    timeframe = mt5_cfg.get("timeframe", "M1")
    bars = int(mt5_cfg.get("bars", 10_000))

    feed = MT5Feed(symbol=symbol, timeframe=timeframe, bars=bars)
    if not feed.connect(
        login=mt5_cfg.get("login"),
        password=mt5_cfg.get("password"),
        server=mt5_cfg.get("server"),
    ):
        logger.error("Failed to connect MT5")
        return
    df = feed.fetch()
    feed.disconnect()

    # resample indien in config
    data_cfg = cfg.get("data") or {}
    resample = data_cfg.get("resample")
    if resample:
        df = df.resample(resample).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()

    engine = _build_engine_from_config(cfg)
    strat, alloc = _build_strategy_from_config(cfg)
    engine.add_strategy(strat, alloc)

    guard_cfg = _build_guards_from_config(cfg)
    logger.info("main: Running backtest with MT5 data")
    results = engine.run_backtest(df, guard_cfg)

    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    results["positions"].to_csv(out_dir / "trades.csv", index=False)
    _print_results(results)


# ----------------- CLI -----------------

def build_arg_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest")
    b.add_argument("--csv", type=str, help="CSV file with OHLCV")
    b.add_argument("--config", type=str, help="YAML config file")
    b.add_argument("--mt5", action="store_true", help="Use MT5 instead of CSV")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = _load_config(args.config)

    if args.cmd == "backtest":
        if args.mt5 and not args.csv:
            run_backtest_mt5(cfg)
        elif args.csv:
            run_backtest_csv(args.csv, cfg)
        else:
            parser.error("Provide --csv or --mt5")


if __name__ == "__main__":
    main()
