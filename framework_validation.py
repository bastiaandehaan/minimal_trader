#!/usr/bin/env python3
# framework_validation.py
# Doel: aantonen dat het FRAMEWORK correct werkt (NEXT_OPEN, guards, exits),
# zonder strategie-parameter-tweaks of repo-wijzigingen.

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# --- Zorg dat het project-rootpad op sys.path staat, ook als het script per ongeluk uit een submap draait ---
_here = Path(__file__).resolve()
_project_root = _here.parent  # Direct use the directory containing this script
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from engine import MultiStrategyEngine, EngineConfig
from strategies.rsi_reversion import RSIReversionStrategy
from utils.engine_guards import GuardConfig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 1) Data: schoon + resample naar 5m
    csv_path = _project_root / "data" / "samples" / "GER40.cash_M1.sample.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV niet gevonden: {csv_path} (verwacht repo-root structuur)")

    df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time").sort_index()
    # dupes weg
    df = df[~df.index.duplicated(keep="first")]
    # resample
    df5 = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    if df5.index.tz is None:
        df5.index = df5.index.tz_localize("UTC")
    print(f"[DATA] {len(df5)} bars | {df5.index[0]} -> {df5.index[-1]}")

    # 2) Engine + default strategy (GEEN RSI param-tuning)
    engine = MultiStrategyEngine(EngineConfig())
    engine.add_strategy(RSIReversionStrategy(), 100.0)

    # 3) Guards: framework-niveau drempel omlaag zodat engine kan handelen
    guard = GuardConfig(
        min_atr_pts=0,               # geen strategie verandering; enkel guard
        cooldown_bars=1,
        max_trades_per_day=10,
        one_trade_per_timestamp=True,
    )

    # 4) Run backtest
    results = engine.run_backtest(df5, guard)
    metrics, positions = results["metrics"], results["positions"]

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== POSITIONS (head) ===")
    print(positions.head() if not positions.empty else "No trades")

    # 5) FRAMEWORK CHECKS
    # 5a) Eén trade per timestamp
    dupes = positions["entry_time"].duplicated().any() if not positions.empty else False
    print("\n[CHECK] one_trade_per_timestamp:", "OK" if not dupes else "FAIL")

    # 5b) Exit-prijs binnen high/low van exit-bar
    def exits_within_bar(df_bars: pd.DataFrame, pos_df: pd.DataFrame) -> bool:
        if pos_df.empty:
            return True
        merged = pos_df.merge(
            df_bars[["high", "low"]].rename(columns={"high": "exit_bar_high", "low": "exit_bar_low"}),
            left_on="exit_time",
            right_index=True,
            how="left",
        )
        merged = merged.dropna(subset=["exit_bar_high", "exit_bar_low"])
        viol = ~((merged["exit_price"] >= merged["exit_bar_low"]) & (merged["exit_price"] <= merged["exit_bar_high"]))
        if viol.any():
            print(merged.loc[viol, ["exit_time", "exit_price", "exit_bar_low", "exit_bar_high"]].head())
        return not viol.any()

    print("[CHECK] exits within bar:", "OK" if exits_within_bar(df5, positions) else "FAIL")

    # 5c) NEXT_OPEN indicatie: entry niet op eerste bar en altijd op een geldige timestamp
    all_on_index = positions["entry_time"].isin(df5.index).all() if not positions.empty else True
    never_first_bar = True if positions.empty else (~(positions["entry_time"] == df5.index[0])).all()
    print("[CHECK] NEXT_OPEN indicatie (entry op bestaande bar, niet eerste bar):",
          "OK" if (all_on_index and never_first_bar) else "FAIL")


if __name__ == "__main__":
    main()
