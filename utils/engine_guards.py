# utils/engine_guards.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import pytz


@dataclass
class GuardConfig:
    trading_hours_tz: str = "Europe/Brussels"
    trading_hours_start: str = "08:00"
    trading_hours_end: str = "22:00"
    min_atr_pts: float = 20.0
    cooldown_bars: int = 3
    max_trades_per_day: int = 10
    one_trade_per_timestamp: bool = True


@dataclass
class GuardState:
    # per-strategy laatste exit bar voor cooldown
    last_exit_bar: Dict[str, int] = field(default_factory=dict)
    # per datum (YYYY-MM-DD) aantal entries
    trades_per_day: Dict[str, int] = field(default_factory=dict)
    # om "one trade per timestamp" te handhaven
    taken_timestamps: Dict[pd.Timestamp, bool] = field(default_factory=dict)


def apply_trading_hours(df: pd.DataFrame, cfg: GuardConfig) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        # aanname: UTC tijden in csv; zet ze expliciet op UTC en converteer daarna
        df = df.copy()
        df.index = df.index.tz_localize("UTC")

    tz = pytz.timezone(cfg.trading_hours_tz)
    df_local = df.tz_convert(tz)

    start_h, start_m = map(int, cfg.trading_hours_start.split(":"))
    end_h, end_m = map(int, cfg.trading_hours_end.split(":"))

    mask = (
        (df_local.index.hour > start_h) | ((df_local.index.hour == start_h) & (df_local.index.minute >= start_m))
    ) & (
        (df_local.index.hour < end_h) | ((df_local.index.hour == end_h) & (df_local.index.minute <= end_m))
    )

    # Belangrijk: rijen selecteren met .loc op df_local (niet kolommen),
    # en daarna terug naar de oorspronkelijke tz.
    filtered = df_local.loc[mask]
    return filtered.tz_convert(df.index.tz)


def should_skip_low_vol(atr_val: float, cfg: GuardConfig) -> bool:
    if atr_val is None:
        return True
    try:
        return float(atr_val) < float(cfg.min_atr_pts)
    except Exception:
        return True


def in_cooldown(strategy_name: str, bar_i: int, cfg: GuardConfig, state: GuardState) -> bool:
    last = state.last_exit_bar.get(strategy_name)
    if last is None:
        return False
    return (bar_i - last) < int(cfg.cooldown_bars)


def allow_entry_at_bar(ts: pd.Timestamp, cfg: GuardConfig, state: GuardState) -> bool:
    """Handhaaf max_trades_per_day en 1 trade per timestamp (indien geactiveerd)."""
    if cfg.one_trade_per_timestamp and state.taken_timestamps.get(ts, False):
        return False

    day_key = ts.tz_convert(cfg.trading_hours_tz).strftime("%Y-%m-%d") if ts.tzinfo else ts.strftime("%Y-%m-%d")
    count = state.trades_per_day.get(day_key, 0)
    if count >= int(cfg.max_trades_per_day):
        return False

    return True


def register_entry(strategy_name: str, ts: pd.Timestamp, bar_i: int, state: GuardState):
    if ts.tzinfo:
        day_key = ts.tz_convert(ts.tzinfo).strftime("%Y-%m-%d")
    else:
        day_key = ts.strftime("%Y-%m-%d")
    state.trades_per_day[day_key] = state.trades_per_day.get(day_key, 0) + 1
    state.taken_timestamps[ts] = True  # claim dit timestamp


def register_exit(strategy_name: str, bar_i: int, state: GuardState):
    state.last_exit_bar[strategy_name] = bar_i
