# utils/engine_guards.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import pandas as pd


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
    opened_timestamps: Set[pd.Timestamp] = field(default_factory=set)
    trades_per_day: Dict[pd.Timestamp, int] = field(default_factory=dict)
    last_exit_bar_by_strategy: Dict[str, int] = field(default_factory=dict)


def apply_trading_hours(df: pd.DataFrame, cfg: GuardConfig) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    local = df.copy()
    local.index = local.index.tz_convert(cfg.trading_hours_tz)

    start = pd.Timestamp(cfg.trading_hours_start).time()
    end = pd.Timestamp(cfg.trading_hours_end).time()
    mask = (local.index.time >= start) & (local.index.time < end)
    return df.loc[mask]


def should_skip_low_vol(atr_val: float, cfg: GuardConfig) -> bool:
    try:
        return float(atr_val) < float(cfg.min_atr_pts)
    except Exception:
        return True


def in_cooldown(strategy_name: str, bar_i: int, cfg: GuardConfig, state: GuardState) -> bool:
    last_exit = state.last_exit_bar_by_strategy.get(strategy_name)
    return (last_exit is not None) and ((bar_i - last_exit) <= cfg.cooldown_bars)


def allow_entry_at_bar(ts: pd.Timestamp, cfg: GuardConfig, state: Optional[GuardState] = None) -> bool:
    # state optioneel om compat te houden
    state = state or GuardState()

    # één trade per bar?
    if cfg.one_trade_per_timestamp and ts in state.opened_timestamps:
        return False

    utc_ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    local_day = utc_ts.tz_convert(cfg.trading_hours_tz).normalize()
    if state.trades_per_day.get(local_day, 0) >= cfg.max_trades_per_day:
        return False

    return True


def register_entry(ts: pd.Timestamp, cfg: GuardConfig, state: GuardState) -> None:
    utc_ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    state.opened_timestamps.add(utc_ts)
    local_day = utc_ts.tz_convert(cfg.trading_hours_tz).normalize()
    state.trades_per_day[local_day] = state.trades_per_day.get(local_day, 0) + 1


def register_exit(strategy_name: str, bar_i: int, state: GuardState) -> None:
    state.last_exit_bar_by_strategy[strategy_name] = bar_i
