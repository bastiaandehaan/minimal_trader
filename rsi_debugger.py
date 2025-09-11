#!/usr/bin/env python3
"""
RSI Strategy Performance Debugger
Analyseert waarom trades falen en geeft concrete verbeteringen
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def analyze_trades(trades_csv: str):
    """Analyseer trade failures en patterns."""

    # Load trades
    df = pd.read_csv(trades_csv)

    print("=" * 60)
    print("🔍 RSI STRATEGY PERFORMANCE DIAGNOSTICS")
    print("=" * 60)

    # Basic stats
    total = len(df)
    wins = len(df[df['pnl'] > 0])
    losses = len(df[df['pnl'] < 0])

    print(f"\n📊 OVERALL STATS:")
    print(f"Total trades: {total}")
    print(f"Winners: {wins} ({wins / total * 100:.1f}%)")
    print(f"Losers: {losses} ({losses / total * 100:.1f}%)")

    # Exit reason analysis
    print(f"\n🚪 EXIT REASONS:")
    for reason, count in df['exit_reason'].value_counts().items():
        pct = count / total * 100
        avg_pnl = df[df['exit_reason'] == reason]['pnl'].mean()
        print(f"  {reason}: {count} trades ({pct:.1f}%) | Avg PnL: ${avg_pnl:.2f}")

    # Side analysis
    print(f"\n📈📉 LONG vs SHORT:")
    for side in ['long', 'short']:
        side_trades = df[df['side'] == side]
        if len(side_trades) > 0:
            wins = len(side_trades[side_trades['pnl'] > 0])
            win_rate = wins / len(side_trades) * 100
            avg_pnl = side_trades['pnl'].mean()
            print(
                f"  {side.upper()}: {len(side_trades)} trades | Win rate: {win_rate:.1f}% | Avg PnL: ${avg_pnl:.2f}")

    # Risk:Reward actual
    print(f"\n💰 RISK:REWARD ANALYSIS:")
    if len(df[df['pnl'] > 0]) > 0:
        avg_win = df[df['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean())
        actual_rr = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"  Average Winner: ${avg_win:.2f}")
        print(f"  Average Loser: -${avg_loss:.2f}")
        print(f"  Actual R:R ratio: 1:{actual_rr:.2f}")

    # Time in trade
    if 'entry_time' in df.columns and 'exit_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60

        print(f"\n⏱️ TIME IN TRADE:")
        print(f"  Average: {df['duration'].mean():.0f} minutes")
        print(f"  Winners avg: {df[df['pnl'] > 0]['duration'].mean():.0f} minutes")
        print(f"  Losers avg: {df[df['pnl'] < 0]['duration'].mean():.0f} minutes")

    # Price movement analysis
    df['price_move'] = df['exit_price'] - df['entry_price']
    df['move_pct'] = (df['price_move'] / df['entry_price']) * 100

    print(f"\n📏 PRICE MOVEMENT:")
    print(f"  Avg move (all): {df['move_pct'].mean():.3f}%")
    print(f"  Avg move (winners): {df[df['pnl'] > 0]['move_pct'].mean():.3f}%")
    print(f"  Avg move (losers): {df[df['pnl'] < 0]['move_pct'].mean():.3f}%")

    return df


def generate_optimized_configs():
    """Genereer 3 geoptimaliseerde configuraties om te testen."""

    configs = []

    # Config 1: Conservative Mean Reversion
    config1 = {'name': 'conservative_mean_reversion',
        'description': 'Meer signalen, realistische targets',
        'engine': {'initial_capital': 10000.0, 'risk_per_trade': 0.5,
            'max_positions': 3, 'commission': 0.0002, 'time_exit_bars': 50,  # Meer tijd
            'allow_shorts': True}, 'strategies': {
            'rsi_reversion': {'enabled': True, 'allocation': 100.0,
                'params': {'rsi_period': 14, 'oversold': 35.0,  # VEEL minder extreem
                    'overbought': 65.0,  # VEEL minder extreem
                    'atr_period': 14, 'sl_multiplier': 2.0,  # Ruimere stop
                    'tp_multiplier': 1.5,  # Realistischer target
                    'use_next_open': False}}}}
    configs.append(config1)

    # Config 2: Balanced Mean Reversion
    config2 = {'name': 'balanced_mean_reversion',
        'description': 'Balans tussen frequentie en kwaliteit',
        'engine': {'initial_capital': 10000.0, 'risk_per_trade': 0.75,
            'max_positions': 2, 'commission': 0.0002, 'time_exit_bars': 100,
            'allow_shorts': True}, 'strategies': {
            'rsi_reversion': {'enabled': True, 'allocation': 100.0,
                'params': {'rsi_period': 10,  # Snellere RSI
                    'oversold': 30.0, 'overbought': 70.0, 'atr_period': 20,
                    # Smoothere ATR
                    'sl_multiplier': 1.5, 'tp_multiplier': 2.0,
                    'use_next_open': False}}}}
    configs.append(config2)

    # Config 3: High Frequency Mean Reversion
    config3 = {'name': 'high_frequency_mean_reversion',
        'description': 'Veel trades, kleine targets',
        'engine': {'initial_capital': 10000.0, 'risk_per_trade': 0.3,
            # Kleiner risico per trade
            'max_positions': 4, 'commission': 0.0002, 'time_exit_bars': 30,
            'allow_shorts': True}, 'strategies': {
            'rsi_reversion': {'enabled': True, 'allocation': 100.0,
                'params': {'rsi_period': 7,  # Zeer responsive
                    'oversold': 40.0,  # Frequente signalen
                    'overbought': 60.0, 'atr_period': 10, 'sl_multiplier': 1.2,
                    'tp_multiplier': 1.0,  # 1:1 R:R voor hoge win rate
                    'use_next_open': False}}}}
    configs.append(config3)

    # Save configs
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for config in configs:
        filename = output_dir / f"config_{config['name']}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"\n✅ Saved: {filename}")
        print(f"   {config['description']}")
        print(
            f"   RSI: {config['strategies']['rsi_reversion']['params']['oversold']}/{config['strategies']['rsi_reversion']['params']['overbought']}")
        print(
            f"   SL/TP: {config['strategies']['rsi_reversion']['params']['sl_multiplier']}/{config['strategies']['rsi_reversion']['params']['tp_multiplier']}")

    return configs


def main():
    """Run de debugger."""
    import sys

    if len(sys.argv) > 1:
        # Analyseer trades CSV
        trades_file = sys.argv[1]
        print(f"Analyzing trades from: {trades_file}")
        analyze_trades(trades_file)
    else:
        print("Generating optimized configurations...")
        generate_optimized_configs()
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("=" * 60)
        print("1. Run backtest with each config:")
        print(
            "   python main.py backtest --config output/config_conservative_mean_reversion.yaml --csv GER40.cash_minutes.csv")
        print("\n2. Then analyze the trades:")
        print("   python rsi_debugger.py output/trades_*.csv")
        print("\n3. Compare results and iterate!")


if __name__ == "__main__":
    main()