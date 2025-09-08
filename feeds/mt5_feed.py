"""MetaTrader 5 data feed for FTMO and other brokers."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class MT5Feed:
    """Real-time and historical data from MT5."""

    TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }

    def __init__(self, symbol: str = "GER40.cash", timeframe: str = "H1"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.mt5_timeframe = self.TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)
        self.connected = False

    def connect(self, login: int = None, password: str = None, server: str = None) -> bool:
        """Connect to MT5/FTMO."""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

            # Login if credentials provided
            if login and password and server:
                authorized = mt5.login(login, password=password, server=server)
                if not authorized:
                    logger.error(f"Login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False

            # Verify symbol exists
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                mt5.shutdown()
                return False

            # Enable symbol for Market Watch
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select {self.symbol}")
                    mt5.shutdown()
                    return False

            self.connected = True
            logger.info(f"Connected to MT5 - Symbol: {self.symbol}, Spread: {symbol_info.spread}")
            return True

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_historical(self, bars: int = 1000, start_date: datetime = None) -> pd.DataFrame:
        """Fetch historical OHLC data."""
        if not self.connected:
            raise RuntimeError("Not connected to MT5")

        try:
            if start_date:
                rates = mt5.copy_rates_from(
                    self.symbol,
                    self.mt5_timeframe,
                    start_date,
                    bars
                )
            else:
                rates = mt5.copy_rates_from_pos(
                    self.symbol,
                    self.mt5_timeframe,
                    0,  # from current bar
                    bars
                )

            if rates is None or len(rates) == 0:
                logger.error(f"No data received: {mt5.last_error()}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Rename columns to standard OHLC
            df.rename(columns={
                'tick_volume': 'volume',
                'real_volume': 'real_volume'
            }, inplace=True)

            # Keep only OHLC + volume
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_latest_tick(self) -> dict:
        """Get latest tick data for live trading."""
        if not self.connected:
            raise RuntimeError("Not connected to MT5")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return {}

        return {
            'time': pd.to_datetime(tick.time, unit='s'),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume
        }

    def get_symbol_info(self) -> dict:
        """Get symbol specifications."""
        if not self.connected:
            raise RuntimeError("Not connected to MT5")

        info = mt5.symbol_info(self.symbol)._asdict()

        return {
            'symbol': info['name'],
            'digits': info['digits'],
            'point': info['point'],
            'tick_size': info['trade_tick_size'],
            'tick_value': info['trade_tick_value'],
            'min_lot': info['volume_min'],
            'max_lot': info['volume_max'],
            'lot_step': info['volume_step'],
            'spread': info['spread'],
            'stops_level': info['trade_stops_level']
        }