# config_mt5_test.yaml
data:
  symbol: GER40.cash
  resample: 5min
  bars: 5000                   # Kleiner aantal voor test

# MT5 connectie zonder credentials (demo mode)
mt5:
  # Geen login/password/server = probeer demo connectie

engine:
  initial_capital: 10000.0
  risk_per_trade: 2.0          
  max_positions: 1             
  commission: 0.0002           
  slippage: 0.0001             
  time_exit_bars: 100          
  allow_shorts: false          
  min_risk_pts: 0.1            

strategies:
  simple_test:
    enabled: true
    allocation: 100.0
    params:
      entry_bar: 100           
      hold_bars: 50            

guards:
  trading_hours_tz: "Europe/Brussels"
  trading_hours_start: "00:00"  
  trading_hours_end: "23:59"
  min_atr_pts: 0                
  cooldown_bars: 0              
  max_trades_per_day: 10
  one_trade_per_timestamp: true