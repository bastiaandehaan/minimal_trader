# 📈 Minimal Trader

A clean, modular multi-strategy trading framework for algorithmic backtesting and live trading with MetaTrader 5 integration.

## 🏗️ Architecture

```
minimal_trader/
├── main.py                 # CLI entry point
├── engine.py              # Multi-strategy execution engine  
├── config.yaml            # Trading configuration
├── strategies/            # Trading strategies
│   ├── abstract.py        # Base strategy interface
│   ├── sma_cross.py       # SMA crossover strategy
│   └── breakout.py        # Breakout strategy
├── feeds/                 # Data sources
│   ├── csv_feed.py        # CSV data loader
│   └── mt5_feed.py        # MetaTrader 5 integration
└── tests/                 # Test suite
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MetaTrader 5 (for live data/trading)
- Windows (required for MT5 integration)

### Installation
```bash
git clone https://github.com/bastiaandehaan/minimal_trader.git
cd minimal_trader
pip install -r requirements.txt
```

### Configuration
1. **Edit `config.yaml`:**
   ```yaml
   # MT5 Connection (optional for backtesting)
   mt5:
     login: your_account_number
     password: "your_password"
     server: "FTMO-Demo"
   
   # Strategy Settings
   strategies:
     sma_cross:
       enabled: true
       allocation: 100.0  # % of capital
       params:
         sma_period: 20
         atr_period: 14
   ```

### Basic Usage

#### 🔍 Test MT5 Connection
```bash
python main.py test
```

#### 📊 Run Backtest (CSV Data)
```bash
# Using CSV file
python main.py backtest --csv data/EURUSD_H1.csv

# Using MT5 historical data
python main.py backtest --symbol GER40.cash
```

#### 📈 Live Trading (Not Yet Implemented)
```bash
python main.py live
```

## 🧠 Strategies

### SMA Cross Strategy
**Logic**: Buy when price crosses above SMA, sell when crosses below  
**Parameters**:
- `sma_period`: Moving average period (default: 20)
- `atr_period`: ATR period for stop/target (default: 14)
- `sl_multiplier`: Stop loss ATR multiplier (default: 1.5)
- `tp_multiplier`: Take profit ATR multiplier (default: 2.5)

### Breakout Strategy  
**Logic**: Buy/sell on price breakouts from recent high/low ranges  
**Parameters**:
- `lookback_period`: Range calculation period (default: 20)
- `breakout_factor`: Breakout threshold multiplier (default: 1.0)
- `atr_period`: ATR period for stop/target (default: 14)

## 🔧 Advanced Usage

### Custom CSV Data Format
Your CSV must contain columns: `open`, `high`, `low`, `close`, `volume`
```python
# Example CSV structure
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.1000,1.1020,1.0980,1.1010,1500
```

### Multi-Strategy Configuration
```yaml
strategies:
  sma_cross:
    enabled: true
    allocation: 60.0    # 60% of capital
  breakout:
    enabled: true  
    allocation: 40.0    # 40% of capital
```

### Risk Management
```yaml
engine:
  initial_capital: 10000.0
  risk_per_trade: 1.0      # % of capital per trade
  max_positions: 3         # Maximum concurrent positions  
  commission: 0.0002       # 2 basis points
  time_exit_bars: 200      # Force exit after N bars
```

## 🔬 Development

### Running Tests
```bash
# Run all tests (recommended method)
python -m pytest tests/ -v

# Alternative: Install as editable package first
pip install -e .
pytest -v

# Run specific test
python -m pytest tests/test_integration.py::test_pipeline_produces_some_trades -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Creating Custom Strategies

1. **Inherit from `AbstractStrategy`:**
```python
from strategies.abstract import AbstractStrategy, Signal, SignalType
import pandas as pd

class MyStrategy(AbstractStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"
    
    @property 
    def required_bars(self) -> int:
        return self.params.get('period', 20)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add your indicators here
        df['my_indicator'] = df['close'].rolling(20).mean()
        return df
    
    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[Signal, Dict]:
        # Generate signals at bar i (no look-ahead!)
        if i < self.required_bars:
            return Signal(SignalType.NONE), {}
            
        # Your signal logic here
        if df.iloc[i]['close'] > df.iloc[i]['my_indicator']:
            return Signal(
                type=SignalType.BUY,
                entry=df.iloc[i]['close'],
                stop=df.iloc[i]['close'] * 0.98,
                target=df.iloc[i]['close'] * 1.05
            ), {}
            
        return Signal(SignalType.NONE), {}
```

2. **Add to config.yaml:**
```yaml
strategies:
  my_strategy:
    enabled: true
    allocation: 50.0
    params:
      period: 20
```

3. **Register in main.py:**
```python
from strategies.my_strategy import MyStrategy

def create_strategies(config: dict) -> list:
    strategies = []
    
    if config.get('strategies', {}).get('my_strategy', {}).get('enabled'):
        params = config['strategies']['my_strategy'].get('params', {})
        allocation = config['strategies']['my_strategy'].get('allocation', 100.0)
        strategies.append((MyStrategy(params), allocation))
        
    return strategies
```

### Adding Custom Data Feeds

```python
from feeds.abstract import DataFeed  # You'd need to create this
import pandas as pd

class MyDataFeed(DataFeed):
    def load(self) -> pd.DataFrame:
        # Your data loading logic
        return df
```

## 📋 Features

### ✅ Implemented
- [x] Multi-strategy backtesting engine
- [x] SMA crossover strategy
- [x] Breakout strategy  
- [x] CSV data loading
- [x] MetaTrader 5 integration
- [x] Risk management (position sizing, stop/target)
- [x] Performance metrics (Sharpe, drawdown, win rate)
- [x] No look-ahead bias protection
- [x] Comprehensive test suite

### 🔄 In Development
- [ ] Live trading execution
- [ ] Advanced portfolio analytics
- [ ] Strategy optimization framework
- [ ] Web dashboard
- [ ] More sophisticated risk management

### 💡 Planned Features
- [ ] Machine learning strategy framework
- [ ] Multi-timeframe analysis
- [ ] Real-time notifications
- [ ] Database integration
- [ ] Cloud deployment support

## 🐛 Troubleshooting

### Common Issues

**1. MT5 Connection Fails**
```
Solution: Ensure MT5 is running and credentials are correct in config.yaml
```

**2. Import Errors**
```bash
# Ensure you're in the project root directory
cd minimal_trader
python main.py --help
```

**3. No Trades Generated**
```
Check your strategy parameters and ensure your data has sufficient volatility
for signal generation. Try adjusting SMA periods or ATR multipliers.
```

**4. Tests Failing**
```bash
# Clean reinstall
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
pytest tests/ -v
```

## 🏛️ Design Principles

### **1. No Look-Ahead Bias**
All strategies operate on bar `i` using only data from bars `0` to `i`. Future data is never accessed.

### **2. Modular Architecture**
Each component (strategies, feeds, engine) is independently testable and replaceable.

### **3. Configuration-Driven**
All parameters are externalized to `config.yaml` for easy experimentation.

### **4. Performance First**
Indicators calculated once (O(n)), not per signal generation (O(n²)).

### **5. Robust Error Handling**
Graceful degradation when strategies fail or data is missing.

## 📊 Performance Metrics

The engine provides comprehensive performance analytics:

- **Total Return %**: Overall portfolio return
- **Sharpe Ratio**: Risk-adjusted return metric  
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss ratio
- **Strategy Breakdown**: Per-strategy performance metrics

## 🔒 Risk Management

### Position Sizing
Uses **fixed risk** model: Each trade risks a fixed % of capital based on stop distance.

```python
risk_amount = capital * (risk_per_trade / 100)  
position_size = risk_amount / abs(entry_price - stop_price)
```

### Exit Rules
1. **Stop Loss**: Hard price-based exit
2. **Take Profit**: Target price-based exit  
3. **Time Exit**: Force close after N bars (prevents stuck positions)
4. **Signal Reversal**: Exit on opposite signal

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-strategy`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit changes (`git commit -m 'Add amazing strategy'`)
6. Push to branch (`git push origin feature/amazing-strategy`)
7. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Never trade with money you cannot afford to lose
- Test thoroughly on paper/demo accounts before live trading
- The authors are not responsible for any financial losses

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/bastiaandehaan/minimal_trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bastiaandehaan/minimal_trader/discussions)
- **Email**: [your.email@domain.com]

---

## 🤖 Claude AI Integration

This README is specifically designed for both human developers and Claude AI assistant interactions.

### For Claude AI Users:

**Key Commands for Development:**
```bash
# Test the current state
python -m pytest tests/ -v

# Run backtest with synthetic data  
python main.py backtest --csv tests/synthetic_data.csv

# Check architecture
find . -name "*.py" -not -path "./.venv/*" | head -20
```

**Important Files to Read First:**
1. `engine.py` - Core backtesting logic
2. `strategies/abstract.py` - Strategy interface
3. `config.yaml` - Configuration structure
4. `tests/test_*.py` - Usage examples and test patterns

**Architecture Patterns Used:**
- **Strategy Pattern**: `AbstractStrategy` base class
- **Factory Pattern**: Strategy creation in `main.py`
- **Configuration Pattern**: YAML-based config
- **Engine Pattern**: `MultiStrategyEngine` orchestrates everything

**Code Guidelines:**
- All strategies must implement `AbstractStrategy` interface
- No look-ahead bias: `get_signal(df, i)` uses only data `[0:i+1]`
- Tests must pass: `pytest tests/ -v`
- Type hints required for new code
- Docstrings for all public methods

**When Modifying:**
1. Read existing code structure first
2. Follow established patterns  
3. Add/update tests for changes
4. Verify no regressions: `pytest tests/ -v`
5. Update this README if architecture changes

**Performance Considerations:**
- Indicators calculated once in `calculate_indicators()`
- Signal generation is O(1) per bar
- Avoid O(n²) loops in backtesting
- Use vectorized pandas operations where possible

---

*Built with ❤️ for algorithmic trading enthusiasts*