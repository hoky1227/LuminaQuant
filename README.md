# Quants Agent - Event-Driven Quantitative Trading System

An advanced, event-driven quantitative trading pipeline designed for robust backtesting and live trading on Binance. Built with a modular architecture to ensure consistency between simulation and real-world execution.

## Features

- **Event-Driven Architecture**: Uses a centralized event loop (`MarkteEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`) to mimic real-world trading mechanics.
- **Accurate Backtesting**:
  - Supports **Trailing Stops**, Stop Loss, and Take Profit.
  - Simulates fill latencies and commissions.
- **Live Trading**:
  - Integration with **Binance** via `ccxt`.
  - Threaded execution for non-blocking real-time data feeding (`LiveBinanceDataHandler`).
  - Secure configuration management via `.env`.
- **Strategy Ready**:
  - Sample `MovingAverageCrossStrategy` included using **TA-Lib**.
  - Easy-to-extend `Strategy` base class.

## Project Structure

```
quants-agent/
├── quants_agent/           # Core Package
│   ├── backtest.py         # Backtesting Engine
│   ├── live_trader.py      # Live Trading Engine
│   ├── events.py           # Event Definitions
│   ├── data.py             # Data Handlers (HistoricCSV, LiveBinance)
│   ├── execution.py        # Execution Handlers (Simulated, Binance)
│   ├── portfolio.py        # Portfolio & Risk Management
│   ├── strategy.py         # Base Strategy Class
│   ├── confg.py            # Configuration Manager
│   └── utils/              # Performance Metrics
├── generate_data.py        # Synthetic Data Generator
├── sample_strategy.py      # Backtest Entry Point
└── README.md
```

## Setup

1.  **Install Dependencies**
    ```bash
    uv sync
    # OR
    pip install .
    ```

2.  **Configuration**
    Copy `.env.example` to `.env` and configure your keys:
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```ini
    BINANCE_API_KEY=your_key
    BINANCE_SECRET_KEY=your_secret
    IS_TESTNET=True  # Set to False for real money
    ```

## Usage

### 1. Backtesting
Generate synthetic data and run a simulation:
```bash
# Generate dummy data for BTCUSDT
uv run generate_data.py

# Run the backtest
uv run sample_strategy.py
```
Results (Sharpe Ratio, Max Drawdown) will be printed to the console, and an `equity.csv` curve will be saved.

### 2. Live Trading
To run the live trader (make sure `.env` is set!):
```python
from quants_agent.live_trader import LiveTrader
from quants_agent.live_data import LiveBinanceDataHandler
from quants_agent.binance_execution import BinanceExecutionHandler
from quants_agent.portfolio import Portfolio
from sample_strategy import MovingAverageCrossStrategy # Or your custom strategy

# Define symbols
symbol_list = ['BTC/USDT', 'ETH/USDT']

trader = LiveTrader(
    symbol_list=symbol_list,
    data_handler_cls=LiveBinanceDataHandler,
    execution_handler_cls=BinanceExecutionHandler,
    portfolio_cls=Portfolio,
    strategy_cls=MovingAverageCrossStrategy
)

trader.run()
```

## Data Analysis
Performance metrics include:
- **Total Return**
- **Sharpe Ratio**
- **Max Drawdown**
- **Drawdown Duration**
