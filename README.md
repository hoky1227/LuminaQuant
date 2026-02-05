[Korean Version (한국어 가이드)](README_KR.md)

# LuminaQuant - Event-Driven Quantitative Trading System

An advanced, event-driven quantitative trading pipeline designed for robust backtesting and live trading on Binance. Built with a modular architecture to ensure consistency between simulation and real-world execution.

## Features

- **Event-Driven Architecture**: Uses a centralized event loop (`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`) to mimic real-world trading mechanics.
- **Accurate Backtesting**:
  - Supports **Trailing Stops**, Stop Loss, and Take Profit.
  - Simulates fill latencies and commissions.
- **Live Trading**:
  - Integration with **Binance** via `ccxt`.
  - **Robust State Management**: Syncs positions with exchange on startup.
  - **Partial Fills**: Handles liquidity constraints by queuing remaining order quantities.
- **Optimization**:
  - Built-in parameter tuning using **Optuna** (Bayesian Optimization) or Grid Search.
  - Walk-Forward Analysis (Train/Validation/Test split).
- **Interactive Dashboard**:
  - Streamlit-based dashboard for performance analysis, equity curves, and trade visualization.
- **Easy Configuration**:
  - Centralized `config.yaml` for all trading, backtesting, and optimization settings.

## Project Structure

```
lumina-quant/
├── lumina_quant/           # Core Package
│   ├── backtest.py         # Backtesting Engine
│   ├── live_trader.py      # Live Trading Engine
│   └── config.py           # Configuration Loader
├── generate_data.py        # Synthetic Data Generator
├── run_backtest.py         # Backtest Entry Point
├── optimize.py             # Strategy Optimizer (Optuna)
├── run_live.py             # Live Trading Entry Point
├── dashboard.py            # Streamlit Dashboard
├── config.yaml             # Main Configuration File
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
    
    **Main Config**: Edit `config.yaml` to set your desired symbols, timeframe, and risk parameters.
    
    **Secrets**: Copy `.env.example` to `.env` and configure your API keys (required for Live Trading).
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```ini
    BINANCE_API_KEY=your_key
    BINANCE_SECRET_KEY=your_secret
    ```

## Configuration Detail (`config.yaml`)

The `config.yaml` file is the central control center. Here is a breakdown of all available settings:

### `system`
- `log_level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

### `trading` (Shared Settings)
- `symbols`: A list of trading pairs. Example: `["BTC/USDT", "ETH/USDT"]`.
- `timeframe`: Candle interval. Common defaults: `1m`, `5m`, `1h`, `4h`, `1d`.
- `initial_capital`: The starting cash balance for simulations.
- `target_allocation`: Position sizing rule. `0.1` means 10% of total equity per trade.
- `min_trade_qty`: Minimum order size to avoid exchange errors.

### `backtest`
- `start_date`: Simulation start date (`YYYY-MM-DD`).
- `end_date`: Simulation end date (`null` for "up to last available").
- `commission_rate`: Transaction cost per side. `0.001` = 0.1% (Binance default).
- `slippage_rate`: Modeled market impact/delay cost. `0.0005` = 0.05%.
- `annual_periods`: Used for annualizing metrics (Sharpe/Volatility).
    - Crypto 24/7 Daily: `365`
    - Crypto 24/7 Hourly: `8760` (365 * 24)
    - Crypto 24/7 Minutely: `525600` (365 * 24 * 60)

### `live`
- `testnet`: `true` for paper trading on Binance Testnet, `false` for real money.
- `poll_interval`: How often to fetch new data (in seconds).
- `order_timeout`: Time to wait for order fills before canceling/retrying.

### `optimization`
- `method`: `OPTUNA` (recommended) or `GRID`.
- `strategy`: The class name of the strategy to optimize (e.g., `RsiStrategy`).
- `optuna`:
    - `n_trials`: Number of experiments to run.
    - `params`: Define hyperparameter ranges.
        - Types: `int`, `float`, `categorical`.
        - Keys must match your strategy's `__init__` arguments.

## Interactive Dashboard

LuminaQuant includes a built-in dashboard to visualize backtest results.

### Launching the Dashboard
After running a backtest, execute:
```bash
uv run streamlit run dashboard.py
```

### Features
1.  **Performance Metrics**: Real-time display of ROI, Max Drawdown, Final Equity, and Total Trades.
2.  **Price Action Chart**: Interactive Plotly charts showing buy (green triangle) and sell (red triangle) points overlaid on the asset price.
3.  **Equity Curve**: Visualizes the growth of your strategy vs. a Buy & Hold benchmark.
4.  **Drawdown Analysis**: "Underwater" plot showing the depth and duration of drawdowns.
5.  **Optimized Parameters**: Displays the best parameters found by the optimizer (if loaded).

## Strategy Development

### Creating a New Strategy

1.  Create a new file in `strategies/` (e.g., `custom_strategy.py`).
2.  Inherit from `lumina_quant.strategy.Strategy`.
3.  Implement `__init__` to accept `bars`, `events`, and parameters.
4.  Implement `calculate_signals(self, event)` to process `MARKET` events.

**Example Template:**

```python
from lumina_quant.strategy import Strategy
from lumina_quant.events import SignalEvent

class CustomStrategy(Strategy):
    def __init__(self, bars, events, my_param=10):
        self.bars = bars      # Data Handler
        self.events = events  # Event Queue
        self.my_param = my_param
        self.symbol_list = self.bars.symbol_list

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for s in self.symbol_list:
                # Get latest close price
                bars = self.bars.get_latest_bars_values(s, "close", N=self.my_param)
                if len(bars) < self.my_param:
                    continue
                
                # Logic
                if bars[-1] > bars[-2]:
                    # SignalType: "LONG", "SHORT", "EXIT"
                    signal = SignalEvent(1, s, event.time, "LONG", 1.0)
                    self.events.put(signal)
```

### Registering the Strategy

1.  Import your strategy in `run_backtest.py` and `optimize.py`.
2.  Add it to the `STRATEGY_MAP` in both files (if using string config) or directly reference the class.

## Parameter Tuning (Optimization)

You can optimize strategy parameters using `config.yaml`.

**Example Config for `CustomStrategy`:**

```yaml
optimization:
  method: "OPTUNA"
  strategy: "CustomStrategy"
  
  optuna:
    n_trials: 50
    params:
      my_param:
        type: "int"
        low: 5
        high: 20
```

Run the optimizer:
```bash
uv run optimize.py
```
This will find the best `my_param` that maximizes the Sharpe Ratio.

## Performance Metrics

The system calculates comprehensive metrics to evaluate strategy performance:

| Metric | Description |
| :--- | :--- |
| **Total Return** | The absolute percentage gain of the portfolio. |
| **Benchmark Return** | Return of a Buy & Hold strategy on the first symbol. |
| **CAGR** | Compound Annual Growth Rate (annualized return). |
| **Ann. Volatility** | Annualized standard deviation of returns (Risk). |
| **Sharpe Ratio** | Risk-adjusted return (Return / Risk). Higher is better. |
| **Sortino Ratio** | Similar to Sharpe, but only penalizes *downside* volatility. |
| **Calmar Ratio** | CAGR divided by Max Drawdown. |
| **Max Drawdown** | The largest peak-to-trough decline in equity. |
| **DD Duration** | The longest time period spent in a drawdown. |
| **Alpha** | Excess return of the strategy compared to the benchmark. |
| **Beta** | Sensitivity/Correlation of the strategy to the benchmark. |
| **Information Ratio** | Active Return divided by Active Risk (Tracking Error). |
| **Daily Win Rate** | Percentage of days with positive returns. |

Outputs are saved to `equity.csv` (equity curve) and `trades.csv` (trade log).
