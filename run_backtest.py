from datetime import datetime
from lumina_quant.backtest import Backtest
from lumina_quant.data import HistoricCSVDataHandler
from lumina_quant.execution import SimulatedExecutionHandler
from lumina_quant.portfolio import Portfolio
from quants_agent.config import BacktestConfig

# ==========================================
# USER CONFIGURATION AREA
# ==========================================

# 1. Strategy Selection
# Import your strategy class and set it here
# Example 1: Moving Average
from strategies.moving_average import MovingAverageCrossStrategy

# Example 2: RSI
from strategies.rsi_strategy import RsiStrategy

# SELECT YOUR STRATEGY HERE
# STRATEGY_CLASS = MovingAverageCrossStrategy
STRATEGY_CLASS = RsiStrategy

# 2. Strategy Parameters
# These are passed to your strategy's __init__
# For Moving Average:
# STRATEGY_PARAMS = {"short_window": 10, "long_window": 30}
# For RSI:
STRATEGY_PARAMS = {"rsi_period": 14, "oversold": 30, "overbought": 70}

# AUTO-LOAD OPTIMIZED PARAMS
# If True, attempts to load from 'best_optimized_parameters/<StrategyName>/best_params.json'
LOAD_OPTIMIZED = True

if LOAD_OPTIMIZED:
    import os
    import json

    strategy_name = STRATEGY_CLASS.__name__
    param_path = os.path.join(
        "best_optimized_parameters", strategy_name, "best_params.json"
    )

    if os.path.exists(param_path):
        try:
            with open(param_path, "r") as f:
                loaded_params = json.load(f)
            print(f"✅ Loaded Optimized Params from {param_path}")
            print(f"   Old: {STRATEGY_PARAMS}")
            STRATEGY_PARAMS = loaded_params
            print(f"   New: {STRATEGY_PARAMS}")
        except Exception as e:
            print(f"⚠️ Failed to load optimized params: {e}")
    else:
        print(f"ℹ️ Optimized params not found at {param_path}. Using Manual Defaults.")

# 3. Data Settings
CSV_DIR = "data"  # Directory containing your .csv files (e.g., BTCUSDT.csv)
SYMBOL_LIST = ["BTCUSDT"]  # Must match filenames in CSV_DIR

# 4. Dates
# Format: datetime(Year, Month, Day)
START_DATE = datetime(2022, 1, 1)

# ==========================================
# EXECUTION (Do not modify generally)
# ==========================================


def run():
    print("------------------------------------------------")
    print(f"Running Backtest for {SYMBOL_LIST}")
    print(f"Strategy: {STRATEGY_CLASS.__name__}")
    print(f"Params: {STRATEGY_PARAMS}")
    print("------------------------------------------------")

    # Initialize Backtest
    backtest = Backtest(
        csv_dir=CSV_DIR,
        symbol_list=SYMBOL_LIST,
        start_date=START_DATE,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=STRATEGY_CLASS,
        strategy_params=STRATEGY_PARAMS,
    )

    backtest.simulate_trading()


if __name__ == "__main__":
    run()
