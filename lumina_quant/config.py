import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BaseConfig:
    """
    Base configuration with shared settings.
    """

    # System
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Strategy Defaults
    SYMBOLS = os.getenv("SYMBOL_LIST", "BTC/USDT").split(",")
    TIMEFRAME = os.getenv("TIMEFRAME", "1d")

    # Shared Risk Settings (Can be overridden)
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000.0"))
    TARGET_ALLOCATION = float(os.getenv("TARGET_ALLOCATION", "0.1"))  # 10% per trade
    MIN_TRADE_QTY = float(os.getenv("MIN_TRADE_QTY", "0.001"))


class BacktestConfig(BaseConfig):
    """
    Configuration for Backtesting Simulation.
    """

    # Simulation Parameters
    START_DATE = os.getenv("BACKTEST_START_DATE", "2022-01-01")
    END_DATE = os.getenv("BACKTEST_END_DATE", None)  # None means until now/end of data

    # Cost Modeling
    COMMISSION_RATE = float(os.getenv("BACKTEST_COMMISSION", "0.001"))  # 0.1%
    SLIPPAGE_RATE = float(os.getenv("BACKTEST_SLIPPAGE", "0.0005"))  # 0.05%

    # Performance Analysis
    RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.0"))
    ANNUAL_PERIODS = int(
        os.getenv("ANNUAL_PERIODS", "252")
    )  # 252 for daily, 252*24*60 for minutely etc.


class LiveConfig(BaseConfig):
    """
    Configuration for Live Trading.
    """

    # Binance API Credentials
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

    # Execution
    IS_TESTNET = os.getenv("IS_TESTNET", "True").lower() in ("true", "1", "t")
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "2"))  # Seconds
    ORDER_TIMEOUT = int(os.getenv("ORDER_TIMEOUT", "10"))  # Seconds

    @classmethod
    def validate(cls):
        if not cls.BINANCE_API_KEY or not cls.BINANCE_SECRET_KEY:
            raise ValueError("API Keys are missing for Live Trading.")
