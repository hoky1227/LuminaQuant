from lumina_quant.live_trader import LiveTrader
from lumina_quant.live_data import LiveBinanceDataHandler
from lumina_quant.binance_execution import BinanceExecutionHandler
from lumina_quant.portfolio import Portfolio
from lumina_quant.config import LiveConfig
from strategies.moving_average import MovingAverageCrossStrategy
import sys


def main():
    # 1. Check Configuration
    print("=== Quants Agent Live Trader ===")
    print(f"Mode: {'TESTNET' if LiveConfig.IS_TESTNET else 'REAL TRADING'}")

    if not LiveConfig.BINANCE_API_KEY or not LiveConfig.BINANCE_SECRET_KEY:
        print(
            "ERROR: API Keys not found. Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in .env file."
        )
        sys.exit(1)

    # 2. Setup
    symbol_list = LiveConfig.SYMBOLS  # e.g. ['BTC/USDT'] from .env
    print(f"Trading Symbols: {symbol_list}")

    # 3. Initialize Trader
    try:
        trader = LiveTrader(
            symbol_list=symbol_list,
            data_handler_cls=LiveBinanceDataHandler,
            execution_handler_cls=BinanceExecutionHandler,
            portfolio_cls=Portfolio,
            strategy_cls=MovingAverageCrossStrategy,
        )

        # 4. Run
        print("Starting engine... Press Ctrl+C to stop.")
        trader.run()

    except KeyboardInterrupt:
        print("\nStopping trader...")
    except Exception as e:
        print(f"\nCritical Error: {e}")


if __name__ == "__main__":
    main()
