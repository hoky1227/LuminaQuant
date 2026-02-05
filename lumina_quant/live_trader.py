import queue
import time
from lumina_quant.config import LiveConfig
from lumina_quant.utils.logging_utils import setup_logging
from lumina_quant.binance_execution import BinanceExecutionHandler
from lumina_quant.utils.persistence import StateManager
import time
import threading
from lumina_quant.engine import TradingEngine


class LiveTrader(TradingEngine):
    """
    The LiveTrader engine.
    """

    def __init__(
        self,
        symbol_list,
        data_handler_cls,
        execution_handler_cls,
        portfolio_cls,
        strategy_cls,
    ):
        self.logger = setup_logging("LiveTrader")
        self.symbol_list = symbol_list
        self.events = queue.Queue()
        self.config = LiveConfig
        self.state_manager = StateManager()

        self.data_handler = data_handler_cls(self.events, self.symbol_list)
        # Execution Handler may need config for API keys (if not generic) or timeouts
        self.execution_handler = execution_handler_cls(
            self.events, self.data_handler, self.config
        )

        self.portfolio = portfolio_cls(
            self.data_handler, self.events, time.time(), self.config
        )
        self.strategy = strategy_cls(self.data_handler, self.events)

        # Initialize Base Engine
        super().__init__(
            self.events,
            self.data_handler,
            self.strategy,
            self.portfolio,
            self.execution_handler,
        )

        # Load State
        self._load_state()

    def _load_state(self):
        state = self.state_manager.load_state()
        if state:
            self.logger.info("Restoring state from disk...")
            try:
                if "portfolio" in state:
                    self.portfolio.set_state(state["portfolio"])
                if "strategy" in state:
                    self.strategy.set_state(state["strategy"])
                self.logger.info("State restored.")
            except Exception as e:
                self.logger.error(f"Failed to restore state: {e}")

    def _save_state(self):
        try:
            state = {
                "portfolio": self.portfolio.get_state(),
                "strategy": self.strategy.get_state(),
            }
            self.state_manager.save_state(state)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def on_fill(self, event):
        """
        Hook from TradingEngine to save state on fill.
        """
        self.logger.info(
            f"Fill Event: {event.direction} {event.quantity} {event.symbol} @ {event.fill_cost}"
        )
        self._save_state()

    def _sync_portfolio(self):
        """
        Syncs the internal portfolio state with the exchange state.
        This runs AFTER load_state, so we should be careful not to overwrite valid persistent state
        with partial exchange data unless warranted.
        """
        if isinstance(self.execution_handler, BinanceExecutionHandler):
            self.logger.info("Syncing Portfolio with Binance Exchange...")

            try:
                # 1. Sync Cash
                balance = self.execution_handler.get_balance()
                if balance > 0:
                    self.logger.info(f"Exchange USDT Balance: {balance}")
                    self.portfolio.current_holdings["cash"] = balance
                    # If total is 0 (first run), init with balance.
                    # If not, let it be recalced below.
                    if (
                        self.portfolio.current_holdings["total"]
                        == self.portfolio.initial_capital
                    ):
                        self.portfolio.initial_capital = balance

                # 2. Sync Positions
                # We need to fetch actual positions from Binance
                # Assuming execution_handler has a way to get all positions, or we access exchange directly here?
                # Best practice: Add get_all_positions to ExecutionHandler interface or specific class.
                # For now, we'll try to use the exchange object from data_handler if accessible, or add method to execution handler.

                # Let's assume we added `get_all_positions` to BinanceExecutionHandler.
                if hasattr(self.execution_handler, "get_all_positions"):
                    exchange_positions = self.execution_handler.get_all_positions()
                    self.logger.info(f"Exchange Positions: {exchange_positions}")

                    for s, qty in exchange_positions.items():
                        if s in self.symbol_list:
                            self.portfolio.current_positions[s] = qty

                # 3. Recalculate Total Holdings (Cash + Position Value)
                total_equity = self.portfolio.current_holdings["cash"]
                for s in self.symbol_list:
                    qty = self.portfolio.current_positions.get(s, 0.0)
                    if qty != 0:
                        last_price = self.data_handler.get_latest_bar_value(s, "close")
                        # If price is 0 (no data yet), we can't value it perfectly.
                        if last_price > 0:
                            total_equity += qty * last_price

                self.portfolio.current_holdings["total"] = total_equity

                # 4. Reconcile Strategy State with Portfolio State
                if hasattr(self.strategy, "bought"):
                    for s in self.symbol_list:
                        position_qty = self.portfolio.current_positions.get(s, 0)
                        strategy_status = self.strategy.bought.get(s, "OUT")

                        if strategy_status != "OUT" and position_qty == 0:
                            self.logger.warning(
                                f"State Mismatch for {s}: Strategy={strategy_status}, Portfolio=0. Resetting Strategy to OUT."
                            )
                            self.strategy.bought[s] = "OUT"
                        elif strategy_status == "OUT" and position_qty != 0:
                            self.logger.warning(
                                f"State Mismatch for {s}: Strategy=OUT, Portfolio={position_qty}. Setting Strategy to matching Side."
                            )
                            self.strategy.bought[s] = (
                                "LONG" if position_qty > 0 else "SHORT"
                            )

                self.logger.info(
                    f"Portfolio Sync Completed. Total Equity: {total_equity}"
                )
            except Exception as e:
                self.logger.error(f"Portfolio Sync Failed: {e}")

    def handle_market_event(self, event):
        """
        Override to save equity curve on every bar.
        """
        super().handle_market_event(event)

        # Save Live Equity
        self.portfolio.create_equity_curve_dataframe()
        self.portfolio.save_equity_curve("live_equity.csv")

    def handle_fill_event(self, event):
        """
        Override to save trades on every fill.
        """
        super().handle_fill_event(event)  # This calls self.on_fill(event) too

        # Save Live Trades
        self.portfolio.output_trade_log("live_trades.csv")

    def run(self):
        """
        Main Live Trading Loop.
        """
        self.logger.info(f"Starting Live Trading on {self.symbol_list}...")
        self._sync_portfolio()

        while True:
            try:
                # In Live mode, data_handler is threaded and pushes events autonomously.
                # We just blocking-wait for events.
                event = self.events.get(True, timeout=10)  # Wait up to 10s

                # Add logging for events if needed (optional overlay on process_event)
                # But engine.py handles basic routing.
                # We might want logging wrappers?
                # For now, rely on engine.
                # But wait, engine doesn't log!
                # I should override handle_xxx methods or add logging to engine.
                # To be Clean OOP, let's keep engine distinct and perhaps add logging there or here.
                # simpler: just log here then call process.

                if event is not None:
                    if event.type == "MARKET":
                        self.logger.debug(f"Market Event: {event.symbol}")
                    elif event.type == "SIGNAL":
                        self.logger.info(f"Signal Event: {event.signal_type}")
                    elif event.type == "ORDER":
                        self.logger.info(f"Order Event: {event.direction}")

                    self.process_event(event)

            except queue.Empty:
                # Heartbeat
                pass
            except KeyboardInterrupt:
                self.logger.info("Stopping Live Trader...")
                self._save_state()  # Save on exit
                self.data_handler.continue_backtest = False
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self._save_state()  # Save on crash attempt
