import polars as pl
from quants_agent.events import OrderEvent


class Portfolio:
    """
    The Portfolio class handles the positions and market value.
    Refactored to use Polars for equity curve storage.
    """

    def __init__(self, bars, events, start_date, config):
        self.bars = bars
        self.events = events
        self.config = config
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = self.config.INITIAL_CAPITAL

        self.all_positions = []
        self.current_positions = {s: 0.0 for s in self.symbol_list}

        self.all_holdings = []
        self.current_holdings = self.construct_current_holdings()

        # Trade Log (for Visualization)
        self.trades = []

        # Circuit Breaker (Safety)
        self.circuit_breaker_tripped = False
        self.day_start_equity = self.initial_capital
        self.max_daily_loss_pct = getattr(
            config, "MAX_DAILY_LOSS_PCT", 0.05
        )  # 5% default

        # Initialize first record
        self.update_initial_record()

    def construct_current_holdings(self):
        d = {s: 0.0 for s in self.symbol_list}
        d["cash"] = self.initial_capital
        d["commission"] = 0.0
        d["total"] = self.initial_capital
        return d

    def update_initial_record(self):
        # Initial positions - Store as Tuple: (datetime, s1, s2, ...)
        # Rely on self.symbol_list order
        pos_row = [self.start_date] + [0.0 for _ in self.symbol_list]
        self.all_positions.append(tuple(pos_row))

        # Initial holdings - Store as Tuple: (datetime, cash, commission, total, s1, s2, ..., benchmark_price)
        h_row = (
            [self.start_date, self.initial_capital, 0.0, self.initial_capital]
            + [0.0 for _ in self.symbol_list]
            + [0.0]
        )  # Benchmark Price Placeholder
        self.all_holdings.append(tuple(h_row))
        self.save_portfolio_state()

    def save_portfolio_state(self):
        # We assume LiveTrader handles file I/O via get_state
        pass

    def get_state(self):
        return {
            "positions": self.current_positions,
            "holdings": self.current_holdings,
            "initial_capital": self.initial_capital,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
        }

    def set_state(self, state):
        if "positions" in state:
            self.current_positions = state["positions"]
        if "holdings" in state:
            self.current_holdings = state["holdings"]
        if "initial_capital" in state:
            self.initial_capital = state["initial_capital"]
        if "circuit_breaker_tripped" in state:
            self.circuit_breaker_tripped = state["circuit_breaker_tripped"]

    def update_timeindex(self, event):
        """
        Updates the positions from the current locations to the
        latest available bar.
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])

        # Update positions
        # Tuple: (datetime, s1, s2...)
        pos_row = [latest_datetime] + [
            self.current_positions[s] for s in self.symbol_list
        ]
        self.all_positions.append(tuple(pos_row))

        # Update holdings
        # Tuple: (datetime, cash, commission, total, s1, s2...)
        dh = {s: 0.0 for s in self.symbol_list}
        dh["datetime"] = latest_datetime
        dh["cash"] = self.current_holdings["cash"]
        dh["commission"] = self.current_holdings["commission"]
        dh["total"] = self.current_holdings["cash"]

        market_vals = []
        for s in self.symbol_list:
            market_value = 0.0
            if self.current_positions[s] != 0:
                market_value = self.current_positions[
                    s
                ] * self.bars.get_latest_bar_value(s, "close")
            market_vals.append(market_value)
            dh["total"] += market_value
            # Update current holdings dict for state (still useful for logic)
            self.current_holdings[s] = market_value

        self.current_holdings["total"] = dh["total"]

        # Store Tuple
        # Schema: (datetime, cash, commission, total, s1_val, s2_val, ..., benchmark_price)
        # Benchmark: Close price of first symbol (Primary Asset)
        bench_price = self.bars.get_latest_bar_value(self.symbol_list[0], "close")

        h_row = (
            [
                latest_datetime,
                dh["cash"],
                dh["commission"],
                dh["total"],
            ]
            + market_vals
            + [bench_price]
        )
        self.all_holdings.append(tuple(h_row))

    def update_positions_from_fill(self, fill):
        fill_dir = 0
        if fill.direction == "BUY":
            fill_dir = 1
        if fill.direction == "SELL":
            fill_dir = -1

        self.current_positions[fill.symbol] += fill_dir * fill.quantity

    def update_holdings_from_fill(self, fill):
        fill_dir = 0
        if fill.direction == "BUY":
            fill_dir = 1
        if fill.direction == "SELL":
            fill_dir = -1

        # USE ACTUAL FILL PRICE (realism)
        # If fill_cost is provided, derive unit price from it.
        # Otherwise fallback to bar close (legacy/compatibility).
        if fill.fill_cost is not None and fill.quantity > 0:
            unit_fill_price = fill.fill_cost / fill.quantity
        else:
            unit_fill_price = self.bars.get_latest_bar_value(fill.symbol, "close")

        cost = fill_dir * unit_fill_price * fill.quantity

        self.current_holdings[fill.symbol] += cost
        self.current_holdings["commission"] += fill.commission
        self.current_holdings["cash"] -= cost + fill.commission
        self.current_holdings["total"] -= fill.commission

    def update_fill(self, event):
        if event.type == "FILL":
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

            # Log Trade
            # FillEvent: timeindex, symbol, exchange, quantity, direction, fill_cost, commission
            self.trades.append(
                {
                    "datetime": event.timeindex,
                    "symbol": event.symbol,
                    "direction": event.direction,
                    "quantity": event.quantity,
                    "fill_cost": event.fill_cost,
                    "commission": event.commission,
                    "price": event.fill_cost / event.quantity
                    if event.quantity > 0
                    else 0.0,
                }
            )

            self._check_circuit_breaker()

    def _check_circuit_breaker(self):
        """
        Circuit Breaker: Halt trading if daily loss exceeds threshold.
        """
        if self.circuit_breaker_tripped:
            return  # Already tripped

        current_equity = self.current_holdings["total"]
        loss_pct = (self.day_start_equity - current_equity) / self.day_start_equity

        if loss_pct >= self.max_daily_loss_pct:
            self.circuit_breaker_tripped = True
            print(
                f"[CIRCUIT BREAKER] Daily loss {loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}. HALTING TRADING."
            )

    def generate_order_from_signal(self, signal) -> OrderEvent:
        """
        Generates an OrderEvent from a SignalEvent.
        Implements Fixed Fractional Position Sizing (Target Allocation).
        """
        order = None
        symbol = signal.symbol
        direction = signal.signal_type

        # Get current price to estimate quantity
        current_price = self.bars.get_latest_bar_value(symbol, "close")
        if current_price == 0:
            return None

        # Position Sizing: Use Target Allocation % of current equity
        target_allocation = self.config.TARGET_ALLOCATION
        equity = self.current_holdings["total"]
        target_value = equity * target_allocation

        # Calculate Quantity
        mkt_quantity = int(target_value / current_price)

        # Minimum Quantity Enforcement (prevent 0 qty orders)
        if mkt_quantity == 0:
            mkt_quantity = self.config.MIN_TRADE_QTY

        if direction == "LONG":
            order = OrderEvent(symbol, "MKT", mkt_quantity, "BUY")
        elif direction == "SHORT":
            order = OrderEvent(symbol, "MKT", mkt_quantity, "SELL")
        elif direction == "EXIT":
            cur_qty = self.current_positions[symbol]
            if cur_qty != 0:
                order = OrderEvent(
                    symbol, "MKT", abs(cur_qty), "SELL" if cur_qty > 0 else "BUY"
                )

        return order

    def update_signal(self, event):
        if self.circuit_breaker_tripped:
            return  # Do not generate orders when breaker is tripped
        if event.type == "SIGNAL":
            order_event = self.generate_order_from_signal(event)
            if order_event is not None:
                self.events.put(order_event)

    def create_equity_curve_dataframe(self):
        """
        Creates a Polars DataFrame from the all_holdings list (list of Tuples).
        """
        # Define Schema matches Tuple order
        # (datetime, cash, commission, total, s1, s2, ..., benchmark_price)
        cols = (
            ["datetime", "cash", "commission", "total"]
            + self.symbol_list
            + ["benchmark_price"]
        )

        # Polars handles list of tuples with 'schema' or 'columns' arg
        # Note: If list is empty, this might crash, but typically not in backtest.
        self.equity_curve = pl.DataFrame(self.all_holdings, schema=cols, orient="row")

        # Calculate returns
        self.equity_curve = self.equity_curve.with_columns(
            [(pl.col("total").diff() / pl.col("total").shift(1)).alias("returns")]
        )

        # Calculate Benchmark Returns (Buy & Hold)
        # Using benchmark_price column
        self.equity_curve = self.equity_curve.with_columns(
            [
                (
                    pl.col("benchmark_price").diff()
                    / pl.col("benchmark_price").shift(1)
                ).alias("benchmark_returns")
            ]
        )

        # Cumprod for equity curve (normalized)
        if len(self.equity_curve) > 0:
            start_val = self.equity_curve["total"][0]
            self.equity_curve = self.equity_curve.with_columns(
                [(pl.col("total") / start_val).alias("equity_curve_norm")]
            )

    def save_equity_curve(self, filename="equity.csv"):
        if hasattr(self, "equity_curve") and not self.equity_curve.is_empty():
            self.equity_curve.write_csv(filename)

    def output_summary_stats(self):
        """
        Creates a list of summary statistics.
        """
        # Convert to numpy for calc
        total_series = self.equity_curve["total"].to_numpy()
        returns = self.equity_curve["returns"].fill_null(0.0).to_numpy()
        benchmark_returns = (
            self.equity_curve["benchmark_returns"].fill_null(0.0).to_numpy()
        )

        if len(total_series) < 2:
            return [("Status", "Not enough data")]

        from quants_agent.utils.performance import (
            create_sharpe_ratio,
            create_drawdowns,
            create_cagr,
            create_sortino_ratio,
            create_calmar_ratio,
            create_annualized_volatility,
            create_alpha_beta,
            create_information_ratio,
        )

        # Use period from config if available (check BacktestConfig)
        periods = getattr(self.config, "ANNUAL_PERIODS", 252)  # Crypto often 365

        # 1. Total Return
        total_return = (total_series[-1] - total_series[0]) / total_series[0]

        # Benchmark Total Return (using last non-zero price vs first)
        # Note: Initial price might be 0 if recorded before first bar.
        # Check indices.
        first_price = (
            self.equity_curve["benchmark_price"][1]
            if len(self.equity_curve) > 1
            else 1.0
        )  # Index 1 is usually first bar
        last_price = self.equity_curve["benchmark_price"][-1]

        if first_price == 0:
            first_price = 1.0  # Safety
        benchmark_unrealized = (last_price - first_price) / first_price

        # 2. CAGR
        cagr = create_cagr(
            total_series[-1], total_series[0], len(total_series), periods
        )

        # 3. Volatility
        volatility = create_annualized_volatility(returns, periods)

        # 4. Sharpe
        sharpe_ratio = create_sharpe_ratio(returns, periods=periods)

        # 5. Sortino
        sortino_ratio = create_sortino_ratio(returns, periods=periods)

        # 6. Drawdowns
        drawdown, max_dd_duration = create_drawdowns(total_series)
        max_dd = max(drawdown)

        # 7. Calmar
        calmar_ratio = create_calmar_ratio(cagr, max_dd)

        # 8. Alpha / Beta
        alpha, beta = create_alpha_beta(returns, benchmark_returns, periods=periods)

        # 9. Information Ratio
        info_ratio = create_information_ratio(returns, benchmark_returns)

        # 10. Daily Win Rate
        winning_days = len(returns[returns > 0])
        total_days = len(returns) - 1  # exclude first 0 return
        win_rate = winning_days / total_days if total_days > 0 else 0.0

        stats = [
            ("Total Return", "%0.2f%%" % (total_return * 100.0)),
            ("Benchmark Return", "%0.2f%%" % (benchmark_unrealized * 100.0)),
            ("CAGR", "%0.2f%%" % (cagr * 100.0)),
            ("Ann. Volatility", "%0.2f%%" % (volatility * 100.0)),
            ("Sharpe Ratio", "%0.4f" % sharpe_ratio),
            ("Sortino Ratio", "%0.4f" % sortino_ratio),
            ("Calmar Ratio", "%0.4f" % calmar_ratio),
            ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
            ("DD Duration", "%d bars" % max_dd_duration),
            ("Alpha", "%0.4f" % alpha),
            ("Beta", "%0.4f" % beta),
            ("Information Ratio", "%0.4f" % info_ratio),
            ("Daily Win Rate", "%0.2f%%" % (win_rate * 100.0)),
        ]

        self.equity_curve.write_csv("equity.csv")
        return stats

    def output_trade_log(self, filename="trades.csv"):
        """
        Outputs the trade log to a CSV file.
        """
        if not self.trades:
            # print("No trades generated.") # Optional: don't spam
            return

        df = pl.DataFrame(self.trades)
        df.write_csv(filename)
        # print(f"Trade log saved to '{filename}'")
