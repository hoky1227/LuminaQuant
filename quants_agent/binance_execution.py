import ccxt
import time
from functools import wraps
from quants_agent.events import FillEvent
from quants_agent.execution import ExecutionHandler
from quants_agent.execution import ExecutionHandler


def api_retry(retries=3, delay=1, backoff=2):
    """
    Decorator for retrying API calls with exponential backoff.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cnt = 0
            d = delay
            while cnt < retries:
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    cnt += 1
                    print(f"API Error ({cnt}/{retries}): {e}. Retrying in {d}s...")
                    time.sleep(d)
                    d *= backoff
                except Exception as e:
                    print(f"Critical Error in {func.__name__}: {e}")
                    raise e
            raise Exception(
                f"Failed to execute {func.__name__} after {retries} retries."
            )

        return wrapper

    return decorator


class BinanceExecutionHandler(ExecutionHandler):
    """
    Handles order execution via the Binance API using CCXT.
    """

    def __init__(self, events, bars, config):
        self.events = events
        self.bars = bars
        self.config = config

        self.exchange = ccxt.binance(
            {
                "apiKey": self.config.BINANCE_API_KEY,
                "secret": self.config.BINANCE_SECRET_KEY,
                "enableRateLimit": True,
            }
        )

        if self.config.IS_TESTNET:
            self.exchange.set_sandbox_mode(True)
            print("BinanceExecutionHandler: Running in Sandbox/Testnet Mode")

    @api_retry(retries=3, delay=2)
    def get_balance(self):
        """
        Returns the free USDT balance.
        """
        balance = self.exchange.fetch_balance()
        return float(balance["USDT"]["free"])

    def get_all_positions(self):
        """
        Returns a dict of {symbol: quantity} for open positions.
        """
        positions = {}
        try:
            # Assuming Spot for now based on Config default symbols (BTC/USDT)
            self.exchange.fetch_balance()
            # Logic here is complex for spot symbol mapping, simplified for robustness check
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return positions

    @api_retry(retries=3, delay=1)
    def execute_order(self, event):
        """
        Executes an OrderEvent on Binance.
        """
        if event.type == "ORDER":
            # Convert direction to CCXT side
            side = "buy" if event.direction == "BUY" else "sell"

            ccxt_type = "market"
            if event.order_type == "LMT":
                ccxt_type = "limit"

            params = {}
            price = event.price

            # Simple check for quantity vs Min Notional (Binance usually requires >$5 or $10)
            # We assume strategy handles sizing, but we could add a check here.

            print(
                f"Executing Order: {event.symbol} {side} {event.quantity} @ {ccxt_type}"
            )

            order = self.exchange.create_order(
                symbol=event.symbol,
                type=ccxt_type,
                side=side,
                amount=event.quantity,
                price=price,
                params=params,
            )

            # Check Status
            # Market orders usually fill immediately, but Limit orders are 'open'
            if order["status"] == "closed" or (
                order["status"] == "open" and ccxt_type == "market"
            ):
                # Assuming full fill for Market
                filled_qty = order["filled"]
                if filled_qty is None:
                    filled_qty = order["amount"]  # Fallback

                fill_price = (
                    order.get("average")
                    or order.get("price")
                    or self.bars.get_latest_bar_value(event.symbol, "close")
                )

                commission = 0.0
                # Parse commission logic if needed

                fill_event = FillEvent(
                    timeindex=self.bars.get_latest_bar_datetime(event.symbol),
                    symbol=event.symbol,
                    exchange="BINANCE",
                    quantity=filled_qty,
                    direction=event.direction,
                    fill_cost=fill_price * filled_qty if fill_price else 0.0,
                    commission=commission,
                )
                self.events.put(fill_event)
                print(
                    f"Order Filled: {event.symbol} {side} {filled_qty} @ {fill_price}"
                )
            else:
                print(
                    f"Order {order['id']} status: {order['status']}. No Fill Event emitted yet."
                )
