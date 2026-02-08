import time
from functools import wraps
from lumina_quant.events import FillEvent
from lumina_quant.execution import ExecutionHandler
from lumina_quant.interfaces import ExchangeInterface


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
                except Exception as e:
                    cnt += 1
                    print(f"API Error ({cnt}/{retries}): {e}. Retrying in {d}s...")
                    time.sleep(d)
                    d *= backoff
            raise Exception(
                f"Failed to execute {func.__name__} after {retries} retries."
            )

        return wrapper

    return decorator


class LiveExecutionHandler(ExecutionHandler):
    """
    Handles order execution via an ExchangeInterface.
    """

    def __init__(self, events, bars, config, exchange: ExchangeInterface):
        self.events = events
        self.bars = bars
        self.config = config
        self.exchange = exchange

    @api_retry(retries=3, delay=2)
    def get_balance(self):
        """
        Returns the free USDT balance.
        """
        return self.exchange.get_balance("USDT")

    @api_retry(retries=3, delay=1)
    def get_all_positions(self):
        """
        Returns a dict of {symbol: quantity} for open positions.
        """
        return self.exchange.get_all_positions()

    @api_retry(retries=3, delay=1)
    def execute_order(self, event):
        """
        Executes an OrderEvent on the Exchange.
        """
        if event.type == "ORDER":
            # Convert direction to side
            side = "buy" if event.direction == "BUY" else "sell"

            # Order Type
            order_type = "market"
            if event.order_type == "LMT":
                order_type = "limit"

            params = {}
            price = event.price

            print(
                f"Executing Order: {event.symbol} {side} {event.quantity} @ {order_type}"
            )

            try:
                order = self.exchange.execute_order(
                    symbol=event.symbol,
                    type=order_type,
                    side=side,
                    quantity=event.quantity,
                    price=price,
                    params=params,
                )

                # Check Status
                # Market orders usually fill immediately, but Limit orders are 'open'
                if order["status"] == "closed" or (
                    order["status"] == "open" and order_type == "market"
                ):
                    # Assuming full fill for Market
                    filled_qty = order["filled"]
                    if filled_qty is None or filled_qty == 0.0:
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
                        exchange="EXCHANGE",  # Generic name
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
            except Exception as e:
                print(f"Execution Failed: {e}")
