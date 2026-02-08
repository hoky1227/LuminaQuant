import ccxt
from typing import Dict, List, Optional
from lumina_quant.interfaces import ExchangeInterface


class CCXTExchange(ExchangeInterface):
    """
    Implementation of ExchangeInterface using CCXT (e.g. for Binance).
    """

    def __init__(self, config):
        self.config = config
        self.exchange = None
        self.connect()

    def connect(self):
        # Determine exchange ID from config or default to binance
        exchange_id = "binance"
        exchange_config = getattr(self.config, "EXCHANGE", None)
        if exchange_config and isinstance(exchange_config, dict):
            exchange_id = exchange_config.get("name", "binance")
        else:
            exchange_id = getattr(self.config, "EXCHANGE_ID", "binance")

        if not exchange_id:
            exchange_id = "binance"

        exchange_class = getattr(ccxt, exchange_id)

        self.exchange = exchange_class(
            {
                "apiKey": getattr(self.config, "BINANCE_API_KEY", ""),
                "secret": getattr(self.config, "BINANCE_SECRET_KEY", ""),
                "enableRateLimit": True,
            }
        )

        if getattr(self.config, "IS_TESTNET", False):
            self.exchange.set_sandbox_mode(True)
            print(f"CCXTExchange ({exchange_id}): Running in Sandbox/Testnet Mode")

    def get_balance(self, currency: str = "USDT") -> float:
        try:
            balance = self.exchange.fetch_balance()
            return float(balance[currency]["free"])
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def get_all_positions(self) -> Dict[str, float]:
        positions = {}
        try:
            bal = self.exchange.fetch_balance()
            if "total" in bal:
                for coin, qty in bal["total"].items():
                    if qty > 0:
                        # Simple mapping assumes USDT pairs for now
                        symbol = f"{coin}/USDT"
                        positions[symbol] = qty
            return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[tuple]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            # ccxt returns [timestamp, open, high, low, close, volume]
            # convert to list of tuples
            return [tuple(candle[:6]) for candle in ohlcv]
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return []

    def execute_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        params: Dict = {},
    ) -> Dict:
        try:
            # Type: market or limit
            # Side: buy or sell
            order = self.exchange.create_order(
                symbol=symbol,
                type=type,
                side=side,
                amount=quantity,
                price=price,
                params=params,
            )

            # Standardize return
            return {
                "id": order["id"],
                "status": order["status"],
                "filled": order.get("filled", 0.0),
                "average": order.get("average", order.get("price")),
                "price": order.get("price"),
                "amount": order.get("amount"),
            }
        except Exception as e:
            print(f"Error executing order: {e}")
            raise e

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            result = []
            for order in orders:
                result.append(
                    {
                        "id": order["id"],
                        "symbol": order["symbol"],
                        "type": order["type"],
                        "side": order["side"],
                        "price": order["price"],
                        "amount": order["amount"],
                        "filled": order["filled"],
                        "status": order["status"],
                        "info": order["info"],
                    }
                )
            return result
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return []

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return False
