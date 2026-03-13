"""Exchange driver factory with lazy imports for optional dependencies."""

from __future__ import annotations


def get_exchange(config: object):
    """Factory function to get the appropriate exchange implementation."""
    exchange_config = getattr(config, "EXCHANGE", None)
    if not isinstance(exchange_config, dict):
        raise ValueError("EXCHANGE config must be a dictionary with a 'driver' field.")
    driver = str(exchange_config.get("driver", "")).strip().lower()

    if driver == "ccxt":
        from .ccxt_exchange import CCXTExchange

        return CCXTExchange(config)
    if driver == "mt5":
        from .mt5_exchange import MT5Exchange

        return MT5Exchange(config)
    if driver == "polymarket":
        from .polymarket_exchange import PolymarketExchange

        return PolymarketExchange(config)
    raise ValueError(f"Unknown exchange driver: {driver}")


__all__ = ["get_exchange"]
