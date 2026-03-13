"""Polymarket exchange adapter with paper/shadow-first semantics."""

from __future__ import annotations

import os
from typing import Any

from lumina_quant.core.protocols import ExchangeInterface


class PolymarketExchange(ExchangeInterface):
    """Polymarket adapter for Phase 1 market-data + paper/shadow workflows."""

    def __init__(self, config):
        self.config = config
        self.connected = False
        self.client: Any = None
        self.connect()

    def connect(self):
        self.connected = True
        try:
            from py_clob_client.client import ClobClient  # type: ignore
        except Exception:
            self.client = None
            return

        private_key = os.getenv(getattr(self.config, "POLYMARKET_PRIVATE_KEY_ENV", "POLYMARKET_PRIVATE_KEY"), "")
        host = str(getattr(self.config, "POLYMARKET_HOST", "https://clob.polymarket.com") or "https://clob.polymarket.com")
        chain_id = int(getattr(self.config, "POLYMARKET_CHAIN_ID", 137) or 137)
        if private_key:
            self.client = ClobClient(host=host, chain_id=chain_id, key=private_key)
        else:
            try:
                self.client = ClobClient(host=host, chain_id=chain_id)
            except TypeError:
                self.client = None

    def get_balance(self, currency: str = "USDC") -> float:
        _ = currency
        return 0.0

    def get_all_positions(self) -> dict[str, float]:
        return {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[tuple]:
        _ = (symbol, timeframe, limit)
        return []

    def execute_order(
        self,
        symbol: str,
        type: str,
        side: str,
        quantity: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        allow_real = bool(getattr(self.config, "POLYMARKET_ALLOW_REAL_EXECUTION", False))
        mode = str(getattr(self.config, "MODE", "paper") or "paper").strip().lower()
        if not allow_real or mode != "real":
            return {
                "id": f"poly-paper-{symbol}-{side}-{quantity}",
                "status": "paper",
                "filled_qty": 0.0,
                "average_price": price,
                "symbol": symbol,
                "type": type,
                "side": side,
                "quantity": quantity,
                "params": dict(params or {}),
            }
        raise RuntimeError(
            "Polymarket real execution is not enabled in Phase 1. Use paper/shadow mode or extend Phase 2 execution support."
        )

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        _ = symbol
        return []

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        _ = (order_id, symbol)
        return True


__all__ = ["PolymarketExchange"]
