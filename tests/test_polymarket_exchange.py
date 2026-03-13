from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumina_quant.exchanges import get_exchange
from lumina_quant.exchanges.polymarket_exchange import PolymarketExchange


def test_get_exchange_supports_polymarket_driver():
    cfg = SimpleNamespace(
        EXCHANGE={"driver": "polymarket", "name": "polymarket"},
        MODE="paper",
        POLYMARKET_ALLOW_REAL_EXECUTION=False,
    )
    exchange = get_exchange(cfg)
    assert isinstance(exchange, PolymarketExchange)


def test_polymarket_exchange_paper_execute_order_returns_stub():
    cfg = SimpleNamespace(
        EXCHANGE={"driver": "polymarket", "name": "polymarket"},
        MODE="paper",
        POLYMARKET_ALLOW_REAL_EXECUTION=False,
    )
    exchange = PolymarketExchange(cfg)
    result = exchange.execute_order("asset-1", "limit", "buy", 5.0, price=0.55)
    assert result["status"] == "paper"


def test_polymarket_exchange_real_execution_requires_phase2_support():
    cfg = SimpleNamespace(
        EXCHANGE={"driver": "polymarket", "name": "polymarket"},
        MODE="real",
        POLYMARKET_ALLOW_REAL_EXECUTION=True,
    )
    exchange = PolymarketExchange(cfg)
    with pytest.raises(RuntimeError):
        exchange.execute_order("asset-1", "limit", "buy", 5.0, price=0.55)
