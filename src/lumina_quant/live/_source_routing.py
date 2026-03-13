"""Shared live market-data source routing helpers."""

from __future__ import annotations


def resolve_market_data_source(config) -> str:
    """Normalize configured live market-data source into a stable token."""
    token = str(getattr(config, "MARKET_DATA_SOURCE", "committed") or "committed")
    token = token.strip().lower().replace("-", "_")
    if token in {"binance_live", "binance", "live"}:
        return "binance_live"
    if token in {"external", "custom"}:
        return "external"
    if token in {"polymarket_live", "polymarket"}:
        return "polymarket_live"
    return "committed"


def binance_live_handler_cls():
    from lumina_quant.live.data_binance_live import BinanceLiveDataHandler

    return BinanceLiveDataHandler


def external_handler_cls():
    from lumina_quant.live.data_external import ExternalWindowDataHandler

    return ExternalWindowDataHandler


def polymarket_live_handler_cls():
    from lumina_quant.live.data_polymarket_live import PolymarketLiveDataHandler

    return PolymarketLiveDataHandler


def committed_handler_cls():
    from lumina_quant.live.data_materialized import CommittedWindowDataHandler

    return CommittedWindowDataHandler


def build_live_data_handler(*, transport: str, events, symbol_list, config, exchange=None):
    """Instantiate the appropriate live data handler for the configured source."""
    source = resolve_market_data_source(config)
    if source == "binance_live":
        return binance_live_handler_cls()(
            events,
            symbol_list,
            config,
            exchange,
            transport=transport,
        )
    if source == "external":
        return external_handler_cls()(events, symbol_list, config, exchange)
    if source == "polymarket_live":
        return polymarket_live_handler_cls()(
            events,
            symbol_list,
            config,
            exchange,
            transport=transport,
        )
    return committed_handler_cls()(events, symbol_list, config, exchange)
