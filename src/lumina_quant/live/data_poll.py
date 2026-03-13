"""Live poll data handler with feature-flagged data-source routing."""

from __future__ import annotations


def _resolve_market_data_source(config) -> str:
    token = str(getattr(config, "MARKET_DATA_SOURCE", "committed") or "committed")
    token = token.strip().lower().replace("-", "_")
    if token in {"binance_live", "binance", "live"}:
        return "binance_live"
    if token in {"external", "custom"}:
        return "external"
    if token in {"polymarket_live", "polymarket"}:
        return "polymarket_live"
    return "committed"


def _binance_live_handler_cls():
    from lumina_quant.live.data_binance_live import BinanceLiveDataHandler

    return BinanceLiveDataHandler


def _external_handler_cls():
    from lumina_quant.live.data_external import ExternalWindowDataHandler

    return ExternalWindowDataHandler


def _polymarket_live_handler_cls():
    from lumina_quant.live.data_polymarket_live import PolymarketLiveDataHandler

    return PolymarketLiveDataHandler


def _committed_handler_cls():
    from lumina_quant.live.data_materialized import CommittedWindowDataHandler

    return CommittedWindowDataHandler


class LiveDataHandler:
    """Factory wrapper preserving legacy constructor signature."""

    def __new__(cls, events, symbol_list, config, exchange=None):
        source = _resolve_market_data_source(config)
        if source == "binance_live":
            return _binance_live_handler_cls()(
                events,
                symbol_list,
                config,
                exchange,
                transport="poll",
            )
        if source == "external":
            return _external_handler_cls()(events, symbol_list, config, exchange)
        if source == "polymarket_live":
            return _polymarket_live_handler_cls()(
                events,
                symbol_list,
                config,
                exchange,
                transport="poll",
            )
        return _committed_handler_cls()(events, symbol_list, config, exchange)


__all__ = ["LiveDataHandler"]
