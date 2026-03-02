"""Live WS handler alias backed by committed materialized windows."""

from __future__ import annotations

from lumina_quant.live.data_materialized import CommittedWindowDataHandler


class BinanceWebSocketDataHandler(CommittedWindowDataHandler):
    """WS entrypoint compatibility wrapper using committed materialized reader."""


__all__ = ["BinanceWebSocketDataHandler"]
