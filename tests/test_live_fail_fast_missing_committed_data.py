from __future__ import annotations

import importlib.util
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.live.data_poll import LiveDataHandler

_RUN_LIVE_PATH = Path(__file__).resolve().parents[1] / "run_live.py"
_RUN_LIVE_SPEC = importlib.util.spec_from_file_location("run_live_module", _RUN_LIVE_PATH)
if _RUN_LIVE_SPEC is None or _RUN_LIVE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_RUN_LIVE_PATH}")
run_live = importlib.util.module_from_spec(_RUN_LIVE_SPEC)
_RUN_LIVE_SPEC.loader.exec_module(run_live)

_RUN_LIVE_WS_PATH = Path(__file__).resolve().parents[1] / "run_live_ws.py"
_RUN_LIVE_WS_SPEC = importlib.util.spec_from_file_location("run_live_ws_module", _RUN_LIVE_WS_PATH)
if _RUN_LIVE_WS_SPEC is None or _RUN_LIVE_WS_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_RUN_LIVE_WS_PATH}")
run_live_ws = importlib.util.module_from_spec(_RUN_LIVE_WS_SPEC)
_RUN_LIVE_WS_SPEC.loader.exec_module(run_live_ws)


class _Config:
    MARKET_DATA_PARQUET_PATH = "data/market_parquet"
    MARKET_DATA_EXCHANGE = "binance"
    LIVE_POLL_SECONDS = 1
    INGEST_WINDOW_SECONDS = 5
    MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
    MARKET_WINDOW_PARITY_V2_ENABLED = False
    MARKET_WINDOW_METRICS_LOG_PATH = "logs/live/market_window_metrics.ndjson"


class _ThreadStub:
    def __init__(self, target, daemon=True):
        self.target = target
        self.daemon = daemon

    def start(self):
        return None

    def is_alive(self):
        return False

    def join(self, timeout=None):
        _ = timeout
        return None


def test_data_handler_fail_fast_propagates_missing_committed_data(monkeypatch):
    class _ReaderStub:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        @staticmethod
        def read_snapshot():
            raise RawFirstDataMissingError("committed manifest missing for BTC/USDT:1s")

    events = queue.Queue()
    monkeypatch.setattr("lumina_quant.live.data_materialized.threading.Thread", _ThreadStub)
    monkeypatch.setattr("lumina_quant.live.data_materialized.MaterializedWindowReader", _ReaderStub)
    handler = LiveDataHandler(events, ["BTC/USDT"], _Config, exchange=SimpleNamespace())
    handler._poll_market_data()

    fatal = handler.consume_fatal_error()
    assert isinstance(fatal, RawFirstDataMissingError)
    assert events.qsize() == 0


def _patch_entrypoint_env(monkeypatch, module, *, strategy_name: str):
    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "ccxt", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            return None

    class _Strategy:
        __name__ = strategy_name

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def _ordered_shutdown():
            return None

        @staticmethod
        def _close_audit_store(status=None):
            _ = status
            return None

        @staticmethod
        def run():
            raise RawFirstDataMissingError("fatal committed data breach")

    monkeypatch.setattr(module, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(module, "STRATEGY_MAP", {strategy_name: _Strategy})
    monkeypatch.setattr(module, "resolve_strategy_class", lambda *_args, **_kwargs: _Strategy)
    monkeypatch.setattr(module, "LiveTrader", _Trader)


def test_run_live_exits_with_code_2_on_fail_fast(monkeypatch):
    _patch_entrypoint_env(monkeypatch, run_live, strategy_name="MovingAverageCrossStrategy")
    monkeypatch.setattr(sys, "argv", ["run_live.py", "--no-selection"])
    with pytest.raises(SystemExit) as exc:
        run_live.main()
    assert int(exc.value.code) == 2


def test_run_live_ws_exits_with_code_2_on_fail_fast(monkeypatch):
    _patch_entrypoint_env(monkeypatch, run_live_ws, strategy_name="RsiStrategy")
    monkeypatch.setattr(sys, "argv", ["run_live_ws.py", "--no-selection"])
    with pytest.raises(SystemExit) as exc:
        run_live_ws.main()
    assert int(exc.value.code) == 2
