from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.cli import live as live_cli
from lumina_quant.live.data_poll import LiveDataHandler


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
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
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
    monkeypatch.setattr(
        module,
        "build_live_runtime_contract",
        lambda **_kwargs: SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport="poll",
        ),
    )


def test_run_live_exits_with_code_2_on_fail_fast(monkeypatch):
    _patch_entrypoint_env(monkeypatch, live_cli, strategy_name="MovingAverageCrossStrategy")
    with pytest.raises(SystemExit) as exc:
        live_cli.main(["--no-selection"])
    assert int(exc.value.code) == 2


def test_run_live_ws_exits_with_code_2_on_fail_fast(monkeypatch):
    _patch_entrypoint_env(monkeypatch, live_cli, strategy_name="RsiStrategy")
    with pytest.raises(SystemExit) as exc:
        live_cli.main(["--transport", "ws", "--no-selection"])
    assert int(exc.value.code) == 2


def test_committed_market_data_forces_poll_transport_even_when_ws_requested(monkeypatch, capsys):
    captured: dict[str, object] = {}

    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            return None

    class _Strategy:
        __name__ = "RsiStrategy"

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def run():
            return None

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "RsiStrategy",
            lambda include_opt_in=True: {"RsiStrategy": _Strategy},
            lambda *_args, **_kwargs: _Strategy,
        ),
    )

    def _capture_contract(*, transport="poll"):
        captured["transport"] = transport
        return SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport=transport,
        )

    monkeypatch.setattr(live_cli, "build_live_runtime_contract", _capture_contract)

    assert live_cli.main(["--transport", "ws", "--no-selection"]) == 0
    assert captured["transport"] == "poll"
    captured_output = capsys.readouterr().out
    assert "Requested Transport: ws" in captured_output
    assert "Effective Transport: poll" in captured_output


def test_selection_overrides_are_applied_before_live_config_validation(monkeypatch):
    observed: dict[str, object] = {}
    validate_calls: list[tuple[list[str], str]] = []

    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            validate_calls.append((list(cls.SYMBOLS), str(cls.TIMEFRAME)))
            return None

    class _Strategy:
        __name__ = "MovingAverageCrossStrategy"

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args
            observed["kwargs"] = kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def run():
            return None

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "MovingAverageCrossStrategy",
            lambda include_opt_in=True: {"MovingAverageCrossStrategy": _Strategy},
            lambda *_args, **_kwargs: _Strategy,
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport="poll",
        ),
    )
    monkeypatch.setattr(live_cli, "resolve_selection_file", lambda _path="": "fake-selection.json")
    monkeypatch.setattr(live_cli, "load_selection_payload", lambda _path: {"ok": True})
    monkeypatch.setattr(
        live_cli,
        "extract_selection_config",
        lambda _payload: {
            "candidate_name": "MovingAverageCrossStrategy",
            "symbols": ["ETH/USDT", "SOL/USDT"],
            "strategy_timeframe": "5m",
            "params": {"fast": 3},
        },
    )

    assert live_cli.main([]) == 0
    assert validate_calls == [(["ETH/USDT", "SOL/USDT"], "5m")]
    assert observed["kwargs"]["symbol_list"] == ["ETH/USDT", "SOL/USDT"]
    assert observed["kwargs"]["strategy_params"] == {"fast": 3}


def test_strategy_helper_resolver_accepts_default_name_keyword():
    _, _, resolver = live_cli._strategy_helpers()

    strategy_cls = resolver(
        "MovingAverageCrossStrategy",
        default_name="MovingAverageCrossStrategy",
    )

    assert strategy_cls.__name__ == "MovingAverageCrossStrategy"
