from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError

_RUN_BACKTEST_PATH = Path(__file__).resolve().parents[1] / "run_backtest.py"
sys.path.insert(0, str(_RUN_BACKTEST_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_backtest_contract_module", _RUN_BACKTEST_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load run_backtest module from {_RUN_BACKTEST_PATH}")
run_backtest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_backtest)


def test_raw_first_requires_parquet_store(monkeypatch):
    monkeypatch.setattr(run_backtest, "is_parquet_market_data_store", lambda *_args, **_kwargs: False)

    with pytest.raises(RawFirstDataMissingError):
        run_backtest._load_data_dict(
            "db",
            "data/market_parquet",
            "binance",
            base_timeframe="1s",
            data_mode="raw-first",
            backtest_mode="windowed",
            auto_collect_db=False,
        )


def test_raw_first_loader_passes_data_mode_to_owner_entrypoint(monkeypatch):
    called: dict[str, object] = {}
    frame = pl.DataFrame(
        {
            "datetime": [1_700_000_000_000],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).with_columns(pl.from_epoch("datetime", time_unit="ms").alias("datetime"))

    monkeypatch.setattr(run_backtest, "is_parquet_market_data_store", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_backtest, "SYMBOL_LIST", ["BTC/USDT"])

    def _loader(*args, **kwargs):
        _ = args
        called.update(kwargs)
        return {run_backtest.SYMBOL_LIST[0]: frame}

    monkeypatch.setattr(run_backtest, "load_data_dict_from_parquet", _loader)
    loaded = run_backtest._load_data_dict(
        "db",
        "data/market_parquet",
        "binance",
        base_timeframe="1s",
        data_mode="raw-first",
        backtest_mode="windowed",
        auto_collect_db=False,
    )

    assert run_backtest.SYMBOL_LIST[0] in loaded
    assert called.get("data_mode") == "raw-first"


def test_raw_first_rejects_non_windowed_backtest_mode():
    with pytest.raises(RawFirstDataMissingError):
        run_backtest.resolve_data_contract(
            data_mode="raw-first",
            backtest_mode="legacy_batch",
            data_source="db",
            default_backtest_mode="windowed",
            default_data_source="auto",
        )
