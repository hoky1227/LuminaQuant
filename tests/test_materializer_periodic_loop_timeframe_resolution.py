from __future__ import annotations

import importlib.util
from pathlib import Path

from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.services.materialize_from_raw import MaterializedCommit

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "materialize_market_windows.py"
_SPEC = importlib.util.spec_from_file_location("materialize_script_module", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load materialize script module from {_SCRIPT_PATH}")
materialize_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(materialize_script)


def test_materializer_periodic_loop_resolves_required_timeframes(monkeypatch):
    captured: dict[str, object] = {}

    def _bundle(**kwargs):
        captured.update(kwargs)
        return {
            "1s": [
                MaterializedCommit(
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1s",
                    partition="2026-01-01",
                    commit_id="c-1s",
                    row_count=1,
                    canonical_row_checksum="abc",
                    manifest_path="manifest-1s.json",
                )
            ],
            "5m": [
                MaterializedCommit(
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="5m",
                    partition="2026-01-01",
                    commit_id="c-5m",
                    row_count=1,
                    canonical_row_checksum="def",
                    manifest_path="manifest-5m.json",
                )
            ],
            "1h": [
                MaterializedCommit(
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1h",
                    partition="2026-01-01",
                    commit_id="c-1h",
                    row_count=1,
                    canonical_row_checksum="ghi",
                    manifest_path="manifest-1h.json",
                )
            ],
        }

    monkeypatch.setattr(materialize_script, "materialize_raw_aggtrades_bundle", _bundle)

    cycles = materialize_script.run_materializer_periodic_loop(
        db_path="data/market_parquet",
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=materialize_script._parse_timeframes("5m,1s,1h,5m"),
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
        periodic_enabled=False,
        poll_seconds=5,
    )

    assert len(cycles) == 1
    assert captured["timeframes"] == ["1s", "5m", "1h"]


def test_materializer_skips_commit_publication_when_bundle_incomplete(monkeypatch):
    def _bundle(**kwargs):
        _ = kwargs
        raise RawFirstDataMissingError("missing BTC/USDT:5m")

    monkeypatch.setattr(materialize_script, "materialize_raw_aggtrades_bundle", _bundle)

    payload = materialize_script.run_materializer_cycle(
        db_path="data/market_parquet",
        exchange="binance",
        symbols=["BTC/USDT"],
        required_timeframes=["1s", "5m"],
        base_timeframe="1s",
        start_date=None,
        end_date=None,
        producer="test",
    )

    assert payload["success"] is False
    symbol_rows = payload["symbols"]
    assert isinstance(symbol_rows, list) and len(symbol_rows) == 1
    assert symbol_rows[0]["status"] == "skipped_incomplete_required_timeframes"
    assert "missing BTC/USDT:5m" in str(symbol_rows[0]["error"])
