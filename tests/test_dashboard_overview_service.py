from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from lumina_quant.dashboard import overview_service


def _contract() -> SimpleNamespace:
    return SimpleNamespace(launch_mode="next", python_backend="streamlit")


def test_empty_overview_payload_tracks_reason() -> None:
    payload = overview_service.empty_overview_payload(contract=_contract(), reason="missing_dsn")

    assert payload["source"]["status"] == "missing_dsn"
    assert payload["recent_runs"] == []
    assert payload["workflow_jobs"] == []
    assert payload["equity_curve"] == []


def test_build_overview_payload_from_frames_exposes_recent_runs_and_curves() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "run-2",
                "mode": "backtest",
                "status": "COMPLETED",
                "metadata": {"strategy": "RsiStrategy"},
                "strategy": "RsiStrategy",
                "started_at": "2026-03-02T00:00:00Z",
            },
            {
                "run_id": "run-1",
                "mode": "live",
                "status": "RUNNING",
                "metadata": {"strategy": "Momentum"},
                "strategy": "Momentum",
                "started_at": "2026-03-01T00:00:00Z",
            },
        ]
    )
    equity = pd.DataFrame(
        [
            {"datetime": "2026-03-02T00:00:00Z", "total": 1000.0},
            {"datetime": "2026-03-03T00:00:00Z", "total": 1100.0},
            {"datetime": "2026-03-04T00:00:00Z", "total": 1050.0},
        ]
    )

    payload = overview_service.build_overview_payload_from_frames(
        contract=_contract(),
        runs_frame=runs,
        equity_frame=equity,
    )

    assert payload["source"]["run_id"] == "run-2"
    assert payload["recent_runs"][0]["run_id"] == "run-2"
    assert payload["recent_runs"][1]["status"] == "RUNNING"
    assert payload["summary_metrics"][4]["value"] == 1000.0
    assert payload["summary_metrics"][5]["value"] == 1050.0
    assert payload["performance_metrics"]["sharpe_ratio"] != 0.0
    assert payload["performance_metrics"]["max_drawdown"] > 0.0
    assert payload["equity_curve"][-1]["equity"] == 1050.0
    assert payload["drawdown_curve"][-1]["drawdown"] < 0.0


def test_load_overview_payload_short_circuits_for_blank_dsn() -> None:
    payload = overview_service.load_overview_payload(contract=_contract(), dsn="")

    assert payload["source"]["status"] == "missing_dsn"
    assert payload["performance_metrics"] == {}
