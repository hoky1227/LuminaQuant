"""Shared dashboard migration compatibility contract helpers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.config import BaseConfig
from lumina_quant.postgres_state import _connect_postgres

DEFAULT_DASHBOARD_COMPAT_PATH = "/api/python/dashboard/overview"


class DashboardCompatibilityError(RuntimeError):
    """Raised when dashboard migration compatibility options are invalid."""


@dataclass(slots=True, frozen=True)
class DashboardSliceContract:
    """JSON contract for the first dashboard slice moved behind the compatibility bridge."""

    slice_id: str
    title: str
    transport: str
    producer: str
    path: str
    payload_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DashboardBridgeContract:
    """Normalized launch + compatibility bridge contract for dashboard migration."""

    launch_mode: str
    python_backend: str
    frontend_target: str
    streamlit_app_path: str
    compatibility_path: str
    slice_contract: DashboardSliceContract

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_version"] = 1
        return payload


def resolve_dashboard_postgres_dsn(dsn: str | None = None) -> str:
    return str(dsn or os.getenv("LQ_POSTGRES_DSN") or getattr(BaseConfig, "POSTGRES_DSN", "") or "").strip()


def normalize_dashboard_launch_mode(value: str | None, default: str = "auto") -> str:
    token = str(value or default).strip().lower()
    if token in {"auto", "streamlit", "next"}:
        return token
    raise DashboardCompatibilityError(
        f"Unsupported dashboard launch mode '{value}'. Expected one of: auto, streamlit, next."
    )


def _normalize_compatibility_path(value: str | None) -> str:
    token = str(value or DEFAULT_DASHBOARD_COMPAT_PATH).strip() or DEFAULT_DASHBOARD_COMPAT_PATH
    normalized = token if token.startswith("/") else f"/{token}"
    if not normalized.startswith("/api/"):
        raise DashboardCompatibilityError(
            "Unsupported dashboard compatibility path "
            f"'{value}'. Expected a stable '/api/...' path."
        )
    return normalized


def resolve_dashboard_bridge_contract(
    *,
    launch_mode: str | None,
    streamlit_app_path: str | Path,
    next_app_dir: str | Path,
    compatibility_path: str | None = None,
) -> DashboardBridgeContract:
    """Resolve the launch mode and minimal bridge contract for the first migrated slice."""
    requested_mode = normalize_dashboard_launch_mode(launch_mode, default="auto")
    resolved_mode = "streamlit" if requested_mode == "auto" else requested_mode
    compat_path = _normalize_compatibility_path(compatibility_path)
    streamlit_target = str(Path(streamlit_app_path).resolve())
    next_target = str(Path(next_app_dir).resolve())

    frontend_target = streamlit_target if resolved_mode == "streamlit" else next_target
    python_backend = "streamlit"
    slice_contract = DashboardSliceContract(
        slice_id="overview",
        title="Dashboard overview compatibility slice",
        transport="json",
        producer="python",
        path=compat_path,
        payload_schema={
            "as_of": "iso8601-datetime",
            "summary_metrics": [
                {"key": "string", "label": "string", "value": "number|string|null"}
            ],
            "equity_curve": [{"timestamp": "iso8601-datetime", "equity": "number"}],
            "drawdown_curve": [{"timestamp": "iso8601-datetime", "drawdown": "number"}],
            "source": {"mode": resolved_mode, "backend": python_backend},
        },
    )
    return DashboardBridgeContract(
        launch_mode=resolved_mode,
        python_backend=python_backend,
        frontend_target=frontend_target,
        streamlit_app_path=streamlit_target,
        compatibility_path=compat_path,
        slice_contract=slice_contract,
    )


def _coerce_datetime_series(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def _empty_overview_payload(
    *,
    contract: DashboardBridgeContract,
    reason: str,
) -> dict[str, Any]:
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "summary_metrics": [],
        "equity_curve": [],
        "drawdown_curve": [],
        "source": {
            "mode": contract.launch_mode,
            "backend": contract.python_backend,
            "status": reason,
        },
    }


def _overview_metric(label: str, value: Any, *, key: str | None = None) -> dict[str, Any]:
    return {
        "key": key or label.lower().replace(" ", "_"),
        "label": label,
        "value": value,
    }


def build_overview_payload_from_frames(
    *,
    contract: DashboardBridgeContract,
    runs_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
) -> dict[str, Any]:
    if runs_frame.empty:
        return _empty_overview_payload(contract=contract, reason="no_runs")

    run = runs_frame.iloc[0]
    run_id = str(run.get("run_id") or "")
    strategy = ""
    metadata = run.get("metadata")
    if isinstance(metadata, dict):
        strategy = str(metadata.get("strategy") or "")
    strategy = strategy or str(run.get("strategy") or "")

    equity = _coerce_datetime_series(equity_frame, "datetime")
    if equity.empty:
        summary_metrics = [
            _overview_metric("Run ID", run_id, key="run_id"),
            _overview_metric("Mode", str(run.get("mode") or "")),
            _overview_metric("Status", str(run.get("status") or "")),
            _overview_metric("Strategy", strategy or "unknown"),
            _overview_metric("Equity Points", 0, key="equity_points"),
        ]
        payload = _empty_overview_payload(contract=contract, reason="no_equity")
        payload["summary_metrics"] = summary_metrics
        payload["source"]["run_id"] = run_id
        return payload

    totals = pd.to_numeric(equity["total"], errors="coerce").fillna(0.0)
    timestamps = equity["datetime"]
    initial_equity = float(totals.iloc[0]) if not totals.empty else 0.0
    latest_equity = float(totals.iloc[-1]) if not totals.empty else 0.0
    total_return = 0.0
    if initial_equity:
        total_return = (latest_equity - initial_equity) / initial_equity

    running_peak = totals.cummax().replace(0.0, pd.NA)
    drawdown = ((totals - running_peak) / running_peak).fillna(0.0)

    summary_metrics = [
        _overview_metric("Run ID", run_id, key="run_id"),
        _overview_metric("Mode", str(run.get("mode") or "")),
        _overview_metric("Status", str(run.get("status") or "")),
        _overview_metric("Strategy", strategy or "unknown"),
        _overview_metric("Initial Equity", round(initial_equity, 4), key="initial_equity"),
        _overview_metric("Latest Equity", round(latest_equity, 4), key="latest_equity"),
        _overview_metric("Total Return", round(total_return, 6), key="total_return"),
        _overview_metric("Equity Points", len(equity), key="equity_points"),
    ]
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "summary_metrics": summary_metrics,
        "equity_curve": [
            {
                "timestamp": value.isoformat(),
                "equity": float(total),
            }
            for value, total in zip(timestamps.tolist(), totals.tolist(), strict=False)
            if value is not pd.NaT
        ],
        "drawdown_curve": [
            {
                "timestamp": value.isoformat(),
                "drawdown": float(level),
            }
            for value, level in zip(timestamps.tolist(), drawdown.tolist(), strict=False)
            if value is not pd.NaT
        ],
        "source": {
            "mode": contract.launch_mode,
            "backend": contract.python_backend,
            "status": "ok",
            "run_id": run_id,
        },
    }


def load_overview_payload(
    *,
    launch_mode: str | None = "next",
    dsn: str | None = None,
    limit: int = 120,
    compatibility_path: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    contract = resolve_dashboard_bridge_contract(
        launch_mode=launch_mode,
        streamlit_app_path=repo_root / "apps" / "dashboard" / "app.py",
        next_app_dir=repo_root / "apps" / "dashboard_web",
        compatibility_path=compatibility_path,
    )
    if dsn is not None and not str(dsn).strip():
        return _empty_overview_payload(contract=contract, reason="missing_dsn")
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return _empty_overview_payload(contract=contract, reason="missing_dsn")

    conn = _connect_postgres(resolved_dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    run_id,
                    mode,
                    started_at,
                    ended_at,
                    status,
                    metadata_json AS metadata,
                    COALESCE(
                        (metadata_json ->> 'strategy'),
                        ''
                    ) AS strategy
                FROM runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            run_rows = cursor.fetchall()
            run_columns = [description[0] for description in cursor.description or ()]
        runs = pd.DataFrame(run_rows, columns=run_columns)
        if runs.empty:
            return _empty_overview_payload(contract=contract, reason="no_runs")
        run_id = str(runs.iloc[0]["run_id"])
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT timeindex AS datetime, total
                FROM (
                    SELECT id, timeindex, total
                    FROM equity
                    WHERE run_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) recent
                ORDER BY id ASC
                """,
                (run_id, int(max(10, limit))),
            )
            equity_rows = cursor.fetchall()
            equity_columns = [description[0] for description in cursor.description or ()]
        equity = pd.DataFrame(equity_rows, columns=equity_columns)
        return build_overview_payload_from_frames(
            contract=contract,
            runs_frame=runs,
            equity_frame=equity,
        )
    finally:
        conn.close()


__all__ = [
    "DEFAULT_DASHBOARD_COMPAT_PATH",
    "DashboardBridgeContract",
    "DashboardCompatibilityError",
    "DashboardSliceContract",
    "build_overview_payload_from_frames",
    "load_overview_payload",
    "normalize_dashboard_launch_mode",
    "resolve_dashboard_bridge_contract",
    "resolve_dashboard_postgres_dsn",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the dashboard migration bridge contract.")
    parser.add_argument("--json", action="store_true", help="Print the contract as JSON.")
    parser.add_argument("--mode", default="auto", help="Dashboard launch mode.")
    parser.add_argument(
        "--streamlit-app-path",
        default=str(Path(__file__).resolve().parents[3] / "apps" / "dashboard" / "app.py"),
    )
    parser.add_argument(
        "--next-app-dir",
        default=str(Path(__file__).resolve().parents[3] / "apps" / "dashboard_web"),
    )
    parser.add_argument("--compat-path", default=DEFAULT_DASHBOARD_COMPAT_PATH)
    parser.add_argument("--overview-json", action="store_true", help="Print overview payload JSON.")
    args = parser.parse_args(argv)

    if args.overview_json:
        print(
            json.dumps(
                load_overview_payload(
                    launch_mode=args.mode,
                    compatibility_path=args.compat_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    contract = resolve_dashboard_bridge_contract(
        launch_mode=args.mode,
        streamlit_app_path=args.streamlit_app_path,
        next_app_dir=args.next_app_dir,
        compatibility_path=args.compat_path,
    )
    if args.json:
        print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    else:
        print(contract.compatibility_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
