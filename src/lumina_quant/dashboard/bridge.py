"""Shared dashboard migration compatibility contract helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


__all__ = [
    "DEFAULT_DASHBOARD_COMPAT_PATH",
    "DashboardBridgeContract",
    "DashboardCompatibilityError",
    "DashboardSliceContract",
    "normalize_dashboard_launch_mode",
    "resolve_dashboard_bridge_contract",
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
    args = parser.parse_args(argv)

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
