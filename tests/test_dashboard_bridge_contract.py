from __future__ import annotations

from pathlib import Path

import pytest

from lumina_quant.dashboard.bridge import (
    DEFAULT_DASHBOARD_COMPAT_PATH,
    DashboardCompatibilityError,
    normalize_dashboard_launch_mode,
    resolve_dashboard_bridge_contract,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "auto"), ("auto", "auto"), ("streamlit", "streamlit"), ("next", "next")],
)
def test_normalize_dashboard_launch_mode_accepts_supported_values(value: str | None, expected: str) -> None:
    assert normalize_dashboard_launch_mode(value) == expected


def test_normalize_dashboard_launch_mode_rejects_unsupported_value() -> None:
    with pytest.raises(DashboardCompatibilityError):
        normalize_dashboard_launch_mode("legacy")


def test_resolve_dashboard_bridge_contract_defaults_to_streamlit() -> None:
    contract = resolve_dashboard_bridge_contract(
        launch_mode="auto",
        streamlit_app_path=ROOT / "apps" / "dashboard" / "app.py",
        next_app_dir=ROOT / "apps" / "dashboard_web",
    )

    assert contract.launch_mode == "streamlit"
    assert contract.python_backend == "streamlit"
    assert contract.frontend_target.endswith("apps/dashboard/app.py")
    assert contract.compatibility_path == DEFAULT_DASHBOARD_COMPAT_PATH
    assert contract.slice_contract.path == DEFAULT_DASHBOARD_COMPAT_PATH


def test_resolve_dashboard_bridge_contract_supports_next_mode_with_api_path() -> None:
    contract = resolve_dashboard_bridge_contract(
        launch_mode="next",
        streamlit_app_path=ROOT / "apps" / "dashboard" / "app.py",
        next_app_dir=ROOT / "apps" / "dashboard_web",
        compatibility_path="api/python/dashboard/overview",
    )
    payload = contract.to_dict()

    assert contract.launch_mode == "next"
    assert contract.frontend_target.endswith("apps/dashboard_web")
    assert contract.compatibility_path == DEFAULT_DASHBOARD_COMPAT_PATH
    assert payload["slice_contract"]["payload_schema"]["source"] == {
        "mode": "next",
        "backend": "streamlit",
    }


def test_resolve_dashboard_bridge_contract_rejects_non_api_compatibility_path() -> None:
    with pytest.raises(DashboardCompatibilityError):
        resolve_dashboard_bridge_contract(
            launch_mode="streamlit",
            streamlit_app_path=ROOT / "apps" / "dashboard" / "app.py",
            next_app_dir=ROOT / "apps" / "dashboard_web",
            compatibility_path="/dashboard/overview",
        )
