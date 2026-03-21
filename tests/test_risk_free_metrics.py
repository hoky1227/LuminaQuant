from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from lumina_quant.config import BacktestConfig
from lumina_quant.strategy_factory import research_runner as rr
from lumina_quant.utils.risk_free import annual_to_periodic_rate, resolve_risk_free_config

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_saved_incumbent_portfolio.py"
SPEC = importlib.util.spec_from_file_location("validate_saved_incumbent_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_annual_to_periodic_rate_matches_compound_formula() -> None:
    annual = 0.05
    periods = 365
    expected = (1.0 + annual) ** (1.0 / periods) - 1.0
    assert math.isclose(float(annual_to_periodic_rate(annual, periods)), expected, rel_tol=1e-12)


def test_resolve_risk_free_config_uses_us_treasury_constant_defaults(monkeypatch) -> None:
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_MODE", "us_treasury_constant", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_TENOR", "3m", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_ANNUAL", 0.0475, raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_MODE", "same_as_rf", raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_ANNUAL", 0.0, raising=False)

    resolved = resolve_risk_free_config(BacktestConfig, periods_per_year=365)

    assert resolved.mode == "us_treasury_constant"
    assert resolved.tenor == "3m"
    assert math.isclose(resolved.annual_rate, 0.0475, rel_tol=1e-12)
    assert math.isclose(resolved.sortino_target_annual, 0.0475, rel_tol=1e-12)


def test_validator_and_research_runner_metrics_share_sharpe_sortino_semantics(monkeypatch) -> None:
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_MODE", "us_treasury_constant", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_TENOR", "3m", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_ANNUAL", 0.03, raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_MODE", "same_as_rf", raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_ANNUAL", 0.0, raising=False)

    returns = np.asarray([0.01, -0.005, 0.004, 0.006, -0.002, 0.003], dtype=float)
    timestamps = np.asarray(
        [
            np.datetime64("2026-01-01"),
            np.datetime64("2026-01-02"),
            np.datetime64("2026-01-03"),
            np.datetime64("2026-01-04"),
            np.datetime64("2026-01-05"),
            np.datetime64("2026-01-06"),
        ],
        dtype="datetime64[ms]",
    )

    validator_metrics = MODULE._metrics(returns, periods_per_year=365, timestamps=timestamps)
    research_metrics = rr._compute_metrics(
        returns,
        turnover=np.zeros_like(returns),
        exposure=np.zeros_like(returns),
        benchmark_returns=np.zeros_like(returns),
        periods_per_year=365,
        num_trials=1,
        metric_config=BacktestConfig,
        timestamps=timestamps,
    )

    assert math.isclose(validator_metrics["sharpe"], research_metrics["sharpe"], rel_tol=1e-12)
    assert math.isclose(validator_metrics["sortino"], research_metrics["sortino"], rel_tol=1e-12)
