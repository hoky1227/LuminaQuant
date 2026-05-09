from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "tune_profit_moonshot_fresh_portfolio.py"
SPEC = importlib.util.spec_from_file_location("tune_profit_moonshot_fresh_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load tune_profit_moonshot_fresh_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_pool_can_reserve_family_balanced_slots() -> None:
    rows = [
        {
            "name": "calendar_a",
            "family": "calendar_rotation",
            "train_total_return": "0.02",
            "val_total_return": "0.05",
            "val_sharpe": "2.0",
            "val_max_drawdown": "0.001",
            "val_round_trips": "5",
        },
        {
            "name": "calendar_b",
            "family": "calendar_rotation",
            "train_total_return": "0.02",
            "val_total_return": "0.04",
            "val_sharpe": "2.0",
            "val_max_drawdown": "0.001",
            "val_round_trips": "5",
        },
        {
            "name": "new_family_a",
            "family": "adaptive_trend_fade",
            "train_total_return": "0.01",
            "val_total_return": "0.005",
            "val_sharpe": "1.0",
            "val_max_drawdown": "0.001",
            "val_round_trips": "5",
        },
    ]

    unbalanced = MODULE._candidate_pool(rows, top_n=2)
    balanced = MODULE._candidate_pool(rows, top_n=3, family_quota=1)

    assert [row["name"] for row in unbalanced] == ["calendar_a", "calendar_b"]
    assert {row["family"] for row in balanced[:2]} == {"calendar_rotation", "adaptive_trend_fade"}


class _FakeMemoryGuard:
    def __init__(self, output_dir: Path) -> None:
        memory_dir = output_dir / "_memory_guard"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self.rss_log_path = memory_dir / "profit_moonshot_fresh_portfolio_tuning_rss_latest.jsonl"
        self.summary_path = memory_dir / "profit_moonshot_fresh_portfolio_tuning_memory_latest.json"
        self.checkpoints: list[tuple[str, dict[str, Any] | None]] = []
        self.finalize_calls: list[dict[str, Any]] = []
        self.released = False

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        self.checkpoints.append((event, context))

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "artifact_kind": "portfolio_followup_memory_summary",
            "run_name": MODULE.RUN_NAME,
            "status": status,
            "error": error,
            "context": dict(context or {}),
            "rss_log_path": str(self.rss_log_path),
            "summary_path": str(self.summary_path),
            "peak_rss_bytes": 123456,
            "memory_policy": MODULE.memory_policy_payload(
                budget_bytes=MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
            ),
        }
        self.finalize_calls.append(payload)
        self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def release(self) -> None:
        self.released = True


def _minimal_payload(tmp_path: Path) -> dict[str, Any]:
    split_payload = {
        "metrics": {
            "total_return": 0.01,
            "max_drawdown": 0.001,
            "sharpe": 1.2,
            "sortino": 1.5,
            "volatility": 0.02,
        },
        "round_trips": 7,
        "fills": 14,
        "final_equity": 10_100.0,
    }
    selected = {
        "name": "fresh_portfolio_equal_weight_stub_a__stub_b",
        "mode": "equal_weight",
        "sleeves": ["stub_a", "stub_b"],
        "sleeve_count": 2,
        "splits": {"train": split_payload, "val": split_payload, "oos": split_payload},
        "gates": {
            "train_positive": True,
            "val_positive": True,
            "oos_return_beats_incumbent": True,
            "oos_mdd_beats_shadow": True,
            "oos_sharpe_gt_1": True,
            "oos_trades_not_starved": True,
        },
        "success_candidate": True,
        "validation_score": 1.0,
    }
    return {
        "artifact_kind": "profit_moonshot_fresh_portfolio_tuning",
        "generated_at_utc": "2026-05-07T00:00:00Z",
        "candidate_csv": str(tmp_path / "candidates.csv"),
        "candidate_sleeve_count": 2,
        "portfolio_spec_count": 1,
        "success_candidate_count": 1,
        "best_success_candidate": selected,
        "selected_by_validation": selected,
        "diagnostic_best_oos": selected,
        "data_metadata": {"source": "unit"},
        "peak_rss_mib": 1.0,
    }


def test_tuning_main_wraps_artifacts_with_memory_guard_and_lockbox_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "fresh_overhaul"
    candidate_csv = tmp_path / "candidates.csv"
    candidate_csv.write_text("name,train_total_return,val_total_return,val_round_trips\n", encoding="utf-8")
    captured: dict[str, Any] = {}
    guards: list[_FakeMemoryGuard] = []

    def _fake_acquire(**kwargs: Any) -> _FakeMemoryGuard:
        captured.update(kwargs)
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(
        MODULE,
        "build_payload",
        lambda args: (
            _minimal_payload(tmp_path),
            [
                {
                    "name": "fresh_portfolio_equal_weight_stub_a__stub_b",
                    "mode": "equal_weight",
                    "sleeve_count": 2,
                    "sleeves": "stub_a,stub_b",
                    "validation_score": 1.0,
                    "success_candidate": True,
                    "failed_gates": "",
                    "train_total_return": 0.01,
                    "train_max_drawdown": 0.001,
                    "train_sharpe": 1.2,
                    "train_sortino": 1.5,
                    "train_volatility": 0.02,
                    "train_round_trips": 7,
                    "val_total_return": 0.01,
                    "val_max_drawdown": 0.001,
                    "val_sharpe": 1.2,
                    "val_sortino": 1.5,
                    "val_volatility": 0.02,
                    "val_round_trips": 7,
                    "oos_total_return": 0.01,
                    "oos_max_drawdown": 0.001,
                    "oos_sharpe": 1.2,
                    "oos_sortino": 1.5,
                    "oos_volatility": 0.02,
                    "oos_round_trips": 7,
                }
            ],
        ),
    )

    assert (
        MODULE.main(
            [
                "--candidate-csv",
                str(candidate_csv),
                "--output-dir",
                str(output_dir),
                "--top-n",
                "2",
                "--max-sleeves",
                "2",
            ]
        )
        == 0
    )

    expected_budget = MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    assert captured["run_name"] == MODULE.RUN_NAME
    assert captured["input_path"] == str(candidate_csv)
    assert captured["budget_bytes"] == expected_budget
    assert captured["metadata"]["locked_oos_label"] == "locked_oos_report_only"
    assert guards[0].checkpoints[0][0] == "start"
    assert guards[0].finalize_calls[0]["status"] == "completed"
    assert guards[0].released is True

    payload = json.loads((output_dir / "fresh_portfolio_tuning_latest.json").read_text(encoding="utf-8"))
    assert payload["lockbox_policy"]["selection_label"] == "train_val_validation_only"
    assert payload["lockbox_policy"]["locked_oos_label"] == "locked_oos_report_only"
    assert payload["memory_policy"]["explicit_budget_bytes"] == expected_budget
    assert payload["memory_summary"]["memory_policy"]["explicit_budget_bytes"] == expected_budget
    assert payload["rss_log_path"].endswith("_rss_latest.jsonl")
    assert payload["memory_summary_path"].endswith("_memory_latest.json")

    rendered = (output_dir / "fresh_portfolio_tuning_latest.md").read_text(encoding="utf-8")
    assert "Locked-OOS label: `locked_oos_report_only`" in rendered
    assert "## Best success candidate" in rendered
    assert "locked OOS" in rendered


def test_tuning_main_finalizes_failed_guard_when_payload_build_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "fresh_overhaul"
    candidate_csv = tmp_path / "candidates.csv"
    candidate_csv.write_text("name\n", encoding="utf-8")
    guards: list[_FakeMemoryGuard] = []

    def _fake_acquire(**kwargs: Any) -> _FakeMemoryGuard:
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    def _boom(args: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raise RuntimeError("candidate build failed")

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(MODULE, "build_payload", _boom)

    with pytest.raises(RuntimeError, match="candidate build failed"):
        MODULE.main(
            [
                "--candidate-csv",
                str(candidate_csv),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert guards[0].finalize_calls[0]["status"] == "failed"
    assert guards[0].finalize_calls[0]["error"] == "candidate build failed"
    assert guards[0].released is True


def _calendar_row(
    name: str,
    *,
    train: float,
    val: float,
    val_mdd: float = 0.001,
    val_sharpe: float = 2.0,
    trips: int = 8,
    long_symbol: str = "TRXUSDT",
    short_symbol: str = "ETHUSDT",
    threshold: float = 0.018,
    hold_bars: int = 168,
    take_profit: float = 0.060,
) -> dict[str, str]:
    return {
        "name": name,
        "family": "calendar_rotation",
        "filters": json.dumps(
            {
                "family": "calendar_rotation",
                "calendar_long_symbol": long_symbol,
                "calendar_short_symbol": short_symbol,
                "threshold": threshold,
                "hold_bars": hold_bars,
                "take_profit_pct": take_profit,
            },
            sort_keys=True,
        ),
        "train_total_return": str(train),
        "val_total_return": str(val),
        "val_max_drawdown": str(val_mdd),
        "val_sharpe": str(val_sharpe),
        "val_round_trips": str(trips),
    }


def test_calendar_neighborhood_representatives_are_train_val_stability_first() -> None:
    stable_a = _calendar_row("stable_a", train=0.055, val=0.041, threshold=0.0180)
    stable_b = _calendar_row("stable_b", train=0.052, val=0.039, threshold=0.0184)
    fragile_spike = _calendar_row("fragile_spike", train=0.060, val=0.070, threshold=0.030)
    fragile_neighbor = _calendar_row("fragile_neighbor", train=0.001, val=0.001, threshold=0.0304)

    selected = MODULE._candidate_pool(
        [fragile_spike, fragile_neighbor, stable_a, stable_b],
        top_n=2,
        calendar_neighborhood_reps=1,
    )

    assert selected[0]["name"] == "stable_a"
    assert {row["name"] for row in selected} == {"stable_a", "fragile_spike"}


def test_cluster_capped_validation_weight_caps_correlated_calendar_cluster() -> None:
    rows = {
        "cal_a": _calendar_row("cal_a", train=0.06, val=0.05),
        "cal_b": _calendar_row("cal_b", train=0.055, val=0.048),
        "cal_c": _calendar_row(
            "cal_c",
            train=0.05,
            val=0.042,
            short_symbol="BTCUSDT",
            threshold=0.024,
            take_profit=0.030,
        ),
    }
    split_payloads = {
        name: {
            "train": {"metrics": {"total_return": 0.05, "max_drawdown": 0.002, "sharpe": 2.0}},
            "val": {"metrics": {"total_return": 0.05, "max_drawdown": 0.002, "sharpe": 2.0}},
        }
        for name in rows
    }
    split_curves = {
        "cal_a": {"train": [10000, 10050, 10100, 10150], "val": [10000, 10040, 10090, 10140]},
        "cal_b": {"train": [10000, 10049, 10099, 10149], "val": [10000, 10041, 10091, 10139]},
        "cal_c": {"train": [10000, 10020, 10010, 10080], "val": [10000, 9990, 10020, 10070]},
    }

    weights, diagnostics = MODULE._cluster_capped_validation_weights(
        combo_names=("cal_a", "cal_b", "cal_c"),
        split_curves=split_curves,
        split_payloads=split_payloads,
        candidate_rows_by_name=rows,
        cluster_cap=0.50,
        sleeve_cap=0.60,
        correlation_threshold=0.95,
    )

    clusters = diagnostics["clusters"]
    shared_cluster = clusters["cal_a"]
    assert clusters["cal_b"] == shared_cluster
    assert clusters["cal_c"] != shared_cluster
    assert sum(weights[:2]) <= 0.500001
    assert abs(sum(weights) - 1.0) < 1e-9
    assert diagnostics["selection_basis"] == "train_val_only_cluster_capped"


def test_train_val_target_return_budget_scales_without_oos_selection() -> None:
    class _Fresh:
        HOURLY_PERIODS_PER_YEAR = 365 * 24

        @staticmethod
        def _metrics_from_equity_totals(equity: list[float], *, periods: int) -> dict[str, float]:
            del periods
            return {"total_return": equity[-1] / 10_000.0 - 1.0}

    split_curves = {
        "sleeve_a": {
            "train": [10_000.0, 10_400.0],
            "val": [10_000.0, 10_350.0],
            "oos": [10_000.0, 10_200.0],
        },
        "sleeve_b": {
            "train": [10_000.0, 10_300.0],
            "val": [10_000.0, 10_250.0],
            "oos": [10_000.0, 10_180.0],
        },
    }

    weights, leverage, diagnostics = MODULE._mode_weights_and_leverage(
        fresh=_Fresh(),
        combo_names=("sleeve_a", "sleeve_b"),
        split_curves=split_curves,
        split_payloads={},
        mode="train_val_target_return_budget",
    )

    assert weights is None
    assert leverage == pytest.approx(MODULE.TARGET_BUDGET_TRAIN_RETURN / 0.07)
    assert diagnostics["selection_basis"] == "train_val_target_return_budget"
    assert diagnostics["uses_locked_oos_for_selection"] is False
    assert diagnostics["raw_train_return"] == pytest.approx(0.07)
    assert diagnostics["raw_val_return"] == pytest.approx(0.06)


def test_mdd_failed_diagnostic_is_quarantined_not_promoted() -> None:
    item = {
        "name": "diagnostic_high_oos_failed_mdd",
        "mode": "additive_sleeves",
        "gates": {
            "train_positive": True,
            "val_positive": True,
            "oos_return_beats_current_champion": True,
            "oos_mdd_beats_shadow": False,
        },
        "success_candidate": False,
        "splits": {
            "train": {"metrics": {"total_return": 0.20}},
            "val": {"metrics": {"total_return": 0.19}},
            "oos": {"metrics": {"total_return": 0.03971, "max_drawdown": 0.008977}},
        },
    }

    labeled = MODULE._with_promotion_labels(item)

    assert labeled["diagnostic_not_promoted"] is True
    assert labeled["promotion_status"] == "diagnostic_not_promoted"
    assert labeled["improved_candidate"] is False
    assert labeled["success_candidate"] is False
    assert "oos_mdd_beats_shadow" in labeled["failed_gates"]


def test_report_order_uses_validation_before_locked_oos_diagnostics() -> None:
    validation_leader = {
        "name": "validation_leader",
        "success_candidate": False,
        "promotion_status": "diagnostic_not_promoted",
        "validation_score": 10.0,
    }
    oos_spike = {
        "name": "oos_spike",
        "success_candidate": False,
        "promotion_status": "diagnostic_not_promoted",
        "validation_score": 4.0,
    }

    ordered = sorted([oos_spike, validation_leader], key=MODULE._portfolio_report_sort_key)

    assert [item["name"] for item in ordered] == ["validation_leader", "oos_spike"]


def test_success_requires_current_champion_oos_threshold_and_return_risk_gate() -> None:
    passing = {
        "gates": {
            "train_positive": True,
            "val_positive": True,
            "train_return_beats_current_champion": True,
            "val_return_beats_current_champion": True,
            "oos_return_beats_current_champion": True,
            "oos_return_risk_beats_current_champion": True,
            "oos_mdd_beats_shadow": True,
            "oos_sharpe_gt_1": True,
            "oos_trades_not_starved": True,
        }
    }
    assert MODULE._improved_candidate_from_gates(passing["gates"])
    assert MODULE.CURRENT_CHAMPION_OOS_RETURN == 0.012181

    failing = dict(passing["gates"], oos_return_beats_current_champion=False)
    assert not MODULE._improved_candidate_from_gates(failing)
