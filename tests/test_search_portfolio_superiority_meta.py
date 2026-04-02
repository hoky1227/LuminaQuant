from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "search_portfolio_superiority_meta.py"
SPEC = importlib.util.spec_from_file_location("search_portfolio_superiority_meta", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load search_portfolio_superiority_meta module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _stream(values: list[float], *, start_day: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "datetime": f"2026-02-{start_day + idx:02d}T00:00:00Z",
            "t": 1_700_000_000_000 + (idx * 86_400_000),
            "v": value,
        }
        for idx, value in enumerate(values)
    ]


def _candidate(
    candidate_key: str,
    *,
    lineage: str | None = None,
    train: list[float] | None = None,
    val: list[float] | None = None,
    oos: list[float] | None = None,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "portfolio_candidate_artifact",
        "portfolio_return_streams": {
            "train": _stream(train or [0.01, 0.01, 0.01], start_day=1),
            "val": _stream(val or [0.01, 0.01, 0.01], start_day=10),
            "oos": _stream(oos or [0.03, 0.02], start_day=20),
        },
    }
    out = {"candidate_key": candidate_key, "label": candidate_key, "payload": payload}
    if lineage is not None:
        out["lineage"] = lineage
    return out


class DummyGuard:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rss_log_path = self.output_dir / "dummy_rss_latest.jsonl"
        self.rss_log_path.write_text("", encoding="utf-8")
        self.summary_path = self.output_dir / "dummy_memory_latest.json"

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        self.rss_log_path.write_text(
            self.rss_log_path.read_text(encoding="utf-8")
            + json.dumps({"event": event, "context": context or {}})
            + "\n",
            encoding="utf-8",
        )

    def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        self.checkpoint(event, context)
        return {"event": event, "context": context or {}}

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "error": error,
            "context": context or {},
            "summary_path": str(self.summary_path.resolve()),
            "memory_policy": {"heavy_lock_path": "/tmp/test.lock", "explicit_budget_bytes": 123},
        }
        self.summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def release(self) -> None:
        return None


def test_build_basis_search_universes_separates_raw_and_derived() -> None:
    universes = MODULE.build_basis_search_universes(
        incumbent=_candidate("incumbent"),
        raw_55_45=_candidate("autoresearch_pair_55_45"),
        derived_80_20=_candidate("incumbent_autoresearch_static_blend"),
        soft_allocator=_candidate("soft_allocator"),
        regime_switch=_candidate("regime_switching_portfolio"),
        grouped_base=_candidate("grouped_base"),
    )

    raw_names = {row["candidate_key"] for row in universes[MODULE.RAW_UNIVERSE_NAME]}
    derived_names = {row["candidate_key"] for row in universes[MODULE.DERIVED_UNIVERSE_NAME]}
    assert "autoresearch_pair_55_45" in raw_names
    assert "incumbent_autoresearch_static_blend" not in raw_names
    assert "incumbent_autoresearch_static_blend" in derived_names
    assert "autoresearch_pair_55_45" not in derived_names


def test_ensure_basis_dedupe_rejects_raw_and_derived_mix() -> None:
    with pytest.raises(MODULE.BasisUniverseError, match="cannot coexist"):
        MODULE.ensure_basis_dedupe(
            [
                _candidate("incumbent"),
                _candidate("autoresearch_pair_55_45"),
                _candidate("incumbent_autoresearch_static_blend"),
            ]
        )


def test_robustness_gate_failures_cover_thresholds() -> None:
    failures = MODULE.robustness_gate_failures(
        candidate_metrics={
            "train": {"total_return": -0.01, "sharpe": -0.2},
            "val": {"total_return": 0.0},
            "oos": {"total_return": 0.01, "sharpe": 1.4, "max_drawdown": 0.11},
        },
        incumbent_oos={"total_return": 0.02, "sharpe": 1.0, "max_drawdown": 0.05},
        monthly_returns=[{"month": "2026-02", "total_return": 0.01, "days": 20}],
    )

    assert failures == [
        "train_total_return<=0",
        "val_total_return<=0",
        "train_sharpe<-0.10",
        "oos_total_return_delta<=0",
        "oos_monthly_mean<0.02",
        "oos_drawdown_worse_without_sharpe_relief",
    ]


def test_robustness_gate_allows_sharpe_relief_with_worse_drawdown() -> None:
    failures = MODULE.robustness_gate_failures(
        candidate_metrics={
            "train": {"total_return": 0.05, "sharpe": 0.2},
            "val": {"total_return": 0.04},
            "oos": {"total_return": 0.08, "sharpe": 1.8, "max_drawdown": 0.08},
        },
        incumbent_oos={"total_return": 0.02, "sharpe": 1.2, "max_drawdown": 0.05},
        monthly_returns=[{"month": "2026-02", "total_return": 0.03, "days": 20}],
    )

    assert failures == []


def test_run_meta_search_writes_artifacts_and_memory_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "acquire_portfolio_memory_guard",
        lambda **kwargs: DummyGuard(Path(kwargs["output_dir"])),
    )

    candidates = MODULE.build_basis_search_universes(
        incumbent=_candidate("incumbent", oos=[0.01, 0.01]),
        raw_55_45=_candidate("autoresearch_pair_55_45", oos=[0.04, 0.03]),
        derived_80_20=_candidate("incumbent_autoresearch_static_blend", oos=[0.03, 0.03]),
        soft_allocator=_candidate("soft_allocator", train=[-0.01, -0.01], val=[-0.01, -0.01], oos=[0.0, 0.0]),
        regime_switch=_candidate("regime_switching_portfolio", oos=[0.015, 0.01]),
        grouped_base=_candidate("grouped_base", oos=[0.012, 0.011]),
    )[MODULE.RAW_UNIVERSE_NAME]

    result = MODULE.run_meta_search(
        universe_name=MODULE.RAW_UNIVERSE_NAME,
        candidates=candidates,
        incumbent_key="incumbent",
        output_dir=tmp_path,
        weight_step=0.5,
        top_k=10,
    )

    assert Path(result["leaderboard_json_path"]).exists()
    assert Path(result["rejections_json_path"]).exists()
    assert Path(result["memory_ledger_path"]).exists()
    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    ledger = json.loads(Path(result["memory_ledger_path"]).read_text(encoding="utf-8"))
    assert summary["universe_name"] == MODULE.RAW_UNIVERSE_NAME
    assert summary["winner_status"] == "promoted_challenger"
    assert summary["winner"]["promotable"] is True
    assert ledger["combination_count"] == summary["combination_count"]
    assert ledger["candidate_count"] == len(candidates)
    assert any(row["lineage"] == MODULE.RAW_55_45_LINEAGE for row in summary["candidate_universe"])
