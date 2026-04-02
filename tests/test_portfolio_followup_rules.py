from __future__ import annotations

from lumina_quant.portfolio_followup_rules import (
    build_basis_search_universes,
    build_memory_ledger_row,
    evaluate_robustness_gates,
    mean_monthly_return,
    parse_memory_ledger_row,
    serialize_memory_ledger_row,
    validate_basis_universe,
)


def _payload(*, train_total_return: float, val_total_return: float, oos_total_return: float, train_sharpe: float, oos_sharpe: float, oos_max_drawdown: float, monthly: list[float]) -> dict[str, object]:
    return {
        "train": {"total_return": train_total_return, "sharpe": train_sharpe},
        "val": {"total_return": val_total_return, "sharpe": 1.0},
        "oos": {
            "total_return": oos_total_return,
            "sharpe": oos_sharpe,
            "sortino": 2.0,
            "calmar": 4.0,
            "max_drawdown": oos_max_drawdown,
            "volatility": 0.12,
        },
        "oos_monthly_returns": [
            {"month": f"2026-0{idx}", "total_return": value, "days": 20}
            for idx, value in enumerate(monthly, start=2)
        ],
    }


def test_validate_basis_universe_rejects_raw_and_derived_same_family() -> None:
    rows = [
        {"candidate_id": "incumbent", "metadata": {"basis_family": "incumbent", "basis_variant": "incumbent", "basis_universe": "shared"}},
        {"candidate_id": "raw", "metadata": {"basis_family": "static_blend", "basis_variant": "raw_55_45", "basis_universe": "raw_basis"}},
        {"candidate_id": "derived", "metadata": {"basis_family": "static_blend", "basis_variant": "derived_80_20", "basis_universe": "derived_basis"}},
    ]

    validation = validate_basis_universe(rows)

    assert validation["ok"] is False
    assert validation["conflicts"][0]["family"] == "static_blend"



def test_build_basis_search_universes_splits_raw_and_derived_rows() -> None:
    rows = [
        {"candidate_id": "incumbent", "metadata": {"basis_family": "incumbent", "basis_variant": "incumbent", "basis_universe": "shared"}},
        {"candidate_id": "soft", "metadata": {"basis_family": "soft_allocator", "basis_variant": "soft_allocator", "basis_universe": "shared"}},
        {"candidate_id": "raw", "metadata": {"basis_family": "static_blend", "basis_variant": "raw_55_45", "basis_universe": "raw_basis"}},
        {"candidate_id": "derived", "metadata": {"basis_family": "static_blend", "basis_variant": "derived_80_20", "basis_universe": "derived_basis"}},
    ]

    universes = build_basis_search_universes(rows)

    assert [row["candidate_id"] for row in universes["raw_basis"]] == ["incumbent", "soft", "raw"]
    assert [row["candidate_id"] for row in universes["derived_basis"]] == ["incumbent", "soft", "derived"]



def test_evaluate_robustness_gates_accepts_candidate_that_clears_all_rules() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.03,
        val_total_return=0.04,
        oos_total_return=0.07,
        train_sharpe=0.6,
        oos_sharpe=2.1,
        oos_max_drawdown=0.05,
        monthly=[0.025, 0.028, 0.031],
    )

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is True
    assert result["rejection_reasons"] == []
    assert result["oos_monthly_mean"] == mean_monthly_return(candidate["oos_monthly_returns"])



def test_evaluate_robustness_gates_rejects_negative_train_return() -> None:
    incumbent = _payload(
        train_total_return=0.02,
        val_total_return=0.03,
        oos_total_return=0.05,
        train_sharpe=0.4,
        oos_sharpe=1.5,
        oos_max_drawdown=0.07,
        monthly=[0.02, 0.02, 0.02],
    )
    candidate = _payload(
        train_total_return=0.0,
        val_total_return=0.04,
        oos_total_return=0.07,
        train_sharpe=0.6,
        oos_sharpe=2.1,
        oos_max_drawdown=0.05,
        monthly=[0.025, 0.028, 0.031],
    )

    result = evaluate_robustness_gates(candidate, incumbent)

    assert result["promotable"] is False
    assert "train_total_return_non_positive" in result["rejection_reasons"]



def test_memory_ledger_row_round_trips() -> None:
    row = build_memory_ledger_row(
        run_name="portfolio_meta_search",
        basis_universe="raw_basis",
        candidate_count=5,
        combination_count=10626,
        budget_bytes=8 * 1024 * 1024 * 1024,
        heavy_lock_path="/tmp/heavy.lock",
        session_memory_lease_path="/tmp/session.lock",
        status="completed",
        started_at="2026-04-02T10:00:00Z",
        completed_at="2026-04-02T10:05:00Z",
        memory_summary_path="/tmp/summary.json",
    )

    parsed = parse_memory_ledger_row(serialize_memory_ledger_row(row))

    assert parsed["basis_universe"] == "raw_basis"
    assert parsed["one_heavy_lane_only"] is True
    assert parsed["combination_count"] == 10626
