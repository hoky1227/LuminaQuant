"""Candidate-research entrypoints extracted from the monolithic runner module.

This module intentionally keeps the public entrypoint contract stable while
moving the orchestration-facing surface out of ``research_runner.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import Any


def _runner_module():
    """Resolve the legacy runner lazily to reduce import-time coupling."""
    return import_module("lumina_quant.strategy_factory.research_runner")


def _report_candidates_from_stage2_results(
    *,
    stage2_results: Sequence[dict[str, Any]],
    candidate_count: int,
    resolved_split: Mapping[str, Any],
    scoring: Any,
) -> list[dict[str, Any]]:
    runner = _runner_module()
    return runner._research_report_builder().report_candidates_from_stage2_results(
        stage2_results=stage2_results,
        candidate_count=candidate_count,
        resolved_split=resolved_split,
        scoring=scoring,
    )


def _attach_cross_candidate_correlations(report_candidates: Sequence[dict[str, Any]]) -> None:
    runner = _runner_module()
    runner._research_report_builder().attach_cross_candidate_correlations(report_candidates)


def _sorted_report_candidates(
    report_candidates: Sequence[dict[str, Any]],
    *,
    scoring: Any,
) -> list[dict[str, Any]]:
    rows = list(report_candidates)
    runner = _runner_module()
    runner._attach_cross_candidate_correlations(rows)
    rows.sort(
        key=lambda item: float(item.get("selection_score", scoring.sort_missing_selection_score)),
        reverse=True,
    )
    return rows


def _candidate_research_report_payload(
    *,
    base_tf: str,
    normalized_timeframes: Sequence[str],
    universe: Sequence[str],
    resolved_split: Mapping[str, Any],
    adapted: Sequence[dict[str, Any]],
    stage2_results: Sequence[dict[str, Any]],
    stage1_keep_ratio: float,
    scoring: Any,
    data_sources: dict[str, list[str]],
    report_candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    runner = _runner_module()
    return {
        "schema_version": "v2",
        "generated_at": runner.datetime.now(runner.UTC).isoformat(),
        "base_timeframe": base_tf,
        "strategy_timeframes": normalized_timeframes,
        "symbol_universe": universe,
        "split": resolved_split,
        "candidates": report_candidates,
        "stage1": {
            "input_count": len(adapted),
            "selected_count": len(stage2_results),
            "keep_ratio": float(stage1_keep_ratio),
            "keep_ratio_applied": float(scoring.keep_ratio_applied),
        },
        "scoring_config": scoring.resolved_scoring_config,
        "data_sources": data_sources,
    }


def _run_candidate_research_with_adapted_candidates(
    *,
    base_tf: str,
    adapted: Sequence[dict[str, Any]],
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
    stage1_keep_ratio: float,
    scoring: Any,
    split: Mapping[str, Any] | None,
    data_mode: str,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bundle_bars: int,
    market_data_settings: Mapping[str, Any],
) -> dict[str, Any]:
    runner = _runner_module()
    normalized_timeframes, universe = runner._resolve_research_run_timeframes_and_universe(
        adapted=adapted,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
    )
    split_timeframe = normalized_timeframes[0] if normalized_timeframes else "1m"
    resolved_split = runner._resolve_split_config(split, strategy_timeframe=split_timeframe)

    cache, data_sources, feature_cache, benchmark = runner._load_research_run_resources(
        adapted=adapted,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        data_mode=str(data_mode or "legacy"),
        allow_csv_fallback=bool(allow_csv_fallback),
        allow_synthetic_fallback=bool(allow_synthetic_fallback),
        min_bundle_bars=max(1, int(min_bundle_bars)),
        market_data_settings=market_data_settings,
    )
    stage2_results = runner._select_stage2_results(
        adapted=adapted,
        cache=cache,
        feature_cache=feature_cache,
        benchmark=benchmark,
        scoring=scoring,
        resolved_split=resolved_split,
    )
    report_candidates = runner._report_candidates_from_stage2_results(
        stage2_results=stage2_results,
        candidate_count=len(adapted),
        resolved_split=resolved_split,
        scoring=scoring,
    )
    return runner._candidate_research_report_payload(
        base_tf=base_tf,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        adapted=adapted,
        stage2_results=stage2_results,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        data_sources=data_sources,
        report_candidates=runner._sorted_report_candidates(report_candidates, scoring=scoring),
    )


def run_candidate_research(
    *,
    candidates: Iterable[dict[str, Any]],
    base_timeframe: str = "1s",
    strategy_timeframes: Sequence[str] | None = None,
    symbol_universe: Sequence[str] | None = None,
    stage1_keep_ratio: float = 0.35,
    max_candidates: int = 512,
    score_config: Mapping[str, Any] | None = None,
    split: Mapping[str, Any] | None = None,
    data_mode: str = "legacy",
    allow_csv_fallback: bool = True,
    allow_synthetic_fallback: bool = True,
    min_bundle_bars: int = 360,
) -> dict[str, Any]:
    """Evaluate candidate manifest into train/val/OOS report contract (v2)."""
    runner = _runner_module()
    base_tf = runner._normalize_candidate_research_base_timeframe(base_timeframe)
    market_data_settings = runner._current_research_market_data_settings()
    scoring = runner._resolve_research_run_scoring_config(
        score_config=score_config,
        stage1_keep_ratio=stage1_keep_ratio,
    )
    adapted = runner._adapt_candidate_inputs(candidates, max_candidates=max_candidates)

    if not adapted:
        return runner._empty_candidate_research_report(
            base_timeframe=base_tf,
            strategy_timeframes=strategy_timeframes,
            symbol_universe=symbol_universe,
            stage1_keep_ratio=stage1_keep_ratio,
            scoring=scoring,
            split=split,
        )

    return _run_candidate_research_with_adapted_candidates(
        base_tf=base_tf,
        adapted=adapted,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        split=split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
        market_data_settings=market_data_settings,
    )


def build_default_candidate_rows(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    max_candidates: int = 512,
) -> list[dict[str, Any]]:
    """Build candidate rows from strategy-factory candidate library."""
    from lumina_quant.strategy_factory.candidate_library import build_binance_futures_candidates
    runner = _runner_module()

    rows = build_binance_futures_candidates(
        symbols=symbols or runner._current_research_market_data_settings()["symbols"],
        timeframes=timeframes or runner.CANONICAL_STRATEGY_TIMEFRAMES,
    )
    out = [runner.adapt_legacy_candidate(item.to_dict()) for item in rows]
    if int(max_candidates) > 0:
        out = out[: int(max_candidates)]
    return out


__all__ = [
    "build_default_candidate_rows",
    "run_candidate_research",
]
