"""Helpers for split-aware candidate evaluation and stage selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchStageSelector:
    """Evaluate candidates and rank the stage-2 shortlist."""

    evaluate_candidate: Callable[..., dict[str, Any]]
    stage1_prefilter_score: Callable[..., float]

    def evaluate_candidate_with_optional_split(
        self,
        candidate: dict[str, Any],
        *,
        cache: Mapping[tuple[str, str], Any],
        feature_cache: Mapping[str, Any] | None,
        benchmark_cache: Mapping[str, Mapping[str, Any] | Any],
        candidate_count: int,
        scoring_config: Mapping[str, Any] | None,
        split: Mapping[str, Any],
    ) -> dict[str, Any]:
        evaluate_kwargs = {
            "cache": cache,
            "feature_cache": feature_cache,
            "benchmark_cache": benchmark_cache,
            "candidate_count": max(1, candidate_count),
            "scoring_config": scoring_config,
        }
        try:
            return self.evaluate_candidate(
                candidate,
                **evaluate_kwargs,
                split=split,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return self.evaluate_candidate(candidate, **evaluate_kwargs)

    def select_stage2_results(
        self,
        *,
        adapted: Sequence[dict[str, Any]],
        cache: Mapping[tuple[str, str], Any],
        feature_cache: Mapping[str, Any] | None,
        benchmark: Mapping[str, Mapping[str, Any] | Any],
        scoring: Any,
        resolved_split: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        scored_stage1: list[tuple[float, dict[str, Any]]] = []
        for row in adapted:
            result = self.evaluate_candidate_with_optional_split(
                row,
                cache=cache,
                feature_cache=feature_cache,
                benchmark_cache=benchmark,
                candidate_count=len(adapted),
                scoring_config=scoring.resolved_scoring_config,
                split=resolved_split,
            )
            score = self.stage1_prefilter_score(
                result,
                stage1_weights=scoring.stage1_weights,
                stage1_error_score=scoring.stage1_error_score,
            )
            scored_stage1.append((float(score), result))

        ranked = sorted(scored_stage1, key=lambda item: item[0], reverse=True)
        keep_count = max(1, round(len(ranked) * scoring.keep_ratio_applied))
        return [item[1] for item in ranked[:keep_count]]
