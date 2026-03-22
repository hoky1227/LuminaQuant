"""Helpers for loading candidate-research resources with optional compatibility fallbacks."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ResearchResourceLoader:
    """Load bundle, feature, and benchmark resources for candidate research."""

    split_window_bounds: Callable[[Mapping[str, Any]], tuple[Any, Any]]
    datetime_to_iso_z: Callable[[Any], str | None]
    load_bundle_cache: Callable[..., tuple[dict[tuple[str, str], Any], dict[str, list[str]]]]
    load_feature_cache: Callable[..., dict[str, pl.DataFrame]]
    benchmark_cache: Callable[
        [Mapping[tuple[str, str], Any], Sequence[str]],
        dict[str, dict[str, np.ndarray] | np.ndarray],
    ]
    canonicalize_symbol_list: Callable[[Any], list[str]]

    def load(
        self,
        *,
        adapted: Sequence[dict[str, Any]],
        normalized_timeframes: Sequence[str],
        universe: Sequence[str],
        resolved_split: Mapping[str, Any],
        data_mode: str,
        allow_csv_fallback: bool,
        allow_synthetic_fallback: bool,
        min_bundle_bars: int,
        market_data_settings: Mapping[str, Any] | None = None,
        feature_support_strategy_classes: Sequence[str] = (
            "PerpCrowdingCarryStrategy",
            "CompositeTrendStrategy",
        ),
    ) -> tuple[
        dict[tuple[str, str], Any],
        dict[str, list[str]],
        dict[str, pl.DataFrame],
        dict[str, dict[str, np.ndarray] | np.ndarray],
    ]:
        load_start, load_end = self.split_window_bounds(resolved_split)
        load_bundle_kwargs = {
            "symbols": universe,
            "timeframes": normalized_timeframes,
        }
        try:
            cache, data_sources = self.load_bundle_cache(
                **load_bundle_kwargs,
                start_date=self.datetime_to_iso_z(load_start),
                end_date=self.datetime_to_iso_z(load_end),
                data_mode=str(data_mode or "legacy"),
                allow_csv_fallback=bool(allow_csv_fallback),
                allow_synthetic_fallback=bool(allow_synthetic_fallback),
                min_bars=max(1, int(min_bundle_bars)),
                market_data_settings=market_data_settings,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            cache, data_sources = self.load_bundle_cache(**load_bundle_kwargs)

        support_feature_symbols = self.canonicalize_symbol_list(
            itertools.chain.from_iterable(
                list(row.get("symbols") or [])
                for row in adapted
                if str(row.get("strategy_class") or row.get("strategy") or "")
                in set(feature_support_strategy_classes)
            )
        )
        try:
            feature_cache = self.load_feature_cache(
                symbols=support_feature_symbols,
                start_date=self.datetime_to_iso_z(load_start),
                end_date=self.datetime_to_iso_z(load_end),
                market_data_settings=market_data_settings,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            feature_cache = self.load_feature_cache(
                symbols=support_feature_symbols,
                start_date=self.datetime_to_iso_z(load_start),
                end_date=self.datetime_to_iso_z(load_end),
            )
        benchmark = self.benchmark_cache(cache, normalized_timeframes)
        return cache, data_sources, feature_cache, benchmark
