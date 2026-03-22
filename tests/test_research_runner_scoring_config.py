from __future__ import annotations

import numpy as np
from lumina_quant.strategy_factory import research_runner


def test_run_candidate_research_scoring_config_defaults_and_override(monkeypatch):
    candidates = [
        {
            "candidate_id": "cand-a",
            "name": "candidate-a",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1m",
            "timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
        {
            "candidate_id": "cand-b",
            "name": "candidate-b",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1m",
            "timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
    ]

    metric_by_id = {
        "cand-a": {"sharpe": 1.8, "return": 0.02, "deflated_sharpe": 0.5},
        "cand-b": {"sharpe": 0.2, "return": 0.08, "deflated_sharpe": 0.5},
    }

    def _mock_load_bundle_cache(*, symbols, timeframes):
        _ = symbols, timeframes
        return {}, {"parquet": [], "csv": [], "synthetic": []}

    def _mock_benchmark_cache(cache, timeframes):
        _ = cache, timeframes
        return {"1m": np.zeros(24, dtype=float)}

    def _mock_evaluate_candidate(
        candidate,
        *,
        cache,
        feature_cache,
        benchmark_cache,
        candidate_count,
        scoring_config=None,
    ):
        _ = cache, feature_cache, benchmark_cache, candidate_count, scoring_config
        cid = str(candidate.get("candidate_id"))
        base = metric_by_id[cid]
        metrics = {
            "return": float(base["return"]),
            "total_return": float(base["return"]),
            "cagr": float(base["return"]),
            "sharpe": float(base["sharpe"]),
            "sortino": float(base["sharpe"]),
            "calmar": float(base["return"]),
            "mdd": 0.1,
            "max_drawdown": 0.1,
            "turnover": 1.0,
            "trades": 20.0,
            "trade_count": 20.0,
            "win_rate": 0.5,
            "avg_trade": 0.01,
            "exposure": 0.5,
            "volatility": 0.2,
            "stability": 0.1,
            "rolling_sharpe_min": 0.0,
            "worst_month": -0.05,
            "benchmark_corr": 0.0,
            "deflated_sharpe": float(base["deflated_sharpe"]),
            "pbo": 0.1,
            "spa_pvalue": 0.2,
        }
        return {
            "candidate": candidate,
            "returns": np.linspace(0.001, 0.002, 24),
            "turnover": np.full(24, 1.0),
            "exposure": np.full(24, 0.2),
            "train": dict(metrics),
            "val": dict(metrics),
            "oos": dict(metrics),
            "oos_cost_stress": {"x2": {"sharpe": 0.5, "return": 0.01}, "x3": {"sharpe": 0.2, "return": 0.005}},
            "hurdle_fields": {"train": {"pass": True, "score": 1.0}, "val": {"pass": True, "score": 1.0}, "oos": {"pass": True, "score": 1.0}},
            "pass": True,
            "hard_reject_reasons": {},
            "metadata": {"cost_rate": 0.0005},
        }

    monkeypatch.setattr(research_runner, "_load_bundle_cache", _mock_load_bundle_cache)
    monkeypatch.setattr(research_runner, "_benchmark_cache", _mock_benchmark_cache)
    monkeypatch.setattr(research_runner, "_evaluate_candidate", _mock_evaluate_candidate)

    report_default = research_runner.run_candidate_research(
        candidates=candidates,
        base_timeframe="1s",
        strategy_timeframes=["1m"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
    )
    rows_default = list(report_default.get("candidates") or [])
    assert rows_default
    assert rows_default[0]["candidate_id"] == "cand-a"
    assert report_default["scoring_config"]["candidate_rank_score_weights"]["sharpe_weight"] == 2.8

    report_override = research_runner.run_candidate_research(
        candidates=candidates,
        base_timeframe="1s",
        strategy_timeframes=["1m"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
        score_config={
            "candidate_rank_score_weights": {
                "sharpe_weight": 0.1,
                "return_weight": 80.0,
            }
        },
    )
    rows_override = list(report_override.get("candidates") or [])
    assert rows_override
    assert rows_override[0]["candidate_id"] == "cand-b"


def test_run_candidate_research_sorts_candidates_and_preserves_stage_metadata(monkeypatch):
    candidates = [
        {
            "candidate_id": "cand-low",
            "name": "candidate-low",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1m",
            "timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
        {
            "candidate_id": "cand-high",
            "name": "candidate-high",
            "strategy_class": "CompositeTrendStrategy",
            "strategy_timeframe": "1m",
            "timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "params": {},
        },
    ]
    resolved_split = {"mode": "default"}
    data_sources = {"parquet": ["BTC/USDT@1m"], "csv": [], "synthetic": []}
    stage2_results = [{"candidate_id": "cand-low"}, {"candidate_id": "cand-high"}]
    report_candidates = [
        {"candidate_id": "cand-low", "selection_score": 0.1},
        {"candidate_id": "cand-high", "selection_score": 0.9},
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(research_runner, "_adapt_candidate_inputs", lambda items, max_candidates: list(items))
    monkeypatch.setattr(
        research_runner,
        "_resolve_research_run_timeframes_and_universe",
        lambda **kwargs: (["1m"], ["BTC/USDT"]),
    )
    monkeypatch.setattr(research_runner, "_resolve_split_config", lambda split, strategy_timeframe: resolved_split)
    monkeypatch.setattr(
        research_runner,
        "_load_research_run_resources",
        lambda **kwargs: ({}, data_sources, {}, {}),
    )
    monkeypatch.setattr(research_runner, "_select_stage2_results", lambda **kwargs: stage2_results)
    monkeypatch.setattr(
        research_runner,
        "_report_candidates_from_stage2_results",
        lambda **kwargs: list(report_candidates),
    )
    monkeypatch.setattr(
        research_runner,
        "_attach_cross_candidate_correlations",
        lambda rows: captured.setdefault("candidate_ids", [row["candidate_id"] for row in rows]),
    )

    report = research_runner.run_candidate_research(
        candidates=candidates,
        base_timeframe="1s",
        strategy_timeframes=["1m"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=0.5,
        max_candidates=8,
    )

    assert [row["candidate_id"] for row in report["candidates"]] == ["cand-high", "cand-low"]
    assert captured["candidate_ids"] == ["cand-low", "cand-high"]
    assert report["data_sources"] == data_sources
    assert report["split"] == resolved_split
    assert report["stage1"] == {
        "input_count": 2,
        "selected_count": 2,
        "keep_ratio": 0.5,
        "keep_ratio_applied": 0.5,
    }


def test_candidate_rank_score_penalizes_validation_to_oos_instability():
    stable = {
        "val": {"sharpe": 1.1, "return": 0.021, "turnover": 1.0},
        "oos": {
            "sharpe": 1.0,
            "return": 0.02,
            "turnover": 1.0,
            "deflated_sharpe": 0.4,
            "pbo": 0.1,
            "mdd": 0.08,
        },
    }
    unstable = {
        "val": {"sharpe": 4.0, "return": 0.09, "turnover": 0.4},
        "oos": dict(stable["oos"]),
    }

    stable_score = research_runner._candidate_rank_score(stable)
    unstable_score = research_runner._candidate_rank_score(unstable)

    assert stable_score > unstable_score


def test_evaluate_candidate_uses_signal_and_metric_payload_helpers(monkeypatch):
    candidate = {
        "candidate_id": "cand-a",
        "strategy_class": "CompositeTrendStrategy",
        "symbols": ["BTC/USDT"],
        "timeframe": "1m",
    }
    signal_payload = research_runner._CandidateSignalPayload(
        symbols=["BTC/USDT"],
        timeframe="1m",
        timestamps=np.asarray(["2026-01-01T00:00:00.000"], dtype="datetime64[ms]"),
        returns_raw=np.asarray([0.01], dtype=float),
        returns=np.asarray([0.0095], dtype=float),
        turnover=np.asarray([0.5], dtype=float),
        exposure=np.asarray([1.0], dtype=float),
        meta={"source": "signal"},
        cost_rate=0.001,
    )
    metric_payload = research_runner._CandidateMetricPayload(
        train_metrics={"sharpe": 1.0, "return": 0.02},
        val_metrics={"sharpe": 0.8, "return": 0.01},
        oos_metrics={"sharpe": 0.7, "return": 0.009},
        oos_stress_x2={"sharpe": 0.5, "return": 0.006},
        oos_stress_x3={"sharpe": 0.2, "return": 0.003},
    )

    monkeypatch.setattr(research_runner, "_load_candidate_signal_payload", lambda *args, **kwargs: signal_payload)
    monkeypatch.setattr(research_runner, "_evaluate_candidate_metric_payload", lambda *args, **kwargs: metric_payload)
    monkeypatch.setattr(
        research_runner,
        "_hurdle_fields",
        lambda *args, **kwargs: (
            {"train": {"pass": True}, "val": {"pass": True}, "oos": {"pass": True}},
            True,
            {},
        ),
    )
    monkeypatch.setattr(research_runner, "_apply_cost_stress_hard_rejects", lambda **kwargs: {})

    result = research_runner._evaluate_candidate(
        candidate,
        cache={},
        feature_cache={},
        benchmark_cache={},
        candidate_count=7,
    )

    assert result["candidate"] is candidate
    assert result["train"] == metric_payload.train_metrics
    assert result["oos"] == metric_payload.oos_metrics
    assert result["oos_cost_stress"]["x2"]["sharpe"] == 0.5
    assert result["metadata"]["strategy_family"] == "trend"
    assert result["metadata"]["cost_rate"] == 0.001
    assert result["metadata"]["aligned_bars"] == 1
    assert result["metadata"]["source"] == "signal"
    assert result["pass"] is True


def test_evaluate_candidate_returns_insufficient_result_when_signal_payload_missing(monkeypatch):
    candidate = {
        "candidate_id": "cand-a",
        "strategy_class": "CompositeTrendStrategy",
        "symbols": ["BTC/USDT"],
        "timeframe": "1m",
    }
    captured: dict[str, object] = {}
    cache = {("BTC/USDT", "1m"): object()}

    monkeypatch.setattr(research_runner, "_load_candidate_signal_payload", lambda *args, **kwargs: None)

    def _capture_insufficient(candidate_arg, *, symbols, timeframe, cache):
        captured["candidate"] = candidate_arg
        captured["symbols"] = symbols
        captured["timeframe"] = timeframe
        captured["cache"] = cache
        return {"candidate": candidate_arg, "error": "insufficient"}

    monkeypatch.setattr(research_runner, "_insufficient_candidate_result", _capture_insufficient)

    result = research_runner._evaluate_candidate(
        candidate,
        cache=cache,
        feature_cache={},
        benchmark_cache={},
        candidate_count=1,
    )

    assert result == {"candidate": candidate, "error": "insufficient"}
    assert captured == {
        "candidate": candidate,
        "symbols": ["BTC/USDT"],
        "timeframe": "1m",
        "cache": cache,
    }


def test_evaluate_candidate_propagates_cost_stress_hard_rejects_to_stage_passes(monkeypatch):
    candidate = {"strategy_class": "CompositeTrendStrategy", "params": {}}
    signal_payload = research_runner._CandidateSignalPayload(
        symbols=["BTC/USDT"],
        timeframe="1m",
        timestamps=np.asarray(["2024-01-01T00:00:00"], dtype="datetime64[ms]"),
        returns_raw=np.asarray([0.02], dtype=float),
        returns=np.asarray([0.019], dtype=float),
        turnover=np.asarray([1.0], dtype=float),
        exposure=np.asarray([1.0], dtype=float),
        meta={},
        cost_rate=0.001,
    )
    metric_payload = research_runner._CandidateMetricPayload(
        train_metrics={"sharpe": 1.0, "return": 0.02},
        val_metrics={"sharpe": 0.8, "return": 0.01},
        oos_metrics={"sharpe": 0.7, "return": 0.009},
        oos_stress_x2={"sharpe": 0.5, "return": 0.006},
        oos_stress_x3={"sharpe": 0.2, "return": 0.003},
    )

    monkeypatch.setattr(research_runner, "_load_candidate_signal_payload", lambda *args, **kwargs: signal_payload)
    monkeypatch.setattr(research_runner, "_evaluate_candidate_metric_payload", lambda *args, **kwargs: metric_payload)
    monkeypatch.setattr(
        research_runner,
        "_hurdle_fields",
        lambda *args, **kwargs: (
            {"train": {"pass": True}, "val": {"pass": True}, "oos": {"pass": True}},
            True,
            {},
        ),
    )
    monkeypatch.setattr(
        research_runner,
        "_apply_cost_stress_hard_rejects",
        lambda **kwargs: {"stress_x3_sharpe": -0.2},
    )

    result = research_runner._evaluate_candidate(
        candidate,
        cache={},
        feature_cache={},
        benchmark_cache={},
        candidate_count=1,
    )

    assert result["pass"] is False
    assert result["hard_reject_reasons"] == {"stress_x3_sharpe": -0.2}
    assert result["hurdle_fields"] == {
        "train": {"pass": False},
        "val": {"pass": False},
        "oos": {"pass": False},
    }
