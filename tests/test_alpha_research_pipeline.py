from __future__ import annotations

import json
from pathlib import Path

from lumina_quant.workflows.alpha_research_pipeline import (
    DEFAULT_METRICS,
    build_alpha_research_pipeline_manifest,
    write_alpha_research_pipeline_manifest,
)


def test_build_alpha_research_pipeline_manifest_contains_expected_sections(tmp_path: Path):
    payload = build_alpha_research_pipeline_manifest(
        output_root=tmp_path,
        article_url="https://example.com/article",
        image_paths=["/tmp/a.jpg", "/tmp/b.jpg"],
        total_memory_cap_gib=8.0,
        heavy_run_cap_gib=6.5,
    )

    assert payload["label"] == "article-inspired llm alpha research pipeline"
    assert payload["execution_policy"]["total_memory_cap_gib"] == 8.0
    assert payload["execution_policy"]["heavy_run_parallelism"] == 1
    assert (
        payload["execution_policy"]["duplicate_policy"]
        == "skip_if_signature_exists_in_exact_window_run_registry_jsonl_unless_forced"
    )
    assert len(payload["families"]) >= 5
    assert set(DEFAULT_METRICS).issubset(set(payload["metric_checklist"]))
    assert any(
        "global across all active sessions/services/workers" in rule for rule in payload["operating_rules"]
    )
    assert payload["recommended_outputs"]["signature_registry"].endswith("exact_window_run_registry.jsonl")
    assert payload["recommended_outputs"]["recovered_run_archive"].endswith("backtest_log_archive_latest.json")


def test_write_alpha_research_pipeline_manifest_persists_json_and_md(tmp_path: Path):
    report_root = tmp_path / "reports"
    result = write_alpha_research_pipeline_manifest(
        report_root=report_root,
        article_url="https://example.com/article",
        image_paths=["/tmp/a.jpg", "/tmp/b.jpg"],
        total_memory_cap_gib=8.0,
        heavy_run_cap_gib=6.5,
    )

    json_path = Path(result["json_path"])
    md_path = Path(result["md_path"])
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["execution_policy"]["total_memory_cap_gib"] == 8.0
    assert payload["execution_policy"]["heavy_run_cap_gib"] == 6.5
    assert result["family_count"] == len(payload["families"])
