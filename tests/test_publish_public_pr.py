from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "publish_public_pr.py"
_SPEC = importlib.util.spec_from_file_location("publish_public_pr_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {_SCRIPT_PATH}")
publish_public_pr = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = publish_public_pr
_SPEC.loader.exec_module(publish_public_pr)


def test_sensitive_path_detection_blocks_private_trees():
    assert publish_public_pr.is_sensitive_path("lumina_quant/strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("lumina_quant/indicators/formulaic_alpha.py")
    assert publish_public_pr.is_sensitive_path("src/lumina_quant/strategy_factory/candidate_library.py")
    assert publish_public_pr.is_sensitive_path("src/lumina_quant/strategy_factory/research_runner.py")
    assert publish_public_pr.is_sensitive_path("src/lumina_quant/workflows/alpha_research_pipeline.py")
    assert publish_public_pr.is_sensitive_path("src/lumina_quant/eval/exact_window_suite.py")
    assert publish_public_pr.is_sensitive_path("apps/dashboard/exact_window_suite.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_research_candidates.py")
    assert publish_public_pr.is_sensitive_path("tests/test_exact_window_suite.py")
    assert publish_public_pr.is_sensitive_path("scripts/research/run_llm_alpha_pipeline.py")
    assert publish_public_pr.is_sensitive_path("scripts/research/write_exact_window_deployment_combo.py")
    assert publish_public_pr.is_sensitive_path("tests/test_strategy_factory_library.py")
    assert publish_public_pr.is_sensitive_path("tests/test_research_runner_feature_support.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_bulk_research.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_research_pipeline.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_research_hurdle.py")
    assert publish_public_pr.is_sensitive_path("tests/test_run_research_pipeline_script.py")
    assert publish_public_pr.is_sensitive_path("reports/benchmarks/latest.json")
    assert publish_public_pr.is_sensitive_path(".agents/skills/alpha-research-pipeline/SKILL.md")
    assert publish_public_pr.is_sensitive_path(".codex/prompts/architect.md")
    assert publish_public_pr.is_sensitive_path(".github/hardcoded_params_baseline.json")
    assert publish_public_pr.is_sensitive_path(".env")


def test_sensitive_path_detection_allows_public_runtime_paths():
    assert not publish_public_pr.is_sensitive_path("run_backtest.py")
    assert not publish_public_pr.is_sensitive_path("lumina_quant/backtesting/chunked_runner.py")
    assert not publish_public_pr.is_sensitive_path("docs/RUNBOOK_1Y_1S_LOCAL.md")


def test_default_branch_name_uses_prefix():
    branch = publish_public_pr._default_branch_name("public-sync")
    assert branch.startswith("public-sync-")


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> None:
    _run_git(repo, "init", "-b", "main")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")


def test_restore_protected_paths_from_base_preserves_public_samples(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    protected = repo / "src" / "lumina_quant" / "strategies"
    protected.mkdir(parents=True)
    sample_file = protected / "sample_public_strategy.py"
    sample_file.write_text("PUBLIC_SAMPLE = True\n", encoding="utf-8")
    safe_file = repo / "README.md"
    safe_file.write_text("public base\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base")

    sample_file.write_text("PRIVATE_SECRET = True\n", encoding="utf-8")
    safe_file.write_text("public safe change\n", encoding="utf-8")
    _run_git(repo, "add", ".")

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        publish_public_pr,
        "PROTECTED_PATHS",
        ("src/lumina_quant/strategies",),
    )

    publish_public_pr._restore_protected_paths_from_base()

    assert sample_file.read_text(encoding="utf-8") == "PUBLIC_SAMPLE = True\n"
    staged = _run_git(repo, "diff", "--cached", "--name-only").stdout.splitlines()
    assert "README.md" in staged
    assert "src/lumina_quant/strategies/sample_public_strategy.py" not in staged


def test_restore_protected_paths_from_base_drops_private_only_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    (repo / "README.md").write_text("public base\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base")

    private_only = repo / "src" / "lumina_quant" / "indicators" / "advanced_alpha.py"
    private_only.parent.mkdir(parents=True)
    private_only.write_text("SECRET_FACTOR = 1\n", encoding="utf-8")
    _run_git(repo, "add", ".")

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        publish_public_pr,
        "PROTECTED_PATHS",
        ("src/lumina_quant/indicators",),
    )

    publish_public_pr._restore_protected_paths_from_base()

    assert not private_only.exists()
    staged = _run_git(repo, "diff", "--cached", "--name-only").stdout.splitlines()
    assert "src/lumina_quant/indicators/advanced_alpha.py" not in staged


def test_content_sensitive_detection_blocks_public_path_with_private_reference(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: ci\njobs:\n  test:\n    steps:\n      - run: python scripts/run_research_pipeline.py\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/ci.yml"])
    assert flagged == [".github/workflows/ci.yml"]


def test_content_sensitive_detection_derives_module_patterns_from_protected_inventory(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    module = repo / "public_probe.py"
    module.write_text(
        "from lumina_quant.eval.exact_window_suite import run_exact_window_suite\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths(["public_probe.py"])
    assert flagged == ["public_probe.py"]


def test_content_sensitive_detection_blocks_root_level_exact_protected_file_references(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "root-files.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: root-files\njobs:\n  check:\n    steps:\n      - run: cat AGENTS.md && cat .env && cat equity.csv\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/root-files.yml"])
    assert flagged == [".github/workflows/root-files.yml"]


def test_content_sensitive_detection_blocks_protected_directory_descendants(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "release.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: release\njobs:\n  publish:\n    steps:\n      - run: cp reports/private.json artifacts/out.json\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/release.yml"])
    assert flagged == [".github/workflows/release.yml"]


def test_content_sensitive_detection_blocks_nested_protected_directory_descendants(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "nested.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: nested\njobs:\n  publish:\n    steps:\n      - run: cat src/lumina_quant/strategies/private_alpha.py\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/nested.yml"])
    assert flagged == [".github/workflows/nested.yml"]


def test_content_sensitive_detection_blocks_dot_prefixed_exact_protected_paths(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "dot-protected.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: dot-protected\njobs:\n  check:\n    steps:\n      - run: cat .github/hardcoded_params_baseline.json\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/dot-protected.yml"])
    assert flagged == [".github/workflows/dot-protected.yml"]


def test_content_sensitive_detection_ignores_publish_sanitizer_self_test_file(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    test_file = repo / "tests" / "test_publish_public_pr.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "assert 'scripts/run_research_pipeline.py'\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths(["tests/test_publish_public_pr.py"])
    assert flagged == []
