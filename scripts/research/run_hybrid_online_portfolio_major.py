"""Expanded major-strategy hybrid online portfolio experiment.

This keeps the main integrated hybrid as the guarded default, but tests a
broader universe where all major previously-promoted portfolio sleeves compete
inside the same causal online governor.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "run_hybrid_online_portfolio",
    ROOT / "run_hybrid_online_portfolio.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load run_hybrid_online_portfolio")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

OUTPUT_DIR = _MOD.GROUP_ROOT / "portfolio_hybrid_online_major_current"


def _majorize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cloned = dict(row)
        meta = dict(cloned.get("metadata") or {})
        name = str(cloned.get("name") or "")
        if name == "risk_off_cash":
            meta["max_weight_cap"] = 1.0
        elif "pair" in name:
            meta["max_weight_cap"] = 0.20
        else:
            meta["max_weight_cap"] = 0.45
        cloned["metadata"] = meta
        out.append(cloned)
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_major_report(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = _MOD.acquire_portfolio_memory_guard(
        run_name="hybrid_online_portfolio_major",
        output_dir=output_dir,
        input_path=_MOD.HISTORICAL_INPUTS["soft_three_way_regime"],
        budget_bytes=_MOD.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    status = "completed"
    error: str | None = None
    try:
        historical_active = _majorize(_MOD._historical_active_rows() + _MOD._historical_benchmark_rows())
        refreshed_active, refreshed_benchmarks = _MOD._refreshed_rows()
        refreshed_active = _majorize(refreshed_active + refreshed_benchmarks)
        refreshed_health_metrics = {row["name"]: dict(row.get("oos") or {}) for row in refreshed_active}

        historical_config = _MOD.HybridOnlineConfig(**{**_MOD.asdict(_MOD.HybridOnlineConfig()), "use_current_health_priors": False})
        historical_result = _MOD.run_hybrid_online_allocator(historical_active, config=historical_config, refreshed_health_metrics=None)
        memory_guard.sample(event="major_historical_done")
        refreshed_result = _MOD.run_hybrid_online_allocator(refreshed_active, config=_MOD.HybridOnlineConfig(), refreshed_health_metrics=refreshed_health_metrics)
        memory_guard.sample(event="major_refreshed_done")
    except _MOD.RSSLimitExceeded as exc:
        status = "aborted_rss_guard"
        error = str(exc)
        raise
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(event="major_finish", context={"status": status, "error": error})
        memory_summary = memory_guard.finalize(status=status, error=error, context={"output_dir": str(output_dir.resolve())})
        memory_guard.release()

    refreshed_rows = _MOD._comparison_rows(hybrid_result=refreshed_result, benchmarks=[], active_rows=refreshed_active)
    hist_rows = _MOD._comparison_rows(hybrid_result=historical_result, benchmarks=[], active_rows=historical_active)
    payload = {
        "artifact_kind": "hybrid_online_portfolio_major",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "memory_summary": memory_summary,
        "config": _MOD.asdict(_MOD.HybridOnlineConfig()),
        "scenarios": {
            "historical_saved_baseline": {
                "active_sleeves": [row["name"] for row in historical_active],
                **historical_result,
                "comparison_rows": hist_rows,
            },
            "refreshed_latest_tail": {
                "active_sleeves": [row["name"] for row in refreshed_active],
                **refreshed_result,
                "comparison_rows": refreshed_rows,
            },
        },
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"hybrid_online_major_{stamp}.json"
    latest_json = output_dir / "hybrid_online_major_latest.json"
    md_path = output_dir / f"hybrid_online_major_{stamp}.md"
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    lines = [
        "# Hybrid Online Portfolio — Major Strategy Universe",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- peak_rss_mib: `{_MOD._safe_float(memory_summary.get('peak_rss_mib'), 0.0):.2f}`",
        "",
    ]
    lines.extend(_MOD._scoreboard_markdown("Historical saved baseline scoreboard", hist_rows))
    lines.extend([""])
    lines.extend(_MOD._scoreboard_markdown("Refreshed latest-tail scoreboard", refreshed_rows))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"latest_json": str(latest_json.resolve()), "md_path": str(md_path.resolve())}


if __name__ == "__main__":
    result = write_major_report()
    print(result["latest_json"])
    print(result["md_path"])
