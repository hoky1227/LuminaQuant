#!/usr/bin/env python3
"""Tune fresh-start multi-sleeve portfolios from replay survivors/candidates.

This is intentionally downstream of replay_profit_moonshot_fresh_start.py.  It
selects sleeve candidates using train/validation evidence only, combines their
stateful replay equity curves, and reports OOS as report-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul"
)
DEFAULT_CANDIDATE_CSV = DEFAULT_OUTPUT_DIR / "fresh_start_overhaul_replay_candidates.csv"
BASELINE_OOS_RETURN = 0.008284
SHADOW_OOS_MDD = 0.001778
SUCCESS_SHARPE = 1.0


def _load_fresh_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_profit_moonshot_fresh_start", FRESH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {FRESH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validation_score(row: dict[str, str]) -> float:
    val_ret = _safe_float(row.get("val_total_return"))
    train_ret = _safe_float(row.get("train_total_return"))
    val_sharpe = _safe_float(row.get("val_sharpe"))
    val_mdd = _safe_float(row.get("val_max_drawdown"), 1.0)
    trips = max(1.0, _safe_float(row.get("val_round_trips"), 0.0))
    return val_ret * 100.0 + train_ret * 25.0 + val_sharpe * 0.15 - val_mdd * 50.0 + math.log1p(trips) * 0.01


def _candidate_pool(rows: list[dict[str, str]], *, top_n: int) -> list[dict[str, str]]:
    eligible = [
        row
        for row in rows
        if _safe_float(row.get("train_total_return")) > 0.0
        and _safe_float(row.get("val_total_return")) > 0.0
        and _safe_float(row.get("val_round_trips")) >= 1.0
    ]
    eligible.sort(key=_validation_score, reverse=True)
    return eligible[:top_n]


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _combine_equity(curves: list[list[float]], *, mode: str) -> list[float]:
    if not curves:
        return []
    min_len = min(len(curve) for curve in curves)
    if min_len <= 0:
        return []
    returns = []
    for curve in curves:
        arr = np.asarray(curve[:min_len], dtype=float)
        returns.append(arr / 10_000.0 - 1.0)
    stacked = np.vstack(returns)
    if mode == "additive_sleeves":
        combined = stacked.sum(axis=0)
    elif mode == "equal_weight":
        combined = stacked.mean(axis=0)
    else:
        raise ValueError(f"unknown combine mode: {mode}")
    return [float(10_000.0 * (1.0 + item)) for item in combined]


def _combo_metrics(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
    mode: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": f"fresh_portfolio_{mode}_" + "__".join(combo_names),
        "mode": mode,
        "sleeves": list(combo_names),
        "sleeve_count": len(combo_names),
        "splits": {},
    }
    for split_name in ("train", "val", "oos"):
        curves = [split_curves[name][split_name] for name in combo_names]
        equity = _combine_equity(curves, mode=mode)
        metrics = fresh._metrics_from_equity_totals(
            equity,
            periods=int(getattr(fresh.BacktestConfig, "ANNUAL_PERIODS", 252)),
        )
        round_trips = sum(int(split_payloads[name][split_name].get("round_trips") or 0) for name in combo_names)
        fills = sum(int(split_payloads[name][split_name].get("fills") or 0) for name in combo_names)
        out["splits"][split_name] = {
            "metrics": metrics,
            "round_trips": round_trips,
            "fills": fills,
            "final_equity": float(equity[-1]) if equity else 10_000.0,
        }
    train = out["splits"]["train"]["metrics"]
    val = out["splits"]["val"]["metrics"]
    oos = out["splits"]["oos"]["metrics"]
    gates = {
        "train_positive": _safe_float(train.get("total_return")) > 0.0,
        "val_positive": _safe_float(val.get("total_return")) > 0.0,
        "oos_return_beats_incumbent": _safe_float(oos.get("total_return")) > BASELINE_OOS_RETURN,
        "oos_mdd_beats_shadow": _safe_float(oos.get("max_drawdown"), 1.0) < SHADOW_OOS_MDD,
        "oos_sharpe_gt_1": _safe_float(oos.get("sharpe")) > SUCCESS_SHARPE,
        "oos_trades_not_starved": int(out["splits"]["oos"].get("round_trips") or 0) >= 5,
    }
    out["gates"] = gates
    out["success_candidate"] = all(gates.values())
    out["validation_score"] = (
        _safe_float(val.get("total_return")) * 100.0
        + _safe_float(train.get("total_return")) * 25.0
        + _safe_float(val.get("sharpe")) * 0.15
        - _safe_float(val.get("max_drawdown"), 1.0) * 50.0
        + math.log1p(max(1, int(out["splits"]["val"].get("round_trips") or 0))) * 0.01
    )
    return out


def _flatten_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": item["name"],
        "mode": item["mode"],
        "sleeve_count": item["sleeve_count"],
        "sleeves": ",".join(item["sleeves"]),
        "validation_score": item["validation_score"],
        "success_candidate": item["success_candidate"],
        "failed_gates": ",".join(key for key, ok in item["gates"].items() if not ok),
    }
    for split_name, split in item["splits"].items():
        metrics = split["metrics"]
        for key in ("total_return", "max_drawdown", "sharpe", "sortino", "volatility"):
            row[f"{split_name}_{key}"] = _safe_float(metrics.get(key), 0.0)
        row[f"{split_name}_round_trips"] = int(split.get("round_trips") or 0)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):+.4%}"


def _fmt_float(value: Any) -> str:
    return f"{_safe_float(value):.6f}"


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    selected = payload.get("selected_by_validation") or {}
    best_oos = payload.get("diagnostic_best_oos") or {}
    lines = [
        "# Profit moonshot fresh portfolio tuning",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Policy",
        "",
        "- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.",
        "- Portfolio selection is validation-primary; OOS is report-only.",
        "- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.",
        "",
        "## Summary",
        "",
        f"- Candidate sleeves considered: `{payload['candidate_sleeve_count']}`",
        f"- Portfolio specs evaluated: `{payload['portfolio_spec_count']}`",
        f"- Success candidates: `{payload['success_candidate_count']}`",
        f"- Peak RSS: `{payload['peak_rss_mib']:.3f} MiB`",
        "",
    ]
    if selected:
        split = selected.get("splits", {})
        lines.extend(
            [
                "## Selected by validation",
                "",
                f"- `{selected.get('name')}`",
                f"- sleeves: `{', '.join(selected.get('sleeves') or [])}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                f"- success: `{selected.get('success_candidate')}` / failed gates: `{','.join(k for k, ok in (selected.get('gates') or {}).items() if not ok)}`",
                "",
            ]
        )
    if best_oos:
        split = best_oos.get("splits", {})
        lines.extend(
            [
                "## Diagnostic best OOS (not selection authority)",
                "",
                f"- `{best_oos.get('name')}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Top rows",
            "",
            "| rank | name | success | train | val | OOS | OOS MDD | OOS Sharpe | failed gates |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(rows[:15], start=1):
        lines.append(
            f"| {idx} | `{row['name']}` | {row['success_candidate']} | "
            f"{_fmt_pct(row['train_total_return'])} | {_fmt_pct(row['val_total_return'])} | "
            f"{_fmt_pct(row['oos_total_return'])} | {_fmt_pct(row['oos_max_drawdown'])} | "
            f"{_fmt_float(row['oos_sharpe'])} | `{row['failed_gates']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fresh = _load_fresh_module()
    rows = _read_rows(Path(args.candidate_csv))
    pool = _candidate_pool(rows, top_n=int(args.top_n))
    oos_end = datetime.fromisoformat(str(args.oos_end_date)).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root), exchange=str(args.exchange), symbols=symbols, start=start, end=end
    )
    arrays = fresh._build_arrays(panel, symbols)
    specs_by_name = {spec.name: spec for spec in fresh._candidate_specs(arrays, symbols)}
    pool = [row for row in pool if row["name"] in specs_by_name]

    split_curves: dict[str, dict[str, list[float]]] = {}
    split_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for row in pool:
        name = row["name"]
        split_curves[name] = {}
        split_payloads[name] = {}
        for split in splits:
            result = fresh._run_split(
                spec=specs_by_name[name], arrays=arrays, split=split, include_equity=True
            )
            split_curves[name][split.name] = list(result.get("equity_history") or [])
            split_payloads[name][split.name] = result

    portfolio_items: list[dict[str, Any]] = []
    names = [row["name"] for row in pool]
    max_k = max(2, min(int(args.max_sleeves), len(names))) if names else 0
    for size in range(2, max_k + 1):
        for combo in itertools.combinations(names, size):
            for mode in ("equal_weight", "additive_sleeves"):
                portfolio_items.append(
                    _combo_metrics(
                        fresh=fresh,
                        combo_names=combo,
                        split_curves=split_curves,
                        split_payloads=split_payloads,
                        mode=mode,
                    )
                )
    portfolio_items.sort(
        key=lambda item: (
            not bool(item["success_candidate"]),
            -_safe_float(item["splits"]["oos"]["metrics"].get("total_return")),
            -_safe_float(item["splits"]["oos"]["metrics"].get("sharpe")),
        )
    )
    csv_rows = [_flatten_row(item) for item in portfolio_items]
    selected_by_validation = max(portfolio_items, key=lambda item: item["validation_score"], default={})
    diagnostic_best_oos = max(
        portfolio_items,
        key=lambda item: (
            _safe_float(item["splits"]["oos"]["metrics"].get("total_return")),
            _safe_float(item["splits"]["oos"]["metrics"].get("sharpe")),
        ),
        default={},
    )
    payload = {
        "artifact_kind": "profit_moonshot_fresh_portfolio_tuning",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "candidate_csv": str(args.candidate_csv),
        "candidate_sleeve_count": len(pool),
        "portfolio_spec_count": len(portfolio_items),
        "success_candidate_count": sum(1 for item in portfolio_items if bool(item["success_candidate"])),
        "selected_by_validation": selected_by_validation,
        "diagnostic_best_oos": diagnostic_best_oos,
        "data_metadata": data_metadata,
        "peak_rss_mib": _rss_mib(),
    }
    return payload, csv_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT")
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=18)
    parser.add_argument("--max-sleeves", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, rows = build_payload(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fresh_portfolio_tuning_latest.json"
    csv_path = output_dir / "fresh_portfolio_tuning_candidates.csv"
    md_path = output_dir / "fresh_portfolio_tuning_latest.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    md_path.write_text(_markdown(payload, rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
                "success_candidate_count": payload["success_candidate_count"],
                "peak_rss_mib": payload["peak_rss_mib"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
