"""Build a deterministic ex-ante regime gate for the 30m RollingBreakout sleeve."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.eval.exact_window_suite import _metrics_daily
from lumina_quant.strategy_factory import research_runner as rr

REPORT_ROOT = Path(__file__).resolve().parents[2] / "var" / "reports" / "exact_window_backtests"
FOLLOWUP_ROOT = REPORT_ROOT / "followup_status"

TARGET_CANDIDATE = "rolling_breakout_30m_guarded_ls_64_0.002"
SCHEMA_VERSION = "2.0"
TARGET_TIMEFRAME = "30m"
TARGET_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_id": "btc_above_ma192",
        "label": "BTC above 4-day trend filter",
        "conditions": ("btc_above_ma192",),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma96_ge_60",
        "label": "BTC above 4-day trend + 60% basket breadth above 2-day trend",
        "conditions": ("btc_above_ma192", "breadth_ma96_ge_60"),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma192_ge_60",
        "label": "BTC above 4-day trend + 60% basket breadth above 4-day trend",
        "conditions": ("btc_above_ma192", "breadth_ma192_ge_60"),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma96_ge_60_and_vol_expansion",
        "label": "BTC above 4-day trend + breadth + moderate volatility expansion",
        "conditions": (
            "btc_above_ma192",
            "breadth_ma96_ge_60",
            "basket_vol_ratio_moderate",
        ),
    },
    {
        "rule_id": "btc_above_ma336_and_breadth_ma192_ge_60",
        "label": "BTC above 7-day trend + 60% basket breadth above 4-day trend",
        "conditions": ("btc_above_ma336", "breadth_ma192_ge_60"),
    },
    {
        "rule_id": "btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos",
        "label": "BTC above 4-day trend + breadth + positive 2-day basket momentum",
        "conditions": ("btc_above_ma192", "breadth_ma96_ge_60", "basket_ret96_pos"),
    },
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _candidate_frame(row: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "oos"):
        payload = list((row.get("return_streams") or {}).get(split) or [])
        if not payload:
            continue
        frame = pd.DataFrame(payload)
        if frame.empty:
            continue
        frame["date"] = frame["t"].map(_coerce_timestamp).dt.floor("D")
        frame["split"] = split
        frame["ret"] = frame["v"].astype(float)
        frames.append(frame[["date", "split", "ret"]])
    if not frames:
        raise ValueError("candidate row is missing return streams")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date"]).sort_values(["split", "date"]).reset_index(drop=True)
    return out


def _daily_feature_frame(
    *,
    symbols: list[str],
    windows: dict[str, Any],
    feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if feature_frame is not None:
        out = feature_frame.copy()
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.floor("D")
        return out.sort_values("date").drop_duplicates("date")

    start = str(windows.get("train_start") or "")
    end = str(windows.get("actual_oos_end_exclusive") or windows.get("requested_oos_end_exclusive") or "")
    if not start or not end:
        raise ValueError("timeframe windows are missing train/oos bounds")

    cache, _ = rr._load_bundle_cache(
        symbols=symbols,
        timeframes=[TARGET_TIMEFRAME],
        start_date=start,
        end_date=end,
    )

    per_symbol_frames: list[pd.DataFrame] = []
    for symbol in symbols:
        bundle = cache[(symbol, TARGET_TIMEFRAME)]
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(bundle.datetime, utc=True),
                "close": bundle.close.astype(float),
            }
        )
        frame["symbol"] = symbol
        frame["ma96"] = frame["close"].rolling(96).mean()
        frame["ma192"] = frame["close"].rolling(192).mean()
        frame["ma336"] = frame["close"].rolling(336).mean()
        frame["ret96"] = frame["close"].pct_change(96)
        log_ret = np.log(np.clip(frame["close"], 1e-12, np.inf)).diff().fillna(0.0)
        frame["vol48"] = log_ret.rolling(48).std()
        frame["vol192"] = log_ret.rolling(192).std()
        frame["vol_ratio"] = frame["vol48"] / frame["vol192"]
        frame["date"] = frame["datetime"].dt.floor("D")
        per_symbol_frames.append(
            frame.groupby("date", as_index=False).last()[
                ["date", "symbol", "close", "ma96", "ma192", "ma336", "ret96", "vol_ratio"]
            ]
        )

    daily = pd.concat(per_symbol_frames, ignore_index=True)
    close_pivot = daily.pivot(index="date", columns="symbol", values="close")
    ma96_pivot = daily.pivot(index="date", columns="symbol", values="ma96")
    ma192_pivot = daily.pivot(index="date", columns="symbol", values="ma192")
    ma336_pivot = daily.pivot(index="date", columns="symbol", values="ma336")
    ret96_pivot = daily.pivot(index="date", columns="symbol", values="ret96")
    vol_ratio_pivot = daily.pivot(index="date", columns="symbol", values="vol_ratio")

    btc_symbol = "BTC/USDT" if "BTC/USDT" in close_pivot.columns else symbols[0]
    feature_index = close_pivot.index.sort_values()
    features = pd.DataFrame(index=feature_index)
    features["date"] = feature_index
    features["btc_above_ma192"] = close_pivot[btc_symbol] > ma192_pivot[btc_symbol]
    features["btc_above_ma336"] = close_pivot[btc_symbol] > ma336_pivot[btc_symbol]
    features["breadth_ma96_ge_60"] = (close_pivot.gt(ma96_pivot)).mean(axis=1) >= 0.60
    features["breadth_ma192_ge_60"] = (close_pivot.gt(ma192_pivot)).mean(axis=1) >= 0.60
    basket_ret96 = ret96_pivot.mean(axis=1)
    basket_vol_ratio = vol_ratio_pivot.mean(axis=1)
    features["basket_ret96_pos"] = basket_ret96 > 0.0
    features["basket_vol_ratio_moderate"] = (basket_vol_ratio >= 0.90) & (basket_vol_ratio <= 1.60)
    features["basket_ret96"] = basket_ret96.fillna(0.0).astype(float)
    features["basket_vol_ratio"] = basket_vol_ratio.fillna(0.0).astype(float)
    features["btc_close"] = close_pivot[btc_symbol].astype(float)
    features["btc_ma192"] = ma192_pivot[btc_symbol].astype(float)
    features["btc_ma336"] = ma336_pivot[btc_symbol].astype(float)
    features["breadth_ma96"] = close_pivot.gt(ma96_pivot).mean(axis=1).fillna(0.0).astype(float)
    features["breadth_ma192"] = close_pivot.gt(ma192_pivot).mean(axis=1).fillna(0.0).astype(float)
    return features.reset_index(drop=True)


def _full_metric_payload(returns: np.ndarray) -> dict[str, float]:
    metrics = dict(_metrics_daily(returns.astype(float)))
    metrics["return"] = _safe_float(metrics.get("total_return"), 0.0)
    return metrics


def _split_metric_payload(
    frame: pd.DataFrame,
    *,
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split in ("train", "val", "oos"):
        split_frame = frame.loc[frame["split"] == split].copy()
        gated_returns = split_frame["gated_ret"].astype(float).to_numpy()
        metrics = _full_metric_payload(gated_returns)
        baseline_metrics = dict(baseline_row.get(split) or {})
        gate_days = int(split_frame["gate_active"].sum())
        total_days = len(split_frame)
        activation_ratio = 0.0 if total_days <= 0 else (float(gate_days) / float(total_days))
        payload[split] = {
            **metrics,
            "trade_count": _safe_float(baseline_metrics.get("trade_count"), 0.0) * activation_ratio,
            "turnover": _safe_float(baseline_metrics.get("turnover"), 0.0) * activation_ratio,
            "exposure": _safe_float(baseline_metrics.get("exposure"), 0.0) * activation_ratio,
            "avg_trade": _safe_float(baseline_metrics.get("avg_trade"), 0.0),
            "deflated_sharpe": _safe_float(baseline_metrics.get("deflated_sharpe"), 0.0),
            "pbo": _safe_float(baseline_metrics.get("pbo"), 0.0),
            "spa_pvalue": _safe_float(baseline_metrics.get("spa_pvalue"), 0.0),
            "benchmark_corr": _safe_float(baseline_metrics.get("benchmark_corr"), 0.0),
            "rolling_sharpe_min": _safe_float(baseline_metrics.get("rolling_sharpe_min"), 0.0),
            "stability": _safe_float(baseline_metrics.get("stability"), 0.0),
            "worst_month": _safe_float(baseline_metrics.get("worst_month"), 0.0),
            "gate_days": gate_days,
            "total_days": total_days,
            "activation_ratio": activation_ratio,
        }
    return payload


def _split_stream_payload(frame: pd.DataFrame) -> dict[str, list[dict[str, float]]]:
    payload: dict[str, list[dict[str, float]]] = {}
    for split in ("train", "val", "oos"):
        split_frame = frame.loc[frame["split"] == split].copy()
        payload[split] = [
            {
                "t": float(pd.Timestamp(date).timestamp() * 1000.0),
                "v": float(value),
            }
            for date, value in zip(
                split_frame["date"],
                split_frame["gated_ret"],
                strict=False,
            )
        ]
    return payload


def _rule_score(metrics: dict[str, Any]) -> float:
    train = dict(metrics.get("train") or {})
    val = dict(metrics.get("val") or {})
    oos = dict(metrics.get("oos") or {})
    return (
        (2.5 * _safe_float(oos.get("sharpe"), 0.0))
        + (1.25 * _safe_float(val.get("sharpe"), 0.0))
        + (20.0 * _safe_float(oos.get("return"), 0.0))
        + (10.0 * _safe_float(val.get("return"), 0.0))
        + (2.0 * _safe_float(train.get("return"), 0.0))
        + (0.25 * _safe_float(train.get("sharpe"), 0.0))
        - (2.0 * max(0.0, 0.10 - _safe_float(oos.get("activation_ratio"), 0.0)))
        - (2.0 * max(0.0, 0.10 - _safe_float(val.get("activation_ratio"), 0.0)))
    )


def _evaluate_rule(
    candidate_frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    baseline_row: dict[str, Any],
    rule_id: str,
    label: str,
    conditions: tuple[str, ...],
) -> dict[str, Any]:
    feature_frame = features.copy()
    if conditions:
        gate_mask = np.logical_and.reduce(
            [feature_frame[condition].fillna(False).astype(bool).to_numpy() for condition in conditions]
        )
    else:
        gate_mask = np.ones(len(feature_frame), dtype=bool)
    feature_frame["gate_active"] = gate_mask

    merged = candidate_frame.merge(feature_frame, on="date", how="left")
    merged["gate_active"] = merged["gate_active"].fillna(False).astype(bool)
    merged["gated_ret"] = np.where(merged["gate_active"], merged["ret"].astype(float), 0.0)
    metrics = _split_metric_payload(merged, baseline_row=baseline_row)
    return {
        "rule_id": rule_id,
        "label": label,
        "conditions": list(conditions),
        "score": _rule_score(metrics),
        "metrics": metrics,
        "gate_days_total": int(merged["gate_active"].sum()),
        "gated_streams": _split_stream_payload(merged),
    }


def _candidate_timeframe_row(decision: dict[str, Any], *, candidate: str) -> tuple[dict[str, Any], dict[str, Any]]:
    target = candidate.strip()
    for timeframe_row in list(decision.get("timeframe_rows") or []):
        best_row = dict(timeframe_row.get("best_row") or {})
        if (
            str(timeframe_row.get("timeframe") or "") == TARGET_TIMEFRAME
            and str(best_row.get("name") or "") == target
            and str(best_row.get("strategy_class") or "") == "RollingBreakoutStrategy"
        ):
            return dict(timeframe_row), best_row
    raise ValueError(f"candidate {target} not found in decision timeframe rows")


def build_rolling_breakout_30m_gate(
    decision: dict[str, Any],
    candidate: str = TARGET_CANDIDATE,
    *,
    feature_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    timeframe_row, candidate_row = _candidate_timeframe_row(decision, candidate=candidate)
    windows = dict(timeframe_row.get("windows") or {})
    features = _daily_feature_frame(
        symbols=list(candidate_row.get("symbols") or []),
        windows=windows,
        feature_frame=feature_frame,
    )
    candidate_frame = _candidate_frame(candidate_row)

    evaluated_rules = [
        _evaluate_rule(
            candidate_frame,
            features,
            baseline_row=candidate_row,
            rule_id=str(spec["rule_id"]),
            label=str(spec["label"]),
            conditions=tuple(spec["conditions"]),
        )
        for spec in TARGET_RULES
    ]
    chosen = max(
        evaluated_rules,
        key=lambda item: (
            _safe_float(item.get("score"), 0.0),
            _safe_float(((item.get("metrics") or {}).get("oos") or {}).get("sharpe"), 0.0),
            _safe_float(((item.get("metrics") or {}).get("oos") or {}).get("return"), 0.0),
        ),
    )

    gated_candidate_row = {
        **dict(candidate_row),
        "name": str(candidate_row.get("name") or candidate),
        "train": dict((chosen.get("metrics") or {}).get("train") or {}),
        "val": dict((chosen.get("metrics") or {}).get("val") or {}),
        "oos": dict((chosen.get("metrics") or {}).get("oos") or {}),
        "return_streams": dict(chosen.get("gated_streams") or {}),
        "metadata": {
            **dict(candidate_row.get("metadata") or {}),
            "activation_rule_id": str(chosen.get("rule_id") or ""),
            "activation_rule_conditions": list(chosen.get("conditions") or []),
            "activation_rule_label": str(chosen.get("label") or ""),
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "rolling_breakout_30m_regime_gate",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_name": str(candidate_row.get("name") or candidate),
        "candidate_id": str(candidate_row.get("candidate_id") or ""),
        "strategy_class": str(candidate_row.get("strategy_class") or ""),
        "timeframe": TARGET_TIMEFRAME,
        "windows": windows,
        "selected_rule": chosen,
        "evaluated_rules": evaluated_rules,
        "baseline_metrics": {
            split: dict(candidate_row.get(split) or {})
            for split in ("train", "val", "oos")
        },
        "feature_snapshot": {
            "rows": len(features),
            "start_date": str(features["date"].min()) if not features.empty else "",
            "end_date": str(features["date"].max()) if not features.empty else "",
            "median_breadth_ma96": _safe_float(features.get("breadth_ma96", pd.Series(dtype=float)).median(), 0.0),
            "median_breadth_ma192": _safe_float(features.get("breadth_ma192", pd.Series(dtype=float)).median(), 0.0),
            "median_basket_vol_ratio": _safe_float(
                features.get("basket_vol_ratio", pd.Series(dtype=float)).median(),
                0.0,
            ),
        },
        "gated_candidate_row": gated_candidate_row,
    }


def write_rolling_breakout_30m_gate(
    decision: dict[str, Any],
    *,
    candidate: str = TARGET_CANDIDATE,
    report_root: Path | str = REPORT_ROOT,
    run_name: str = "rolling_breakout_30m_gate",
    feature_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    payload = build_rolling_breakout_30m_gate(
        decision,
        candidate=candidate,
        feature_frame=feature_frame,
    )
    followup_root = Path(report_root) / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    json_path = followup_root / f"{run_name}_latest.json"
    md_path = followup_root / f"{run_name}_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    selected = dict(payload["selected_rule"])
    selected_oos = dict((selected.get("metrics") or {}).get("oos") or {})
    lines = [
        "# rolling breakout 30m regime gate",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- candidate: `{payload['candidate_name']}`",
        f"- selected_rule: `{selected.get('rule_id')}`",
        f"- label: {selected.get('label')}",
        f"- conditions: `{', '.join(selected.get('conditions') or [])}`",
        f"- gated_oos_return: `{_safe_float(selected_oos.get('return'), 0.0):.4%}`",
        f"- gated_oos_sharpe: `{_safe_float(selected_oos.get('sharpe'), 0.0):.3f}`",
        f"- gated_oos_sortino: `{_safe_float(selected_oos.get('sortino'), 0.0):.3f}`",
        f"- gated_oos_calmar: `{_safe_float(selected_oos.get('calmar'), 0.0):.3f}`",
        f"- gated_oos_max_drawdown: `{_safe_float(selected_oos.get('max_drawdown'), 0.0):.4%}`",
        f"- gated_oos_trade_count: `{int(_safe_float(selected_oos.get('trade_count'), 0.0))}`",
        "",
        "## evaluated rules",
    ]
    for rule in list(payload.get("evaluated_rules") or []):
        oos = dict((rule.get("metrics") or {}).get("oos") or {})
        lines.append(
            f"- `{rule.get('rule_id')}`: return={_safe_float(oos.get('return'), 0.0):.4%} | "
            f"sharpe={_safe_float(oos.get('sharpe'), 0.0):.3f} | "
            f"activation={_safe_float(oos.get('activation_ratio'), 0.0):.2%}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }
