#!/usr/bin/env python3
"""Liquidation-aware validation for the profit-moonshot current-base tuple.

The fresh portfolio tuner historically scaled sleeve equity curves linearly.
This script keeps the same train/validation selection boundary, but replays the
known current-base sleeves through a conservative USDⓈ-M perpetual-style margin
model with intrabar high/low liquidation checks before any levered row can be
treated as deployable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.portfolio_split_contract import (  # noqa: E402
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

TUNER_PATH = REPO_ROOT / "scripts/research/tune_profit_moonshot_fresh_portfolio.py"
FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "liquidation_aware_5x_20260510"
)
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market_parquet"
DEFAULT_CURRENT_BASE_ARTIFACT = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json"
)
DEFAULT_INTEGER_AUDIT_ARTIFACT = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "integer_leverage_alpha_v2_top40_20260509/fresh_portfolio_tuning_latest.json"
)
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT"
RUN_NAME = "profit_moonshot_liquidation_aware_validation"
STARTING_EQUITY = 10_000.0
CURRENT_BASE_SLEEVES = (
    "fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600",
    "fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600",
    "fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all",
    "fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600",
)
VALIDATION_SPLIT = "validation"
BINANCE_SOURCE_REFS = {
    "notional_leverage_brackets": (
        "https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/"
        "Notional-and-Leverage-Brackets"
    ),
    "funding_rate_history": (
        "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/"
        "Get-Funding-Rate-History"
    ),
}


@dataclass(frozen=True, slots=True)
class MarginModel:
    """Conservative scalar fallback for Binance-style USDⓈ-M margin checks."""

    margin_mode: str = "cross"
    maintenance_margin_rate: float = 0.01
    taker_fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    funding_rate_per_8h: float = 0.0001
    stress_buffer_rate: float = 0.0025
    liquidation_fee_rate: float = 0.005
    source: str = "conservative_scalar_fallback_without_authenticated_symbol_brackets"

    @property
    def liquidation_reserve_rate(self) -> float:
        return float(
            self.maintenance_margin_rate
            + self.taker_fee_rate
            + self.slippage_rate
            + self.funding_rate_per_8h
            + self.stress_buffer_rate
            + self.liquidation_fee_rate
        )


@dataclass(slots=True)
class OpenLeg:
    sleeve: str
    symbol: str
    side: str
    qty: float
    entry_price: float


@dataclass(slots=True)
class SleeveState:
    spec: Any
    legs: list[OpenLeg]
    gross_entry_notional: float = 0.0
    entry_equity: float = STARTING_EQUITY
    bars_held: int = 0
    cooldown: int = 0
    fills: int = 0
    round_trips: int = 0


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _json_safe(value: Any) -> Any:
    """Return a strict-JSON-safe copy with non-finite floats converted to null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _display_split_name(split_name: Any) -> str:
    raw = str(split_name)
    return VALIDATION_SPLIT if raw == "val" else raw


def _split_payload(splits: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    if split_name == VALIDATION_SPLIT:
        return dict(splits.get(VALIDATION_SPLIT) or splits.get("val") or {})
    return dict(splits.get(split_name) or {})


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _liquidation_adverse_fraction(*, leverage: float, model: MarginModel) -> float:
    lev = max(1e-9, float(leverage))
    return max(0.0, (1.0 / lev) - float(model.liquidation_reserve_rate))


def _liquidation_price(entry_price: float, side: str, *, leverage: float, model: MarginModel) -> float:
    entry = max(1e-12, float(entry_price))
    adverse_fraction = _liquidation_adverse_fraction(leverage=leverage, model=model)
    if str(side).upper() == "LONG":
        return entry * max(0.0, 1.0 - adverse_fraction)
    if str(side).upper() == "SHORT":
        return entry * (1.0 + adverse_fraction)
    raise ValueError(f"unknown side: {side}")


def _intrabar_liquidation_event(
    leg: OpenLeg,
    *,
    high: float,
    low: float,
    leverage: float,
    model: MarginModel,
    split_name: str,
    timestamp: str,
) -> dict[str, Any] | None:
    liq = _liquidation_price(leg.entry_price, leg.side, leverage=leverage, model=model)
    side = str(leg.side).upper()
    if side == "LONG" and math.isfinite(low) and float(low) <= liq:
        return {
            "split": split_name,
            "timestamp": timestamp,
            "sleeve": leg.sleeve,
            "symbol": leg.symbol,
            "side": side,
            "entry_price": float(leg.entry_price),
            "liquidation_price": float(liq),
            "trigger_price": float(low),
            "leverage": float(leverage),
            "reason": "intrabar_low_breached_liquidation_threshold",
        }
    if side == "SHORT" and math.isfinite(high) and float(high) >= liq:
        return {
            "split": split_name,
            "timestamp": timestamp,
            "sleeve": leg.sleeve,
            "symbol": leg.symbol,
            "side": side,
            "entry_price": float(leg.entry_price),
            "liquidation_price": float(liq),
            "trigger_price": float(high),
            "leverage": float(leverage),
            "reason": "intrabar_high_breached_liquidation_threshold",
        }
    return None


def _split_margin_summary(
    *,
    split_name: str,
    snapshots: list[Mapping[str, Any]],
    liquidation_events: list[Mapping[str, Any]],
) -> dict[str, Any]:
    split_snapshots = [dict(item) for item in snapshots if str(item.get("split")) in {"", split_name} or "split" not in item]
    if not split_snapshots:
        split_snapshots = [dict(item) for item in snapshots]
    buffers = [_safe_float(item.get("margin_buffer"), STARTING_EQUITY) for item in split_snapshots]
    ratios = [_safe_float(item.get("margin_ratio"), math.inf) for item in split_snapshots]
    minimum_buffer = min(buffers) if buffers else STARTING_EQUITY
    finite_ratios = [item for item in ratios if math.isfinite(item)]
    minimum_ratio = min(finite_ratios) if finite_ratios else math.inf
    count = sum(1 for event in liquidation_events if str(event.get("split")) == split_name)
    return {
        "liquidation_count": int(count),
        "minimum_margin_buffer": float(minimum_buffer),
        "minimum_margin_ratio": float(minimum_ratio) if math.isfinite(minimum_ratio) else math.inf,
        "margin_buffer_positive": bool(minimum_buffer > 0.0),
    }


def _split_is_liquidation_safe(split_payload: Mapping[str, Any]) -> bool:
    return int(split_payload.get("liquidation_count") or 0) == 0 and _safe_float(
        split_payload.get("minimum_margin_buffer"), 0.0
    ) > 0.0


def _liquidation_promotion_gates(candidate: Mapping[str, Any]) -> dict[str, bool]:
    splits = dict(candidate.get("splits") or {})
    train = _split_is_liquidation_safe(_split_payload(splits, "train"))
    val = _split_is_liquidation_safe(_split_payload(splits, VALIDATION_SPLIT))
    oos = _split_is_liquidation_safe(_split_payload(splits, "oos"))
    return {
        "train_validation_liquidation_safe": bool(train and val),
        "all_splits_liquidation_safe": bool(train and val and oos),
        "liquidation_free": all(int(dict(payload).get("liquidation_count") or 0) == 0 for payload in splits.values()),
        "margin_buffer_positive": all(
            _safe_float(dict(payload).get("minimum_margin_buffer"), 0.0) > 0.0 for payload in splits.values()
        ),
    }


def _liquidation_safe_for_promotion(gates: Mapping[str, Any]) -> bool:
    return bool(gates.get("liquidation_free")) and bool(gates.get("margin_buffer_positive")) and bool(
        gates.get("all_splits_liquidation_safe")
    )


def _select_train_validation_leverage(grid: list[dict[str, Any]]) -> dict[str, Any]:
    train_val_safe = [
        dict(item)
        for item in grid
        if _split_is_liquidation_safe(_split_payload(dict(item.get("splits") or {}), "train"))
        and _split_is_liquidation_safe(_split_payload(dict(item.get("splits") or {}), VALIDATION_SPLIT))
    ]
    selection_pool = train_val_safe or [dict(item) for item in grid]
    selected = max(
        selection_pool,
        key=lambda item: (
            _safe_float(item.get("train_val_score")),
            _safe_float(item.get("leverage")),
        ),
    )
    selected.setdefault("selection_policy", {})
    selected["selection_policy"] = {
        **dict(selected.get("selection_policy") or {}),
        "selection_inputs": ["train", "validation"],
        "locked_oos": "report_only_gate_only",
        "uses_locked_oos_for_selection": False,
    }
    return selected


def _timestamp_for_idx(arrays: Mapping[str, Any], idx: int) -> str:
    raw = arrays["datetime"][idx]
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    return str(raw)


def _array_value(arrays: Mapping[str, Any], key: str, idx: int, default: float = math.nan) -> float:
    values = arrays.get(key)
    if values is None:
        return float(default)
    try:
        value = float(values[idx])
    except (IndexError, TypeError, ValueError):
        return float(default)
    return float(value) if math.isfinite(value) else float(default)


def _symbol_prefix(fresh: Any, symbol: str) -> str:
    return str(fresh._symbol_prefix(symbol))


def _leg_unrealized_pnl(arrays: Mapping[str, Any], leg: OpenLeg, idx: int, fresh: Any) -> float:
    prefix = _symbol_prefix(fresh, leg.symbol)
    close = _array_value(arrays, f"{prefix}_close", idx, leg.entry_price)
    if str(leg.side).upper() == "LONG":
        return float(leg.qty) * (close - float(leg.entry_price))
    return float(leg.qty) * (float(leg.entry_price) - close)


def _state_unrealized_pnl(arrays: Mapping[str, Any], state: SleeveState, idx: int, fresh: Any) -> float:
    return sum(_leg_unrealized_pnl(arrays, leg, idx, fresh) for leg in state.legs)


def _portfolio_equity(cash: float, states: list[SleeveState], arrays: Mapping[str, Any], idx: int, fresh: Any) -> float:
    return float(cash) + sum(_state_unrealized_pnl(arrays, state, idx, fresh) for state in states)


def _open_legs(states: list[SleeveState]) -> list[OpenLeg]:
    return [leg for state in states for leg in state.legs]


def _portfolio_notional(states: list[SleeveState], arrays: Mapping[str, Any], idx: int, fresh: Any) -> float:
    total = 0.0
    for leg in _open_legs(states):
        prefix = _symbol_prefix(fresh, leg.symbol)
        close = _array_value(arrays, f"{prefix}_close", idx, leg.entry_price)
        total += abs(float(leg.qty) * close)
    return float(total)


def _margin_snapshot(
    *,
    cash: float,
    states: list[SleeveState],
    arrays: Mapping[str, Any],
    idx: int,
    split_name: str,
    leverage: float,
    model: MarginModel,
    fresh: Any,
) -> dict[str, Any]:
    equity = _portfolio_equity(cash, states, arrays, idx, fresh)
    notional = _portfolio_notional(states, arrays, idx, fresh)
    maintenance = notional * float(model.maintenance_margin_rate)
    estimated_exit = notional * (float(model.taker_fee_rate) + float(model.slippage_rate))
    funding_buffer = notional * float(model.funding_rate_per_8h)
    stress_buffer = notional * float(model.stress_buffer_rate)
    requirement = maintenance + estimated_exit + funding_buffer + stress_buffer
    margin_buffer = equity - requirement
    margin_ratio = equity / requirement if requirement > 0.0 else math.inf
    return {
        "split": split_name,
        "timestamp": _timestamp_for_idx(arrays, idx),
        "equity": float(equity),
        "open_notional": float(notional),
        "maintenance_margin": float(maintenance),
        "estimated_exit_cost": float(estimated_exit),
        "funding_buffer": float(funding_buffer),
        "stress_buffer": float(stress_buffer),
        "margin_requirement": float(requirement),
        "margin_buffer": float(margin_buffer),
        "margin_ratio": float(margin_ratio) if math.isfinite(margin_ratio) else math.inf,
        "leverage": float(leverage),
    }


def _apply_funding_cost(
    *,
    cash: float,
    states: list[SleeveState],
    arrays: Mapping[str, Any],
    idx: int,
    model: MarginModel,
    fresh: Any,
) -> float:
    hourly_cost = 0.0
    for leg in _open_legs(states):
        prefix = _symbol_prefix(fresh, leg.symbol)
        close = _array_value(arrays, f"{prefix}_close", idx, leg.entry_price)
        funding = abs(_array_value(arrays, f"{prefix}_funding_ffill", idx, model.funding_rate_per_8h))
        funding = max(float(model.funding_rate_per_8h), funding)
        hourly_cost += abs(float(leg.qty) * close) * funding / 8.0
    return float(cash) - float(hourly_cost)


def _realize_leg(
    *,
    cash: float,
    leg: OpenLeg,
    fill_price: float,
    action: str,
    high_low_vol: float,
    fresh: Any,
    model: MarginModel,
    liquidation: bool = False,
) -> float:
    fill, fee_rate = fresh._fill_price(float(fill_price), str(action), high_low_vol=float(high_low_vol))
    effective_fee_rate = max(float(fee_rate), float(model.taker_fee_rate))
    if liquidation:
        effective_fee_rate += float(model.liquidation_fee_rate)
    qty = float(leg.qty)
    pnl = qty * (float(fill) - float(leg.entry_price)) if leg.side == "LONG" else qty * (
        float(leg.entry_price) - float(fill)
    )
    return float(cash) + pnl - abs(qty * float(fill)) * effective_fee_rate


def _close_state_at_idx(
    *,
    cash: float,
    state: SleeveState,
    arrays: Mapping[str, Any],
    idx: int,
    fresh: Any,
    model: MarginModel,
    liquidation: bool = False,
    override_prices: Mapping[str, float] | None = None,
) -> float:
    if not state.legs:
        return float(cash)
    override = dict(override_prices or {})
    for leg in list(state.legs):
        prefix = _symbol_prefix(fresh, leg.symbol)
        close = override.get(leg.symbol, _array_value(arrays, f"{prefix}_close", idx, leg.entry_price))
        high = _array_value(arrays, f"{prefix}_high", idx, close)
        low = _array_value(arrays, f"{prefix}_low", idx, close)
        open_ = _array_value(arrays, f"{prefix}_open", idx, close)
        high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0
        action = "SELL" if leg.side == "LONG" else "BUY"
        cash = _realize_leg(
            cash=cash,
            leg=leg,
            fill_price=close,
            action=action,
            high_low_vol=high_low_vol,
            fresh=fresh,
            model=model,
            liquidation=liquidation,
        )
    state.fills += len(state.legs)
    state.round_trips += 1
    state.legs = []
    state.gross_entry_notional = 0.0
    state.entry_equity = float(cash)
    state.bars_held = 0
    return float(cash)


def _plan_order(
    *,
    symbol: str,
    action: str,
    scale: float,
    idx: int,
    equity: float,
    leverage: float,
    arrays: Mapping[str, Any],
    fresh: Any,
    model: MarginModel,
) -> dict[str, Any] | None:
    prefix = _symbol_prefix(fresh, symbol)
    close = _array_value(arrays, f"{prefix}_close", idx)
    if not math.isfinite(close) or close <= 0.0 or scale <= 0.0:
        return None
    high = _array_value(arrays, f"{prefix}_high", idx, close)
    low = _array_value(arrays, f"{prefix}_low", idx, close)
    open_ = _array_value(arrays, f"{prefix}_open", idx, close)
    volume = max(0.0, _array_value(arrays, f"{prefix}_volume", idx, 0.0))
    high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0
    base_notional = min(fresh.TARGET_ALLOCATION * float(scale) * float(equity), fresh.MAX_ORDER_VALUE * float(scale))
    notional = base_notional * max(0.0, float(leverage))
    raw_qty = math.floor((notional / close) / 0.001) * 0.001
    order_qty = min(raw_qty, volume * 0.10)
    if order_qty * close < 5.0 or order_qty <= 0.0:
        return None
    fill, fee_rate = fresh._fill_price(close, action, high_low_vol=high_low_vol)
    return {
        "symbol": symbol,
        "action": action,
        "qty": float(order_qty),
        "fill": float(fill),
        "fee_rate": max(float(fee_rate), float(model.taker_fee_rate)),
        "side": "LONG" if action == "BUY" else "SHORT",
    }


def _spread_signal_for_state(fresh: Any, state: SleeveState, arrays: Mapping[str, Any], idx: int) -> tuple[str, str, str, str]:
    family = str(state.spec.family)
    if family == "calendar_spread":
        return fresh._calendar_spread_signal(state.spec, arrays, idx)
    if family == "residual_pair_reversion_spread":
        return fresh._residual_pair_spread_signal(state.spec, arrays, idx)
    if family == "residual_pair_momentum_spread":
        return fresh._residual_pair_momentum_spread_signal(state.spec, arrays, idx)
    return "", "", "", "unsupported_spread_family"


def _run_liquidation_split(
    *,
    fresh: Any,
    specs: list[Any],
    arrays: Mapping[str, Any],
    split: Any,
    leverage: float,
    model: MarginModel,
) -> dict[str, Any]:
    split_name = _display_split_name(split.name)
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = int(datetime.combine(split.end, datetime.min.time(), tzinfo=UTC).timestamp()) + 24 * 60 * 60 - 1
    indices = np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))
    if indices.size == 0:
        return {
            "metrics": {},
            "liquidation_events": [],
            "margin_snapshots": [],
            **_split_margin_summary(split_name=split_name, snapshots=[], liquidation_events=[]),
        }

    cash = STARTING_EQUITY
    states = [SleeveState(spec=spec, legs=[]) for spec in specs]
    equity_history: list[float] = []
    margin_snapshots: list[dict[str, Any]] = []
    liquidation_events: list[dict[str, Any]] = []

    for raw_idx in indices:
        idx = int(raw_idx)
        cash = _apply_funding_cost(cash=cash, states=states, arrays=arrays, idx=idx, model=model, fresh=fresh)

        for state in states:
            if not state.legs:
                continue
            state.bars_held += 1
            state_events: list[dict[str, Any]] = []
            override_prices: dict[str, float] = {}
            for leg in list(state.legs):
                prefix = _symbol_prefix(fresh, leg.symbol)
                high = _array_value(arrays, f"{prefix}_high", idx, leg.entry_price)
                low = _array_value(arrays, f"{prefix}_low", idx, leg.entry_price)
                event = _intrabar_liquidation_event(
                    leg,
                    high=high,
                    low=low,
                    leverage=leverage,
                    model=model,
                    split_name=split_name,
                    timestamp=_timestamp_for_idx(arrays, idx),
                )
                if event is not None:
                    state_events.append(event)
                    override_prices[leg.symbol] = _safe_float(event.get("liquidation_price"), leg.entry_price)
            if state_events:
                liquidation_events.extend(state_events)
                cash = _close_state_at_idx(
                    cash=cash,
                    state=state,
                    arrays=arrays,
                    idx=idx,
                    fresh=fresh,
                    model=model,
                    liquidation=True,
                    override_prices=override_prices,
                )
                state.cooldown = max(0, int(state.spec.cooldown_bars))

        pre_exit_snapshot = _margin_snapshot(
            cash=cash,
            states=states,
            arrays=arrays,
            idx=idx,
            split_name=split_name,
            leverage=leverage,
            model=model,
            fresh=fresh,
        )
        if _open_legs(states) and _safe_float(pre_exit_snapshot.get("margin_buffer"), 1.0) <= 0.0:
            liquidation_events.append(
                {
                    "split": split_name,
                    "timestamp": _timestamp_for_idx(arrays, idx),
                    "reason": "cross_margin_buffer_non_positive",
                    "margin_buffer": pre_exit_snapshot["margin_buffer"],
                    "margin_ratio": pre_exit_snapshot["margin_ratio"],
                    "leverage": float(leverage),
                }
            )
            for state in states:
                cash = _close_state_at_idx(
                    cash=cash,
                    state=state,
                    arrays=arrays,
                    idx=idx,
                    fresh=fresh,
                    model=model,
                    liquidation=True,
                )

        for state in states:
            if state.legs:
                spread_return = _state_unrealized_pnl(arrays, state, idx, fresh) / max(
                    1e-9, float(state.gross_entry_notional)
                )
                exit_reason = ""
                if float(state.spec.stop_loss_pct) > 0.0 and spread_return <= -float(state.spec.stop_loss_pct):
                    exit_reason = "stop"
                elif float(state.spec.take_profit_pct) > 0.0 and spread_return >= float(state.spec.take_profit_pct):
                    exit_reason = "take_profit"
                elif state.bars_held >= int(state.spec.hold_bars):
                    exit_reason = "max_hold"
                if exit_reason:
                    cash = _close_state_at_idx(
                        cash=cash,
                        state=state,
                        arrays=arrays,
                        idx=idx,
                        fresh=fresh,
                        model=model,
                    )
                    state.cooldown = max(0, int(state.spec.cooldown_bars))

            if state.legs:
                continue
            if state.cooldown > 0:
                state.cooldown -= 1
                continue
            long_symbol, short_symbol, direction, _reason = _spread_signal_for_state(fresh, state, arrays, idx)
            if not long_symbol or not short_symbol or not direction:
                continue
            hedge = max(0.0, float(state.spec.spread_hedge_ratio))
            if direction == "LONG_SPREAD":
                plans = (
                    (long_symbol, "BUY", max(0.0, float(state.spec.long_allocation_scale))),
                    (short_symbol, "SELL", max(0.0, float(state.spec.short_allocation_scale)) * hedge),
                )
            else:
                plans = (
                    (long_symbol, "SELL", max(0.0, float(state.spec.long_allocation_scale))),
                    (short_symbol, "BUY", max(0.0, float(state.spec.short_allocation_scale)) * hedge),
                )
            equity = _portfolio_equity(cash, states, arrays, idx, fresh)
            orders = [
                order
                for symbol, action, scale in plans
                if (
                    order := _plan_order(
                        symbol=symbol,
                        action=action,
                        scale=scale,
                        idx=idx,
                        equity=equity,
                        leverage=leverage,
                        arrays=arrays,
                        fresh=fresh,
                        model=model,
                    )
                )
                is not None
            ]
            if len(orders) != 2:
                continue
            state.gross_entry_notional = sum(abs(float(order["qty"]) * float(order["fill"])) for order in orders)
            for order in orders:
                cash -= abs(float(order["qty"]) * float(order["fill"])) * float(order["fee_rate"])
                state.legs.append(
                    OpenLeg(
                        sleeve=str(state.spec.name),
                        symbol=str(order["symbol"]),
                        side=str(order["side"]),
                        qty=float(order["qty"]),
                        entry_price=float(order["fill"]),
                    )
                )
            state.fills += len(orders)
            state.entry_equity = _portfolio_equity(cash, states, arrays, idx, fresh)
            state.bars_held = 0

        snapshot = _margin_snapshot(
            cash=cash,
            states=states,
            arrays=arrays,
            idx=idx,
            split_name=split_name,
            leverage=leverage,
            model=model,
            fresh=fresh,
        )
        margin_snapshots.append(snapshot)
        equity_history.append(float(snapshot["equity"]))

    if indices.size:
        last_idx = int(indices[-1])
        for state in states:
            cash = _close_state_at_idx(
                cash=cash,
                state=state,
                arrays=arrays,
                idx=last_idx,
                fresh=fresh,
                model=model,
            )
        if equity_history:
            equity_history[-1] = float(cash)

    metrics = fresh._metrics_from_equity_totals(equity_history, periods=int(fresh.HOURLY_PERIODS_PER_YEAR))
    summary = _split_margin_summary(
        split_name=split_name,
        snapshots=margin_snapshots,
        liquidation_events=liquidation_events,
    )
    return {
        "metrics": metrics,
        "round_trips": int(sum(state.round_trips for state in states)),
        "fills": int(sum(state.fills for state in states)),
        "final_equity": float(equity_history[-1]) if equity_history else STARTING_EQUITY,
        "liquidation_events": liquidation_events[:25],
        "liquidation_event_count_total": len(liquidation_events),
        "margin_snapshot_count": len(margin_snapshots),
        "margin_tail": margin_snapshots[-5:],
        **summary,
    }


def _monthlyized_return(tuner: Any, metrics: Mapping[str, Any]) -> float:
    return float(tuner._monthlyized_return(dict(metrics)))


def _smart_sortino(tuner: Any, metrics: Mapping[str, Any]) -> float:
    return float(tuner._smart_sortino(dict(metrics)))


def _train_val_score(tuner: Any, item: Mapping[str, Any]) -> float:
    splits = dict(item.get("splits") or {})
    train = dict(_split_payload(splits, "train").get("metrics") or {})
    val = dict(_split_payload(splits, VALIDATION_SPLIT).get("metrics") or {})
    components = {
        "train_monthlyized_return": _monthlyized_return(tuner, train),
        "validation_monthlyized_return": _monthlyized_return(tuner, val),
        "train_sharpe": _safe_float(train.get("sharpe")),
        "validation_sharpe": _safe_float(val.get("sharpe")),
        "train_sortino": _safe_float(train.get("sortino")),
        "validation_sortino": _safe_float(val.get("sortino")),
        "train_calmar": _safe_float(train.get("calmar")),
        "validation_calmar": _safe_float(val.get("calmar")),
        "train_max_drawdown": _safe_float(train.get("max_drawdown"), 1.0),
        "validation_max_drawdown": _safe_float(val.get("max_drawdown"), 1.0),
        "leverage": _safe_float(item.get("leverage"), 1.0),
        "sleeve_count": float(len(CURRENT_BASE_SLEEVES)),
    }
    return float(tuner._train_val_stability_score_from_components(components))


def _build_leverage_result(
    *,
    leverage: float,
    fresh: Any,
    tuner: Any,
    specs: list[Any],
    arrays: Mapping[str, Any],
    splits: list[Any],
    model: MarginModel,
) -> dict[str, Any]:
    split_payloads = {
        _display_split_name(split.name): _run_liquidation_split(
            fresh=fresh,
            specs=specs,
            arrays=arrays,
            split=split,
            leverage=leverage,
            model=model,
        )
        for split in splits
    }
    train = dict(split_payloads["train"]["metrics"])
    val = dict(split_payloads[VALIDATION_SPLIT]["metrics"])
    oos = dict(split_payloads["oos"]["metrics"])
    oos_return = _safe_float(oos.get("total_return"))
    oos_mdd = _safe_float(oos.get("max_drawdown"), 1.0)
    return_risk = tuner._return_risk_score(oos_return, oos_mdd)
    result = {
        "name": f"current_base_liquidation_aware_{float(leverage):g}x",
        "mode": "train_val_monthly_return_budget",
        "sleeves": list(CURRENT_BASE_SLEEVES),
        "leverage": float(leverage),
        "splits": split_payloads,
        "return_quality": {
            "train_monthlyized_return": _monthlyized_return(tuner, train),
            "validation_monthlyized_return": _monthlyized_return(tuner, val),
            "val_monthlyized_return": _monthlyized_return(tuner, val),
            "oos_monthlyized_return": _monthlyized_return(tuner, oos),
            "locked_oos_total_return": oos_return,
            "locked_oos_max_drawdown": oos_mdd,
            "locked_oos_return_risk": return_risk,
            "locked_oos_sharpe": _safe_float(oos.get("sharpe")),
            "locked_oos_sortino": _safe_float(oos.get("sortino")),
            "locked_oos_smart_sortino": _smart_sortino(tuner, oos),
            "locked_oos_calmar": _safe_float(oos.get("calmar")),
        },
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
        },
    }
    result["train_val_score"] = _train_val_score(tuner, result)
    result["liquidation_gates"] = _liquidation_promotion_gates(result)
    return result


def _performance_gates_against_current_base(
    *,
    tuner: Any,
    result: Mapping[str, Any],
    current_base_result: Mapping[str, Any],
) -> dict[str, bool]:
    oos = dict(_split_payload(dict(result.get("splits") or {}), "oos").get("metrics") or {})
    base_oos = dict(_split_payload(dict(current_base_result.get("splits") or {}), "oos").get("metrics") or {})
    oos_return = _safe_float(oos.get("total_return"))
    oos_mdd = _safe_float(oos.get("max_drawdown"), 1.0)
    base_return = _safe_float(base_oos.get("total_return"))
    base_mdd = _safe_float(base_oos.get("max_drawdown"), 1.0)
    return_risk = tuner._return_risk_score(oos_return, oos_mdd)
    base_return_risk = tuner._return_risk_score(base_return, base_mdd)
    smart_sortino = _smart_sortino(tuner, oos)
    return {
        "oos_mdd_within_25pct_budget": oos_mdd <= tuner.MAX_ACCEPTABLE_OOS_MDD,
        "oos_return_beats_current_base": oos_return > base_return,
        "oos_return_risk_beats_current_base": return_risk > base_return_risk,
        "oos_sharpe_high": _safe_float(oos.get("sharpe")) >= tuner.SUCCESS_SHARPE,
        "oos_sortino_high": _safe_float(oos.get("sortino")) >= tuner.SUCCESS_SORTINO,
        "oos_smart_sortino_high": smart_sortino >= tuner.SUCCESS_SMART_SORTINO,
        "oos_calmar_high": _safe_float(oos.get("calmar")) >= tuner.SUCCESS_CALMAR,
    }


def _comparison_to_current_base(tuner: Any, result: Mapping[str, Any], current_base_result: Mapping[str, Any]) -> dict[str, Any]:
    oos = dict(_split_payload(dict(result.get("splits") or {}), "oos").get("metrics") or {})
    base_oos = dict(_split_payload(dict(current_base_result.get("splits") or {}), "oos").get("metrics") or {})
    oos_return = _safe_float(oos.get("total_return"))
    oos_mdd = _safe_float(oos.get("max_drawdown"), 1.0)
    base_return = _safe_float(base_oos.get("total_return"))
    base_mdd = _safe_float(base_oos.get("max_drawdown"), 1.0)
    return_risk = tuner._return_risk_score(oos_return, oos_mdd)
    base_return_risk = tuner._return_risk_score(base_return, base_mdd)
    return {
        "baseline": "liquidation_aware_current_base_replay",
        "current_base_leverage": _safe_float(current_base_result.get("leverage")),
        "candidate_leverage": _safe_float(result.get("leverage")),
        "candidate_oos_return": oos_return,
        "current_base_oos_return": base_return,
        "candidate_oos_mdd": oos_mdd,
        "current_base_oos_mdd": base_mdd,
        "candidate_oos_return_risk": return_risk,
        "current_base_oos_return_risk": base_return_risk,
        "oos_return_delta": oos_return - base_return,
        "oos_return_risk_delta": return_risk - base_return_risk,
    }


def _apply_reference_gates(tuner: Any, results: list[dict[str, Any]], current_base_result: Mapping[str, Any]) -> None:
    for result in results:
        result["comparison_to_current_base"] = _comparison_to_current_base(tuner, result, current_base_result)
        result["performance_gates"] = _performance_gates_against_current_base(
            tuner=tuner,
            result=result,
            current_base_result=current_base_result,
        )
        result["deployable_success"] = bool(
            _liquidation_safe_for_promotion(dict(result.get("liquidation_gates") or {}))
            and all(bool(item) for item in dict(result.get("performance_gates") or {}).values())
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _markdown(payload: Mapping[str, Any]) -> str:
    forced = dict(payload.get("forced_5x") or {})
    selected = dict(payload.get("selected_by_train_validation") or {})
    current = dict(payload.get("current_base_reference_result") or {})
    decision = dict(payload.get("decision") or {})

    def _split_line(item: Mapping[str, Any], split: str) -> str:
        split_payload = _split_payload(dict(item.get("splits") or {}), split)
        metrics = dict(split_payload.get("metrics") or {})
        return (
            f"- {split}: return `{_safe_float(metrics.get('total_return')):+.4%}`, "
            f"MDD `{_safe_float(metrics.get('max_drawdown')):.4%}`, "
            f"liq `{int(split_payload.get('liquidation_count') or 0)}`, "
            f"min buffer `{_safe_float(split_payload.get('minimum_margin_buffer')):.4f}`, "
            f"min ratio `{_safe_float(split_payload.get('minimum_margin_ratio')):.4f}`"
        )

    lines = [
        "# Profit moonshot liquidation-aware current-base validation",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        f"- decision outcome: `{decision.get('outcome')}`",
        f"- deployable improvement: `{bool(decision.get('deployable_improvement'))}`",
        f"- memory peak RSS: `{_safe_float((payload.get('memory_summary') or {}).get('peak_rss_mib')):.3f} MiB`",
        "",
        "## Margin model",
        "",
        f"- mode: `{(payload.get('margin_model') or {}).get('margin_mode')}`",
        f"- maintenance margin rate: `{_safe_float((payload.get('margin_model') or {}).get('maintenance_margin_rate')):.4%}`",
        f"- stress/funding/fee reserve: `{_safe_float((payload.get('margin_model') or {}).get('liquidation_reserve_rate')):.4%}`",
        "- Binance docs references recorded in JSON under `source_references`.",
        "",
        "## Current base reference replay",
        "",
        f"- leverage: `{_safe_float(current.get('leverage')):.6f}x`",
        _split_line(current, "oos") if current else "- missing current base replay",
        "",
        "## Forced 5x replay",
        "",
        f"- deployable_success: `{bool(forced.get('deployable_success'))}`",
        f"- train/validation score: `{_safe_float(forced.get('train_val_score')):.6f}`",
        f"- OOS return delta vs current-base replay: `{_safe_float((forced.get('comparison_to_current_base') or {}).get('oos_return_delta')):+.4%}`",
        f"- OOS return/MDD delta vs current-base replay: `{_safe_float((forced.get('comparison_to_current_base') or {}).get('oos_return_risk_delta')):+.6f}`",
        _split_line(forced, "train") if forced else "- missing 5x replay",
        _split_line(forced, VALIDATION_SPLIT) if forced else "",
        _split_line(forced, "oos") if forced else "",
        "",
        "## Selected by train/validation safety",
        "",
        f"- leverage: `{_safe_float(selected.get('leverage')):.6f}x`",
        f"- locked-OOS used for selection: `{bool((selected.get('selection_policy') or {}).get('uses_locked_oos_for_selection'))}`",
        "",
        "## Decision",
        "",
        f"- `{decision.get('summary')}`",
    ]
    return "\n".join(lines).strip() + "\n"


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    fresh = _load_module(FRESH_PATH, "replay_profit_moonshot_fresh_start")
    tuner = _load_module(TUNER_PATH, "tune_profit_moonshot_fresh_portfolio")
    model = MarginModel(
        margin_mode=str(args.margin_mode),
        maintenance_margin_rate=float(args.maintenance_margin_rate),
        taker_fee_rate=float(args.taker_fee_rate),
        slippage_rate=float(args.slippage_rate),
        funding_rate_per_8h=float(args.funding_rate_per_8h),
        stress_buffer_rate=float(args.stress_buffer_rate),
        liquidation_fee_rate=float(args.liquidation_fee_rate),
    )
    current_base = _load_json(Path(args.current_base_artifact))
    integer_audit = _load_json(Path(args.integer_audit_artifact))
    oos_end = datetime.fromisoformat(str(args.oos_end_date)).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
    )
    arrays = fresh._build_arrays(panel, symbols)
    specs_by_name = {spec.name: spec for spec in fresh._candidate_specs(arrays, symbols)}
    missing = [name for name in CURRENT_BASE_SLEEVES if name not in specs_by_name]
    if missing:
        raise RuntimeError(f"missing current-base sleeve specs: {missing}")
    specs = [specs_by_name[name] for name in CURRENT_BASE_SLEEVES]

    leverage_values = [float(tuner.CURRENT_BASE_LEVERAGE), *[float(item) for item in range(1, int(args.max_leverage) + 1)]]
    results = [
        _build_leverage_result(
            leverage=leverage,
            fresh=fresh,
            tuner=tuner,
            specs=specs,
            arrays=arrays,
            splits=splits,
            model=model,
        )
        for leverage in leverage_values
    ]
    by_leverage = {round(float(item["leverage"]), 9): item for item in results}
    current_base_result = dict(by_leverage.get(round(float(tuner.CURRENT_BASE_LEVERAGE), 9)) or {})
    _apply_reference_gates(tuner, results, current_base_result)
    current_base_result = dict(by_leverage.get(round(float(tuner.CURRENT_BASE_LEVERAGE), 9)) or {})
    integer_grid = [item for item in results if abs(float(item["leverage"]) - round(float(item["leverage"]))) <= 1e-9]
    selected = _select_train_validation_leverage(integer_grid)
    forced_5x = dict(by_leverage.get(5.0) or {})
    zero_liq = [
        item
        for item in integer_grid
        if _liquidation_safe_for_promotion(dict(item.get("liquidation_gates") or {}))
    ]
    highest_zero_liq = max(zero_liq, key=lambda item: float(item.get("leverage") or 0.0), default={})
    forced_5x_success = bool(forced_5x.get("deployable_success"))
    selected_success = bool(selected.get("deployable_success"))
    outcome = (
        "current_base_5x_deployable_improvement"
        if forced_5x_success
        else "alternate_integer_leverage_deployable"
        if selected_success
        else "current_base_retained_liquidation_or_performance_gate_failed"
    )
    summary = (
        "Forced current-base 5x passes liquidation and performance gates."
        if forced_5x_success
        else "Forced current-base 5x is not deployable under the liquidation-aware gate; retain current base unless a train/validation-safe integer row is explicitly accepted."
    )
    payload = {
        "artifact_kind": "profit_moonshot_liquidation_aware_validation",
        "generated_at_utc": _utc_now_iso(),
        "policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
            "forced_candidate": "current_base_integer_5x",
            "integer_grid": [int(item) for item in range(1, int(args.max_leverage) + 1)],
            "promotion_requires_liquidation_count_zero": True,
            "promotion_requires_positive_margin_buffer": True,
            "maximum_oos_mdd": tuner.MAX_ACCEPTABLE_OOS_MDD,
            "memory_budget_bytes": PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
        },
        "source_references": dict(BINANCE_SOURCE_REFS),
        "margin_model": {**asdict(model), "liquidation_reserve_rate": model.liquidation_reserve_rate},
        "current_base_artifact": str(Path(args.current_base_artifact)),
        "integer_audit_artifact": str(Path(args.integer_audit_artifact)),
        "current_base_source_metrics": dict(current_base.get("metrics") or {}),
        "baseline_preservation": {
            "pushed_green_handoff_head": "77f10d54174628c24f1a6bbba34a74505a2a40b5",
            "performance_baseline_commit": "02f4520cf906f48089b8852c2651a0f1e4bd0c1c",
            "comparison_baseline": "liquidation_aware_current_base_replay",
            "baseline_preserved": True,
        },
        "integer_audit_forced_current_base_row": dict(
            ((integer_audit.get("runs") or {}).get("alpha_v2_top40") or {}).get("forced_current_base_integer_row") or {}
        ),
        "current_base_reference_result": current_base_result,
        "forced_5x": forced_5x,
        "integer_grid_results": integer_grid,
        "selected_by_train_validation": selected,
        "highest_zero_liquidation_integer": highest_zero_liq,
        "data_metadata": data_metadata,
        "memory_policy": memory_policy_payload(budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES),
        "memory_summary": {
            "peak_rss_mib": _rss_mib(),
            "under_8gib": _rss_mib() * 1024.0 * 1024.0 < PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
        },
        "decision": {
            "outcome": outcome,
            "deployable_improvement": forced_5x_success,
            "selected_integer_deployable": selected_success,
            "current_base_retained": not forced_5x_success,
            "summary": summary,
        },
    }
    return payload


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"liquidation_aware_current_base_{timestamp}.json"
    md_path = output_dir / f"liquidation_aware_current_base_{timestamp}.md"
    latest_json = output_dir / "liquidation_aware_current_base_latest.json"
    latest_md = output_dir / "liquidation_aware_current_base_latest.md"
    text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    markdown = _markdown(payload)
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return {"json": latest_json, "markdown": latest_md, "timestamped_json": json_path, "timestamped_markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(DEFAULT_MARKET_ROOT))
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--current-base-artifact", default=str(DEFAULT_CURRENT_BASE_ARTIFACT))
    parser.add_argument("--integer-audit-artifact", default=str(DEFAULT_INTEGER_AUDIT_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-leverage", type=int, default=6)
    parser.add_argument("--margin-mode", default="cross")
    parser.add_argument("--maintenance-margin-rate", type=float, default=0.01)
    parser.add_argument("--taker-fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--funding-rate-per-8h", type=float, default=0.0001)
    parser.add_argument("--stress-buffer-rate", type=float, default=0.0025)
    parser.add_argument("--liquidation-fee-rate", type=float, default=0.005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    memory_guard = acquire_portfolio_memory_guard(
        run_name=RUN_NAME,
        output_dir=output_dir,
        input_path=args.current_base_artifact,
        metadata={
            "script": Path(__file__).name,
            "current_base_artifact": str(args.current_base_artifact),
            "integer_audit_artifact": str(args.integer_audit_artifact),
            "max_leverage": int(args.max_leverage),
            "selection": "train_validation_only",
            "locked_oos": "report_only_gate_only",
        },
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    finalized = False
    try:
        memory_guard.checkpoint("start", {"output_dir": str(output_dir)})
        payload = run_validation(args)
        paths = write_outputs(payload, output_dir)
        memory_summary = memory_guard.finalize(
            status="completed",
            context={key: str(value) for key, value in paths.items()},
        )
        finalized = True
        memory_summary["summary_path"] = str(memory_guard.summary_path)
        payload = dict(payload)
        payload["memory_summary"] = {
            **dict(payload.get("memory_summary") or {}),
            **dict(memory_summary),
            "peak_rss_mib": max(
                _safe_float((payload.get("memory_summary") or {}).get("peak_rss_mib")),
                _safe_float(memory_summary.get("peak_rss_bytes")) / (1024.0 * 1024.0),
            ),
        }
        paths = write_outputs(payload, output_dir)
    except Exception as exc:
        if not finalized:
            memory_guard.finalize(status="failed", error=str(exc), context={"script": Path(__file__).name})
        raise
    finally:
        memory_guard.release()
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
