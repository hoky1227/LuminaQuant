from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT
from lumina_quant.strategy import Strategy
from lumina_quant.strategies import resolve_strategy_class

GROUP_ROOT = FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped"
REFRESH_ROOT = GROUP_ROOT / "current_switch_validation_current"
HYBRID_PATH = GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
LEGACY_NO_HIGHVOL_HYBRID_PATH = (
    GROUP_ROOT
    / "legacy_metric_live_materialization_20260426"
    / "legacy_metric_live_materialization_latest.json"
)
RETUNED_LIVE_PORTFOLIO_HYBRID_PATH = (
    GROUP_ROOT
    / "live_portfolio_hybrid_retune_20260426"
    / "live_portfolio_hybrid_retune_latest.json"
)
WAVE2_PAIR_PATH = (
    GROUP_ROOT
    / "legacy_metric_live_materialization_20260426"
    / "wave2_pair_live_candidate_latest.json"
)
REFRESHED_INCUMBENT_PATH = REFRESH_ROOT / "refreshed_current_one_shot_incumbent_portfolio_latest.json"
REFRESHED_BLEND_PATH = REFRESH_ROOT / "refreshed_grouped_static_blend_latest.json"
REFRESHED_AUTORESEARCH_55_45_PATH = (
    REFRESH_ROOT / "refreshed_autoresearch_pair_55_45_portfolio_latest.json"
)
SOFT_THREE_WAY_ALLOCATOR_PATH = (
    REFRESH_ROOT / "refreshed_soft_three_way_allocator_current" / "soft_three_way_market_regime_allocator_latest.json"
)
THREE_WAY_ALLOCATOR_PATH = (
    REFRESH_ROOT / "refreshed_three_way_allocator_current" / "three_way_market_regime_allocator_latest.json"
)
PAIR_TACTICAL_PATH = (
    REFRESH_ROOT / "refreshed_pair_fast_exit_candidate_latest.json"
)
PRODUCTION_GUARDED_PATH = (
    GROUP_ROOT / "portfolio_production_guarded_current" / "production_guarded_portfolio_latest.json"
)
STATE_VWAP_PAIR_PATH = (
    GROUP_ROOT
    / "portfolio_superiority_dense_pairs_current"
    / "state_vwap_pair_candidate_latest.json"
)
STRICT_AUTORESEARCH_1X_PATH = (
    GROUP_ROOT
    / "strict_blend_76_24_leverage_sweep_rerun_current"
    / "inc_1_auto_1"
    / "strict_autoresearch_portfolio_current"
    / "strict_autoresearch_portfolio_latest.json"
)


@dataclass(frozen=True, slots=True)
class PortfolioModeComponent:
    component_id: str
    label: str
    strategy_class: str
    symbols: tuple[str, ...]
    params: dict[str, Any]
    weight: float
    source: str


@dataclass(frozen=True, slots=True)
class PortfolioModeDefinition:
    portfolio_mode: str
    components: tuple[PortfolioModeComponent, ...]
    cash_weight: float
    source_artifacts: dict[str, str]
    watch_symbols: tuple[str, ...] = ()

    @property
    def symbols(self) -> list[str]:
        ordered: list[str] = []
        for component in self.components:
            for symbol in component.symbols:
                if symbol not in ordered:
                    ordered.append(symbol)
        if not ordered:
            for symbol in self.watch_symbols:
                if symbol not in ordered:
                    ordered.append(symbol)
        return ordered


class _BarsSubsetProxy:
    def __init__(self, bars: Any, symbols: list[str]) -> None:
        self._bars = bars
        self.symbol_list = list(symbols or [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bars, name)


class _SignalCaptureQueue:
    def __init__(self) -> None:
        self._items: deque[Any] = deque()

    def put(self, item: Any) -> None:
        self._items.append(item)

    def drain(self) -> list[Any]:
        out = list(self._items)
        self._items.clear()
        return out


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _ordered_unique(items: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for item in items:
        token = str(item).strip()
        if token and token not in ordered:
            ordered.append(token)
    return tuple(ordered)


def _component_from_row(
    row: dict[str, Any],
    *,
    weight: float,
    source: str,
) -> PortfolioModeComponent:
    strategy_class = str(row.get("strategy_class") or "").strip()
    if not strategy_class:
        raise ValueError(f"component row is missing strategy_class: {row}")
    params = dict(row.get("params") or {})
    return PortfolioModeComponent(
        component_id=str(row.get("candidate_id") or row.get("name") or strategy_class),
        label=str(row.get("name") or strategy_class),
        strategy_class=strategy_class,
        symbols=tuple(str(symbol) for symbol in list(row.get("symbols") or []) if str(symbol).strip()),
        params=params,
        weight=float(weight),
        source=source,
    )


def _pair_component(weight: float) -> PortfolioModeComponent:
    row = _read_json(PAIR_TACTICAL_PATH)
    return _component_from_row(row, weight=weight, source="pair_tactical")


def _state_vwap_pair_component(weight: float) -> PortfolioModeComponent:
    row = _read_json(STATE_VWAP_PAIR_PATH)
    return _component_from_row(row, weight=weight, source="state_vwap_pair")


def _wave2_pair_component(weight: float) -> PortfolioModeComponent:
    row = _read_json(WAVE2_PAIR_PATH)
    return _component_from_row(row, weight=weight, source="wave2_pair")


def _portfolio_weight_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = [dict(item) for item in list(payload.get("weights") or []) if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        if "weight_share" in normalized:
            normalized["weight"] = _safe_float(normalized.get("weight_share"), 0.0)
        else:
            normalized["weight"] = _safe_float(normalized.get("weight"), 0.0)
        out.append(normalized)
    cash_weight = _safe_float(payload.get("cash_weight"), 0.0)
    if cash_weight > 1e-12:
        out.append({"candidate_id": "cash", "name": "cash", "weight": cash_weight})
    return out


def _state_weight_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    state_weights = dict(dict(payload.get("current_state") or {}).get("weights") or {})
    out: list[dict[str, Any]] = []
    for name, weight in state_weights.items():
        out.append(
            {
                "candidate_id": str(name),
                "name": str(name),
                "weight": _safe_float(weight, 0.0),
            }
        )
    return out


def _hybrid_weight_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    final_allocation = dict(
        dict((payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get("final_allocation")
        or {}
    )
    state_weights = dict(final_allocation.get("weights") or {})
    out: list[dict[str, Any]] = []
    for name, weight in state_weights.items():
        out.append(
            {
                "candidate_id": str(name),
                "name": str(name),
                "weight": _safe_float(weight, 0.0),
            }
        )
    cash_weight = _safe_float(final_allocation.get("cash_weight"), 0.0)
    if cash_weight > 1e-12:
        out.append({"candidate_id": "cash", "name": "cash", "weight": cash_weight})
    return out


def _alias_rows(token: str) -> list[dict[str, Any]] | None:
    portfolio_paths = {
        "incumbent": REFRESHED_INCUMBENT_PATH,
        "incumbent_only": REFRESHED_INCUMBENT_PATH,
        "autoresearch_55_45": REFRESHED_AUTORESEARCH_55_45_PATH,
        "blend_85_15": REFRESHED_BLEND_PATH,
        "static_blend_76_24": REFRESHED_BLEND_PATH,
        "production_guarded_portfolio": PRODUCTION_GUARDED_PATH,
        "strict_autoresearch_1x": STRICT_AUTORESEARCH_1X_PATH,
    }
    state_paths = {
        "soft_three_way_regime": SOFT_THREE_WAY_ALLOCATOR_PATH,
        "three_way_regime": THREE_WAY_ALLOCATOR_PATH,
    }
    hybrid_paths = {
        "hybrid_guarded_mode": HYBRID_PATH,
        "legacy_no_highvol_hybrid_mode": LEGACY_NO_HIGHVOL_HYBRID_PATH,
        "retuned_live_portfolio_hybrid_mode": RETUNED_LIVE_PORTFOLIO_HYBRID_PATH,
    }
    synthetic_rows = {
        "core_mode": [
            {"candidate_id": "soft_three_way_regime", "name": "soft_three_way_regime", "weight": 1.0},
        ],
        "balanced_overlay_mode": [
            {"candidate_id": "balanced_overlay_80_20", "name": "balanced_overlay_80_20", "weight": 1.0},
        ],
        "defensive_overlay_mode": [
            {"candidate_id": "soft_three_way_regime", "name": "soft_three_way_regime", "weight": 0.7},
            {"candidate_id": "pair_tactical_mode", "name": "pair_tactical_mode", "weight": 0.3},
        ],
        "aggressive_realized_mode": [
            {"candidate_id": "three_way_regime", "name": "three_way_regime", "weight": 1.0},
        ],
        "risk_off_mode": [
            {"candidate_id": "cash", "name": "cash", "weight": 1.0},
        ],
        "balanced_overlay_80_20": [
            {"candidate_id": "soft_three_way_regime", "name": "soft_three_way_regime", "weight": 0.8},
            {"candidate_id": "pair_fast_exit", "name": "pair_fast_exit", "weight": 0.2},
        ],
        "pair_tactical_mode": [
            {"candidate_id": "pair_fast_exit", "name": "pair_fast_exit", "weight": 1.0},
        ],
        "strict_autoresearch_practical_mode": [
            {"candidate_id": "production_guarded_portfolio", "name": "production_guarded_portfolio", "weight": 0.8},
            {"candidate_id": "strict_autoresearch_1x", "name": "strict_autoresearch_1x", "weight": 0.2},
        ],
        "production_guarded_state_vwap_pair_mode": [
            {"candidate_id": "production_guarded_portfolio", "name": "production_guarded_portfolio", "weight": 0.4},
            {"candidate_id": "state_vwap_pair_leaf", "name": "state_vwap_pair_leaf", "weight": 0.25},
            {"candidate_id": "cash", "name": "cash", "weight": 0.35},
        ],
        "state_vwap_pair": [
            {"candidate_id": "state_vwap_pair_leaf", "name": "state_vwap_pair_leaf", "weight": 1.0},
        ],
        "wave2_pair": [
            {"candidate_id": "wave2_pair_leaf", "name": "wave2_pair_leaf", "weight": 1.0},
        ],
        "pair_fast_exit": [
            {"candidate_id": "pair_fast_exit_leaf", "name": "pair_fast_exit_leaf", "weight": 1.0},
        ],
    }
    if token in portfolio_paths:
        return _portfolio_weight_rows(portfolio_paths[token])
    if token in state_paths:
        return _state_weight_rows(state_paths[token])
    if token in hybrid_paths:
        return _hybrid_weight_rows(hybrid_paths[token])
    if token in synthetic_rows:
        return [dict(item) for item in synthetic_rows[token]]
    return None


def _watch_symbols() -> tuple[str, ...]:
    symbols: list[str] = []
    for path in (REFRESHED_INCUMBENT_PATH, REFRESHED_AUTORESEARCH_55_45_PATH, PAIR_TACTICAL_PATH):
        payload = _read_json(path)
        rows = [payload] if isinstance(payload.get("strategy_class"), str) else list(payload.get("weights") or [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            for symbol in list(row.get("symbols") or []):
                token = str(symbol).strip()
                if token and token not in symbols:
                    symbols.append(token)
    return tuple(symbols)


def _expand_reference(
    token: str,
    *,
    weight_scale: float,
    source: str,
    stack: tuple[str, ...] = (),
) -> tuple[list[PortfolioModeComponent], float]:
    token = str(token or "").strip()
    if not token or weight_scale <= 0.0:
        return [], 0.0
    if token in stack:
        raise ValueError(f"cyclic artifact portfolio reference: {' -> '.join([*stack, token])}")
    if token in {"cash", "risk_off_cash"}:
        return [], float(weight_scale)
    if token in {"pair_fast_exit_leaf"}:
        return [_pair_component(float(weight_scale))], 0.0
    if token in {"state_vwap_pair_leaf"}:
        return [_state_vwap_pair_component(float(weight_scale))], 0.0
    if token in {"wave2_pair_leaf"}:
        return [_wave2_pair_component(float(weight_scale))], 0.0

    rows = _alias_rows(token)
    if rows is None:
        raise ValueError(f"unsupported artifact portfolio reference: {token}")

    components: list[PortfolioModeComponent] = []
    cash_weight = 0.0
    next_stack = (*stack, token)
    for row in rows:
        row_weight = _safe_float(row.get("weight"), 0.0)
        if row_weight <= 0.0:
            continue
        scaled_weight = float(weight_scale) * row_weight
        if str(row.get("strategy_class") or "").strip():
            components.append(
                _component_from_row(
                    row,
                    weight=scaled_weight,
                    source=f"{source}:{token}",
                )
            )
            continue
        child_token = str(row.get("candidate_id") or row.get("name") or "").strip()
        child_components, child_cash = _expand_reference(
            child_token,
            weight_scale=scaled_weight,
            source=f"{source}:{token}",
            stack=next_stack,
        )
        components.extend(child_components)
        cash_weight += child_cash
    return components, cash_weight


def _merge_components(components: list[PortfolioModeComponent]) -> list[PortfolioModeComponent]:
    merged: dict[str, PortfolioModeComponent] = {}
    for component in components:
        existing = merged.get(component.component_id)
        if existing is None:
            merged[component.component_id] = component
            continue
        merged[component.component_id] = PortfolioModeComponent(
            component_id=component.component_id,
            label=component.label,
            strategy_class=component.strategy_class,
            symbols=component.symbols,
            params=dict(component.params),
            weight=float(existing.weight + component.weight),
            source=f"{existing.source}+{component.source}",
        )
    return sorted(merged.values(), key=lambda item: item.weight, reverse=True)


def resolve_portfolio_mode_definition(portfolio_mode: str) -> PortfolioModeDefinition:
    token = str(portfolio_mode or "").strip()
    if not token:
        raise ValueError("portfolio_mode is required")

    source_artifacts = {
        "hybrid_path": str(HYBRID_PATH.resolve()),
        "legacy_no_highvol_hybrid_path": str(LEGACY_NO_HIGHVOL_HYBRID_PATH.resolve()),
        "retuned_live_portfolio_hybrid_path": str(RETUNED_LIVE_PORTFOLIO_HYBRID_PATH.resolve()),
        "refreshed_incumbent_path": str(REFRESHED_INCUMBENT_PATH.resolve()),
        "refreshed_blend_path": str(REFRESHED_BLEND_PATH.resolve()),
        "refreshed_autoresearch_55_45_path": str(REFRESHED_AUTORESEARCH_55_45_PATH.resolve()),
        "soft_three_way_allocator_path": str(SOFT_THREE_WAY_ALLOCATOR_PATH.resolve()),
        "three_way_allocator_path": str(THREE_WAY_ALLOCATOR_PATH.resolve()),
        "pair_tactical_path": str(PAIR_TACTICAL_PATH.resolve()),
        "production_guarded_path": str(PRODUCTION_GUARDED_PATH.resolve()),
        "state_vwap_pair_path": str(STATE_VWAP_PAIR_PATH.resolve()),
        "wave2_pair_path": str(WAVE2_PAIR_PATH.resolve()),
        "strict_autoresearch_1x_path": str(STRICT_AUTORESEARCH_1X_PATH.resolve()),
    }

    components: list[PortfolioModeComponent] = []
    cash_weight = 0.0
    watch_symbols: tuple[str, ...] = ()

    if token == "risk_off_mode":
        cash_weight = 1.0
        watch_symbols = _watch_symbols()
    elif token == "pair_tactical_mode":
        components, cash_weight = _expand_reference("pair_tactical_mode", weight_scale=1.0, source=token)
    elif token == "core_mode":
        components, cash_weight = _expand_reference("soft_three_way_regime", weight_scale=1.0, source=token)
    elif token == "balanced_overlay_mode":
        components, cash_weight = _expand_reference("balanced_overlay_80_20", weight_scale=1.0, source=token)
    elif token == "defensive_overlay_mode":
        soft_components, soft_cash = _expand_reference("soft_three_way_regime", weight_scale=0.7, source=token)
        pair_components, pair_cash = _expand_reference("pair_tactical_mode", weight_scale=0.3, source=token)
        components.extend(soft_components)
        components.extend(pair_components)
        cash_weight = soft_cash + pair_cash
    elif token == "aggressive_realized_mode":
        components, cash_weight = _expand_reference("three_way_regime", weight_scale=1.0, source=token)
    elif token in {
        "hybrid_guarded_mode",
        "legacy_no_highvol_hybrid_mode",
        "retuned_live_portfolio_hybrid_mode",
    }:
        hybrid_paths = {
            "hybrid_guarded_mode": HYBRID_PATH,
            "legacy_no_highvol_hybrid_mode": LEGACY_NO_HIGHVOL_HYBRID_PATH,
            "retuned_live_portfolio_hybrid_mode": RETUNED_LIVE_PORTFOLIO_HYBRID_PATH,
        }
        hybrid_path = hybrid_paths[token]
        hybrid_payload = _read_json(hybrid_path)
        final_allocation = dict(
            dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get("final_allocation")
            or {}
        )
        sleeve_weights = {
            str(key): _safe_float(value, 0.0)
            for key, value in dict(final_allocation.get("weights") or {}).items()
        }
        cash_weight = _safe_float(final_allocation.get("cash_weight"), 0.0)
        for sleeve_name, sleeve_weight in sleeve_weights.items():
            if sleeve_weight <= 0.0:
                continue
            sleeve_components, sleeve_cash = _expand_reference(
                sleeve_name,
                weight_scale=sleeve_weight,
                source=token,
            )
            components.extend(sleeve_components)
            cash_weight += sleeve_cash
    elif token == "strict_autoresearch_practical_mode":
        components, cash_weight = _expand_reference(
            "strict_autoresearch_practical_mode",
            weight_scale=1.0,
            source=token,
        )
    elif token == "production_guarded_state_vwap_pair_mode":
        components, cash_weight = _expand_reference(
            "production_guarded_state_vwap_pair_mode",
            weight_scale=1.0,
            source=token,
        )
    else:
        raise ValueError(f"unsupported live portfolio mode: {token}")

    merged = tuple(_merge_components([item for item in components if item.weight > 1e-12]))
    return PortfolioModeDefinition(
        portfolio_mode=token,
        components=merged,
        cash_weight=float(max(0.0, min(1.0, cash_weight))),
        source_artifacts=source_artifacts,
        watch_symbols=watch_symbols,
    )


def supported_portfolio_modes() -> set[str]:
    return {
        "aggressive_realized_mode",
        "hybrid_guarded_mode",
        "legacy_no_highvol_hybrid_mode",
        "retuned_live_portfolio_hybrid_mode",
        "balanced_overlay_mode",
        "defensive_overlay_mode",
        "core_mode",
        "pair_tactical_mode",
        "production_guarded_state_vwap_pair_mode",
        "strict_autoresearch_practical_mode",
        "risk_off_mode",
    }


class ArtifactPortfolioModeStrategy(Strategy):
    preferred_contract = "market_window"

    def __init__(self, bars, events, *, portfolio_mode: str):
        self.bars = bars
        self.events = events
        self.portfolio_mode = str(portfolio_mode or "").strip()
        self.definition = resolve_portfolio_mode_definition(self.portfolio_mode)
        self.symbol_list = list(self.definition.symbols)
        self.decision_cadence_seconds = 60
        self._children: list[tuple[PortfolioModeComponent, Any, _SignalCaptureQueue]] = []
        required_timeframes: set[str] = set()
        for component in self.definition.components:
            strategy_cls = resolve_strategy_class(component.strategy_class, default_name=component.strategy_class)
            child_queue = _SignalCaptureQueue()
            child_bars = _BarsSubsetProxy(self.bars, list(component.symbols))
            child = strategy_cls(child_bars, child_queue, **dict(component.params))
            raw_timeframes = getattr(child, "required_timeframes", ()) or ()
            required_timeframes.update(str(token) for token in raw_timeframes if str(token).strip())
            self._children.append((component, child, child_queue))
        self.required_timeframes = tuple(sorted(required_timeframes))

    def get_state(self) -> dict[str, Any]:
        return {
            "portfolio_mode": self.portfolio_mode,
            "children": {
                component.component_id: dict(getattr(child, "get_state", lambda: {})() or {})
                for component, child, _queue in self._children
            },
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        raw_children = dict(state.get("children") or {})
        for component, child, _queue in self._children:
            child_state = raw_children.get(component.component_id)
            setter = getattr(child, "set_state", None)
            if callable(setter) and isinstance(child_state, dict):
                setter(child_state)

    def _component_client_order_id(self, *, component: PortfolioModeComponent, signal: SignalEvent) -> str:
        base = str(signal.client_order_id or "").strip()
        if base:
            return f"{component.component_id[:12]}-{base}"
        token = "|".join(
            [
                component.component_id,
                str(signal.symbol),
                str(signal.signal_type),
                str(signal.datetime),
                str(getattr(signal, "position_side", "") or ""),
            ]
        )
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
        return f"LQPM-{digest}"

    def _forward_child_signal(self, component: PortfolioModeComponent, signal: SignalEvent) -> None:
        metadata = dict(signal.metadata or {})
        metadata.update(
            {
                "portfolio_mode": self.portfolio_mode,
                "component_id": component.component_id,
                "component_label": component.label,
                "component_weight": float(component.weight),
                "target_allocation_scale": float(component.weight),
            }
        )
        forwarded = SignalEvent(
            strategy_id=f"artifact_portfolio_mode::{self.portfolio_mode}",
            symbol=str(signal.symbol),
            datetime=signal.datetime,
            signal_type=str(signal.signal_type),
            strength=float(getattr(signal, "strength", 1.0) or 1.0) * float(component.weight),
            price=getattr(signal, "price", None),
            stop_loss=getattr(signal, "stop_loss", None),
            take_profit=getattr(signal, "take_profit", None),
            position_side=getattr(signal, "position_side", None),
            client_order_id=self._component_client_order_id(component=component, signal=signal),
            time_in_force=getattr(signal, "time_in_force", None),
            metadata=metadata,
            trailing_percent=getattr(signal, "trailing_percent", None),
        )
        self.events.put(forwarded)

    def _drain_child_queue(self, component: PortfolioModeComponent, queue: _SignalCaptureQueue) -> None:
        for item in queue.drain():
            if isinstance(item, SignalEvent):
                self._forward_child_signal(component, item)

    def calculate_signals(self, event: Any) -> None:
        for component, child, child_queue in self._children:
            child.calculate_signals(event)
            self._drain_child_queue(component, child_queue)

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        for component, child, child_queue in self._children:
            handler = getattr(child, "calculate_signals_window", None)
            if callable(handler):
                handler(event, aggregator)
            else:
                child.calculate_signals(event)
            self._drain_child_queue(component, child_queue)

    def calculate_signals_context(self, context: Any) -> None:
        for component, child, child_queue in self._children:
            handler = getattr(child, "calculate_signals_context", None)
            if callable(handler):
                handler(context)
            else:
                child.calculate_signals_window(context.event, context.aggregator)
            self._drain_child_queue(component, child_queue)


__all__ = [
    "ArtifactPortfolioModeStrategy",
    "PortfolioModeComponent",
    "PortfolioModeDefinition",
    "resolve_portfolio_mode_definition",
    "supported_portfolio_modes",
]
