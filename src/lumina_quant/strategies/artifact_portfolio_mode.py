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
HYBRID_PATH = GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
GROUPED_SOFT_PATH = GROUP_ROOT / "grouped_incumbent_autoresearch_portfolio_latest.json"
PAIR_TACTICAL_PATH = (
    GROUP_ROOT / "current_switch_validation_current" / "refreshed_pair_fast_exit_candidate_latest.json"
)
INCUMBENT_BUNDLE_PATH = FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.json"
AUTORESEARCH_OPT_PATH = FOLLOWUP_ROOT / "autoresearch_candidate_portfolio_opt" / "portfolio_optimization_latest.json"


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

    @property
    def symbols(self) -> list[str]:
        ordered: list[str] = []
        for component in self.components:
            for symbol in component.symbols:
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


def _incumbent_component_lookup() -> dict[str, dict[str, Any]]:
    payload = _read_json(INCUMBENT_BUNDLE_PATH)
    rows = []
    rows.extend([dict(item) for item in list(payload.get("selected_team") or []) if isinstance(item, dict)])
    rows.extend([dict(item) for item in list(payload.get("candidates") or []) if isinstance(item, dict)])
    return {
        str(row.get("candidate_id") or row.get("name")): row
        for row in rows
        if str(row.get("candidate_id") or row.get("name")).strip()
    }


def _autoresearch_component_lookup() -> dict[str, dict[str, Any]]:
    if not AUTORESEARCH_OPT_PATH.exists():
        return {}
    payload = _read_json(AUTORESEARCH_OPT_PATH)
    return {
        str(row.get("candidate_id") or row.get("name")): dict(row)
        for row in list(payload.get("weights") or [])
        if isinstance(row, dict) and str(row.get("candidate_id") or row.get("name")).strip()
    }


def _portfolio_candidate_lookup() -> dict[str, dict[str, Any]]:
    merged = _incumbent_component_lookup()
    merged.update(_autoresearch_component_lookup())
    return merged


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


def _soft_sleeve_definition() -> tuple[list[PortfolioModeComponent], float]:
    payload = _read_json(GROUPED_SOFT_PATH)
    exposures = [dict(item) for item in list(payload.get("final_component_exposure") or []) if isinstance(item, dict)]
    candidate_lookup = _portfolio_candidate_lookup()
    components: list[PortfolioModeComponent] = []
    invested_total = 0.0
    for exposure in exposures:
        weight = _safe_float(exposure.get("weight"), 0.0)
        if weight <= 0.0:
            continue
        candidate_id = str(exposure.get("candidate_id") or exposure.get("source_candidate_id") or exposure.get("name") or "")
        row = candidate_lookup.get(candidate_id)
        if row is None:
            raise ValueError(f"missing candidate artifact for grouped soft sleeve component: {candidate_id}")
        components.append(
            _component_from_row(
                row,
                weight=weight,
                source=f"grouped_soft:{candidate_id}",
            )
        )
        invested_total += weight
    cash_weight = max(0.0, 1.0 - invested_total)
    return components, cash_weight


def _pair_component(weight: float) -> PortfolioModeComponent:
    row = _read_json(PAIR_TACTICAL_PATH)
    return _component_from_row(row, weight=weight, source="pair_tactical")


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

    soft_components, soft_cash = _soft_sleeve_definition()
    pair_component = _pair_component(1.0)
    source_artifacts = {
        "hybrid_path": str(HYBRID_PATH.resolve()),
        "grouped_soft_path": str(GROUPED_SOFT_PATH.resolve()),
        "pair_tactical_path": str(PAIR_TACTICAL_PATH.resolve()),
        "incumbent_bundle_path": str(INCUMBENT_BUNDLE_PATH.resolve()),
    }

    components: list[PortfolioModeComponent] = []
    cash_weight = 0.0

    if token == "risk_off_mode":
        cash_weight = 1.0
    elif token == "pair_tactical_mode":
        components.append(_pair_component(1.0))
    elif token == "core_mode":
        components.extend(soft_components)
        cash_weight = soft_cash
    elif token == "balanced_overlay_mode":
        components.extend(
            PortfolioModeComponent(
                component_id=item.component_id,
                label=item.label,
                strategy_class=item.strategy_class,
                symbols=item.symbols,
                params=dict(item.params),
                weight=float(item.weight * 0.8),
                source=f"{item.source}:balanced80",
            )
            for item in soft_components
        )
        components.append(
            PortfolioModeComponent(
                component_id=pair_component.component_id,
                label=pair_component.label,
                strategy_class=pair_component.strategy_class,
                symbols=pair_component.symbols,
                params=dict(pair_component.params),
                weight=0.2,
                source="pair_tactical:balanced20",
            )
        )
        cash_weight = soft_cash * 0.8
    elif token == "hybrid_guarded_mode":
        hybrid_payload = _read_json(HYBRID_PATH)
        final_allocation = dict(
            dict((hybrid_payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get("final_allocation")
            or {}
        )
        sleeve_weights = {str(key): _safe_float(value, 0.0) for key, value in dict(final_allocation.get("weights") or {}).items()}
        cash_weight = _safe_float(final_allocation.get("cash_weight"), 0.0)
        for sleeve_name, sleeve_weight in sleeve_weights.items():
            if sleeve_weight <= 0.0:
                continue
            if sleeve_name == "balanced_overlay_80_20":
                components.extend(
                    PortfolioModeComponent(
                        component_id=item.component_id,
                        label=item.label,
                        strategy_class=item.strategy_class,
                        symbols=item.symbols,
                        params=dict(item.params),
                        weight=float(item.weight * 0.8 * sleeve_weight),
                        source=f"{item.source}:hybrid-balanced",
                    )
                    for item in soft_components
                )
                components.append(
                    PortfolioModeComponent(
                        component_id=pair_component.component_id,
                        label=pair_component.label,
                        strategy_class=pair_component.strategy_class,
                        symbols=pair_component.symbols,
                        params=dict(pair_component.params),
                        weight=float(0.2 * sleeve_weight),
                        source="pair_tactical:hybrid-balanced",
                    )
                )
                cash_weight += soft_cash * 0.8 * sleeve_weight
            elif sleeve_name == "soft_three_way_regime":
                components.extend(
                    PortfolioModeComponent(
                        component_id=item.component_id,
                        label=item.label,
                        strategy_class=item.strategy_class,
                        symbols=item.symbols,
                        params=dict(item.params),
                        weight=float(item.weight * sleeve_weight),
                        source=f"{item.source}:hybrid-soft",
                    )
                    for item in soft_components
                )
                cash_weight += soft_cash * sleeve_weight
            elif sleeve_name == "pair_tactical_mode":
                components.append(
                    PortfolioModeComponent(
                        component_id=pair_component.component_id,
                        label=pair_component.label,
                        strategy_class=pair_component.strategy_class,
                        symbols=pair_component.symbols,
                        params=dict(pair_component.params),
                        weight=float(sleeve_weight),
                        source="pair_tactical:hybrid-direct",
                    )
                )
            elif sleeve_name == "risk_off_cash":
                cash_weight += sleeve_weight
            else:
                raise ValueError(f"unsupported hybrid sleeve in artifact-driven live mode: {sleeve_name}")
    else:
        raise ValueError(f"unsupported live portfolio mode: {token}")

    merged = tuple(_merge_components([item for item in components if item.weight > 1e-12]))
    return PortfolioModeDefinition(
        portfolio_mode=token,
        components=merged,
        cash_weight=float(max(0.0, min(1.0, cash_weight))),
        source_artifacts=source_artifacts,
    )


def supported_portfolio_modes() -> set[str]:
    return {
        "hybrid_guarded_mode",
        "balanced_overlay_mode",
        "core_mode",
        "pair_tactical_mode",
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
