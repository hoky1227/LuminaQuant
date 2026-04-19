import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lumina_quant.core.events import MarketEvent, SignalEvent

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lumina_quant"
    / "strategies"
    / "artifact_portfolio_mode.py"
)
SPEC = importlib.util.spec_from_file_location("artifact_portfolio_mode", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_portfolio_mode_strategy_forwards_component_weighted_signals(monkeypatch) -> None:
    class _ChildStrategy:
        required_timeframes = ("1h",)

        def __init__(self, bars, events, **params):
            _ = bars, params
            self.events = events

        def calculate_signals(self, event):
            self.events.put(
                SignalEvent(
                    strategy_id="child",
                    symbol="BNB/USDT",
                    datetime=event.time,
                    signal_type="LONG",
                )
            )

    monkeypatch.setattr(
        MODULE,
        "resolve_portfolio_mode_definition",
        lambda portfolio_mode: MODULE.PortfolioModeDefinition(
            portfolio_mode=portfolio_mode,
            components=(
                MODULE.PortfolioModeComponent(
                    component_id="comp-a",
                    label="component-a",
                    strategy_class="MovingAverageCrossStrategy",
                    symbols=("BNB/USDT",),
                    params={},
                    weight=0.3,
                    source="test",
                ),
            ),
            cash_weight=0.7,
            source_artifacts={},
        ),
    )
    monkeypatch.setattr(MODULE, "resolve_strategy_class", lambda name, default_name=None: _ChildStrategy)

    events = []
    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0),
        events=SimpleNamespace(put=lambda item: events.append(item)),
        portfolio_mode="hybrid_guarded_mode",
    )
    strategy.calculate_signals(
        MarketEvent(
            time="2026-04-17T00:00:00Z",
            symbol="BNB/USDT",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
    )

    assert len(events) == 1
    signal = events[0]
    assert signal.metadata["component_id"] == "comp-a"
    assert signal.metadata["target_allocation_scale"] == 0.3
    assert signal.client_order_id.startswith("LQPM-") or signal.client_order_id.startswith("comp-a-")


def test_resolve_portfolio_mode_definition_supports_recursive_allocator_sleeves(monkeypatch, tmp_path: Path) -> None:
    def _write(path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    incumbent_path = _write(
        tmp_path / "incumbent.json",
        {
            "weights": [
                {
                    "candidate_id": "leaf_a",
                    "name": "leaf_a",
                    "strategy_class": "MovingAverageCrossStrategy",
                    "symbols": ["BTC/USDT"],
                    "weight": 0.42,
                    "weight_share": 0.6,
                },
                {
                    "candidate_id": "leaf_b",
                    "name": "leaf_b",
                    "strategy_class": "RsiStrategy",
                    "symbols": ["ETH/USDT"],
                    "weight": 0.28,
                    "weight_share": 0.4,
                },
            ],
            "cash_weight": 0.3,
        },
    )
    autoresearch_path = _write(
        tmp_path / "autoresearch.json",
        {
            "weights": [
                {
                    "candidate_id": "leaf_c",
                    "name": "leaf_c",
                    "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                    "symbols": ["SOL/USDT"],
                    "weight": 1.0,
                }
            ]
        },
    )
    blend_path = _write(
        tmp_path / "blend.json",
        {
            "weights": [
                {"candidate_id": "incumbent_only", "name": "incumbent_only", "weight": 0.7},
                {"candidate_id": "autoresearch_55_45", "name": "autoresearch_55_45", "weight": 0.3},
            ]
        },
    )
    soft_path = _write(
        tmp_path / "soft.json",
        {
            "current_state": {
                "weights": {
                    "incumbent": 0.5,
                    "blend_85_15": 0.5,
                    "autoresearch_55_45": 0.0,
                }
            }
        },
    )
    three_way_path = _write(
        tmp_path / "three.json",
        {
            "current_state": {
                "weights": {
                    "incumbent": 0.0,
                    "blend_85_15": 1.0,
                    "autoresearch_55_45": 0.0,
                }
            }
        },
    )
    pair_path = _write(
        tmp_path / "pair.json",
        {
            "candidate_id": "leaf_pair",
            "name": "leaf_pair",
            "strategy_class": "PairSpreadZScoreStrategy",
            "symbols": ["BNB/USDT", "TRX/USDT"],
        },
    )
    hybrid_path = _write(
        tmp_path / "hybrid.json",
        {
            "scenarios": {
                "refreshed_latest_tail": {
                    "final_allocation": {
                        "weights": {
                            "soft_three_way_regime": 0.4,
                            "balanced_overlay_80_20": 0.3,
                            "pair_tactical_mode": 0.1,
                        },
                        "cash_weight": 0.2,
                    }
                }
            }
        },
    )

    monkeypatch.setattr(MODULE, "REFRESHED_INCUMBENT_PATH", incumbent_path)
    monkeypatch.setattr(MODULE, "REFRESHED_AUTORESEARCH_55_45_PATH", autoresearch_path)
    monkeypatch.setattr(MODULE, "REFRESHED_BLEND_PATH", blend_path)
    monkeypatch.setattr(MODULE, "SOFT_THREE_WAY_ALLOCATOR_PATH", soft_path)
    monkeypatch.setattr(MODULE, "THREE_WAY_ALLOCATOR_PATH", three_way_path)
    monkeypatch.setattr(MODULE, "PAIR_TACTICAL_PATH", pair_path)
    monkeypatch.setattr(MODULE, "HYBRID_PATH", hybrid_path)
    monkeypatch.setattr(MODULE, "PRODUCTION_GUARDED_PATH", _write(
        tmp_path / "production_guarded.json",
        {
            "weights": [
                {"candidate_id": "incumbent_only", "name": "incumbent_only", "weight": 0.4},
                {"candidate_id": "blend_85_15", "name": "blend_85_15", "weight": 0.35},
                {"candidate_id": "autoresearch_55_45", "name": "autoresearch_55_45", "weight": 0.2},
            ],
            "cash_weight": 0.05,
        },
    ))
    monkeypatch.setattr(MODULE, "STRICT_AUTORESEARCH_1X_PATH", autoresearch_path)

    defensive = MODULE.resolve_portfolio_mode_definition("defensive_overlay_mode")
    aggressive = MODULE.resolve_portfolio_mode_definition("aggressive_realized_mode")
    hybrid = MODULE.resolve_portfolio_mode_definition("hybrid_guarded_mode")
    practical = MODULE.resolve_portfolio_mode_definition("strict_autoresearch_practical_mode")
    risk_off = MODULE.resolve_portfolio_mode_definition("risk_off_mode")

    defensive_weights = {item.component_id: round(item.weight, 6) for item in defensive.components}
    aggressive_weights = {item.component_id: round(item.weight, 6) for item in aggressive.components}
    hybrid_weights = {item.component_id: round(item.weight, 6) for item in hybrid.components}
    practical_weights = {item.component_id: round(item.weight, 6) for item in practical.components}

    assert defensive_weights == {
        "leaf_a": 0.357,
        "leaf_b": 0.238,
        "leaf_c": 0.105,
        "leaf_pair": 0.3,
    }
    assert aggressive_weights == {
        "leaf_a": 0.42,
        "leaf_b": 0.28,
        "leaf_c": 0.3,
    }
    assert aggressive.cash_weight == 0.21
    assert hybrid_weights == {
        "leaf_a": 0.3264,
        "leaf_b": 0.2176,
        "leaf_c": 0.096,
        "leaf_pair": 0.16,
    }
    assert practical_weights == {
        "leaf_a": 0.3096,
        "leaf_b": 0.2064,
        "leaf_c": 0.444,
    }
    assert abs(hybrid.cash_weight - 0.3632) < 1e-12
    assert abs(practical.cash_weight - 0.1948) < 1e-6
    assert risk_off.cash_weight == 1.0
    assert risk_off.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT"]
