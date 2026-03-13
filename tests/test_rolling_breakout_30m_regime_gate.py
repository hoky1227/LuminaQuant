from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "rolling_breakout_30m_regime_gate.py"
SPEC = importlib.util.spec_from_file_location("rolling_breakout_30m_regime_gate", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load rolling_breakout_30m_regime_gate module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TARGET_CANDIDATE = MODULE.TARGET_CANDIDATE
build_rolling_breakout_30m_gate = MODULE.build_rolling_breakout_30m_gate
write_rolling_breakout_30m_gate = MODULE.write_rolling_breakout_30m_gate


def _decision_payload() -> dict[str, object]:
    stream = [
        {"t": 1735689600000.0, "v": -0.0100},
        {"t": 1735776000000.0, "v": -0.0060},
        {"t": 1767225600000.0, "v": 0.0080},
        {"t": 1767312000000.0, "v": 0.0070},
        {"t": 1769904000000.0, "v": 0.0300},
        {"t": 1769990400000.0, "v": 0.0200},
        {"t": 1770076800000.0, "v": -0.0200},
    ]
    return {
        "timeframe_rows": [
            {
                "timeframe": "30m",
                "windows": {
                    "train_start": "2025-01-01T00:00:00Z",
                    "actual_oos_end_exclusive": "2026-03-07T10:00:00.001Z",
                },
                "best_row": {
                    "candidate_id": "cand-rolling",
                    "name": TARGET_CANDIDATE,
                    "strategy_class": "RollingBreakoutStrategy",
                    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
                    "train": {"return": -0.0150, "sharpe": -1.20},
                    "val": {"return": 0.0150, "sharpe": 1.90},
                    "oos": {"return": 0.0290, "sharpe": 1.00},
                    "return_streams": {
                        "train": stream[:2],
                        "val": stream[2:4],
                        "oos": stream[4:],
                    },
                },
            }
        ]
    }


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-01-01T00:00:00Z",
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": False,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "basket_vol_ratio": 1.1,
            },
            {
                "date": "2025-01-02T00:00:00Z",
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": False,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "basket_vol_ratio": 1.1,
            },
            {
                "date": "2026-01-01T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": True,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2026-01-02T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": True,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2026-02-01T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": True,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2026-02-02T00:00:00Z",
                "btc_above_ma192": True,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": True,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": True,
                "breadth_ma96": 0.8,
                "breadth_ma192": 0.4,
                "basket_vol_ratio": 1.2,
            },
            {
                "date": "2026-02-03T00:00:00Z",
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "breadth_ma96_ge_60": False,
                "breadth_ma192_ge_60": False,
                "basket_vol_ratio_moderate": True,
                "basket_ret96_pos": False,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "basket_vol_ratio": 1.1,
            },
        ]
    )


def test_build_rolling_breakout_gate_prefers_ex_ante_market_rule():
    gate = build_rolling_breakout_30m_gate(
        _decision_payload(),
        feature_frame=_feature_frame(),
    )

    selected = dict(gate["selected_rule"])
    assert selected["rule_id"] == "btc_above_ma192"
    assert selected["metrics"]["oos"]["gate_days"] == 2
    assert float(selected["metrics"]["oos"]["return"]) > 0.04
    assert float(selected["metrics"]["oos"]["activation_ratio"]) > 0.60


def test_write_rolling_breakout_30m_gate_writes_files(tmp_path: Path):
    result = write_rolling_breakout_30m_gate(
        _decision_payload(),
        report_root=tmp_path,
        feature_frame=_feature_frame(),
    )

    json_path = Path(result["json_path"])
    md_path = Path(result["md_path"])
    assert json_path.exists()
    assert md_path.exists()
    text = json_path.read_text(encoding="utf-8")
    assert TARGET_CANDIDATE in text
    assert "selected_rule" in text
