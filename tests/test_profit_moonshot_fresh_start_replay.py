from __future__ import annotations

from datetime import UTC, date, datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts" / "research" / "replay_profit_moonshot_fresh_start.py"
SPEC = importlib.util.spec_from_file_location("replay_profit_moonshot_fresh_start", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load replay_profit_moonshot_fresh_start module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from scripts.research.replay_eth_shock_filters import _materialized_paths
from scripts.research.replay_profit_moonshot_fresh_start import FreshSpec


def test_fresh_start_hourly_loader_matches_raw_first_1s_aggregation_when_data_exists() -> None:
    market_root = Path("data/market_parquet")
    day = date(2026, 5, 5)
    one_second_paths = _materialized_paths(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start=day,
        end=day,
    )
    one_hour_paths = _materialized_paths(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start=day,
        end=day,
    )
    if not one_second_paths or not one_hour_paths:
        pytest.skip("raw-first materialized fixture data is not present")

    loaded = MODULE._load_symbol_hourly(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        start=day,
        end=day,
    )
    aggregated = (
        pl.scan_parquet(one_second_paths)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("open").first().alias("btcusdt_open"),
                pl.col("high").max().alias("btcusdt_high"),
                pl.col("low").min().alias("btcusdt_low"),
                pl.col("close").last().alias("btcusdt_close"),
                pl.col("volume").sum().alias("btcusdt_volume"),
            ]
        )
        .drop_nulls(["btcusdt_open", "btcusdt_close"])
        .collect()
        .sort("datetime")
    )

    assert loaded.height == aggregated.height
    assert loaded.select("datetime", "btcusdt_close").to_dicts() == aggregated.select(
        "datetime", "btcusdt_close"
    ).to_dicts()


def test_candidate_signal_adaptive_trend_and_persistence_rules() -> None:
    arrays = {
        "datetime": [datetime(2026, 5, 1, h, tzinfo=UTC) for h in range(6)],
        "symbols": ("BTC/USDT",),
        "btcusdt_open": [100.0] * 6,
        "btcusdt_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "btcusdt_ret_12h": np.array([np.nan] * 5 + [0.05]),
        "btcusdt_ret_3h": np.array([np.nan, np.nan, np.nan, 0.01, 0.01, 0.01]),
        "btcusdt_resid_z_12h": [np.nan] * 6,
        "btcusdt_resid_z_3h": [np.nan] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "btcusdt_open_interest_ffill": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
        "market_ret_12h": np.array([np.nan] * 5 + [0.05]),
        "btcusdt_flow_imbalance_3h": [0.25] * 6,
        "btcusdt_flow_imbalance_2h": [0.25] * 6,
    }
    arrays["btcusdt_oi_delta_12h"] = np.full(6, 0.02)
    arrays["btcusdt_ret_6h"] = np.array([np.nan] * 5 + [0.03])
    arrays["market_ret_6h"] = np.array([np.nan] * 5 + [0.02])
    spec = FreshSpec(
        name="adaptive_trend_test",
        family="adaptive_trend",
        lookback_bars=12,
        threshold=0.01,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        adaptive_lookback_bars=12,
    )
    symbol, side, reason = MODULE._candidate_signal(spec, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"

    persistence = FreshSpec(
        name="flow_persistence_test",
        family="flow_imbalance_persistence",
        lookback_bars=3,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        flow_lookback_bars=2,
        flow_threshold=0.0,
        flow_persistence_bars=3,
        flow_persistence_threshold=0.2,
        min_abs_return=0.001,
    )
    arrays["btcusdt_ret_3h"] = np.array([np.nan, np.nan, 0.002, 0.003, 0.003, 0.003])
    arrays["btcusdt_resid_z_3h"] = np.array([np.nan] * 6)
    symbol, side, reason = MODULE._candidate_signal(persistence, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"


def test_candidate_signal_cross_sharpe_rank_and_funding_oi() -> None:
    dt = [datetime(2026, 5, 1, h, tzinfo=UTC) for h in range(6)]
    arrays = {
        "datetime": dt,
        "symbols": ("BTC/USDT", "ETH/USDT"),
        "btcusdt_open": [100.0] * 6,
        "btcusdt_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "ethusdt_open": [50.0] * 6,
        "ethusdt_close": [50.0, 49.0, 49.0, 49.0, 49.0, 49.0],
        "btcusdt_ret_6h": np.array([np.nan] * 4 + [0.02, 0.03]),
        "ethusdt_ret_6h": np.array([np.nan] * 4 + [-0.01, -0.03]),
        "btcusdt_resid_z_6h": [np.nan] * 6,
        "ethusdt_resid_z_6h": [np.nan] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "ethusdt_funding_ffill": [0.0] * 6,
        "btcusdt_oi_delta_12h": np.array([np.nan] * 5 + [0.5]),
        "ethusdt_oi_delta_12h": np.array([np.nan] * 5 + [-0.5]),
        "btcusdt_open_interest_ffill": [20.0] * 6,
        "ethusdt_open_interest_ffill": [20.0] * 6,
        "btcusdt_flow_imbalance_3h": [0.0] * 6,
        "ethusdt_flow_imbalance_3h": [0.0] * 6,
        "market_ret_24h": np.array([np.nan] * 6),
        "market_ret_6h": np.array([np.nan] * 5 + [0.005]),
    }
    arrays["btcusdt_ret_24h"] = np.array([np.nan] * 5 + [0.03])
    arrays["ethusdt_ret_24h"] = np.array([np.nan] * 5 + [-0.03])
    arrays["btcusdt_ret_3h"] = np.array([np.nan] * 3 + [0.01, 0.01, 0.01])
    arrays["ethusdt_ret_3h"] = np.array([np.nan] * 3 + [0.01, 0.01, 0.01])
    arrays["btcusdt_resid_z_3h"] = np.array([np.nan] * 3 + [0.4, 0.6, 0.8])
    arrays["ethusdt_resid_z_3h"] = np.array([np.nan] * 3 + [0.1, 0.2, 0.3])

    cross = FreshSpec(
        name="cross_sharpe_rank",
        family="cross_sectional_sharpe_rank",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        sharpe_lookback_bars=6,
        sharpe_rank_min=0.0,
        min_abs_return=0.001,
    )
    symbol, side, reason = MODULE._candidate_signal(cross, arrays, 5)
    assert (symbol, side) == ("BTC/USDT", "LONG")
    assert reason == ""

    oi = FreshSpec(
        name="funding_oi",
        family="funding_oi_carry_fade",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        funding_rank_min=0.01,
        oi_rank_min=0.1,
        sharpe_lookback_bars=12,
        min_abs_return=0.001,
    )
    arrays["btcusdt_funding_ffill"] = [0.02] * 6
    arrays["ethusdt_funding_ffill"] = [-0.02] * 6
    symbol, side, reason = MODULE._candidate_signal(oi, arrays, 5)
    assert (symbol, side) in {("BTC/USDT", "SHORT"), ("ETH/USDT", "LONG")}
    assert reason == ""


def test_candidate_signal_flow_imbalance_persistence_and_exhaustion() -> None:
    dt = [datetime(2026, 5, 1, h, tzinfo=UTC) for h in range(6)]
    arrays = {
        "datetime": dt,
        "symbols": ("BTC/USDT",),
        "btcusdt_open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "btcusdt_high": [101.0] * 6,
        "btcusdt_low": [99.0] * 6,
        "btcusdt_close": [100.0, 101.0, 100.0, 101.0, 100.0, 101.0],
        "btcusdt_volume": [1000.0] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "btcusdt_open_interest_ffill": [10.0] * 6,
        "btcusdt_oi_delta_12h": np.full(6, 0.1),
        "market_ret_6h": np.array([np.nan, np.nan, 0.003, 0.003, 0.004, 0.004]),
        "btcusdt_ret_6h": np.array([np.nan, np.nan, 0.0, 0.0, -0.003, -0.002]),
        "btcusdt_flow_imbalance_3h": np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]),
        "btcusdt_flow_imbalance_6h": np.array([np.nan, np.nan, 0.04, 0.05, 0.04, 0.03]),
        "btcusdt_flow_imbalance_12h": np.array([np.nan] * 6),
        "btcusdt_resid_z_6h": np.array([np.nan] * 6),
        "btcusdt_resid_z_12h": np.array([np.nan] * 6),
        "btcusdt_ret_3h": np.array([np.nan, np.nan, 0.002, 0.002, 0.002, 0.002]),
        "btcusdt_ret_12h": np.array([np.nan, np.nan, 0.0, 0.0, -0.0, 0.0]),
    }

    persistence = FreshSpec(
        name="flow_imbalance_persistence_smoke",
        family="flow_imbalance_persistence",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=6,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        flow_lookback_bars=6,
        flow_threshold=0.03,
        flow_persistence_bars=3,
        flow_persistence_threshold=0.03,
    )
    symbol, side, reason = MODULE._candidate_signal(persistence, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"

    exhaustion = FreshSpec(
        name="flow_imbalance_exhaustion_smoke",
        family="flow_imbalance_exhaustion",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=6,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        flow_lookback_bars=3,
        flow_threshold=0.0,
        flow_persistence_threshold=0.02,
        min_abs_return=0.001,
    )
    arrays["btcusdt_flow_imbalance_3h"] = np.array([np.nan, np.nan, -0.05, -0.05, -0.06, -0.07])
    arrays["btcusdt_ret_6h"] = np.array([np.nan, np.nan, -0.003, -0.0025, -0.004, -0.0035])
    symbol, side, reason = MODULE._candidate_signal(exhaustion, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"


def test_candidate_signal_calendar_rotation_selects_month_side() -> None:
    arrays = {
        "datetime": [
            datetime(2026, 1, 15, tzinfo=UTC),
            datetime(2026, 3, 15, tzinfo=UTC),
        ],
        "symbols": ("BTC/USDT", "TRX/USDT"),
        "symbol_prefixes": ("btcusdt", "trxusdt"),
        "btcusdt_close": np.array([100.0, 100.0]),
        "trxusdt_close": np.array([20.0, 20.0]),
        "btcusdt_ret_72h": np.array([-0.05, 0.01]),
        "trxusdt_ret_72h": np.array([-0.01, 0.04]),
        "market_ret_72h": np.array([-0.03, 0.02]),
    }
    spec = FreshSpec(
        name="calendar_rotation",
        family="calendar_rotation",
        lookback_bars=72,
        threshold=0.002,
        hold_bars=48,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        min_abs_return=0.002,
        calendar_long_months=(3, 4, 5),
        calendar_short_months=(1, 2),
    )

    assert MODULE._candidate_signal(spec, arrays, 0) == ("BTC/USDT", "SHORT", "")
    assert MODULE._candidate_signal(spec, arrays, 1) == ("TRX/USDT", "LONG", "")

    fixed_long = FreshSpec(
        name="calendar_rotation_fixed",
        family="calendar_rotation",
        lookback_bars=72,
        threshold=0.002,
        hold_bars=48,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        min_abs_return=0.002,
        calendar_long_months=(3, 4, 5),
        calendar_short_months=(1, 2),
        calendar_long_symbol="BTCUSDT",
    )
    assert MODULE._candidate_signal(fixed_long, arrays, 1) == ("BTC/USDT", "LONG", "")


def test_candidate_specs_include_external_inspired_families() -> None:
    arrays = {
        "btcusdt_rv_24h": np.linspace(0.002, 0.003, 6),
        "btcusdt_ret_6h": np.zeros(6),
        "btcusdt_ret_12h": np.zeros(6),
        "btcusdt_ret_24h": np.zeros(6),
        "btcusdt_ret_48h": np.zeros(6),
        "btcusdt_ret_72h": np.zeros(6),
    }
    specs = MODULE._candidate_specs(arrays=arrays, symbols=["BTC/USDT"])
    families = {spec.family for spec in specs}
    assert "adaptive_trend" in families
    assert "cross_sectional_sharpe_rank" in families
    assert "flow_imbalance_persistence" in families
    assert "flow_imbalance_exhaustion" in families
    assert "funding_oi_carry_fade" in families
    assert "calendar_rotation" in families


def test_joined_panel_reuses_cache_without_reloading_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"load": 0}
    datetimes = [datetime(2026, 1, 1, hour, tzinfo=UTC) for hour in range(2)]

    def _fake_load_symbol_hourly(**_kwargs: object) -> pl.DataFrame:
        calls["load"] += 1
        return pl.DataFrame(
            {
                "datetime": datetimes,
                "btcusdt_open": [100.0, 101.0],
                "btcusdt_high": [101.0, 102.0],
                "btcusdt_low": [99.0, 100.0],
                "btcusdt_close": [100.5, 101.5],
                "btcusdt_volume": [10.0, 11.0],
            }
        )

    def _fake_load_feature_hourly(**_kwargs: object) -> tuple[pl.DataFrame, dict[str, object]]:
        return pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")}), {
            "symbol": "BTC/USDT",
            "rows": 0,
        }

    monkeypatch.setattr(MODULE, "_load_symbol_hourly", _fake_load_symbol_hourly)
    monkeypatch.setattr(MODULE, "_load_feature_hourly", _fake_load_feature_hourly)

    panel, meta = MODULE._joined_panel(
        market_root=tmp_path / "market",
        exchange="binance",
        symbols=["BTC/USDT"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        cache_dir=tmp_path / "cache",
    )
    assert panel.height == 2
    assert meta["panel_cache"]["cache_hit"] is False
    assert calls["load"] == 1

    monkeypatch.setattr(MODULE, "_load_symbol_hourly", lambda **_kwargs: pytest.fail("cache miss"))
    cached, cached_meta = MODULE._joined_panel(
        market_root=tmp_path / "market",
        exchange="binance",
        symbols=["BTC/USDT"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        cache_dir=tmp_path / "cache",
    )
    assert cached.height == 2
    assert cached_meta["panel_cache"]["cache_hit"] is True


class _FakeMemoryGuard:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.rss_log_path = output_dir / "_memory_guard" / "fresh_replay_rss_latest.jsonl"
        self.summary_path = output_dir / "_memory_guard" / "fresh_replay_memory_latest.json"
        self.checkpoints: list[tuple[str, dict[str, object] | None]] = []
        self.finalize_calls: list[dict[str, object]] = []
        self.released = False

    def checkpoint(self, event: str, context: dict[str, object] | None = None) -> None:
        self.checkpoints.append((event, context))

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = {
            "status": status,
            "error": error,
            "context": context or {},
            "peak_rss_bytes": 256 * 1024 * 1024,
            "rss_log_path": str(self.rss_log_path),
        }
        self.finalize_calls.append(payload)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text("{}", encoding="utf-8")
        return payload

    def release(self) -> None:
        self.released = True


def _tiny_replay_payload() -> dict[str, object]:
    return {
        "artifact_kind": "profit_moonshot_fresh_start_overhaul_replay",
        "generated_at_utc": "2026-05-07T00:00:00Z",
        "market_root": "data/market_parquet",
        "exchange": "binance",
        "symbols": ["BTC/USDT"],
        "oos_end_date": "2026-05-06",
        "split_windows": [],
        "data_metadata": {},
        "gate_policy": {},
        "spec_count": 1,
        "replay_survivor_count": 0,
        "success_candidate_count": 0,
        "top_results": [],
        "peak_rss_mib": 1.0,
    }


def test_fresh_replay_main_wraps_execution_with_memory_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guards: list[_FakeMemoryGuard] = []
    captured: dict[str, object] = {}

    def _fake_acquire(**kwargs: object) -> _FakeMemoryGuard:
        captured.update(kwargs)
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(MODULE, "build_payload", lambda args: (_tiny_replay_payload(), []))

    assert MODULE.main(["--output-dir", str(tmp_path / "external_overhaul")]) == 0

    payload = json.loads((tmp_path / "external_overhaul" / "fresh_start_overhaul_replay_latest.json").read_text(encoding="utf-8"))
    assert captured["run_name"] == MODULE.RUN_NAME
    assert captured["budget_bytes"] == MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    assert payload["memory_policy"]["explicit_budget_bytes"] == MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    assert payload["rss_log_path"].endswith("fresh_replay_rss_latest.jsonl")
    assert payload["memory_summary_path"].endswith("fresh_replay_memory_latest.json")
    assert payload["memory_summary"]["status"] == "completed"
    assert guards[0].checkpoints[0][0] == "start"
    assert guards[0].finalize_calls[0]["status"] == "completed"
    assert guards[0].released is True


def test_fresh_replay_main_finalizes_failed_guard_when_payload_build_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guards: list[_FakeMemoryGuard] = []

    def _fake_acquire(**kwargs: object) -> _FakeMemoryGuard:
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    def _boom(args: object) -> tuple[dict[str, object], list[dict[str, object]]]:
        raise RuntimeError("replay build failed")

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(MODULE, "build_payload", _boom)

    with pytest.raises(RuntimeError, match="replay build failed"):
        MODULE.main(["--output-dir", str(tmp_path / "external_overhaul")])

    assert guards[0].finalize_calls[0]["status"] == "failed"
    assert guards[0].finalize_calls[0]["error"] == "replay build failed"
    assert guards[0].released is True
