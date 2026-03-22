from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lumina_quant.strategy_factory import research_runner


def _write_runtime_config(
    tmp_path: Path,
    *,
    symbols: list[str],
    root_path: str,
    exchange: str,
) -> str:
    cfg = textwrap.dedent(
        f"""
        trading:
          symbols: {symbols!r}
          timeframe: \"15m\"
        storage:
          backend: \"local\"
          market_data_parquet_path: \"{root_path}\"
          market_data_exchange: \"{exchange}\"
        live:
          mode: \"paper\"
          exchange:
            driver: \"binance_futures\"
            name: \"binance\"
            market_type: \"future\"
            position_mode: \"HEDGE\"
            margin_mode: \"isolated\"
            leverage: 2
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return str(cfg_path)


def test_run_candidate_research_uses_runtime_symbols_when_symbol_universe_is_omitted(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    report = research_runner.run_candidate_research(
        candidates=[],
        strategy_timeframes=["1m"],
        symbol_universe=None,
    )

    assert report["symbol_universe"] == ["ETH/USDT", "SOL/USDT"]


def test_build_default_candidate_rows_uses_runtime_symbols_when_symbols_are_omitted(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured: dict[str, object] = {}

    def _stub_build_candidates(*, symbols, timeframes):
        captured.update({"symbols": list(symbols), "timeframes": list(timeframes)})
        return []

    monkeypatch.setattr(
        "lumina_quant.strategy_factory.candidate_library.build_binance_futures_candidates",
        _stub_build_candidates,
    )

    rows = research_runner.build_default_candidate_rows(symbols=None, timeframes=["1m"])

    assert rows == []
    assert captured["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert captured["timeframes"] == ["1m"]


def test_load_bundle_cache_uses_runtime_market_data_settings(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured: dict[str, object] = {}

    def _stub_load_data_dict(
        root_path,
        *,
        exchange,
        symbol_list,
        timeframe,
        start_date=None,
        end_date=None,
        data_mode="legacy",
    ):
        captured.update(
            {
                "root_path": root_path,
                "exchange": exchange,
                "symbol_list": list(symbol_list),
                "timeframe": timeframe,
                "data_mode": data_mode,
            }
        )
        return {}

    monkeypatch.setattr(research_runner, "load_data_dict_from_parquet", _stub_load_data_dict)

    with pytest.raises(research_runner.RawFirstDataMissingError):
        research_runner._load_bundle_cache(
            symbols=["ETH/USDT"],
            timeframes=["1m"],
            allow_csv_fallback=False,
            allow_synthetic_fallback=False,
        )

    assert captured["root_path"] == "var/data/research_runtime"
    assert captured["exchange"] == "kraken"


def test_load_bundle_cache_honors_explicit_market_data_settings(
    monkeypatch,
):
    captured: dict[str, object] = {}

    def _stub_load_data_dict(
        root_path,
        *,
        exchange,
        symbol_list,
        timeframe,
        start_date=None,
        end_date=None,
        data_mode="legacy",
    ):
        captured.update(
            {
                "root_path": root_path,
                "exchange": exchange,
                "symbol_list": list(symbol_list),
                "timeframe": timeframe,
                "data_mode": data_mode,
            }
        )
        return {}

    monkeypatch.setattr(research_runner, "load_data_dict_from_parquet", _stub_load_data_dict)

    with pytest.raises(research_runner.RawFirstDataMissingError):
        research_runner._load_bundle_cache(
            symbols=["ETH/USDT"],
            timeframes=["1m"],
            allow_csv_fallback=False,
            allow_synthetic_fallback=False,
            market_data_settings={
                "symbols": ["BTC/USDT"],
                "market_data_parquet_path": "explicit/runtime/root",
                "market_data_exchange": "bybit",
            },
        )

    assert captured["root_path"] == "explicit/runtime/root"
    assert captured["exchange"] == "bybit"
    assert captured["symbol_list"] == ["ETH/USDT"]


def test_load_bundle_cache_uses_csv_fallback_with_date_bounds(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "BTCUSDT.csv").write_text(
        "\n".join(
            [
                "datetime,open,high,low,close,volume",
                "2024-01-01T00:00:00,100,101,99,100.5,10",
                "2024-01-02T00:00:00,101,102,100,101.5,12",
                "2024-01-03T00:00:00,102,103,101,102.5,11",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(research_runner, "load_data_dict_from_parquet", lambda *args, **kwargs: {})

    cache, source_map = research_runner._load_bundle_cache(
        symbols=["BTC/USDT"],
        timeframes=["1m"],
        start_date="2024-01-02",
        end_date="2024-01-02",
        allow_synthetic_fallback=False,
        min_bars=1,
        market_data_settings={
            "symbols": ["BTC/USDT"],
            "market_data_parquet_path": "unused",
            "market_data_exchange": "binance",
        },
    )

    bundle = cache[("BTC/USDT", "1m")]
    assert source_map["csv"] == ["BTC/USDT@1m"]
    assert bundle.close.tolist() == [101.5]


def test_load_feature_cache_uses_runtime_market_data_settings(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    captured: dict[str, object] = {}

    def _stub_load_feature_points(
        db_path,
        *,
        exchange,
        symbol,
        start_date=None,
        end_date=None,
    ):
        captured.update({"db_path": db_path, "exchange": exchange, "symbol": symbol})
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(research_runner, "load_futures_feature_points_from_db", _stub_load_feature_points)

    cache = research_runner._load_feature_cache(symbols=["ETH/USDT"])
    assert "ETH/USDT" in cache
    assert cache["ETH/USDT"].is_empty()
    assert captured["db_path"] == "var/data/research_runtime"
    assert captured["exchange"] == "kraken"
    assert captured["symbol"] == "ETH/USDT"


def test_current_research_market_data_settings_accepts_explicit_runtime_mapping():
    settings = research_runner._current_research_market_data_settings(
        {
            "symbols": ["eth/usdt", "sol/usdt"],
            "market_data_parquet_path": "explicit/runtime/root",
            "market_data_exchange": "kraken",
        }
    )

    assert settings["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert settings["parquet_root"] == "explicit/runtime/root"
    assert settings["exchange"] == "kraken"


def test_resolve_feature_points_path_uses_runtime_market_data_path(
    tmp_path: Path,
    monkeypatch,
):
    cfg_path = _write_runtime_config(
        tmp_path,
        symbols=["ETH/USDT", "SOL/USDT"],
        root_path="var/data/research_runtime",
        exchange="kraken",
    )
    feature_points_dir = tmp_path / "var" / "data" / "research_runtime" / "feature_points"
    feature_points_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LQ_CONFIG_PATH", cfg_path)

    assert research_runner._resolve_feature_points_path() == feature_points_dir.resolve()
