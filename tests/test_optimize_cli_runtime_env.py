from __future__ import annotations

from lumina_quant.cli import optimize


def test_optimize_runtime_settings_reads_env_and_config_at_call_time(monkeypatch):
    monkeypatch.setenv("LQ_DATA_MODE", "legacy")
    monkeypatch.setenv("LQ_BASE_TIMEFRAME", "5m")
    monkeypatch.setenv("LQ_AUTO_COLLECT_DB", "1")
    monkeypatch.setenv("LQ_BACKTEST_MODE", "legacy_batch")
    monkeypatch.setattr(optimize.BaseConfig, "MARKET_DATA_PARQUET_PATH", "var/data/runtime_parquet")
    monkeypatch.setattr(optimize.BaseConfig, "MARKET_DATA_EXCHANGE", "kraken")
    monkeypatch.setattr(optimize.BaseConfig, "STORAGE_BACKEND", "parquet")
    monkeypatch.setattr(optimize.BaseConfig, "SYMBOLS", ["BTC/USDT", "ETH/USDT"])
    monkeypatch.setattr(optimize.BaseConfig, "TIMEFRAME", "15m")
    monkeypatch.setattr(optimize.OptimizationConfig, "MAX_WORKERS", 7)

    settings = optimize._current_optimize_runtime_settings()

    assert settings["data_mode"] == "legacy"
    assert settings["base_timeframe"] == "5m"
    assert settings["auto_collect_db"] is True
    assert settings["backtest_mode"] == "legacy_batch"
    assert settings["market_db_path"] == "var/data/runtime_parquet"
    assert settings["market_db_exchange"] == "kraken"
    assert settings["market_db_backend"] == "parquet"
    assert settings["symbol_list"] == ["BTC/USDT", "ETH/USDT"]
    assert settings["strategy_timeframe"] == "15m"
    assert settings["max_workers"] == 2
