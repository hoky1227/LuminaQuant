from __future__ import annotations

import sys
import types

from lumina_quant.strategy_factory import runtime_settings


def test_current_research_market_data_settings_uses_explicit_runtime_mapping() -> None:
    settings = runtime_settings.current_research_market_data_settings(
        {
            "symbols": ["eth/usdt", "sol/usdt"],
            "market_data_parquet_path": "explicit/runtime/root",
            "market_data_exchange": "kraken",
        }
    )

    assert settings["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert settings["parquet_root"] == "explicit/runtime/root"
    assert settings["exchange"] == "kraken"


def test_default_research_symbol_universe_falls_back_when_config_import_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "lumina_quant.config", None)

    assert runtime_settings.default_research_symbol_universe() == (
        "BTC/USDT",
        "ETH/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "TRX/USDT",
        "DOGE/USDT",
        "ADA/USDT",
        "TON/USDT",
        "AVAX/USDT",
        "XAU/USDT",
        "XAG/USDT",
        "XPT/USDT",
        "XPD/USDT",
    )


def test_runtime_settings_fall_back_when_base_config_lookup_raises_file_not_found(
    monkeypatch,
) -> None:
    class _MissingSymbolsMeta(type):
        MARKET_DATA_PARQUET_PATH = "tmp/parquet"
        MARKET_DATA_EXCHANGE = "bybit"

        def __getattr__(cls, _name: str):
            raise FileNotFoundError("config missing")

    class _BaseConfig(metaclass=_MissingSymbolsMeta):
        pass

    config_module = types.ModuleType("lumina_quant.config")
    config_module.BaseConfig = _BaseConfig
    monkeypatch.setitem(sys.modules, "lumina_quant.config", config_module)

    assert runtime_settings.default_research_symbol_universe()[0] == "BTC/USDT"
    assert runtime_settings.current_research_market_data_settings() == {
        "symbols": list(runtime_settings._DEFAULT_SYMBOL_FALLBACK),
        "parquet_root": "tmp/parquet",
        "exchange": "bybit",
    }
