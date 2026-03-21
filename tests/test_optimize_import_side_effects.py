from __future__ import annotations

import importlib
import sys


def test_optimize_import_does_not_materialize_runtime_config_snapshots(monkeypatch):
    monkeypatch.delenv("LQ__BACKTEST__DECISION_CADENCE_SECONDS", raising=False)

    import lumina_quant.config as config_module

    config_module = importlib.reload(config_module)
    assert "SYMBOLS" not in config_module.BaseConfig.__dict__
    assert "MAX_WORKERS" not in config_module.OptimizationConfig.__dict__

    sys.modules.pop("lumina_quant.cli.optimize", None)
    optimize_module = importlib.import_module("lumina_quant.cli.optimize")

    assert optimize_module.CSV_DIR == "data"
    assert "SYMBOLS" not in config_module.BaseConfig.__dict__
    assert "MAX_WORKERS" not in config_module.OptimizationConfig.__dict__
    assert "LQ__BACKTEST__DECISION_CADENCE_SECONDS" not in config_module.os.environ
