from __future__ import annotations

from lumina_quant.cli import main as cli_main


def test_backtest_help_has_no_strategy_setup_noise(capsys):
    assert cli_main.main(["backtest", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Optimized params not found" not in captured.out
    assert "Run LuminaQuant backtest." in captured.out


def test_optimize_help_has_no_optuna_import_warning(capsys):
    assert cli_main.main(["optimize", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Optuna not found" not in captured.out
    assert "Run LuminaQuant walk-forward optimization." in captured.out
