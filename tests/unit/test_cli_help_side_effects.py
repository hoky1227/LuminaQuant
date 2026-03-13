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


def test_live_help_runs_without_name_errors(capsys):
    assert cli_main.main(["live", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Run LuminaQuant live trader." in captured.out


def test_live_help_does_not_force_runtime_or_strategy_loading(monkeypatch):
    from lumina_quant.cli import live as live_cli

    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (_ for _ in ()).throw(AssertionError("strategy helpers loaded during help")),
    )
    monkeypatch.setattr(
        live_cli,
        "_runtime_classes",
        lambda: (_ for _ in ()).throw(AssertionError("runtime classes loaded during help")),
    )
    assert cli_main.main(["live", "--help"]) == 0
