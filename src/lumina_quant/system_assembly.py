"""Mode-based system assembly shared by backtest/live entrypoints."""

from __future__ import annotations

from lumina_quant.backtesting.portfolio_backtest import Portfolio as LivePortfolio


def build_live_runtime_contract(*, transport: str = "poll") -> dict[str, object]:
    selected_transport = str(transport or "poll").strip().lower() or "poll"
    if selected_transport not in {"poll", "ws"}:
        raise ValueError(f"Unsupported live transport: {transport}")

    if selected_transport == "ws":
        from lumina_quant.live.data_ws import BinanceWebSocketDataHandler as LiveDataHandler
    else:
        from lumina_quant.live.data_poll import LiveDataHandler

    from lumina_quant.live.execution_live import LiveExecutionHandler
    from lumina_quant.live.trader import LiveDataFatalError, LiveTrader

    return {
        "engine_cls": LiveTrader,
        "data_handler_cls": LiveDataHandler,
        "execution_handler_cls": LiveExecutionHandler,
        "portfolio_cls": LivePortfolio,
        "fatal_error_cls": LiveDataFatalError,
        "transport": selected_transport,
    }


def build_system(mode: str) -> dict[str, object]:
    selected = str(mode or "").strip().lower()
    if selected == "backtest":
        from lumina_quant.backtesting.backtest import Backtest
        from lumina_quant.backtesting.data import HistoricCSVDataHandler
        from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
        from lumina_quant.backtesting.portfolio_backtest import Portfolio

        return {
            "mode": "backtest",
            "engine_cls": Backtest,
            "data_handler_cls": HistoricCSVDataHandler,
            "execution_handler_cls": SimulatedExecutionHandler,
            "portfolio_cls": Portfolio,
        }

    if selected in {"live", "sandbox"}:
        contract = build_live_runtime_contract()
        contract["mode"] = selected
        return contract

    raise ValueError(f"Unsupported system mode: {mode}")
