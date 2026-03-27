from __future__ import annotations

import argparse
import os

from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.cli._strategy_registry_fallback import (
    import_private_strategy_registry,
    load_strategy_registry,
)
from lumina_quant.config import LiveConfig
from lumina_quant.core.market_window_contract import MarketWindowContractError
from lumina_quant.live_selection import (
    extract_selection_config,
    infer_strategy_class_name,
    load_selection_payload,
    resolve_selection_file,
)
from lumina_quant.system_assembly import build_live_runtime_contract

DEFAULT_LIVE_STRATEGY_NAME = "MovingAverageCrossStrategy"
DEFAULT_WS_STRATEGY_NAME = "RsiStrategy"
STRATEGY_MAP = None
resolve_strategy_class = None


def _strategy_helpers():
    global STRATEGY_MAP, resolve_strategy_class
    registry = load_strategy_registry(
        import_private_strategy_registry
    )
    default_name = getattr(registry, "DEFAULT_STRATEGY_NAME", "PublicStubStrategy")

    def _get_live_strategy_map(include_opt_in=True):
        if hasattr(registry, "get_live_strategy_map"):
            return registry.get_live_strategy_map(include_opt_in=include_opt_in)
        return registry.get_strategy_map()

    def _resolve_strategy_class(name: str, default_name_override: str | None = None):
        return registry.resolve_strategy_class(name, default_name=default_name_override or default_name)

    if STRATEGY_MAP is None:
        STRATEGY_MAP = _get_live_strategy_map(include_opt_in=True)
    if resolve_strategy_class is None:
        resolve_strategy_class = _resolve_strategy_class
    return default_name, _get_live_strategy_map, resolve_strategy_class


def _resolve_transport(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "poll", "ws"}:
        return token or "poll"
    raise ValueError(f"Unsupported --transport: {value}")


def _shutdown_on_fatal(trader, exc: Exception) -> None:
    print(f"\nCritical live-data contract breach: {exc}")
    ordered_shutdown = getattr(trader, "_ordered_shutdown", None)
    if callable(ordered_shutdown):
        ordered_shutdown()
    close_audit = getattr(trader, "_close_audit_store", None)
    if callable(close_audit):
        close_audit(status="FAILED")
    raise SystemExit(2) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LuminaQuant live trader.")
    parser.add_argument(
        "--transport",
        choices=["poll", "ws"],
        default=str(os.getenv("LQ_LIVE_TRANSPORT", "poll") or "poll"),
        help="Live market-data transport (poll or ws).",
    )
    parser.add_argument(
        "--enable-live-real",
        action="store_true",
        help="Explicitly allow real trading mode.",
    )
    parser.add_argument(
        "--strategy",
        default="",
        help="Strategy class override. If omitted, live selection artifact is used when available.",
    )
    parser.add_argument(
        "--selection-file",
        default="",
        help=(
            "Optional live selection JSON path. If omitted, newest file under "
            "best_optimized_parameters/live/ is used."
        ),
    )
    parser.add_argument(
        "--no-selection",
        action="store_true",
        help="Disable loading live selection artifact and run with config/manual strategy only.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional external run_id for audit trail correlation.",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help="Optional stop-file path for graceful shutdown signal.",
    )
    args = parser.parse_args(argv)

    default_strategy_registry_name, get_live_strategy_map, resolve_strategy_class = _strategy_helpers()
    strategy_map = STRATEGY_MAP if isinstance(STRATEGY_MAP, dict) else get_live_strategy_map(include_opt_in=True)

    transport = _resolve_transport(args.transport)
    if args.enable_live_real:
        os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"

    # 1. Check Configuration
    print("=== LuminaQuant Live Trader ===")

    # 2. Setup
    symbol_list = list(LiveConfig.SYMBOLS)  # e.g. ['BTC/USDT'] from config.yaml
    resolved_timeframe = str(LiveConfig.TIMEFRAME)
    strategy_params = {}
    default_strategy_name = (
        DEFAULT_WS_STRATEGY_NAME if transport == "ws" else DEFAULT_LIVE_STRATEGY_NAME
    )
    strategy_cls = resolve_strategy_class(
        default_strategy_name,
        default_name=default_strategy_registry_name,
    )
    strategy_name = strategy_cls.__name__
    selection_path = None
    selection_cfg = None
    LiveDataFatalError = RuntimeError
    trader = None

    if not bool(args.no_selection):
        selection_path = resolve_selection_file(args.selection_file)
        if selection_path is not None:
            selection_payload = load_selection_payload(selection_path)
            selection_cfg = extract_selection_config(selection_payload)
            candidate_name = str(selection_cfg.get("candidate_name") or "").strip()
            inferred_name = infer_strategy_class_name(candidate_name)
            if inferred_name and inferred_name in strategy_map:
                strategy_cls = strategy_map[inferred_name]
                strategy_name = inferred_name
            selected_symbols = list(selection_cfg.get("symbols") or [])
            if selected_symbols:
                symbol_list = selected_symbols
            selected_timeframe = selection_cfg.get("strategy_timeframe")
            if selected_timeframe:
                resolved_timeframe = str(selected_timeframe)
            selected_params = selection_cfg.get("params")
            if isinstance(selected_params, dict):
                strategy_params = dict(selected_params)

    manual_strategy = str(args.strategy or "").strip()
    if manual_strategy:
        if manual_strategy not in strategy_map:
            raise ValueError(f"Unknown strategy override: {manual_strategy}")
        strategy_cls = strategy_map[manual_strategy]
        strategy_name = manual_strategy
        if selection_cfg is not None:
            inferred_name = infer_strategy_class_name(
                str(selection_cfg.get("candidate_name") or "")
            )
            if inferred_name != manual_strategy:
                strategy_params = {}
                print(
                    "[WARN] --strategy overrides selection strategy. "
                    "Selection params were ignored to avoid mismatch."
                )

    LiveConfig.SYMBOLS = list(symbol_list)
    LiveConfig.TIMEFRAME = str(resolved_timeframe)
    LiveConfig.validate()

    print(f"Mode: {'TESTNET/PAPER' if LiveConfig.IS_TESTNET else 'REAL TRADING'}")
    print(f"Exchange: {LiveConfig.EXCHANGE}")
    market_data_source = str(getattr(LiveConfig, "MARKET_DATA_SOURCE", "committed"))
    order_state_source = str(getattr(LiveConfig, "ORDER_STATE_SOURCE", "polling"))
    print(f"Market Data Source: {market_data_source}")
    print(f"Order State Source: {order_state_source}")
    print(f"Transport: {transport}")
    if market_data_source == "committed" and transport == "ws":
        print(
            "[WARN] --transport=ws is ignored when live.market_data_source=committed "
            "(committed reader path remains active)."
        )

    if selection_path is not None:
        print(f"Selection File: {selection_path}")
        if selection_cfg is not None:
            print(f"Selection Candidate: {selection_cfg.get('candidate_name')}")

    print(f"Trading Symbols: {symbol_list}")
    print(f"Strategy Timeframe: {LiveConfig.TIMEFRAME}")
    print(
        "Materialized Staleness Gate: "
        f"threshold={LiveConfig.MATERIALIZED_STALENESS_THRESHOLD_SECONDS}s, "
        f"alert_cooldown={LiveConfig.MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS}s"
    )
    print(f"Strategy Params: {strategy_params}")

    try:
        live_runtime = build_live_runtime_contract(transport=transport)
        LiveTrader = live_runtime["engine_cls"]
        LiveDataFatalError = live_runtime["fatal_error_cls"]
        data_handler_cls = live_runtime["data_handler_cls"]
        execution_handler_cls = live_runtime["execution_handler_cls"]
        portfolio_cls = live_runtime["portfolio_cls"]

        trader = LiveTrader(
            symbol_list=symbol_list,
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
            strategy_name=strategy_name,
            stop_file=args.stop_file,
            external_run_id=args.run_id,
        )

        print("Starting engine... Press Ctrl+C to stop.")
        trader.run()
        consume = getattr(trader.data_handler, "consume_fatal_error", None)
        if callable(consume):
            fatal_exc = consume()
            if fatal_exc is not None:
                _shutdown_on_fatal(trader, fatal_exc)

    except KeyboardInterrupt:
        print("\nStopping trader...")
        return 130
    except (RawFirstDataMissingError, MarketWindowContractError, LiveDataFatalError) as exc:
        if trader is not None:
            _shutdown_on_fatal(trader, exc)
        return 2
    except Exception as exc:
        print(f"\nCritical Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
