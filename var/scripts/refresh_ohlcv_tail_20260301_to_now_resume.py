from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.config import BaseConfig
from lumina_quant.data_sync import create_binance_exchange, sync_symbol_ohlcv
from lumina_quant.storage.parquet import normalize_symbol

START_DT = datetime(2026, 3, 1, tzinfo=UTC)
REPORT_PATH = Path("var/reports/ohlcv_tail_refresh_20260301_to_now_summary.json")


def _load_report() -> dict[str, object]:
    if REPORT_PATH.exists():
        try:
            payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _save_report(payload: dict[str, object]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _supports_symbol(exchange, symbol: str) -> bool:
    return normalize_symbol(symbol) in getattr(exchange, "markets", {})


def main() -> int:
    now_dt = datetime.now(UTC)
    requested_symbols = list(BaseConfig.SYMBOLS)
    report = _load_report()
    completed_rows = report.get("symbols")
    completed_rows = completed_rows if isinstance(completed_rows, list) else []
    completed_symbols = {
        str(item.get("symbol"))
        for item in completed_rows
        if isinstance(item, dict) and str(item.get("status", "ok")) == "ok"
    }

    exchange = create_binance_exchange(market_type="future", testnet=False)
    try:
        exchange.load_markets()
        supported_symbols = [symbol for symbol in requested_symbols if _supports_symbol(exchange, symbol)]
        skipped_symbols = [symbol for symbol in requested_symbols if symbol not in supported_symbols]

        report = {
            "generated_at_utc": now_dt.isoformat(),
            "db_path": str(BaseConfig.MARKET_DATA_PARQUET_PATH),
            "exchange_id": str(BaseConfig.MARKET_DATA_EXCHANGE),
            "timeframe": "1s",
            "start_utc": START_DT.isoformat(),
            "end_utc": now_dt.isoformat(),
            "requested_symbols": requested_symbols,
            "supported_symbols": supported_symbols,
            "skipped_symbols": skipped_symbols,
            "symbols": completed_rows,
        }
        _save_report(report)

        remaining = [symbol for symbol in supported_symbols if symbol not in completed_symbols]
        print(
            f"[TAIL-REFRESH-RESUME] start={START_DT.isoformat()} end={now_dt.isoformat()} "
            f"supported={len(supported_symbols)} skipped={len(skipped_symbols)} remaining={len(remaining)}",
            flush=True,
        )
        if skipped_symbols:
            print(
                f"[TAIL-REFRESH-RESUME] skipped unsupported symbols: {', '.join(skipped_symbols)}",
                flush=True,
            )

        for index, symbol in enumerate(remaining, start=1):
            try:
                stats = sync_symbol_ohlcv(
                    exchange=exchange,
                    db_path=str(BaseConfig.MARKET_DATA_PARQUET_PATH),
                    exchange_id=str(BaseConfig.MARKET_DATA_EXCHANGE),
                    symbol=symbol,
                    timeframe="1s",
                    start_ms=int(START_DT.timestamp() * 1000),
                    end_ms=int(now_dt.timestamp() * 1000),
                    limit=1000,
                    max_batches=250000,
                    retries=2,
                    base_wait_sec=0.25,
                )
                payload = {
                    "symbol": symbol,
                    "status": "ok",
                    "fetched_rows": int(stats.fetched_rows),
                    "upserted_rows": int(stats.upserted_rows),
                    "first_timestamp_ms": stats.first_timestamp_ms,
                    "last_timestamp_ms": stats.last_timestamp_ms,
                    "first_utc": datetime.fromtimestamp(stats.first_timestamp_ms / 1000, tz=UTC).isoformat()
                    if stats.first_timestamp_ms
                    else None,
                    "last_utc": datetime.fromtimestamp(stats.last_timestamp_ms / 1000, tz=UTC).isoformat()
                    if stats.last_timestamp_ms
                    else None,
                }
                completed_rows.append(payload)
                _save_report(report)
                print(
                    f"[TAIL-REFRESH-RESUME][{index}/{len(remaining)}] {symbol} "
                    f"fetched={payload['fetched_rows']} upserted={payload['upserted_rows']} "
                    f"last={payload['last_utc']}",
                    flush=True,
                )
            except Exception as exc:
                payload = {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                }
                completed_rows.append(payload)
                _save_report(report)
                print(
                    f"[TAIL-REFRESH-RESUME][{index}/{len(remaining)}] {symbol} error={exc}",
                    flush=True,
                )

    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    print(f"[TAIL-REFRESH-RESUME] summary written to {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
