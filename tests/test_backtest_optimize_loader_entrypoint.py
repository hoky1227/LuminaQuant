from __future__ import annotations

import re
from pathlib import Path


def _parquet_import_block(source: str) -> str:
    match = re.search(
        r"from\s+lumina_quant\.parquet_market_data\s+import\s*\((.*?)\)",
        source,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def test_run_backtest_imports_owner_loader_entrypoint():
    source = Path("run_backtest.py").read_text(encoding="utf-8")
    assert "from lumina_quant.market_data import (" in source
    assert "load_data_dict_from_parquet" in source
    parquet_block = _parquet_import_block(source)
    assert "load_data_dict_from_parquet" not in parquet_block


def test_optimize_imports_owner_loader_entrypoint():
    source = Path("optimize.py").read_text(encoding="utf-8")
    assert "from lumina_quant.market_data import (" in source
    assert "load_data_dict_from_parquet" in source
    parquet_block = _parquet_import_block(source)
    assert "load_data_dict_from_parquet" not in parquet_block
