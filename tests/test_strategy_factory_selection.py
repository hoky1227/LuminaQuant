from __future__ import annotations

import pytest

from lumina_quant.strategy_factory.selection import safe_float


def test_safe_float_only_falls_back_for_coercion_errors() -> None:
    assert safe_float(None, default=1.5) == 1.5
    assert safe_float("bad", default=1.5) == 1.5

    class ExplodingFloat:
        def __float__(self) -> float:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        safe_float(ExplodingFloat(), default=1.5)
