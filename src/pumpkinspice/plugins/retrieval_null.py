"""No-op retrieval: returns no belief nodes.

For bringing the loop up before any corpus is seeded, and as the control's
"no retrieval" baseline. Still reports latency so capture stays uniform.
"""

from __future__ import annotations

import time
from typing import Any

from ..contracts import RetrievalResult


class NullRetrieval:
    name = "null"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def retrieve(self, query: str, *, top_k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        return RetrievalResult(
            query=query,
            nodes=[],
            latency_ms=(time.perf_counter() - t0) * 1e3,
            backend=self.name,
        )
