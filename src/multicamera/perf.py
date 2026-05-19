"""Small performance timing helpers."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def perf_timer(label: str, threshold_ms: float = 50.0) -> Iterator[None]:
    """Log elapsed time when an operation is slower than ``threshold_ms``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms >= threshold_ms:
            logger.info("perf: %s took %.1f ms", label, elapsed_ms)

