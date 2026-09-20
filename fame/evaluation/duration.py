from __future__ import annotations

import time


def start_timer() -> float:
    return time.time()


def elapsed_seconds(start: float) -> float:
    return time.time() - start
