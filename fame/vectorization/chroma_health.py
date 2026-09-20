from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

from fame.exceptions import UserMessageError
from fame.utils.dirs import ensure_dir


class ChromaNotRunningError(UserMessageError):
    def __init__(self, host: str, detail: str = ""):
        msg = f"Chroma is not reachable at {host}. Please start the Chroma server."
        if detail:
            msg += f" Details: {detail}"
        super().__init__(msg)


def assert_chroma_running() -> None:
    """
    Health-check Chroma using env settings:
      - CHROMA_MODE: persistent | http (default: persistent)
      - CHROMA_PATH: for persistent mode
      - CHROMA_HOST / CHROMA_PORT: for http mode
    Raises a user-friendly error if unreachable.
    """
    mode = os.getenv("CHROMA_MODE", "persistent").lower()

    if mode == "http":
        host = os.getenv("CHROMA_HOST", "127.0.0.1")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        url = f"http://{host}:{port}/api/v1/heartbeat"
        try:
            r = requests.get(url, timeout=2)
            if r.status_code != 200:
                raise ChromaNotRunningError(url, f"status={r.status_code}")
        except Exception as e:
            raise ChromaNotRunningError(url, str(e))
    else:  # persistent
        path = Path(os.getenv("CHROMA_PATH", "data/chroma_db")).expanduser().resolve()
        try:
            ensure_dir(path)
        except Exception as e:
            raise ChromaNotRunningError(str(path), f"cannot access path: {e}")
        # Minimal check: presence of the directory; we assume client will initialize LMDB if needed.

