from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fame.utils.dirs import build_paths, ensure_for_stage
from fame.config.load import load_config


def _default_log_dir() -> Path:
    paths = build_paths()
    ensure_for_stage("logs", paths)
    return paths.logs


class JsonFormatter(logging.Formatter):
    """
    Structured JSON formatter.
    Fields:
      - ts: ISO 8601 UTC timestamp
      - level: INFO/WARNING/ERROR
      - logger: logger name
      - msg: log message
      - extra: any structured extras passed via logger.info(..., extra={...})
    """

    def __init__(self, include_exc: bool = True) -> None:
        super().__init__()
        self.include_exc = include_exc

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if self.include_exc and record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # record.__dict__ may contain 'extra' keys; include safe extras
        for key, val in record.__dict__.items():
            if key in ("args", "msg", "levelname", "name", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "stacklevel"):
                continue
            payload.setdefault("extra", {})[key] = val
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "fame", level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    cfg = load_config().logging
    log_dir = _default_log_dir()
    log_path = log_dir / "fame.log"

    # derive level precedence: argument > env > config
    cfg_level = getattr(cfg, "level", "INFO")
    lvl = (level or os.getenv("FAME_LOG_LEVEL") or cfg_level or "INFO").upper()
    logger.setLevel(getattr(logging, lvl, logging.INFO))

    file_formatter = JsonFormatter(include_exc=True)
    console_formatter = JsonFormatter(include_exc=cfg.console_include_exc)

    if cfg.to_file:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    if cfg.console:
        ch = logging.StreamHandler()
        ch.setFormatter(console_formatter)
        ch.setLevel(getattr(logging, cfg.console_level, logger.level))
        logger.addHandler(ch)
    logger.propagate = False
    return logger


def log_exception(logger: logging.Logger, exc: Exception, **extra: Any) -> None:
    logger.error(str(exc), exc_info=exc, **extra)
