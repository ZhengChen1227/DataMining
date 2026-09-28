"""统一的日志出口，保证不同模块的输出格式一致。"""

from __future__ import annotations

import logging
import sys

_ROOT = "tsad"
_configured = False


def get_logger(name: str = "tsad", level: int = logging.INFO) -> logging.Logger:
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger(_ROOT)
        root.handlers = [handler]
        root.setLevel(level)
        root.propagate = False
        _configured = True

    full = name if name.startswith(_ROOT) else f"{_ROOT}.{name}"
    logger = logging.getLogger(full)
    logger.setLevel(level)
    return logger