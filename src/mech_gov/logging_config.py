# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Logging configuration for mech_gov v2.

Usage in notebooks:
    from mech_gov.logging_config import setup_logging
    setup_logging()           # INFO level (default — see per-case progress)
    setup_logging("DEBUG")    # DEBUG level (see every LLM call, parse step)
    setup_logging("WARNING")  # quiet — only problems

Usage in scripts:
    import logging
    from mech_gov.logging_config import setup_logging
    setup_logging("INFO", log_file="experiment.log")  # also write to file
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    fmt: str = "%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """Configure the mech_gov logger hierarchy.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional path to also write logs to file
        fmt: Log message format
        datefmt: Timestamp format
    """
    global _CONFIGURED

    root_logger = logging.getLogger("mech_gov")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if _CONFIGURED:
        root_logger.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    _CONFIGURED = True

    root_logger.info("Logging configured: level=%s  file=%s", level, log_file or "(console only)")
