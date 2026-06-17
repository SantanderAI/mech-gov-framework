# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Policy template loader utilities."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent


def load_template(name: str) -> str:
    """Load a policy template by filename (without extension).

    Args:
        name: Template name, e.g. 'r1_system_prompt' or 'r2_system_prompt'

    Returns:
        Template text content
    """
    path = _TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")
