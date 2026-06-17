# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
I6Q — Argument Quality Enforcement for R2 Mechanical regime.

Syntactic enforcement of rationale quality (the design spec §2.4).
Checks minimum argument count, length, and lexical diversity.
If I6Q fails after MAX_RETRIES, forces ESCALATE.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class I6QConfig:
    """I6Q enforcement thresholds. From config/experiment_config.yaml."""

    min_arg_tokens: int = 10
    min_diversity: float = 0.4
    max_retries: int = 2


@dataclass
class I6QResult:
    """Result of I6Q quality check."""

    passed: bool
    checks: dict[str, bool]
    details: str


def check_i6q(
    pro_arguments: list[str],
    con_arguments: list[str],
    config: I6QConfig | None = None,
) -> I6QResult:
    """Check argument quality against I6Q thresholds.

    the design spec §2.4:
      - Minimum 1 pro argument and 1 con argument
      - Each argument >= MIN_ARG_TOKENS words
      - Lexical diversity (unique tokens / total tokens) >= MIN_DIVERSITY

    Args:
        pro_arguments: List of pro argument strings
        con_arguments: List of con argument strings
        config: I6Q thresholds (defaults from paper)

    Returns:
        I6QResult with pass/fail and per-check details
    """
    cfg = config or I6QConfig()
    checks = {}
    details_parts = []

    # Check 1: minimum argument count
    has_pro = len(pro_arguments) >= 1
    has_con = len(con_arguments) >= 1
    checks["has_pro_argument"] = has_pro
    checks["has_con_argument"] = has_con
    if not has_pro:
        details_parts.append("Missing pro argument(s)")
    if not has_con:
        details_parts.append("Missing con argument(s)")

    # Check 2: minimum argument length (token count per argument)
    all_args = pro_arguments + con_arguments
    args_long_enough = True
    for i, arg in enumerate(all_args):
        tokens = arg.split()
        if len(tokens) < cfg.min_arg_tokens:
            args_long_enough = False
            details_parts.append(
                f"Argument {i} too short: {len(tokens)} tokens " f"(min {cfg.min_arg_tokens})"
            )
    checks["args_min_length"] = args_long_enough

    # Check 3: lexical diversity
    if all_args:
        all_tokens = " ".join(all_args).lower().split()
        if len(all_tokens) > 0:
            diversity = len(set(all_tokens)) / len(all_tokens)
        else:
            diversity = 0.0
        diversity_ok = diversity >= cfg.min_diversity
        checks["lexical_diversity"] = diversity_ok
        if not diversity_ok:
            details_parts.append(
                f"Lexical diversity too low: {diversity:.3f} " f"(min {cfg.min_diversity})"
            )
    else:
        checks["lexical_diversity"] = False
        details_parts.append("No arguments to check diversity")

    passed = all(checks.values())
    details = "; ".join(details_parts) if details_parts else "All I6Q checks passed"

    return I6QResult(passed=passed, checks=checks, details=details)
