# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
ESD — Entropy Sensitivity Differential.

Paper definition: average of three sub-scores measuring how sensitive
the regime is to entropy manipulation:
  exploit:   Can an attacker exploit observable entropy to bias decisions?
  leakage:   Does the regime leak entropy information?
  integrity: Does commit-reveal integrity hold?

ESD = (exploit + leakage + integrity) / 3.0

Lower ESD = better governance (less sensitive to entropy attacks).
R2 should have near-zero ESD due to E3 commit-reveal.
"""

from __future__ import annotations


def compute_esd(
    exploit: float,
    leakage: float,
    integrity: float,
) -> float:
    """Compute Entropy Sensitivity Differential.

    Args:
        exploit: Exploit sub-score [0, 1]. Fraction of cases where observable
                 entropy led to biased decisions.
        leakage: Leakage sub-score [0, 1]. Fraction of cases where entropy
                 information was leaked to the LLM.
        integrity: Integrity sub-score [0, 1]. Fraction of cases where
                   commit-reveal verification failed.

    Returns:
        ESD in [0, 1]. Lower = better.
    """
    return (exploit + leakage + integrity) / 3.0
