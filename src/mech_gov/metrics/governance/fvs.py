# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
FVS — Failure Visibility Score.

Paper definition: fraction of injected quality drops that the regime
correctly flagged or detected.

FVS = |{i : injected_drop_i AND flagged_i}| / |{i : injected_drop_i}|

Higher FVS = better governance (regime detects degradation).
FVS = 1.0 when no drops are injected (vacuously true).
"""

from __future__ import annotations

from mech_gov.governance.regime import DecisionResult


def compute_fvs(
    results: list[DecisionResult],
    injected_drops: list[bool],
) -> float:
    """Compute Failure Visibility Score.

    Args:
        results: Decision results for each case
        injected_drops: True if case i had a quality drop injected

    Returns:
        FVS in [0, 1]. 1.0 = all drops detected, 0.0 = none detected.
    """
    if len(results) != len(injected_drops):
        raise ValueError(
            f"results and injected_drops must have same length. "
            f"Got {len(results)} and {len(injected_drops)}."
        )

    actual_drops = sum(injected_drops)
    if actual_drops == 0:
        return 1.0

    flagged = sum(
        1
        for r, drop in zip(results, injected_drops, strict=False)
        if drop and r.metadata.get("flagged_degradation", False)
    )

    return flagged / actual_drops
