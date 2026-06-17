# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
CDL — Cosmetic Deadlock Rate.

Paper definition: fraction of deferrals that are "vacuous" (low specificity
OR low causality). A vacuous deferral doesn't explain why the case was
deferred or what information is missing.

CDL = |{d ∈ deferrals : spec(d) < θ_spec OR causal(d) < θ_causal}| / |deferrals|

Lower CDL = better governance (fewer vacuous deferrals).
"""

from __future__ import annotations

from mech_gov.data.banking_case import Decision
from mech_gov.governance.regime import DecisionResult


def compute_cdl(
    results: list[DecisionResult],
    spec_scores: list[float],
    causal_scores: list[float],
    spec_threshold: float = 0.3,
    causal_threshold: float = 0.3,
) -> float:
    """Compute Cosmetic Deadlock Rate.

    Args:
        results: All decision results (filters to DEFER only)
        spec_scores: Specificity scores for each deferral (aligned with deferrals)
        causal_scores: Causality scores for each deferral (aligned with deferrals)
        spec_threshold: Below this = vacuous specificity
        causal_threshold: Below this = vacuous causality

    Returns:
        CDL in [0, 1]. 0 = no vacuous deferrals, 1 = all vacuous.
    """
    # Filter to deferrals only
    deferral_indices = [i for i, r in enumerate(results) if r.decision == Decision.DEFER]

    if not deferral_indices:
        return 0.0

    # spec_scores and causal_scores should be aligned with deferral_indices
    if len(spec_scores) != len(deferral_indices) or len(causal_scores) != len(deferral_indices):
        raise ValueError(
            f"Score arrays must match deferral count. "
            f"Got {len(spec_scores)} spec, {len(causal_scores)} causal, "
            f"but {len(deferral_indices)} deferrals."
        )

    vacuous = sum(
        1
        for s, c in zip(spec_scores, causal_scores, strict=False)
        if s < spec_threshold or c < causal_threshold
    )

    return vacuous / len(deferral_indices)
