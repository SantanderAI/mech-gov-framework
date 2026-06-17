# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
DIU — Deferral Information Utilisation.

Paper definition: geometric-mean composite of the three deferral
sub-scores (spec, causal, bshift), averaged across all deferrals.

DIU = mean( (spec_i · causal_i · bshift_i)^{1/3} )  for i in deferrals

We use the geometric mean (cube root of the product) rather than the raw
product because multiplying three [0,1] scores collapses values
exponentially (e.g. 0.5 × 0.4 × 0.3 = 0.06 vs (0.06)^{1/3} = 0.39).
The geometric mean preserves the "all dimensions must be nonzero"
property while keeping the composite on the same scale as the inputs.

Justification:
  - The geometric mean is the standard aggregator for multi-dimensional
    quality indices when all dimensions are required but compensatory
    (UNDP Human Development Index uses it for exactly this reason;
    see Klugman et al., 2011, "Human Development Report 2011").
  - In information retrieval, the geometric mean of precision and recall
    (G-mean) is preferred over the product when dimensions have
    different base rates (Kubat & Matwin, 1997).
  - For composite governance scores, Kaufmann et al. (2010,
    "The Worldwide Governance Indicators") recommend geometric
    aggregation to avoid a single weak dimension dominating.

Higher DIU = better governance (deferrals contain useful information).
"""

from __future__ import annotations

import numpy as np


def compute_diu(
    spec_scores: list[float],
    causal_scores: list[float],
    bshift_scores: list[float],
) -> float:
    """Compute Deferral Information Utilisation.

    Uses the geometric mean (cube root) of the three sub-scores per
    deferral, then averages across all deferrals.

    Args:
        spec_scores: Specificity scores per deferral
        causal_scores: Causality scores per deferral
        bshift_scores: Bias-shift scores per deferral

    Returns:
        DIU in [0, 1]. Higher = better deferral quality.
    """
    if not spec_scores:
        return 0.0

    if not (len(spec_scores) == len(causal_scores) == len(bshift_scores)):
        raise ValueError(
            f"All score arrays must have the same length. "
            f"Got spec={len(spec_scores)}, causal={len(causal_scores)}, "
            f"bshift={len(bshift_scores)}."
        )

    # Geometric mean: (s · c · b)^{1/3} per deferral
    geo_means = [
        (s * c * b) ** (1.0 / 3.0)
        for s, c, b in zip(spec_scores, causal_scores, bshift_scores, strict=False)
    ]

    return float(np.mean(geo_means))
