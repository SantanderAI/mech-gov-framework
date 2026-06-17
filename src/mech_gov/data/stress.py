# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Stress condition transforms for mech_gov v2 synthetic dataset.

Implements S0-S3 from the paper:
  S0 (Baseline): No transform
  S1 (HighRisk): Risk score upward shift
  S2 (LowInfo): Completeness reduction + flag removal
  S3 (Threshold): Concentrate cases near decision boundaries

All transforms preserve original values for Δ tracking.
Parameters are loaded from config/experiment_config.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mech_gov.data.banking_case import BankingCase, StressCondition

# ---------------------------------------------------------------------------
# Default stress parameters (overridable via config)
# ---------------------------------------------------------------------------

DEFAULT_STRESS_PARAMS: dict[str, dict[str, Any]] = {
    "S1_HighRisk": {
        "risk_shift_range": [-0.15, 0.15],
        "positive_bias": 0.90,
    },
    "S2_LowInfo": {
        "completeness_multiplier_range": [0.3, 0.7],
        "flags_to_remove": [1, 2],
    },
    "S3_Threshold": {
        "epsilon_risk": 0.05,
        "epsilon_completeness": 0.10,
        "target_proximity_pct": 0.60,
    },
}


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


def apply_s0_baseline(case: BankingCase, rng: np.random.Generator) -> BankingCase:
    """S0: No transform — pass through."""
    return case


def apply_s1_high_risk(
    case: BankingCase,
    rng: np.random.Generator,
    params: dict | None = None,
) -> BankingCase:
    """S1 (HighRisk): Shift risk scores upward.

    Paper: "risk scores shifted upward by ±15%"
    Implementation: 60% chance of positive shift, 40% negative, within ±15%.
    """
    p = params or DEFAULT_STRESS_PARAMS["S1_HighRisk"]
    lo, hi = p["risk_shift_range"]
    positive_bias = p["positive_bias"]

    case.original_risk_score = case.risk_score

    # Biased shift: more likely to increase risk
    if rng.random() < positive_bias:
        shift = rng.uniform(0, hi)
    else:
        shift = rng.uniform(lo, 0)

    case.risk_score = float(np.clip(case.risk_score + shift, 0.0, 1.0))
    case.stress_condition = StressCondition.S1_HIGH_RISK
    return case


def apply_s2_low_info(
    case: BankingCase,
    rng: np.random.Generator,
    params: dict | None = None,
) -> BankingCase:
    """S2 (LowInfo): Reduce completeness and remove flags.

    Paper: "completeness reduction × uniform(0.3, 0.7) + flag removal"
    """
    p = params or DEFAULT_STRESS_PARAMS["S2_LowInfo"]
    lo, hi = p["completeness_multiplier_range"]
    min_remove, max_remove = p["flags_to_remove"]

    case.original_completeness = case.completeness

    # Reduce completeness
    multiplier = rng.uniform(lo, hi)
    case.completeness = float(np.clip(case.completeness * multiplier, 0.0, 1.0))

    # Remove some flags (simulating information loss)
    if case.regulatory_flags:
        n_remove = min(
            rng.integers(min_remove, max_remove + 1),
            len(case.regulatory_flags),
        )
        if n_remove > 0:
            indices_to_remove = rng.choice(
                len(case.regulatory_flags), size=int(n_remove), replace=False
            )
            case.regulatory_flags = [
                f for i, f in enumerate(case.regulatory_flags) if i not in indices_to_remove
            ]

    case.stress_condition = StressCondition.S2_LOW_INFO
    return case


def apply_s3_threshold(
    case: BankingCase,
    rng: np.random.Generator,
    thresholds: list[float] | None = None,
    params: dict | None = None,
) -> BankingCase:
    """S3 (Threshold): Concentrate cases near decision boundaries.

    For each hard gate threshold θ, move case parameters so that
    |param - θ| < ε. This creates cases that are maximally challenging
    for the governance system.
    """
    p = params or DEFAULT_STRESS_PARAMS["S3_Threshold"]
    eps_risk = p["epsilon_risk"]
    eps_comp = p["epsilon_completeness"]
    target_pct = p["target_proximity_pct"]

    # Default thresholds from paper's hard gates
    if thresholds is None:
        thresholds = [0.3, 0.7, 0.85, 0.9]

    case.original_risk_score = case.risk_score
    case.original_completeness = case.completeness

    if rng.random() < target_pct:
        # Move risk_score near a random threshold
        target_threshold = rng.choice(thresholds)
        case.risk_score = float(
            np.clip(
                target_threshold + rng.uniform(-eps_risk, eps_risk),
                0.0,
                1.0,
            )
        )

        # Move completeness near the ambiguity gate threshold (0.3)
        case.completeness = float(
            np.clip(
                0.3 + rng.uniform(-eps_comp, eps_comp),
                0.0,
                1.0,
            )
        )

    case.stress_condition = StressCondition.S3_THRESHOLD
    return case


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: dict[StressCondition, Callable[..., BankingCase]] = {
    StressCondition.S0_BASELINE: apply_s0_baseline,
    StressCondition.S1_HIGH_RISK: apply_s1_high_risk,
    StressCondition.S2_LOW_INFO: apply_s2_low_info,
    StressCondition.S3_THRESHOLD: apply_s3_threshold,
}


def apply_stress(
    cases: list[BankingCase],
    condition: StressCondition,
    rng: np.random.Generator,
    stress_params: dict | None = None,
) -> list[BankingCase]:
    """Apply a stress condition to a list of cases.

    Makes deep copies to avoid mutating originals.

    Args:
        cases: Original cases (will not be modified)
        condition: Which stress condition to apply
        rng: Random generator for stochastic transforms
        stress_params: Override default stress parameters

    Returns:
        New list of stressed cases
    """
    transform_fn = TRANSFORM_REGISTRY[condition]
    stressed = []

    for case in cases:
        stressed_case = case.model_copy(deep=True)

        if condition == StressCondition.S0_BASELINE:
            stressed_case = transform_fn(stressed_case, rng)
        elif condition == StressCondition.S3_THRESHOLD:
            stressed_case = transform_fn(stressed_case, rng, params=stress_params)
        else:
            stressed_case = transform_fn(stressed_case, rng, params=stress_params)

        stressed.append(stressed_case)

    return stressed
