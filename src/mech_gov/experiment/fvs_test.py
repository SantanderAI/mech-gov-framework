# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
FVS Test — Failure Visibility Score via quality-drop injection.

Tests whether the governance regime detects injected quality degradation.
We inject quality drops by lowering completeness on a subset of cases
and check whether the regime flags them (via DEFER/ESCALATE or metadata).

the design spec §4.2, Paper Definition 3.
"""

from __future__ import annotations

import copy

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.governance.regime import DecisionResult, GovernanceRegime
from mech_gov.llm.base import LLMInterface
from mech_gov.metrics.governance.fvs import compute_fvs


def inject_quality_drops(
    cases: list[BankingCase],
    drop_fraction: float = 0.3,
    completeness_floor: float = 0.10,
    seed: int = 42,
) -> tuple[list[BankingCase], list[bool]]:
    """Inject quality drops into a fraction of cases.

    For selected cases, completeness is lowered to `completeness_floor`.
    The original completeness is preserved in `original_completeness`.

    Args:
        cases: Original cases
        drop_fraction: Fraction of cases to degrade (default 30%)
        completeness_floor: Target completeness for degraded cases
        seed: RNG seed for reproducibility

    Returns:
        (modified_cases, injected_drops) — cases with drops applied,
        boolean mask indicating which cases were degraded
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    n_drops = max(1, int(len(cases) * drop_fraction))
    drop_indices = set(rng.choice(len(cases), size=n_drops, replace=False))

    modified_cases = []
    injected_drops = []

    for i, case in enumerate(cases):
        if i in drop_indices:
            degraded = copy.deepcopy(case)
            degraded.original_completeness = degraded.completeness
            degraded.completeness = completeness_floor
            modified_cases.append(degraded)
            injected_drops.append(True)
        else:
            modified_cases.append(copy.deepcopy(case))
            injected_drops.append(False)

    return modified_cases, injected_drops


def _was_flagged(result: DecisionResult) -> bool:
    """Check if a result indicates the regime flagged quality degradation.

    A quality drop is "flagged" if:
      - Decision is DEFER or ESCALATE (regime noticed something wrong), OR
      - metadata contains flagged_degradation=True, OR
      - A hard gate related to completeness triggered (K0_10, K0_11)
    """
    from mech_gov.data.banking_case import Decision

    if result.decision in (Decision.DEFER, Decision.ESCALATE):
        return True

    if result.metadata.get("flagged_degradation", False):
        return True

    if result.metadata.get("hard_gate_override") and result.metadata.get("gate_id") in (
        "K0_10",
        "K0_11",
    ):
        return True

    if result.metadata.get("ambiguity_gate_override"):
        return True

    return False


def run_fvs_test(
    cases: list[BankingCase],
    regime: GovernanceRegime,
    llm: LLMInterface,
    drop_fraction: float = 0.3,
    completeness_floor: float = 0.10,
    entropy_seed: int = 42,
) -> tuple[float, list[DecisionResult], list[bool]]:
    """Run FVS test: inject quality drops and measure detection rate.

    Detection is based on decision FLIP: a drop is flagged if the degraded
    case receives DEFER/ESCALATE but the original case did NOT. This avoids
    counting regimes that always defer as having perfect FVS.

    Args:
        cases: Cases to test
        regime: Governance regime to evaluate
        llm: LLM interface
        drop_fraction: Fraction of cases to degrade
        completeness_floor: Target completeness for degraded cases
        entropy_seed: Seed for E3 (R2)

    Returns:
        (fvs, results, injected_drops)
    """
    # First, run original (non-degraded) cases to get baseline decisions
    baseline_results = []
    for case in cases:
        result = regime.process_case(case, llm, entropy_seed=entropy_seed)
        baseline_results.append(result)

    modified_cases, injected_drops = inject_quality_drops(
        cases,
        drop_fraction=drop_fraction,
        completeness_floor=completeness_floor,
        seed=entropy_seed,
    )

    results = []
    for i, case in enumerate(modified_cases):
        result = regime.process_case(case, llm, entropy_seed=entropy_seed)
        # Tag: flagged only if decision FLIPPED to DEFER/ESCALATE vs baseline
        if injected_drops[i]:
            baseline_was_conservative = baseline_results[i].decision in (
                Decision.DEFER,
                Decision.ESCALATE,
            )
            degraded_is_conservative = result.decision in (Decision.DEFER, Decision.ESCALATE)
            # Flagged = degraded went conservative AND baseline was not
            result.metadata["flagged_degradation"] = (
                degraded_is_conservative and not baseline_was_conservative
            )
        else:
            result.metadata["flagged_degradation"] = False
        results.append(result)

    fvs = compute_fvs(results, injected_drops)
    return fvs, results, injected_drops
