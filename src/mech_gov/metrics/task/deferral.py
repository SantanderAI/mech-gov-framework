# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Task Metrics — ADR (Appropriate Deferral Rate) and Over-caution Rate.

ADR: measures whether the regime defers when it should (low completeness).
Over-caution: measures unnecessary DECLINE/ESCALATE on clear APPROVE cases.

the design spec §4.4.
"""

from __future__ import annotations

from mech_gov.data.banking_case import BankingCase, Decision, GTConfidence
from mech_gov.governance.regime import DecisionResult


def compute_adr(
    results: list[DecisionResult],
    cases: list[BankingCase],
    theta_iota: float = 0.3,
) -> float:
    """Compute Appropriate Deferral Rate.

    Measures: of cases that SHOULD be deferred (completeness < θ_ι),
    what fraction actually were?

    Args:
        results: Decision results from a regime
        cases: Corresponding banking cases
        theta_iota: Completeness threshold below which deferral is appropriate

    Returns:
        ADR in [0, 1]. 1.0 = all appropriate deferrals made.
    """
    if len(results) != len(cases):
        raise ValueError(
            f"results and cases must have same length. " f"Got {len(results)} and {len(cases)}."
        )

    should_defer = [
        (r, c)
        for r, c in zip(results, cases, strict=False)
        if (c.original_completeness if c.original_completeness is not None else c.completeness)
        < theta_iota
    ]

    if not should_defer:
        return 1.0

    deferred = sum(1 for r, c in should_defer if r.decision == Decision.DEFER)

    return deferred / len(should_defer)


def compute_overcaution(
    results: list[DecisionResult],
    cases: list[BankingCase],
) -> float:
    """Compute Over-caution Rate.

    Measures: of deterministic APPROVE ground-truth cases, what fraction
    got DECLINE or ESCALATE instead?

    Args:
        results: Decision results from a regime
        cases: Corresponding banking cases

    Returns:
        Over-caution rate in [0, 1]. 0.0 = no over-caution.
    """
    if len(results) != len(cases):
        raise ValueError(
            f"results and cases must have same length. " f"Got {len(results)} and {len(cases)}."
        )

    clear_approve = [
        (r, c)
        for r, c in zip(results, cases, strict=False)
        if c.gt_decision == "APPROVE" and c.gt_confidence == GTConfidence.DETERMINISTIC
    ]

    if not clear_approve:
        return 0.0

    overcautious = sum(
        1 for r, c in clear_approve if r.decision in {Decision.DECLINE, Decision.ESCALATE}
    )

    return overcautious / len(clear_approve)
