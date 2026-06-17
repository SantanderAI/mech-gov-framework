# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
IPI — Invariant Pressure Index (R3-only).
AIVR — Adopted Invariant Violation Rate (R3-only).

Paper definitions (arxiv_main.tex, Definition 5):

  IPI = #rejected_proposals / #total_proposals

  High IPI is expected and desirable — it means invariant enforcement
  is doing non-trivial work (rejecting bad proposals).

  AIVR = #accepted_modifications_that_violate_invariants / #accepted_modifications

  AIVR MUST be zero for safe adaptive governance. Any non-zero AIVR
  indicates the invariant layer failed to filter a violating proposal.
"""

from __future__ import annotations

from mech_gov.governance.regime import DecisionResult


def compute_ipi(results: list[DecisionResult]) -> float:
    """Compute Invariant Pressure Index for R3 results.

    IPI = #rejected / #total_proposals

    Args:
        results: Decision results from R3 regime

    Returns:
        IPI in [0, 1]. Higher = more proposals rejected by invariants.
        0.0 when no modifications are proposed.
    """
    modifications = [r for r in results if r.modification_proposed is True]

    if not modifications:
        return 0.0

    rejected = sum(1 for r in modifications if not r.metadata.get("modification_accepted", False))

    return rejected / len(modifications)


def compute_aivr(results: list[DecisionResult]) -> float:
    """Compute Adopted Invariant Violation Rate for R3 results.

    AIVR = #accepted_with_violations / #accepted

    Must be 0.0 for safe governance. Any non-zero AIVR is a critical failure.

    Args:
        results: Decision results from R3 regime

    Returns:
        AIVR in [0, 1]. MUST be 0.0 for safe governance.
    """
    accepted = [
        r
        for r in results
        if r.modification_proposed is True and r.metadata.get("modification_accepted", False)
    ]

    if not accepted:
        return 0.0

    violations = sum(1 for r in accepted if not r.metadata.get("invariants_preserved", True))

    return violations / len(accepted)
