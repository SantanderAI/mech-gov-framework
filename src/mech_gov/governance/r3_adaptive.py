# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
R3 Adaptive Policy Governance Regime for mech_gov v2.

Exploratory / stub implementation (the design spec §2.5):
  - Extends R2 Mechanical
  - Adds modification proposal mechanism (LLM can propose policy changes)
  - Invariant checking (hard constraints that CANNOT be violated)
  - Drift budget δ tracking (bounds cumulative modification)

NOT a development priority for v2. Kept for reproducing paper v1 results
and as a forward reference (Appendix B).

Key invariants:
  1. SANCTIONS flag → never APPROVE (hard gate K0_6 equivalent)
  2. Risk > 0.95 → never APPROVE
  3. Completeness < 0.1 → always DEFER
  4. Drift budget cannot exceed δ_max
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.governance.regime import DecisionResult
from mech_gov.llm.base import LLMInterface

logger = logging.getLogger("mech_gov.governance.r3")

# =========================================================================
# Invariant System
# =========================================================================


@dataclass
class Invariant:
    """A hard constraint that cannot be violated by R3 modifications."""

    invariant_id: str
    description: str
    check_fn: str  # Name of check function (for serialization)


@dataclass
class InvariantCheckResult:
    """Result of checking all invariants against a proposed modification."""

    all_preserved: bool
    violations: list[str]
    checked: int


DEFAULT_INVARIANTS = [
    Invariant(
        invariant_id="INV_1",
        description="SANCTIONS flag → never APPROVE",
        check_fn="check_inv_sanctions",
    ),
    Invariant(
        invariant_id="INV_2",
        description="Risk > 0.95 → never APPROVE",
        check_fn="check_inv_high_risk",
    ),
    Invariant(
        invariant_id="INV_3",
        description="Completeness < 0.1 → always DEFER",
        check_fn="check_inv_low_completeness",
    ),
]


def check_inv_sanctions(case: BankingCase, decision: Decision) -> str | None:
    """INV_1: SANCTIONS flag → never APPROVE."""
    if "SANCTIONS" in case.regulatory_flags and decision == Decision.APPROVE:
        return "INV_1 violated: APPROVE with SANCTIONS flag"
    return None


def check_inv_high_risk(case: BankingCase, decision: Decision) -> str | None:
    """INV_2: Risk > 0.95 → never APPROVE."""
    if case.risk_score > 0.95 and decision == Decision.APPROVE:
        return "INV_2 violated: APPROVE with risk > 0.95"
    return None


def check_inv_low_completeness(case: BankingCase, decision: Decision) -> str | None:
    """INV_3: Completeness < 0.1 → always DEFER."""
    if case.completeness < 0.1 and decision != Decision.DEFER:
        return f"INV_3 violated: {decision.value} with completeness < 0.1"
    return None


INVARIANT_CHECKS = {
    "check_inv_sanctions": check_inv_sanctions,
    "check_inv_high_risk": check_inv_high_risk,
    "check_inv_low_completeness": check_inv_low_completeness,
}


def check_all_invariants(
    case: BankingCase,
    decision: Decision,
    invariants: list[Invariant] | None = None,
) -> InvariantCheckResult:
    """Check all invariants against a proposed decision.

    Returns:
        InvariantCheckResult with pass/fail and violations list
    """
    if invariants is None:
        invariants = DEFAULT_INVARIANTS

    violations = []
    for inv in invariants:
        check_fn = INVARIANT_CHECKS.get(inv.check_fn)
        if check_fn:
            violation = check_fn(case, decision)
            if violation:
                violations.append(violation)

    return InvariantCheckResult(
        all_preserved=len(violations) == 0,
        violations=violations,
        checked=len(invariants),
    )


# =========================================================================
# Drift Budget
# =========================================================================


@dataclass
class DriftBudget:
    """Tracks cumulative modification drift.

    δ_current starts at 0 and increases with each accepted modification.
    Once δ_current ≥ δ_max, no further modifications are permitted.
    """

    delta_max: float = 1.0  # Maximum cumulative drift
    delta_current: float = 0.0  # Current drift
    modifications_accepted: int = 0
    modifications_rejected: int = 0

    @property
    def remaining(self) -> float:
        return max(self.delta_max - self.delta_current, 0.0)

    @property
    def exhausted(self) -> bool:
        return self.delta_current >= self.delta_max

    def propose(self, cost: float) -> bool:
        """Check if a modification with given cost is within budget.

        Args:
            cost: Drift cost of the proposed modification [0, 1]

        Returns:
            True if modification is within budget
        """
        return (self.delta_current + cost) <= self.delta_max

    def accept(self, cost: float):
        """Accept a modification and update budget."""
        self.delta_current += cost
        self.modifications_accepted += 1

    def reject(self):
        """Reject a modification (no budget change)."""
        self.modifications_rejected += 1


# =========================================================================
# R3 Adaptive Regime
# =========================================================================

MODIFICATION_PROMPT_SUFFIX = """

ADDITIONAL INSTRUCTION (R3 Adaptive Mode):
After providing your decision, you may OPTIONALLY propose a policy modification 
if you believe the current policy is suboptimal for this type of case.

If you propose a modification, add this to your JSON response:
  "modification_proposed": true,
  "modification_description": "<what you would change>",
  "modification_justification": "<why this improves governance>",
  "modification_cost": <float 0.0-1.0, how significant is this change>

If you do NOT propose a modification, set:
  "modification_proposed": false
"""


class R3Adaptive(R2Mechanical):
    """R3: Adaptive governance — bounded self-modification over R2.

    Extends R2 with:
      1. LLM can propose policy modifications
      2. All modifications checked against invariants
      3. Drift budget limits cumulative change
      4. Modifications that violate invariants or exceed budget are rejected
    """

    def __init__(
        self,
        drift_budget_max: float = 1.0,
        invariants: list[Invariant] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._drift_budget = DriftBudget(delta_max=drift_budget_max)
        self._invariants = invariants or DEFAULT_INVARIANTS
        self._accepted_modifications: list[dict[str, Any]] = []

    @property
    def regime_name(self) -> str:
        return "R3"

    @property
    def drift_budget(self) -> DriftBudget:
        return self._drift_budget

    def process_case(
        self,
        case: BankingCase,
        llm: LLMInterface,
        entropy_seed: int | None = None,
    ) -> DecisionResult:
        """Process case through R3 pipeline: R2 + modification proposal."""

        # Run full R2 pipeline first
        logger.debug("[R3] %s: running R2 base pipeline", case.case_id)
        result = super().process_case(case, llm, entropy_seed)

        # Override regime name
        result.regime = self.regime_name

        # Check if LLM proposed a modification (from raw response)
        modification = self._extract_modification(result)

        if modification and modification.get("modification_proposed"):
            result.modification_proposed = True
            cost = float(modification.get("modification_cost", 0.5))
            logger.info(
                "[R3] %s: modification proposed (cost=%.2f, budget_remaining=%.2f)",
                case.case_id,
                cost,
                self._drift_budget.remaining,
            )

            # Check invariants
            inv_result = check_all_invariants(case, result.decision, self._invariants)

            # Check drift budget
            within_budget = self._drift_budget.propose(cost)

            if inv_result.all_preserved and within_budget:
                # Accept modification
                self._drift_budget.accept(cost)
                logger.info(
                    "[R3] %s: modification ACCEPTED (budget now %.2f/%.2f)",
                    case.case_id,
                    self._drift_budget.delta_current,
                    self._drift_budget.delta_max,
                )
                self._accepted_modifications.append(
                    {
                        "case_id": case.case_id,
                        "description": modification.get("modification_description", ""),
                        "justification": modification.get("modification_justification", ""),
                        "cost": cost,
                    }
                )
                result.metadata["modification_accepted"] = True
                result.metadata["modification_description"] = modification.get(
                    "modification_description", ""
                )
                result.metadata["invariants_preserved"] = True
            else:
                # Reject modification
                self._drift_budget.reject()
                result.metadata["modification_accepted"] = False
                result.metadata["invariants_preserved"] = inv_result.all_preserved
                if not inv_result.all_preserved:
                    result.metadata["invariant_violations"] = inv_result.violations
                    logger.warning(
                        "[R3] %s: modification REJECTED — invariant violations: %s",
                        case.case_id,
                        inv_result.violations,
                    )
                if not within_budget:
                    result.metadata["drift_budget_exceeded"] = True
                    logger.warning(
                        "[R3] %s: modification REJECTED — budget exceeded "
                        "(cost=%.2f, remaining=%.2f)",
                        case.case_id,
                        cost,
                        self._drift_budget.remaining,
                    )

            result.metadata["drift_budget_current"] = round(self._drift_budget.delta_current, 4)
            result.metadata["drift_budget_remaining"] = round(self._drift_budget.remaining, 4)
        else:
            result.modification_proposed = False
            result.metadata["invariants_preserved"] = True

        return result

    def _extract_modification(self, result: DecisionResult) -> dict | None:
        """Extract modification proposal from LLM response."""
        raw = result.llm_raw_response or ""
        if not raw:
            return None

        # Try to parse from the JSON response
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if "modification_proposed" in parsed:
                    return parsed
        except (json.JSONDecodeError, AttributeError):
            pass

        return None
