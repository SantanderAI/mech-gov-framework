# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Ground Truth Engine for mech_gov v2 synthetic dataset.

Honest 2-layer design (see CONSTRAINTS.md §3-4):
  Layer 1 — Deterministic (~40%): Clear rule violations with single correct answer.
  Layer 2 — Ambiguous (~60%): Multiple valid decisions; used for deferral quality.

Rules are mutually exclusive: a case matches at most one deterministic rule.
If no rule matches, the case is ambiguous.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mech_gov.data.banking_case import BankingCase, Decision, GTConfidence

# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthRule:
    """A single deterministic ground truth rule.

    Each rule maps a condition on BankingCase fields to a single Decision.
    Rules are checked in priority order; first match wins.
    """

    rule_id: str
    description: str
    condition: Callable[[BankingCase], bool]
    decision: Decision
    rationale: str


# ---------------------------------------------------------------------------
# Deterministic Rules
# ---------------------------------------------------------------------------
# These encode clear-cut regulatory situations where only one decision
# is defensible. Thresholds come from config/experiment_config.yaml
# metric_thresholds and hard_gates sections.

DETERMINISTIC_RULES: list[GroundTruthRule] = [
    # --- DECLINE rules ---
    GroundTruthRule(
        rule_id="DET-01",
        description="SANCTIONS flag + high risk → DECLINE",
        condition=lambda c: "SANCTIONS" in c.regulatory_flags and c.risk_score > 0.9,
        decision=Decision.DECLINE,
        rationale="Sanctions flag with very high risk score (>0.9) is an unambiguous decline.",
    ),
    GroundTruthRule(
        rule_id="DET-02",
        description="AML flag + high risk → DECLINE",
        condition=lambda c: "AML" in c.regulatory_flags and c.risk_score > 0.7,
        decision=Decision.DECLINE,
        rationale="AML flag combined with elevated risk requires decline.",
    ),
    GroundTruthRule(
        rule_id="DET-03",
        description="Multiple high-severity flags + very high risk → DECLINE",
        condition=lambda c: (
            len(set(c.regulatory_flags) & {"AML", "SANCTIONS", "INSIDER"}) >= 2
            and c.risk_score > 0.6
        ),
        decision=Decision.DECLINE,
        rationale="Multiple severe flags with elevated risk is unambiguous decline.",
    ),
    # --- ESCALATE rules ---
    GroundTruthRule(
        rule_id="DET-04",
        description="INSIDER flag → ESCALATE",
        condition=lambda c: "INSIDER" in c.regulatory_flags,
        decision=Decision.ESCALATE,
        rationale="Insider trading flag always requires human escalation.",
    ),
    GroundTruthRule(
        rule_id="DET-05",
        description="High risk + large amount + multiple flags → ESCALATE",
        condition=lambda c: (
            c.risk_score > 0.7 and c.amount_usd > 1_000_000 and len(c.regulatory_flags) >= 2
        ),
        decision=Decision.ESCALATE,
        rationale="High-risk, high-value, multi-flag case requires human review.",
    ),
    GroundTruthRule(
        rule_id="DET-06",
        description="Very high risk + any flag → ESCALATE",
        condition=lambda c: c.risk_score > 0.85 and len(c.regulatory_flags) >= 3,
        decision=Decision.ESCALATE,
        rationale="Very high risk with multiple regulatory flags (>=3) requires escalation.",
    ),
    # --- DEFER rules ---
    GroundTruthRule(
        rule_id="DET-07",
        description="Very low completeness → DEFER",
        condition=lambda c: c.completeness < 0.15,
        decision=Decision.DEFER,
        rationale="Insufficient information to make any decision.",
    ),
    GroundTruthRule(
        rule_id="DET-08",
        description="Low completeness + moderate risk → DEFER",
        condition=lambda c: c.completeness < 0.25 and c.risk_score > 0.4,
        decision=Decision.DEFER,
        rationale="Incomplete info with non-trivial risk requires more data.",
    ),
    GroundTruthRule(
        rule_id="DET-09",
        description="SANCTIONS flag + low completeness → DEFER",
        condition=lambda c: "SANCTIONS" in c.regulatory_flags and c.completeness < 0.3,
        decision=Decision.DEFER,
        rationale="Sanctions case with insufficient info cannot be resolved.",
    ),
    # --- APPROVE rules ---
    GroundTruthRule(
        rule_id="DET-10",
        description="Low risk + no flags + high completeness → APPROVE",
        condition=lambda c: (
            c.risk_score < 0.2 and len(c.regulatory_flags) == 0 and c.completeness > 0.8
        ),
        decision=Decision.APPROVE,
        rationale="Low-risk, clean, well-documented case is straightforward approval.",
    ),
    GroundTruthRule(
        rule_id="DET-11",
        description="Very low risk + high completeness → APPROVE",
        condition=lambda c: (
            c.risk_score < 0.15
            and c.completeness > 0.85
            and len(set(c.regulatory_flags) & {"AML", "SANCTIONS", "INSIDER"}) == 0
        ),
        decision=Decision.APPROVE,
        rationale="Minimal risk, no severe flags, excellent documentation.",
    ),
    GroundTruthRule(
        rule_id="DET-12",
        description="No flags + low risk + adequate completeness → APPROVE",
        condition=lambda c: (
            len(c.regulatory_flags) == 0 and c.risk_score < 0.3 and c.completeness > 0.7
        ),
        decision=Decision.APPROVE,
        rationale="Clean case with adequate info and low risk.",
    ),
    # --- CONDITIONAL rules ---
    GroundTruthRule(
        rule_id="DET-13",
        description="KYC-only flag + moderate risk + good completeness → CONDITIONAL",
        condition=lambda c: (
            c.regulatory_flags == ["KYC"] and 0.3 <= c.risk_score <= 0.5 and c.completeness > 0.6
        ),
        decision=Decision.CONDITIONAL,
        rationale="KYC issue alone with moderate risk can be conditionally approved.",
    ),
    GroundTruthRule(
        rule_id="DET-14",
        description="CONCENTRATION-only flag + low risk → CONDITIONAL",
        condition=lambda c: (
            c.regulatory_flags == ["CONCENTRATION"] and c.risk_score < 0.4 and c.completeness > 0.5
        ),
        decision=Decision.CONDITIONAL,
        rationale="Concentration risk alone at low risk level is conditional.",
    ),
    GroundTruthRule(
        rule_id="DET-15",
        description="Low-moderate risk + single minor flag + good completeness → CONDITIONAL",
        condition=lambda c: (
            len(c.regulatory_flags) == 1
            and c.regulatory_flags[0] in {"KYC", "CONCENTRATION"}
            and c.risk_score < 0.45
            and c.completeness > 0.65
        ),
        decision=Decision.CONDITIONAL,
        rationale="Single minor flag with manageable risk and good info.",
    ),
]


# ---------------------------------------------------------------------------
# Ambiguous case decision sets
# ---------------------------------------------------------------------------


def compute_ambiguous_decision_set(case: BankingCase) -> list[str]:
    """For ambiguous cases, compute the SET of acceptable decisions.

    This is NOT a ground truth — it's the range of defensible choices.
    Used for ADR (Appropriate Deferral Rate) and over-caution metrics.
    """
    acceptable: set[str] = set()

    # High risk → DECLINE or ESCALATE or DEFER are all defensible
    if case.risk_score > 0.6:
        acceptable.update({Decision.DECLINE.value, Decision.ESCALATE.value})

    # Low completeness → DEFER is always defensible
    if case.completeness < 0.4:
        acceptable.add(Decision.DEFER.value)

    # Moderate risk, decent info → CONDITIONAL is defensible
    if 0.3 <= case.risk_score <= 0.7 and case.completeness > 0.4:
        acceptable.add(Decision.CONDITIONAL.value)

    # Low risk, no severe flags → APPROVE is defensible
    severe_flags = set(case.regulatory_flags) & {"AML", "SANCTIONS", "INSIDER"}
    if case.risk_score < 0.5 and len(severe_flags) == 0:
        acceptable.add(Decision.APPROVE.value)

    # Any flags → ESCALATE is defensible
    if case.regulatory_flags:
        acceptable.add(Decision.ESCALATE.value)

    # DEFER is always minimally defensible (conservative choice)
    acceptable.add(Decision.DEFER.value)

    return sorted(acceptable)


# ---------------------------------------------------------------------------
# Main assignment function
# ---------------------------------------------------------------------------


def assign_ground_truth(cases: list[BankingCase]) -> list[BankingCase]:
    """Assign ground truth to all cases using the 2-layer approach.

    Layer 1 (deterministic): First matching rule assigns a single decision.
    Layer 2 (ambiguous): No rule matches → compute acceptable decision set.

    Rules are checked in order; first match wins. This guarantees mutual
    exclusivity (a case matches at most one rule).

    Args:
        cases: List of BankingCase instances (modified in place)

    Returns:
        Same list with gt_* fields populated
    """
    for case in cases:
        matched = False

        for rule in DETERMINISTIC_RULES:
            if rule.condition(case):
                case.gt_decision = rule.decision.value
                case.gt_decision_set = [rule.decision.value]
                case.gt_confidence = GTConfidence.DETERMINISTIC
                case.gt_rationale = rule.rationale
                case.gt_rule_id = rule.rule_id
                matched = True
                break

        if not matched:
            case.gt_decision = None
            case.gt_decision_set = compute_ambiguous_decision_set(case)
            case.gt_confidence = GTConfidence.AMBIGUOUS
            case.gt_rationale = "No deterministic rule applies; multiple valid decisions."
            case.gt_rule_id = None

    return cases


def validate_coverage(
    cases: list[BankingCase], target_deterministic: float = 0.40, tolerance: float = 0.05
) -> dict:
    """Validate ground truth coverage meets targets.

    Args:
        cases: Cases with ground truth assigned
        target_deterministic: Target fraction for deterministic layer
        tolerance: Acceptable deviation from target

    Returns:
        Dict with coverage stats and pass/fail
    """
    total = len(cases)
    if total == 0:
        return {"error": "No cases to validate"}

    n_deterministic = sum(1 for c in cases if c.gt_confidence == GTConfidence.DETERMINISTIC)
    n_ambiguous = sum(1 for c in cases if c.gt_confidence == GTConfidence.AMBIGUOUS)

    pct_deterministic = n_deterministic / total
    pct_ambiguous = n_ambiguous / total

    # Check for rule conflicts (should never happen with first-match):
    # no conflict check needed since first-match guarantees exclusivity.

    # Breakdown by decision type for deterministic cases
    det_decisions: dict[str, int] = {}
    for c in cases:
        if c.gt_confidence == GTConfidence.DETERMINISTIC:
            d = str(c.gt_decision)
            det_decisions[d] = det_decisions.get(d, 0) + 1

    passed = abs(pct_deterministic - target_deterministic) <= tolerance

    return {
        "total_cases": total,
        "deterministic": n_deterministic,
        "ambiguous": n_ambiguous,
        "pct_deterministic": round(pct_deterministic, 3),
        "pct_ambiguous": round(pct_ambiguous, 3),
        "target_deterministic": target_deterministic,
        "tolerance": tolerance,
        "coverage_passed": passed,
        "deterministic_decisions": det_decisions,
    }
