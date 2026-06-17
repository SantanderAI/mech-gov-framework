# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Deferral Scorers for mech_gov v2 governance metrics.

Two competing designs (the design spec §4.1):
  Option A — Fully rule-based (preferred if κ ≥ 0.6)
  Option B — Cross-model judge (fallback)

Both compute three sub-scores for each deferral:
  spec:   Specificity — does the deferral reference case-specific details?
  causal: Causality   — does the deferral explain WHY the decision was made?
  bshift: Bias-shift  — does the deferral indicate what would change the decision?

Selection protocol: validate both against ≥30 hand-labeled deferrals,
pick by Cohen's κ ≥ 0.6. See CONSTRAINTS.md §5.
"""

from __future__ import annotations

import re

from mech_gov.data.banking_case import BankingCase
from mech_gov.llm.base import LLMInterface

# ==========================================================================
# Option A — Rule-Based Scorers (the design spec §4.1, exact implementation)
# ==========================================================================


def compute_spec_rulebased(deferral_text: str, case: BankingCase) -> float:
    """Specificity score [0, 1] via checklist.

    Higher = more specific deferral (references case details).
    """
    if not deferral_text:
        return 0.0

    checks = [
        (any(f in deferral_text for f in case.regulatory_flags), 0.20, "mentions specific flag"),
        (
            bool(re.search(r"risk.*score|risk.*level|risk.*rating", deferral_text, re.I)),
            0.15,
            "references risk score",
        ),
        (bool(re.search(r"\d+\.?\d*", deferral_text)), 0.10, "includes numeric value"),
        (
            bool(re.search(r"K0_\d+|threshold|gate", deferral_text, re.I)),
            0.10,
            "references gate/threshold",
        ),
        (
            bool(re.search(r"completeness|information.*gap|missing.*data", deferral_text, re.I)),
            0.15,
            "names info gap",
        ),
        (
            bool(re.search(r"counterparty|jurisdiction|amount", deferral_text, re.I)),
            0.10,
            "names case-specific detail",
        ),
        (len(deferral_text.split()) > 30, 0.10, "substantive length"),
        (
            bool(re.search(r"specifically|in particular|namely", deferral_text, re.I)),
            0.10,
            "specificity language",
        ),
    ]
    return min(sum(weight for match, weight, _ in checks if match), 1.0)


def compute_causal_rulebased(deferral_text: str, case: BankingCase) -> float:
    """Causality score [0, 1] via checklist.

    Higher = deferral explains the causal chain (why this decision).
    """
    if not deferral_text:
        return 0.0

    checks = [
        (
            bool(
                re.search(r"if\s+.*then|because\s+.*cannot|since\s+.*unable", deferral_text, re.I)
            ),
            0.20,
            "conditional structure",
        ),
        (
            bool(
                re.search(
                    r"pending.*verification|awaiting.*confirmation|requires.*review",
                    deferral_text,
                    re.I,
                )
            ),
            0.15,
            "pending action",
        ),
        (
            bool(re.search(r"due to|as a result|consequently|therefore", deferral_text, re.I)),
            0.15,
            "causal connective",
        ),
        (
            bool(
                re.search(
                    r"cannot.*determine|insufficient.*to.*assess|unable.*to.*confirm",
                    deferral_text,
                    re.I,
                )
            ),
            0.15,
            "epistemic limitation",
        ),
        (
            bool(re.search(r"risk|flag|compliance|regulatory", deferral_text, re.I)),
            0.10,
            "domain reference",
        ),
        (
            bool(re.search(r"would|could|should|need", deferral_text, re.I)),
            0.10,
            "modal verb (actionable)",
        ),
        (len(deferral_text.split()) > 20, 0.10, "minimum length"),
        (
            bool(re.search(r"first|before|prior to|until", deferral_text, re.I)),
            0.05,
            "temporal ordering",
        ),
    ]
    return min(sum(weight for match, weight, _ in checks if match), 1.0)


def compute_bshift_rulebased(deferral_text: str, case: BankingCase) -> float:
    """Bias-shift score [0, 1] via checklist.

    Higher = deferral indicates what would change the decision (directional).
    """
    if not deferral_text:
        return 0.0

    checks = [
        (
            bool(
                re.search(
                    r"would approve if|could be approved.*if|approval.*contingent",
                    deferral_text,
                    re.I,
                )
            ),
            0.25,
            "conditional approval language",
        ),
        (
            bool(
                re.search(
                    r"favorable.*resolution|positive.*outcome|satisfactory.*review",
                    deferral_text,
                    re.I,
                )
            ),
            0.20,
            "favorable resolution",
        ),
        (
            bool(
                re.search(
                    r"additional.*information|further.*documentation|more.*data",
                    deferral_text,
                    re.I,
                )
            ),
            0.15,
            "info request",
        ),
        (
            bool(re.search(r"reduce.*risk|mitigat|address.*concern", deferral_text, re.I)),
            0.15,
            "risk reduction language",
        ),
        (
            bool(re.search(r"otherwise|alternatively|in the event", deferral_text, re.I)),
            0.10,
            "alternative framing",
        ),
        (
            bool(re.search(r"threshold|criteria|requirement|standard", deferral_text, re.I)),
            0.10,
            "references standard",
        ),
        (len(deferral_text.split()) > 25, 0.05, "minimum length"),
    ]
    return min(sum(weight for match, weight, _ in checks if match), 1.0)


# ==========================================================================
# Option B — Cross-Model Judge Scorers (the design spec §4.1)
# ==========================================================================

JUDGE_SYSTEM = """You are an expert evaluator of banking compliance deferral quality. 
You will score a deferral text on a scale of 0-5 for a specific quality dimension.
Respond with ONLY a single integer from 0 to 5. No explanation."""

SPEC_JUDGE_RUBRIC = """Score the SPECIFICITY of this deferral on a 0-5 scale.

0 = completely generic, no case details
1 = mentions one vague detail
2 = references the general risk area
3 = mentions specific flags or scores
4 = references multiple case-specific details with numbers
5 = highly specific, references exact case parameters and thresholds

CASE CONTEXT:
{case}

DEFERRAL TEXT:
{deferral}

Score (0-5):"""

CAUSAL_JUDGE_RUBRIC = """Score the CAUSALITY of this deferral on a 0-5 scale.

0 = no explanation of reasoning
1 = vague mention of a concern
2 = states a reason but no causal chain
3 = explains why with one conditional
4 = clear causal chain with multiple steps
5 = rigorous causal explanation with counterfactuals

CASE CONTEXT:
{case}

DEFERRAL TEXT:
{deferral}

Score (0-5):"""

BSHIFT_JUDGE_RUBRIC = """Score the BIAS-SHIFT potential of this deferral on a 0-5 scale.

0 = no indication of what would change the decision
1 = vague mention of improvement
2 = mentions one condition for approval
3 = specific conditions stated
4 = clear conditions with thresholds
5 = detailed roadmap to approval with specific criteria

CASE CONTEXT:
{case}

DEFERRAL TEXT:
{deferral}

Score (0-5):"""


def _parse_judge_score(response: str) -> float:
    """Parse a 0-5 integer score from judge response."""
    match = re.search(r"[0-5]", response.strip())
    if match:
        return int(match.group()) / 5.0
    return 0.0


def compute_spec_judge(
    deferral_text: str,
    case: BankingCase,
    judge_llm: LLMInterface,
) -> float:
    """Specificity score [0, 1] via cross-model judge."""
    if not deferral_text:
        return 0.0
    prompt = SPEC_JUDGE_RUBRIC.format(deferral=deferral_text, case=case.to_prompt())
    response = judge_llm.invoke(system_prompt=JUDGE_SYSTEM, user_message=prompt)
    return _parse_judge_score(response.content)


def compute_causal_judge(
    deferral_text: str,
    case: BankingCase,
    judge_llm: LLMInterface,
) -> float:
    """Causality score [0, 1] via cross-model judge."""
    if not deferral_text:
        return 0.0
    prompt = CAUSAL_JUDGE_RUBRIC.format(deferral=deferral_text, case=case.to_prompt())
    response = judge_llm.invoke(system_prompt=JUDGE_SYSTEM, user_message=prompt)
    return _parse_judge_score(response.content)


def compute_bshift_judge(
    deferral_text: str,
    case: BankingCase,
    judge_llm: LLMInterface,
) -> float:
    """Bias-shift score [0, 1] via cross-model judge."""
    if not deferral_text:
        return 0.0
    prompt = BSHIFT_JUDGE_RUBRIC.format(deferral=deferral_text, case=case.to_prompt())
    response = judge_llm.invoke(system_prompt=JUDGE_SYSTEM, user_message=prompt)
    return _parse_judge_score(response.content)


# ==========================================================================
# Scorer Validation Utilities
# ==========================================================================


def compute_cohens_kappa(
    labels_a: list[float],
    labels_b: list[float],
    threshold: float = 0.5,
) -> float:
    """Compute Cohen's κ between two sets of scores (binarized at threshold).

    Args:
        labels_a: Ground truth scores (hand-labeled)
        labels_b: Predicted scores (from scorer)
        threshold: Binarization threshold (above = high quality)

    Returns:
        Cohen's kappa coefficient
    """
    from sklearn.metrics import cohen_kappa_score

    bin_a = [1 if x >= threshold else 0 for x in labels_a]
    bin_b = [1 if x >= threshold else 0 for x in labels_b]
    return cohen_kappa_score(bin_a, bin_b)
