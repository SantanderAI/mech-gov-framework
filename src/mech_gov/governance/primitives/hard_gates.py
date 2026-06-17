# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Hard Gates (K0_6, K0_7, K0_8, K0_10, K0_12, K0_13, K0_14) for R2 Mechanical regime.

Evaluated BEFORE the LLM call. If any gate triggers, the decision is
mechanical and the LLM is NOT consulted. Matches the design spec §2.4.

Gate thresholds are loaded from config/experiment_config.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mech_gov.data.banking_case import BankingCase, Decision


@dataclass
class HardGate:
    """A single hard gate: condition → forced decision."""

    gate_id: str
    description: str
    condition: Callable[[BankingCase], bool]
    forced_decision: Decision
    rationale_template: str


def build_default_gates(config: dict | None = None) -> list[HardGate]:
    """Build the default hard gate list from config.

    Args:
        config: Optional dict from experiment_config.yaml['metric_thresholds']['hard_gates'].
                If None, uses paper defaults.

    Returns:
        Ordered list of HardGate instances. Evaluated in order; first match wins.
    """
    # Defaults from paper / experiment_config.yaml
    cfg = config or {}
    k06 = cfg.get("K0_6", {})
    k07 = cfg.get("K0_7", {})
    k08 = cfg.get("K0_8", {})
    k10 = cfg.get("K0_10", {})
    k12 = cfg.get("K0_12", {})
    k14 = cfg.get("K0_14", {})

    k06_risk = k06.get("risk_threshold", 0.9)
    k07_risk = k07.get("risk_threshold", 0.85)
    k07_min_flags = k07.get("min_flags", 3)
    k08_risk = k08.get("risk_threshold", 0.7)
    k08_amount = k08.get("amount_threshold_usd", 1_000_000)
    k10_comp = k10.get("completeness_threshold", 0.15)
    k12_risk = k12.get("risk_threshold", 0.7)
    k14_risk = k14.get("risk_threshold", 0.6)
    k14_min_severe = k14.get("min_severe_flags", 2)

    return [
        HardGate(
            gate_id="K0_6",
            description="High risk + sanctions → DECLINE",
            condition=lambda c: (c.risk_score > k06_risk and "SANCTIONS" in c.regulatory_flags),
            forced_decision=Decision.DECLINE,
            rationale_template=(
                "Hard gate K0_6 triggered: because the risk score ({risk_score:.3f}) exceeds "
                "the threshold of {threshold} and a SANCTIONS regulatory flag is present, "
                "this case cannot be approved under compliance requirements. Due to the "
                "combination of elevated risk level and sanctions exposure, the case "
                "requires an automatic DECLINE. A favorable resolution would only be "
                "possible if the sanctions flag were cleared and risk score reduced "
                "below the threshold criteria, pending verification by compliance."
            ),
        ),
        HardGate(
            gate_id="K0_7",
            description="High risk + multiple flags → ESCALATE",
            condition=lambda c: (
                c.risk_score > k07_risk and len(c.regulatory_flags) >= k07_min_flags
            ),
            forced_decision=Decision.ESCALATE,
            rationale_template=(
                "Hard gate K0_7 triggered: because the risk score ({risk_score:.3f}) exceeds "
                "the threshold of {threshold} and {n_flags} regulatory flags are present "
                "(minimum {min_flags} required for escalation), this case is unable to be "
                "assessed at the standard level. Due to the concentration of risk indicators, "
                "the case requires review by a senior compliance officer. A favorable "
                "resolution would need additional information to reduce risk and address "
                "the flagged concerns, specifically clearing the regulatory requirements "
                "before the standard criteria could be met."
            ),
        ),
        HardGate(
            gate_id="K0_8",
            description="Large amount + high risk → ESCALATE",
            condition=lambda c: (c.amount_usd > k08_amount and c.risk_score > k08_risk),
            forced_decision=Decision.ESCALATE,
            rationale_template=(
                "Hard gate K0_8 triggered: because the transaction amount (${amount:,.2f}) "
                "exceeds ${amount_threshold:,.0f} and the risk score ({risk_score:.3f}) exceeds "
                "the threshold of {threshold}, the case cannot be determined at this level. "
                "Due to the combination of high monetary exposure and elevated risk, "
                "this requires escalation for senior review. A favorable resolution could "
                "be approved if additional documentation were provided to mitigate the "
                "counterparty risk and reduce the overall risk score below the threshold "
                "standard, pending verification of the transaction purpose."
            ),
        ),
        HardGate(
            gate_id="K0_10",
            description="Very low completeness → DEFER",
            condition=lambda c: c.completeness < k10_comp,
            forced_decision=Decision.DEFER,
            rationale_template=(
                "Hard gate K0_10 triggered: because the information completeness "
                "({completeness:.3f}) falls below the minimum threshold of {threshold}, "
                "the system is unable to confirm the legitimacy of the transaction. "
                "Due to this critical information gap, the case cannot be assessed "
                "and requires deferral pending verification of missing data. "
                "Specifically, additional information is needed to reduce the "
                "completeness risk and meet the minimum threshold criteria. "
                "A favorable resolution would be possible if the completeness score "
                "were raised above {threshold} through further documentation."
            ),
        ),
        # K0_11 REMOVED from pre-LLM gates (Bug 2 fix).
        # It was shadowed by K0_10 (completeness < 0.15 ⊂ completeness < 0.3)
        # and is now only applied post-LLM via ambiguity_gate.py.
        HardGate(
            gate_id="K0_12",
            description="AML flag + high risk → DECLINE",
            condition=lambda c: ("AML" in c.regulatory_flags and c.risk_score > k12_risk),
            forced_decision=Decision.DECLINE,
            rationale_template=(
                "Hard gate K0_12 triggered: AML regulatory flag present and risk "
                "score ({risk_score:.3f}) exceeds threshold {threshold}. AML-flagged "
                "transactions with elevated risk require automatic decline."
            ),
        ),
        HardGate(
            gate_id="K0_13",
            description="INSIDER flag → ESCALATE",
            condition=lambda c: "INSIDER" in c.regulatory_flags,
            forced_decision=Decision.ESCALATE,
            rationale_template=(
                "Hard gate K0_13 triggered: INSIDER regulatory flag present. "
                "Insider trading indicators always require human escalation "
                "regardless of other case attributes."
            ),
        ),
        HardGate(
            gate_id="K0_14",
            description="Multiple severe flags + elevated risk → DECLINE",
            condition=lambda c: (
                len(set(c.regulatory_flags) & {"AML", "SANCTIONS", "INSIDER"}) >= k14_min_severe
                and c.risk_score > k14_risk
            ),
            forced_decision=Decision.DECLINE,
            rationale_template=(
                "Hard gate K0_14 triggered: {n_flags} severe regulatory flags "
                "present with risk score ({risk_score:.3f}) above threshold "
                "{threshold}. Multiple severe flags with elevated risk require "
                "automatic decline."
            ),
        ),
    ]


def evaluate_hard_gates(
    case: BankingCase,
    gates: list[HardGate] | None = None,
    config: dict | None = None,
) -> tuple | None:
    """Evaluate all hard gates against a case.

    Args:
        case: The banking case to evaluate
        gates: Pre-built gate list (if None, builds from config)
        config: Gate config dict (used only if gates is None)

    Returns:
        (gate_id, forced_decision, rationale) if a gate triggers, else None.
        First matching gate wins.
    """
    if gates is None:
        gates = build_default_gates(config)

    cfg = config or {}

    # Actual threshold values for rationale formatting
    gate_thresholds = {
        "K0_6": cfg.get("K0_6", {}).get("risk_threshold", 0.9),
        "K0_7": cfg.get("K0_7", {}).get("risk_threshold", 0.85),
        "K0_8": cfg.get("K0_8", {}).get("risk_threshold", 0.7),
        "K0_10": cfg.get("K0_10", {}).get("completeness_threshold", 0.15),
        "K0_12": cfg.get("K0_12", {}).get("risk_threshold", 0.7),
        "K0_13": 0.0,  # No threshold — INSIDER flag is unconditional
        "K0_14": cfg.get("K0_14", {}).get("risk_threshold", 0.6),
    }
    gate_min_flags = cfg.get("K0_7", {}).get("min_flags", 3)
    gate_amount_threshold = cfg.get("K0_8", {}).get("amount_threshold_usd", 1_000_000)

    for gate in gates:
        if gate.condition(case):
            decision = gate.forced_decision

            rationale = gate.rationale_template.format(
                risk_score=case.risk_score,
                completeness=case.completeness,
                amount=case.amount_usd,
                n_flags=len(case.regulatory_flags),
                threshold=gate_thresholds.get(gate.gate_id, 0.0),
                min_flags=gate_min_flags,
                amount_threshold=gate_amount_threshold,
                decision=decision.value,
            )

            return gate.gate_id, decision, rationale

    return None
