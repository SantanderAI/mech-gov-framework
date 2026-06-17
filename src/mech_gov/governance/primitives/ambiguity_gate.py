# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Ambiguity Gate K0_11 — Post-LLM completeness check for R2 Mechanical regime.

Forces DEFER or ESCALATE when information completeness is below θ_ι.
Applied AFTER the LLM call as a final override. Matches the design spec §2.4.

Note: K0_11 also appears in hard_gates.py as a pre-LLM gate. This module
provides the post-LLM version that can override the LLM's decision even
if the pre-LLM gate didn't trigger (e.g., if completeness is borderline).
"""

from __future__ import annotations

from mech_gov.data.banking_case import BankingCase, Decision


def ambiguity_gate(
    case: BankingCase,
    theta_iota: float = 0.3,
    risk_escalation_threshold: float = 0.7,
) -> Decision | None:
    """Force DEFER or ESCALATE when information is too incomplete.

    the design spec §2.4:
      if completeness < θ_ι:
          if risk_score > 0.7 → ESCALATE (high risk + low info)
          else → DEFER (low risk + low info)

    Args:
        case: The banking case to check
        theta_iota: Completeness threshold (default 0.3 from paper)
        risk_escalation_threshold: Risk threshold for ESCALATE vs DEFER

    Returns:
        Decision.ESCALATE or Decision.DEFER if gate triggers, else None
    """
    if case.completeness < theta_iota:
        if case.risk_score > risk_escalation_threshold:
            return Decision.ESCALATE
        return Decision.DEFER
    return None
