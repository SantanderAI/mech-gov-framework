# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
R1 Text-Only Governance Regime for mech_gov v2.

Pipeline (the design spec §2.2):
  1. Construct prompt: system_prompt (policy) + case description
  2. Send to LLM
  3. Parse response: extract decision, rationale, pro/con arguments
  4. Return DecisionResult

No mechanical enforcement — all interpretation is by the LLM.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.governance.policy_templates import load_template
from mech_gov.governance.regime import DecisionResult, GovernanceRegime
from mech_gov.llm.base import LLMInterface

logger = logging.getLogger("mech_gov.governance.r1")

# Valid decision strings for parsing
_VALID_DECISIONS = {d.value for d in Decision}


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _extract_decision(parsed: dict[str, Any]) -> Decision | None:
    """Extract and validate decision from parsed response."""
    raw_decision = parsed.get("decision", "").strip().upper()
    if raw_decision in _VALID_DECISIONS:
        return Decision(raw_decision)
    return None


class R1TextOnly(GovernanceRegime):
    """R1: Text-only governance — LLM interprets policy without mechanical enforcement.

    The LLM receives the full governance policy as a system prompt and the
    case as a user message. All decision-making is delegated to the LLM's
    interpretation of the policy text. No constraints are enforced mechanically.
    """

    def __init__(self, template_name: str = "r1_system_prompt", temperature: float = 0.7):
        self._system_prompt = load_template(template_name)
        self._temperature = temperature

    @property
    def regime_name(self) -> str:
        return "R1"

    def process_case(
        self,
        case: BankingCase,
        llm: LLMInterface,
        entropy_seed: int | None = None,
    ) -> DecisionResult:
        """Process a case through R1 text-only pipeline.

        Steps:
          1. Format case as text prompt
          2. Invoke LLM with system prompt + case
          3. Parse structured JSON response
          4. Handle non-compliance (missing fields, invalid decision)
        """
        user_message = (
            "Please evaluate the following banking transaction case and provide "
            "your decision in the required JSON format.\n\n"
            f"{case.to_prompt()}"
        )

        logger.debug("[R1] %s: invoking LLM...", case.case_id)
        start_ms = time.perf_counter() * 1000

        llm_response = llm.invoke(
            system_prompt=self._system_prompt,
            user_message=user_message,
            temperature=self._temperature,
        )

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        logger.debug(
            "[R1] %s: LLM responded in %.0fms (%d+%d tok)",
            case.case_id,
            elapsed_ms,
            llm_response.input_tokens,
            llm_response.output_tokens,
        )

        # Parse response
        parsed = _parse_llm_json(llm_response.content)
        decision = _extract_decision(parsed)

        # Handle non-compliance: if we can't parse a valid decision, flag it
        metadata: dict[str, Any] = {}
        if decision is None:
            decision = Decision.ESCALATE
            metadata["parse_failure"] = True
            metadata["raw_decision"] = parsed.get("decision", "UNPARSEABLE")
            logger.warning(
                "[R1] %s: parse failure, raw='%s' → forced ESCALATE",
                case.case_id,
                parsed.get("decision", "UNPARSEABLE"),
            )
        else:
            logger.debug("[R1] %s: decision=%s", case.case_id, decision.value)

        rationale = parsed.get("rationale", "")
        pro_args = parsed.get("pro_arguments", [])
        con_args = parsed.get("con_arguments", [])

        # Ensure lists
        if not isinstance(pro_args, list):
            pro_args = [str(pro_args)] if pro_args else []
        if not isinstance(con_args, list):
            con_args = [str(con_args)] if con_args else []

        # Flag missing rationale components (for FVS/FSR metrics later)
        if not pro_args:
            metadata["missing_pro_arguments"] = True
        if not con_args:
            metadata["missing_con_arguments"] = True

        deferral_text = parsed.get("deferral_info_needed")
        conditions_text = parsed.get("conditions")

        return DecisionResult(
            case_id=case.case_id,
            regime=self.regime_name,
            decision=decision,
            rationale=rationale,
            pro_arguments=pro_args,
            con_arguments=con_args,
            deferral_text=deferral_text,
            conditions_text=conditions_text,
            metadata=metadata,
            llm_raw_response=llm_response.content,
            processing_time_ms=elapsed_ms,
            tokens_used=llm_response.input_tokens + llm_response.output_tokens,
        )
