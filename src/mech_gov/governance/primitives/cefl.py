# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
CEFL — Candidate Expansion and Freezing for R2 Mechanical regime.

Non-agentic implementation (the design spec §2.4):
  1. Generate N candidate decisions via independent LLM calls
  2. Freeze candidates (no further modification)
  3. Score each candidate against policy criteria
  4. Select highest-scoring candidate

Key: the LLM cannot influence which candidate is selected.
"""

from __future__ import annotations

import json
from typing import Any

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.llm.base import LLMInterface


def _score_candidate(parsed: dict[str, Any], case: BankingCase) -> float:
    """Score a single candidate response against policy criteria.

    Scoring rubric:
      - Valid decision: +1.0
      - Has pro arguments: +0.5 per argument (max 1.0)
      - Has con arguments: +0.5 per argument (max 1.0)
      - Rationale present and substantive (>30 words): +1.0
      - Decision-specific bonus:
          - DEFER with deferral_info_needed: +0.5
          - CONDITIONAL with conditions: +0.5

    Returns:
        Score in [0, ~4.0]
    """
    score = 0.0

    # Valid decision
    raw_decision = parsed.get("decision", "").strip().upper()
    valid_decisions = {d.value for d in Decision}
    if raw_decision in valid_decisions:
        score += 1.0

    # Pro arguments
    pro = parsed.get("pro_arguments", [])
    if isinstance(pro, list):
        score += min(len(pro) * 0.5, 1.0)

    # Con arguments
    con = parsed.get("con_arguments", [])
    if isinstance(con, list):
        score += min(len(con) * 0.5, 1.0)

    # Rationale quality
    rationale = parsed.get("rationale", "")
    if isinstance(rationale, str) and len(rationale.split()) > 30:
        score += 1.0

    # Decision-specific completeness
    if raw_decision == "DEFER" and parsed.get("deferral_info_needed"):
        score += 0.5
    if raw_decision == "CONDITIONAL" and parsed.get("conditions"):
        score += 0.5

    return score


def _parse_candidate(raw: str) -> dict[str, Any]:
    """Parse a candidate LLM response into a dict."""
    import re

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    # Strip trailing commas before } or ] (common LLM JSON error)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def generate_cefl_candidates(
    case: BankingCase,
    llm: LLMInterface,
    system_prompt: str,
    user_message: str,
    n_candidates: int = 3,
    temperature: float = 0.7,
) -> list[dict[str, Any]]:
    """Generate N independent candidate responses.

    Each candidate is generated with temperature > 0 to get diversity.
    Candidates are frozen immediately after generation.

    Args:
        case: The banking case
        llm: LLM interface
        system_prompt: Governance policy prompt
        user_message: Case description message
        n_candidates: Number of candidates to generate
        temperature: Sampling temperature for candidate diversity

    Returns:
        List of (parsed_dict, raw_response, score) tuples, sorted by score desc
    """
    candidates: list[dict[str, Any]] = []

    for i in range(n_candidates):
        response = llm.invoke(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
        )

        parsed = _parse_candidate(response.content)
        score = _score_candidate(parsed, case)

        candidates.append(
            {
                "index": i,
                "parsed": parsed,
                "raw": response.content,
                "score": score,
                "tokens": response.input_tokens + response.output_tokens,
                "latency_ms": response.latency_ms,
            }
        )

    # Sort by score descending — best candidate first
    candidates.sort(key=lambda c: float(c["score"]), reverse=True)
    return candidates


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the highest-scoring frozen candidate.

    This selection is mechanical — the LLM cannot influence it.

    Returns:
        The best candidate dict
    """
    if not candidates:
        return {"parsed": {}, "raw": "", "score": 0.0, "tokens": 0, "latency_ms": 0.0}
    return candidates[0]
