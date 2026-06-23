# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
R2 Mechanical Governance Regime for mech_gov v2.

Pipeline (the design spec §2.3):
  1. Pre-LLM: Hard gates check — if any gate triggers, override mechanically
  2. E3 Entropy: Commit seed before LLM call
  3. LLM call with CEFL: generate N candidates, freeze, score, select best
  4. Post-LLM: I6Q check — verify argument quality (retry up to MAX_RETRIES)
  5. Post-LLM: Ambiguity gate K0_11 — force DEFER/ESCALATE if ι < θ_ι
  6. E3 Entropy: Reveal — verify seed wasn't conditioned on output
  7. Return DecisionResult with all metadata
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.governance.policy_templates import load_template
from mech_gov.governance.primitives.ambiguity_gate import ambiguity_gate
from mech_gov.governance.primitives.cefl import (
    generate_cefl_candidates,
    select_best_candidate,
)
from mech_gov.governance.primitives.entropy_e3 import e3_commit, e3_reveal
from mech_gov.governance.primitives.hard_gates import (
    build_default_gates,
    evaluate_hard_gates,
)
from mech_gov.governance.primitives.i6q import I6QConfig, check_i6q
from mech_gov.governance.primitives.privacy_gate import (
    PRIVACY_GATE_ID,
    PiiRecognizer,
    PrivacyConfig,
    RegexRecognizer,
    privacy_gate,
)
from mech_gov.governance.regime import DecisionResult, GovernanceRegime
from mech_gov.llm.base import LLMInterface

logger = logging.getLogger("mech_gov.governance.r2")


class R2Mechanical(GovernanceRegime):
    """R2: Mechanical governance — LLM output constrained by primitives.

    All five primitives are composed into a single pipeline:
      hard_gates → E3_commit → CEFL(LLM) → I6Q → ambiguity_gate → E3_reveal
    """

    def __init__(
        self,
        template_name: str = "r2_system_prompt",
        hard_gates_config: dict | None = None,
        i6q_config: I6QConfig | None = None,
        n_cefl_candidates: int = 3,
        theta_iota: float = 0.3,
        risk_escalation_threshold: float = 0.7,
        privacy_config: PrivacyConfig | None = None,
        privacy_recognizer: PiiRecognizer | None = None,
    ):
        self._system_prompt = load_template(template_name)
        self._gates = build_default_gates(hard_gates_config)
        self._gates_config = hard_gates_config
        self._i6q_config = i6q_config or I6QConfig()
        self._n_cefl_candidates = n_cefl_candidates
        self._theta_iota = theta_iota
        self._risk_escalation_threshold = risk_escalation_threshold
        self._privacy = privacy_config or PrivacyConfig()
        self._privacy_recognizer = privacy_recognizer or RegexRecognizer()

    @property
    def regime_name(self) -> str:
        return "R2"

    def process_case(
        self,
        case: BankingCase,
        llm: LLMInterface,
        entropy_seed: int | None = None,
    ) -> DecisionResult:
        """Process a case through the full R2 mechanical pipeline."""
        logger.debug("[R2] %s: starting pipeline", case.case_id)
        start_ms = time.perf_counter() * 1000
        metadata: dict[str, Any] = {}
        gates_triggered: list[str] = []
        total_tokens = 0

        # =====================================================================
        # Step 1: Pre-LLM Hard Gates
        # =====================================================================
        gate_result = evaluate_hard_gates(case, self._gates, self._gates_config)

        if gate_result is not None:
            gate_id, forced_decision, rationale = gate_result
            gates_triggered.append(gate_id)
            logger.info(
                "[R2] %s: hard gate %s triggered → %s", case.case_id, gate_id, forced_decision.value
            )
            elapsed_ms = time.perf_counter() * 1000 - start_ms

            return DecisionResult(
                case_id=case.case_id,
                regime=self.regime_name,
                decision=forced_decision,
                rationale=rationale,
                deferral_text=rationale,  # Bug 8 fix: populate deferral_text for consistent scoring
                metadata={"hard_gate_override": True, "gate_id": gate_id},
                processing_time_ms=elapsed_ms,
                tokens_used=0,
                gates_triggered=gates_triggered,
            )

        # =====================================================================
        # Step 1b: Privacy gate — minimize PII before the model is consulted
        # =====================================================================
        prompt_body = case.to_prompt()
        if self._privacy.enabled:
            pr = privacy_gate(prompt_body, self._privacy, self._privacy_recognizer)
            metadata["privacy_entities_found"] = pr.entities_found
            metadata["privacy_residual_pii"] = pr.residual_pii
            if pr.forced_decision is not None:
                gates_triggered.append(PRIVACY_GATE_ID)
                metadata["privacy_gate_override"] = True
                logger.info(
                    "[R2] %s: privacy gate %s → %s (no model call)",
                    case.case_id,
                    PRIVACY_GATE_ID,
                    pr.forced_decision.value,
                )
                rationale = (
                    "Privacy gate PRIV_0: the case could not be safely minimized "
                    "(residual direct identifiers above the configured budget), so "
                    "it was deferred without sending any content to the model."
                )
                elapsed_ms = time.perf_counter() * 1000 - start_ms
                return DecisionResult(
                    case_id=case.case_id,
                    regime=self.regime_name,
                    decision=pr.forced_decision,
                    rationale=rationale,
                    deferral_text=rationale,
                    metadata=metadata,
                    processing_time_ms=elapsed_ms,
                    tokens_used=0,
                    gates_triggered=gates_triggered,
                )
            prompt_body = pr.redacted_text

        # =====================================================================
        # Step 2: E3 Entropy — Commit
        # =====================================================================
        commit = e3_commit(entropy_seed)
        metadata["e3_nonce_hash"] = commit.nonce_hash
        logger.debug("[R2] %s: E3 commit hash=%s", case.case_id, commit.nonce_hash[:16])

        # =====================================================================
        # Step 3: CEFL — Candidate Expansion and Freezing
        # =====================================================================
        user_message = (
            "Please evaluate the following banking transaction case and provide "
            "your decision in the required JSON format.\n\n"
            f"{prompt_body}"
        )

        logger.debug(
            "[R2] %s: CEFL generating %d candidates...", case.case_id, self._n_cefl_candidates
        )
        candidates = generate_cefl_candidates(
            case=case,
            llm=llm,
            system_prompt=self._system_prompt,
            user_message=user_message,
            n_candidates=self._n_cefl_candidates,
            temperature=0.7,
        )

        total_tokens += sum(c["tokens"] for c in candidates)
        logger.debug(
            "[R2] %s: CEFL %d candidates, scores=%s",
            case.case_id,
            len(candidates),
            [round(c["score"], 2) for c in candidates],
        )
        cefl_scores = [{"index": c["index"], "score": c["score"]} for c in candidates]
        metadata["cefl_candidate_scores"] = cefl_scores

        best = select_best_candidate(candidates)
        parsed = best["parsed"]

        # Extract fields from best candidate
        raw_decision = parsed.get("decision", "").strip().upper()
        valid_decisions = {d.value for d in Decision}
        if raw_decision in valid_decisions:
            decision = Decision(raw_decision)
        else:
            decision = Decision.ESCALATE
            metadata["parse_failure"] = True

        rationale = parsed.get("rationale", "")
        pro_args = parsed.get("pro_arguments", [])
        con_args = parsed.get("con_arguments", [])
        if not isinstance(pro_args, list):
            pro_args = [str(pro_args)] if pro_args else []
        if not isinstance(con_args, list):
            con_args = [str(con_args)] if con_args else []

        # =====================================================================
        # Step 4: I6Q — Argument Quality Check (with retries)
        # =====================================================================
        i6q_result = check_i6q(pro_args, con_args, self._i6q_config)
        retries = 0

        while not i6q_result.passed and retries < self._i6q_config.max_retries:
            retries += 1
            logger.info(
                "[R2] %s: I6Q failed (%s), retry %d/%d",
                case.case_id,
                i6q_result.details,
                retries,
                self._i6q_config.max_retries,
            )
            metadata[f"i6q_retry_{retries}"] = i6q_result.details

            # Re-invoke LLM with explicit quality reminder
            retry_message = (
                f"{user_message}\n\n"
                "IMPORTANT: Your previous response failed the argument quality check. "
                f"Reason: {i6q_result.details}. "
                "Please provide more specific, detailed arguments with at least "
                f"{self._i6q_config.min_arg_tokens} words each."
            )

            response = llm.invoke(
                system_prompt=self._system_prompt,
                user_message=retry_message,
            )
            total_tokens += response.input_tokens + response.output_tokens

            # Re-parse
            import json
            import re

            text = response.content.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
            # Strip trailing commas before } or ] (common LLM JSON error)
            text = re.sub(r",\s*([}\]])", r"\1", text)
            try:
                retry_parsed = json.loads(text)
            except json.JSONDecodeError:
                retry_parsed = {}
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        retry_parsed = json.loads(match.group())
                    except json.JSONDecodeError:
                        logger.warning("[R2] %s: retry JSON parse failed, skipping", case.case_id)

            pro_args = retry_parsed.get("pro_arguments", [])
            con_args = retry_parsed.get("con_arguments", [])
            if not isinstance(pro_args, list):
                pro_args = []
            if not isinstance(con_args, list):
                con_args = []

            rationale = retry_parsed.get("rationale", rationale)

            raw_d = retry_parsed.get("decision", "").strip().upper()
            if raw_d in valid_decisions:
                decision = Decision(raw_d)

            i6q_result = check_i6q(pro_args, con_args, self._i6q_config)

        # If I6Q still fails after all retries → force ESCALATE
        i6q_passed = i6q_result.passed
        if not i6q_passed:
            decision = Decision.ESCALATE
            metadata["i6q_forced_escalate"] = True
            metadata["i6q_final_failure"] = i6q_result.details
            logger.warning(
                "[R2] %s: I6Q failed after %d retries → forced ESCALATE", case.case_id, retries
            )
        else:
            logger.debug("[R2] %s: I6Q passed (retries=%d)", case.case_id, retries)

        metadata["i6q_retries"] = retries

        # =====================================================================
        # Step 5: Ambiguity Gate K0_11 (post-LLM)
        # =====================================================================
        gate_override = ambiguity_gate(
            case,
            theta_iota=self._theta_iota,
            risk_escalation_threshold=self._risk_escalation_threshold,
        )

        if gate_override is not None:
            gates_triggered.append("K0_11_post")
            metadata["ambiguity_gate_override"] = True
            metadata["hard_gate_override"] = True
            metadata["original_llm_decision"] = decision.value
            logger.info(
                "[R2] %s: ambiguity gate K0_11 override %s → %s",
                case.case_id,
                metadata["original_llm_decision"],
                gate_override.value,
            )
            decision = gate_override

        # =====================================================================
        # Step 6: E3 Entropy — Reveal
        # =====================================================================
        reveal = e3_reveal(commit)
        metadata["e3_verified"] = reveal.verified
        if not reveal.verified:
            logger.error("[R2] %s: E3 commit-reveal FAILED", case.case_id)
        else:
            logger.debug("[R2] %s: E3 verified OK", case.case_id)

        elapsed_ms = time.perf_counter() * 1000 - start_ms
        logger.debug(
            "[R2] %s: pipeline done in %.0fms → %s (%d tok)",
            case.case_id,
            elapsed_ms,
            decision.value,
            total_tokens,
        )

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
            llm_raw_response=best.get("raw", ""),
            processing_time_ms=elapsed_ms,
            tokens_used=total_tokens,
            gates_triggered=gates_triggered,
            cefl_candidates=len(candidates),
            cefl_candidate_scores=cefl_scores,
            i6q_passed=i6q_passed,
            entropy_nonce=commit.nonce,
        )
