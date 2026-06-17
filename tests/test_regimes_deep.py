# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Branch-level tests for R1/R2/R3 pipelines, driven by a scripted callable LLM."""

import json

from mech_gov.data.banking_case import BankingCase, Decision, TransactionType
from mech_gov.governance.primitives.i6q import I6QConfig as _I6QConfig
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.governance.r3_adaptive import R3Adaptive
from mech_gov.llm.providers.callable_provider import CallableLLM

_GOOD_ARGS = {
    "pro_arguments": ["customer has strong verified history and stable income profile"],
    "con_arguments": ["counterparty exposure introduces moderate residual regulatory concern"],
}


def _llm_returning(payload: dict) -> CallableLLM:
    text = json.dumps(payload)

    def fn(system_prompt, user_message, temperature=0.0, max_tokens=2048):
        return text

    return CallableLLM(fn, model_id="scripted")


def _case(**kw) -> BankingCase:
    base = dict(
        case_id="c",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.4,
        completeness=0.6,
        regulatory_flags=[],
    )
    base.update(kw)
    return BankingCase(**base)


# ------------------------------- R1 -------------------------------


def test_r1_valid_decision():
    llm = _llm_returning({"decision": "APPROVE", "rationale": "ok", **_GOOD_ARGS})
    r = R1TextOnly().process_case(_case(), llm)
    assert r.decision == Decision.APPROVE
    assert r.pro_arguments


def test_r1_parse_failure_forces_escalate():
    llm = _llm_returning({"decision": "BANANA", "rationale": "x"})
    r = R1TextOnly().process_case(_case(), llm)
    assert r.decision == Decision.ESCALATE
    assert r.metadata.get("parse_failure") is True


def test_r1_strips_markdown_fences():
    payload = json.dumps({"decision": "DECLINE", "rationale": "y", **_GOOD_ARGS})

    def fn(system_prompt, user_message, temperature=0.0, max_tokens=2048):
        return f"```json\n{payload}\n```"

    r = R1TextOnly().process_case(_case(), CallableLLM(fn))
    assert r.decision == Decision.DECLINE


def test_r1_non_list_args_coerced_and_flagged():
    llm = _llm_returning({"decision": "APPROVE", "rationale": "z", "pro_arguments": "single"})
    r = R1TextOnly().process_case(_case(), llm)
    assert r.pro_arguments == ["single"]
    assert r.metadata.get("missing_con_arguments") is True


# ------------------------------- R2 -------------------------------


def _r2(**kw) -> R2Mechanical:
    kw.setdefault("i6q_config", _I6QConfig(min_arg_tokens=1, min_diversity=0.0, max_retries=1))
    return R2Mechanical(**kw)


def test_r2_approve_passes_i6q():
    llm = _llm_returning({"decision": "APPROVE", "rationale": "ok", **_GOOD_ARGS})
    r = _r2().process_case(_case(completeness=0.6), llm)
    assert r.decision == Decision.APPROVE
    assert r.i6q_passed is True
    assert r.entropy_nonce is not None
    assert r.cefl_candidates == 3
    assert r.metadata.get("e3_verified") is True


def test_r2_i6q_failure_forces_escalate():
    # Empty arguments always fail I6Q (missing pro/con) → forced ESCALATE.
    llm = _llm_returning({"decision": "APPROVE", "rationale": "ok"})
    r = _r2(i6q_config=_I6QConfig(min_arg_tokens=99, max_retries=1)).process_case(_case(), llm)
    assert r.decision == Decision.ESCALATE
    assert r.metadata.get("i6q_forced_escalate") is True


def test_r2_ambiguity_gate_defer_on_low_completeness():
    llm = _llm_returning({"decision": "APPROVE", "rationale": "ok", **_GOOD_ARGS})
    r = _r2().process_case(_case(completeness=0.2, risk_score=0.4), llm)
    assert r.decision == Decision.DEFER
    assert r.metadata.get("ambiguity_gate_override") is True


def test_r2_ambiguity_gate_escalate_on_low_completeness_high_risk():
    llm = _llm_returning({"decision": "APPROVE", "rationale": "ok", **_GOOD_ARGS})
    r = _r2().process_case(_case(completeness=0.2, risk_score=0.8), llm)
    assert r.decision == Decision.ESCALATE


def test_r2_i6q_retry_then_pass():
    # First 3 CEFL candidates have too-short args (fail I6Q); the retry call
    # returns substantive args that pass — exercises the retry+re-parse loop.
    from mech_gov.llm.providers.mock import MockLLM

    short = json.dumps(
        {"decision": "APPROVE", "rationale": "x", "pro_arguments": ["a"], "con_arguments": ["b"]}
    )
    good = json.dumps({"decision": "APPROVE", "rationale": "ok", **_GOOD_ARGS})
    llm = MockLLM(responses=[short, short, short, good])
    regime = R2Mechanical(i6q_config=_I6QConfig(min_arg_tokens=5, min_diversity=0.0, max_retries=2))
    r = regime.process_case(_case(completeness=0.6), llm)
    assert r.i6q_passed is True
    assert r.metadata.get("i6q_retries", 0) >= 1


def test_r2_parse_failure_marks_metadata():
    llm = _llm_returning({"decision": "NONSENSE", "rationale": "x", **_GOOD_ARGS})
    r = _r2().process_case(_case(), llm)
    assert r.metadata.get("parse_failure") is True


# ------------------------------- R3 -------------------------------


def _r3(**kw) -> R3Adaptive:
    kw.setdefault("i6q_config", _I6QConfig(min_arg_tokens=1, min_diversity=0.0, max_retries=1))
    return R3Adaptive(**kw)


def test_r3_no_modification():
    llm = _llm_returning(
        {"decision": "APPROVE", "rationale": "ok", **_GOOD_ARGS, "modification_proposed": False}
    )
    r = _r3().process_case(_case(), llm)
    assert r.regime == "R3"
    assert r.modification_proposed is False
    assert r.metadata.get("invariants_preserved") is True


def test_r3_modification_accepted():
    llm = _llm_returning(
        {
            "decision": "APPROVE",
            "rationale": "ok",
            **_GOOD_ARGS,
            "modification_proposed": True,
            "modification_description": "loosen tenure weight",
            "modification_justification": "improves recall",
            "modification_cost": 0.1,
        }
    )
    r = _r3(drift_budget_max=1.0).process_case(_case(risk_score=0.3), llm)
    assert r.modification_proposed is True
    assert r.metadata.get("modification_accepted") is True


def test_r3_modification_rejected_budget():
    llm = _llm_returning(
        {
            "decision": "APPROVE",
            "rationale": "ok",
            **_GOOD_ARGS,
            "modification_proposed": True,
            "modification_cost": 0.9,
        }
    )
    r = _r3(drift_budget_max=0.05).process_case(_case(risk_score=0.3), llm)
    assert r.metadata.get("modification_accepted") is False
    assert r.metadata.get("drift_budget_exceeded") is True


def test_r3_modification_rejected_invariant_violation():
    # SANCTIONS + APPROVE violates INV_1; risk 0.5 avoids the pre-LLM K0_6 gate.
    llm = _llm_returning(
        {
            "decision": "APPROVE",
            "rationale": "ok",
            **_GOOD_ARGS,
            "modification_proposed": True,
            "modification_cost": 0.1,
        }
    )
    r = _r3().process_case(_case(risk_score=0.5, regulatory_flags=["SANCTIONS"]), llm)
    assert r.metadata.get("modification_accepted") is False
    assert r.metadata.get("invariants_preserved") is False
    assert r.metadata.get("invariant_violations")
