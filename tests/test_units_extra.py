# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for small branches across primitives and providers."""

import json

import pytest

from mech_gov.data.banking_case import BankingCase, Decision, TransactionType
from mech_gov.experiment.fvs_test import _was_flagged
from mech_gov.governance.primitives.cefl import (
    _parse_candidate,
    _score_candidate,
    generate_cefl_candidates,
    select_best_candidate,
)
from mech_gov.governance.r1_text_only import R1TextOnly, _parse_llm_json
from mech_gov.governance.regime import DecisionResult
from mech_gov.llm.base import LLMResponse
from mech_gov.llm.providers.callable_provider import CallableLLM
from mech_gov.llm.providers.callable_provider import build as build_callable
from mech_gov.llm.providers.mock import MockLLM


def _case() -> BankingCase:
    return BankingCase(
        case_id="c",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.4,
        completeness=0.6,
    )


# ----------------------------- cefl -----------------------------


def test_score_candidate_full_rubric():
    parsed = {
        "decision": "DEFER",
        "pro_arguments": ["a", "b", "c"],
        "con_arguments": ["d", "e"],
        "rationale": " ".join(["word"] * 35),
        "deferral_info_needed": "need docs",
    }
    assert _score_candidate(parsed, _case()) == pytest.approx(1.0 + 1.0 + 1.0 + 1.0 + 0.5)


def test_score_candidate_conditional_bonus():
    parsed = {"decision": "CONDITIONAL", "conditions": "x"}
    assert _score_candidate(parsed, _case()) >= 1.5


def test_score_candidate_invalid_decision():
    assert _score_candidate({"decision": "ZZZ"}, _case()) == 0.0


def test_parse_candidate_trailing_comma_and_fences():
    assert _parse_candidate('```json\n{"decision": "APPROVE",}\n```') == {"decision": "APPROVE"}


def test_parse_candidate_prose_wrapped():
    assert _parse_candidate('answer: {"decision": "DECLINE"} done')["decision"] == "DECLINE"


def test_parse_candidate_garbage_returns_empty():
    assert _parse_candidate("no json here") == {}


def test_select_best_candidate_empty_returns_default():
    best = select_best_candidate([])
    assert best["score"] == 0.0


def test_generate_cefl_candidates_scores_and_sorts():
    payload = json.dumps(
        {"decision": "DEFER", "rationale": "x", "deferral_info_needed": "more docs"}
    )
    llm = MockLLM(response=payload)
    cands = generate_cefl_candidates(_case(), llm, "sys", "user", n_candidates=2)
    assert len(cands) == 2
    assert cands[0]["score"] >= cands[-1]["score"]


# ----------------------------- callable provider -----------------------------


def test_callable_requires_callable():
    with pytest.raises(TypeError):
        CallableLLM(fn="not-callable")


def test_callable_passthrough_llmresponse():
    resp = LLMResponse(
        content="x",
        model_id="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=0.0,
        stop_reason="stop",
    )
    llm = CallableLLM(lambda system_prompt, user_message, temperature=0.0, max_tokens=2048: resp)
    assert llm.invoke("s", "u") is resp
    assert llm.model_id == "callable"


def test_build_callable_missing_fn():
    with pytest.raises(ValueError):
        build_callable({})


# ----------------------------- mock provider modes -----------------------------


def test_mock_responder_mode():
    llm = MockLLM(responder=lambda s, u: "RESP")
    assert llm.invoke("s", "u").content == "RESP"


def test_mock_responses_cycle():
    llm = MockLLM(responses=["a", "b"])
    assert [llm.invoke("s", "u").content for _ in range(3)] == ["a", "b", "a"]


# ----------------------------- fvs flagging helper -----------------------------


def _res(decision=Decision.APPROVE, **meta) -> DecisionResult:
    return DecisionResult(case_id="c", regime="R2", decision=decision, rationale="r", metadata=meta)


def test_was_flagged_branches():
    assert _was_flagged(_res(Decision.DEFER)) is True
    assert _was_flagged(_res(Decision.APPROVE, flagged_degradation=True)) is True
    assert _was_flagged(_res(Decision.APPROVE, hard_gate_override=True, gate_id="K0_10")) is True
    assert _was_flagged(_res(Decision.APPROVE, ambiguity_gate_override=True)) is True
    assert _was_flagged(_res(Decision.APPROVE)) is False


# ----------------------------- r1 JSON fallback -----------------------------


def test_parse_llm_json_fallback_regex():
    parsed = _parse_llm_json('Sure! {"decision": "APPROVE", "rationale": "ok"} hope this helps')
    assert parsed["decision"] == "APPROVE"


def test_parse_llm_json_unparseable_returns_empty():
    assert _parse_llm_json("totally not json") == {}


def test_r1_handles_empty_response():
    llm = MockLLM(response="not json at all")
    r = R1TextOnly().process_case(_case(), llm)
    assert r.decision == Decision.ESCALATE


# ----------------------------- i6q diversity branch -----------------------------


def test_i6q_low_diversity_fails():
    from mech_gov.governance.primitives.i6q import I6QConfig, check_i6q

    repeated = "same same same same same same"
    res = check_i6q([repeated], [repeated], I6QConfig(min_arg_tokens=1, min_diversity=0.9))
    assert res.passed is False
    assert res.checks["lexical_diversity"] is False


# ----------------------------- seed attack sub-scores -----------------------------


def test_seed_attack_leakage_and_integrity():
    from mech_gov.experiment.seed_attack import (
        compute_integrity_score,
        compute_leakage_score,
    )

    leaky = DecisionResult(
        case_id="c",
        regime="R2",
        decision=Decision.APPROVE,
        rationale="r",
        llm_raw_response="seed 42 leaked into output",
    )
    assert compute_leakage_score({42: [leaky]}) == 1.0

    failed = DecisionResult(
        case_id="c",
        regime="R2",
        decision=Decision.APPROVE,
        rationale="r",
        entropy_nonce="abc",
        metadata={"e3_verified": False},
    )
    assert compute_integrity_score({1: [failed]}) == 1.0
