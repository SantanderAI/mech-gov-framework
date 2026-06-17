# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the R2 pre-LLM hard gates (offline, no LLM)."""

from mech_gov.data.banking_case import BankingCase, Decision, TransactionType
from mech_gov.governance.primitives.hard_gates import (
    build_default_gates,
    evaluate_hard_gates,
)


def _case(**overrides) -> BankingCase:
    base = dict(
        case_id="g-0001",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.30,
        completeness=0.80,
        regulatory_flags=["KYC"],
        amount_usd=1_000.0,
    )
    base.update(overrides)
    return BankingCase(**base)


def test_build_default_gates_has_expected_ids():
    ids = [g.gate_id for g in build_default_gates()]
    assert ids == ["K0_6", "K0_7", "K0_8", "K0_10", "K0_12", "K0_13", "K0_14"]


def test_k0_6_high_risk_sanctions_declines():
    gid, decision, rationale = evaluate_hard_gates(
        _case(risk_score=0.95, regulatory_flags=["SANCTIONS"])
    )
    assert gid == "K0_6"
    assert decision == Decision.DECLINE
    assert "K0_6" in rationale


def test_k0_7_high_risk_multiple_flags_escalates():
    gid, decision, _ = evaluate_hard_gates(
        _case(risk_score=0.88, regulatory_flags=["AML", "KYC", "SANCTIONS"])
    )
    assert gid == "K0_7"
    assert decision == Decision.ESCALATE


def test_k0_8_large_amount_high_risk_escalates():
    gid, decision, _ = evaluate_hard_gates(
        _case(risk_score=0.75, amount_usd=2_000_000.0, regulatory_flags=["KYC"])
    )
    assert gid == "K0_8"
    assert decision == Decision.ESCALATE


def test_k0_10_low_completeness_defers():
    gid, decision, _ = evaluate_hard_gates(
        _case(risk_score=0.30, completeness=0.10, regulatory_flags=[])
    )
    assert gid == "K0_10"
    assert decision == Decision.DEFER


def test_k0_12_aml_high_risk_declines():
    gid, decision, _ = evaluate_hard_gates(_case(risk_score=0.75, regulatory_flags=["AML"]))
    assert gid == "K0_12"
    assert decision == Decision.DECLINE


def test_k0_13_insider_escalates():
    gid, decision, _ = evaluate_hard_gates(_case(risk_score=0.20, regulatory_flags=["INSIDER"]))
    assert gid == "K0_13"
    assert decision == Decision.ESCALATE


def test_k0_14_multiple_severe_flags_declines():
    gid, decision, _ = evaluate_hard_gates(
        _case(risk_score=0.65, regulatory_flags=["AML", "SANCTIONS"])
    )
    assert gid == "K0_14"
    assert decision == Decision.DECLINE


def test_no_gate_triggers_returns_none():
    assert evaluate_hard_gates(_case()) is None


def test_custom_config_overrides_threshold():
    # Lower the K0_6 risk threshold so a 0.6 risk + SANCTIONS triggers it.
    gid, decision, _ = evaluate_hard_gates(
        _case(risk_score=0.60, regulatory_flags=["SANCTIONS"]),
        config={"K0_6": {"risk_threshold": 0.5}},
    )
    assert gid == "K0_6"
    assert decision == Decision.DECLINE
