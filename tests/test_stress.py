# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for stress transforms (pure, seeded, offline)."""

from mech_gov.data.banking_case import BankingCase, StressCondition, TransactionType
from mech_gov.data.generator import make_rng
from mech_gov.data.stress import (
    apply_s0_baseline,
    apply_s1_high_risk,
    apply_s2_low_info,
    apply_s3_threshold,
    apply_stress,
)


def _case(**kw) -> BankingCase:
    base = dict(
        case_id="x",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.5,
        completeness=0.8,
        regulatory_flags=["KYC", "AML"],
        amount_usd=50_000.0,
    )
    base.update(kw)
    return BankingCase(**base)


def test_s0_baseline_is_passthrough():
    c = _case()
    out = apply_s0_baseline(c, make_rng(1))
    assert out.risk_score == 0.5


def test_s1_high_risk_preserves_originals_and_bounds():
    out = apply_s1_high_risk(_case(), make_rng(7))
    assert out.original_risk_score == 0.5
    assert 0.0 <= out.risk_score <= 1.0
    assert out.stress_condition == StressCondition.S1_HIGH_RISK


def test_s2_low_info_reduces_completeness():
    out = apply_s2_low_info(_case(completeness=0.9), make_rng(3))
    assert out.original_completeness == 0.9
    assert out.completeness <= 0.9
    assert out.stress_condition == StressCondition.S2_LOW_INFO


def test_s3_threshold_concentrates_near_boundaries():
    out = apply_s3_threshold(_case(), make_rng(5))
    assert out.original_risk_score == 0.5
    assert 0.0 <= out.risk_score <= 1.0
    assert out.stress_condition == StressCondition.S3_THRESHOLD


def test_s3_threshold_accepts_custom_params_and_thresholds():
    out = apply_s3_threshold(
        _case(),
        make_rng(5),
        thresholds=[0.5],
        params={"epsilon_risk": 0.01, "epsilon_completeness": 0.01, "target_proximity_pct": 1.0},
    )
    assert out.stress_condition == StressCondition.S3_THRESHOLD


def test_apply_stress_dispatch_all_conditions():
    cases = [_case(case_id=f"c{i}") for i in range(5)]
    for cond in StressCondition:
        out = apply_stress(cases, cond, make_rng(11))
        assert len(out) == len(cases)
        # Originals must be untouched (deep copy semantics).
        assert all(c.risk_score == 0.5 for c in cases)
