# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for deferral scorers (rule-based + judge + validation)."""

from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.llm.registry import create_llm
from mech_gov.metrics.governance.scorers import (
    _parse_judge_score,
    compute_bshift_judge,
    compute_bshift_rulebased,
    compute_causal_judge,
    compute_causal_rulebased,
    compute_cohens_kappa,
    compute_spec_judge,
    compute_spec_rulebased,
)


def _case() -> BankingCase:
    return BankingCase(
        case_id="s-1",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.7,
        completeness=0.4,
        regulatory_flags=["AML", "KYC"],
    )


def test_rulebased_scores_empty_text_is_zero():
    case = _case()
    assert compute_spec_rulebased("", case) == 0.0
    assert compute_causal_rulebased("", case) == 0.0
    assert compute_bshift_rulebased("", case) == 0.0


def test_spec_rulebased_rewards_specific_text():
    case = _case()
    rich = (
        "Specifically, the AML flag and a risk score of 0.70 exceed the K0_12 "
        "threshold; the counterparty jurisdiction and amount add further "
        "completeness concerns about missing data in this particular case."
    )
    assert compute_spec_rulebased(rich, case) > compute_spec_rulebased("deferred", case)


def test_causal_rulebased_rewards_causal_language():
    case = _case()
    causal = (
        "Because the case cannot be determined due to insufficient information, "
        "the decision requires review pending verification before approval; "
        "the applicant would need to address the regulatory concern first."
    )
    assert compute_causal_rulebased(causal, case) > 0.5


def test_bshift_rulebased_rewards_directional_language():
    case = _case()
    bshift = (
        "This could be approved if additional documentation is provided to "
        "reduce risk and meet the required threshold; a favorable resolution "
        "is possible once the criteria are satisfied."
    )
    assert compute_bshift_rulebased(bshift, case) > 0.5


def test_scores_are_bounded():
    case = _case()
    text = "risk score threshold gate K0_12 because due to would approve if " * 5
    for fn in (compute_spec_rulebased, compute_causal_rulebased, compute_bshift_rulebased):
        assert 0.0 <= fn(text, case) <= 1.0


def test_parse_judge_score():
    assert _parse_judge_score("4") == 0.8
    assert _parse_judge_score("score: 5") == 1.0
    assert _parse_judge_score("no digit here") == 0.0


def test_judge_scorers_run_with_mock_llm():
    case = _case()
    judge = create_llm({"provider": "mock"})
    for fn in (compute_spec_judge, compute_causal_judge, compute_bshift_judge):
        score = fn("deferral text", case, judge)
        assert 0.0 <= score <= 1.0
        assert fn("", case, judge) == 0.0


def test_cohens_kappa_perfect_agreement():
    a = [0.9, 0.1, 0.8, 0.2]
    assert compute_cohens_kappa(a, a) == 1.0
