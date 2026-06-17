# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the BankingCase data model (validation + serialization)."""

import pytest
from pydantic import ValidationError

from mech_gov.data.banking_case import BankingCase, TransactionType


def _case(**overrides) -> BankingCase:
    base = dict(
        case_id="c-0001",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.50,
        completeness=0.60,
        regulatory_flags=["KYC"],
    )
    base.update(overrides)
    return BankingCase(**base)


def test_valid_case_constructs():
    case = _case()
    assert case.case_id == "c-0001"
    assert case.transaction_type == TransactionType.CREDIT_APPROVAL


def test_unknown_flag_raises():
    with pytest.raises(ValidationError):
        _case(regulatory_flags=["NOT_A_FLAG"])


def test_flags_are_deduplicated_and_sorted():
    case = _case(regulatory_flags=["SANCTIONS", "AML", "AML"])
    assert case.regulatory_flags == ["AML", "SANCTIONS"]


def test_risk_score_out_of_range_raises():
    with pytest.raises(ValidationError):
        _case(risk_score=1.5)


def test_json_roundtrip():
    case = _case(amount_usd=12_345.0, jurisdiction="ES")
    restored = BankingCase.from_json(case.to_json())
    assert restored == case


def test_dict_roundtrip():
    case = _case()
    restored = BankingCase.from_dict(case.to_dict())
    assert restored == case


def test_to_prompt_contains_key_fields():
    prompt = _case(risk_score=0.42, regulatory_flags=["AML", "KYC"]).to_prompt()
    assert "Risk Score: 0.420" in prompt
    assert "AML" in prompt and "KYC" in prompt
