# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for experiment harnesses (seed attack, framing, FVS, ablation) — offline."""

import pytest

from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.experiment.ablation import (
    ABLATION_REGISTRY,
    A1_NoI6Q,
    create_ablation_regime,
)
from mech_gov.experiment.framing_test import apply_framing_manipulation, run_framing_test
from mech_gov.experiment.fvs_test import inject_quality_drops, run_fvs_test
from mech_gov.experiment.seed_attack import (
    compute_exploit_score,
    compute_integrity_score,
    compute_leakage_score,
    run_seed_attack_test,
)
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.llm.registry import create_llm


def _cases(n=6):
    return [
        BankingCase(
            case_id=f"c{i}",
            transaction_type=TransactionType.CREDIT_APPROVAL,
            risk_score=0.3 + 0.1 * (i % 5),
            completeness=0.5,
            regulatory_flags=["KYC"],
        )
        for i in range(n)
    ]


def _llm():
    return create_llm({"provider": "mock"})


# ----------------------------- seed attack -----------------------------


def test_run_seed_attack_test_r2():
    esd, subs = run_seed_attack_test(_cases(), R2Mechanical(), _llm(), seeds=[1, 2, 3])
    assert 0.0 <= esd <= 1.0
    assert set(subs) == {"exploit", "leakage", "integrity"}


def test_run_seed_attack_default_seeds():
    esd, _ = run_seed_attack_test(_cases(3), R2Mechanical(), _llm())
    assert 0.0 <= esd <= 1.0


def test_exploit_score_single_seed_is_zero():
    assert compute_exploit_score({1: []}) == 0.0


def test_leakage_and_integrity_empty():
    assert compute_leakage_score({}) == 0.0
    assert compute_integrity_score({}) == 0.0


# ----------------------------- framing -----------------------------


def test_apply_framing_manipulation_preserves_values():
    c = _cases(1)[0]
    framed = apply_framing_manipulation(c)
    assert framed.risk_score == c.risk_score
    assert framed.case_id.endswith("_framed")
    assert framed.to_prompt() != c.to_prompt()


def test_run_framing_test_r1():
    fsr, a, b = run_framing_test(_cases(), R1TextOnly(), _llm())
    assert 0.0 <= fsr <= 1.0
    assert len(a) == len(b) == 6


# ----------------------------- FVS -----------------------------


def test_inject_quality_drops_mask():
    cases = _cases(10)
    modified, drops = inject_quality_drops(cases, drop_fraction=0.3, seed=1)
    assert len(modified) == 10
    assert sum(drops) >= 1
    for m, dropped in zip(modified, drops, strict=False):
        if dropped:
            assert m.completeness == 0.10


def test_run_fvs_test_r2():
    fvs, results, drops = run_fvs_test(_cases(10), R2Mechanical(), _llm())
    assert 0.0 <= fvs <= 1.0
    assert len(results) == len(drops) == 10


# ----------------------------- ablation -----------------------------


@pytest.mark.parametrize("name", list(ABLATION_REGISTRY))
def test_each_ablation_regime_runs(name):
    regime = create_ablation_regime(name)
    result = regime.process_case(_cases(1)[0], _llm(), entropy_seed=42)
    assert result.decision is not None
    assert regime.regime_name.startswith("R2_")


def test_unknown_ablation_raises():
    with pytest.raises(ValueError):
        create_ablation_regime("A99_nonexistent")


def test_a1_no_i6q_config_defaults():
    regime = A1_NoI6Q()
    assert regime.regime_name == "R2_A1_no_i6q"
