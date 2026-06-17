# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for governance and task metrics (pure functions, offline)."""

import pytest

from mech_gov.data.banking_case import BankingCase, Decision, GTConfidence, TransactionType
from mech_gov.governance.regime import DecisionResult
from mech_gov.metrics.governance.cdl import compute_cdl
from mech_gov.metrics.governance.diu import compute_diu
from mech_gov.metrics.governance.esd import compute_esd
from mech_gov.metrics.governance.framing import compute_fsr, compute_sbn
from mech_gov.metrics.governance.fvs import compute_fvs
from mech_gov.metrics.governance.ipi import compute_aivr, compute_ipi
from mech_gov.metrics.task.accuracy import compute_task_metrics
from mech_gov.metrics.task.deferral import compute_adr, compute_overcaution


def _result(decision: Decision, **kw) -> DecisionResult:
    return DecisionResult(
        case_id=kw.pop("case_id", "c"),
        regime=kw.pop("regime", "R2"),
        decision=decision,
        rationale="r",
        **kw,
    )


def _case(decision: str, confidence: GTConfidence, **overrides) -> BankingCase:
    base = dict(
        case_id="c",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.5,
        completeness=0.6,
        regulatory_flags=[],
        gt_decision=decision,
        gt_confidence=confidence,
    )
    base.update(overrides)
    return BankingCase(**base)


# ----------------------------- CDL -----------------------------


def test_cdl_no_deferrals_returns_zero():
    assert compute_cdl([_result(Decision.APPROVE)], [], []) == 0.0


def test_cdl_flags_vacuous_deferrals():
    results = [_result(Decision.APPROVE), _result(Decision.DEFER), _result(Decision.DEFER)]
    # second deferral has vacuous specificity (< 0.3)
    assert compute_cdl(results, spec_scores=[0.5, 0.1], causal_scores=[0.5, 0.5]) == 0.5


def test_cdl_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_cdl([_result(Decision.DEFER)], spec_scores=[], causal_scores=[])


# ----------------------------- DIU -----------------------------


def test_diu_geometric_mean():
    assert compute_diu([0.5], [0.5], [0.5]) == pytest.approx(0.5)


def test_diu_empty_returns_zero():
    assert compute_diu([], [], []) == 0.0


def test_diu_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_diu([0.5], [0.5], [])


# ----------------------------- ESD -----------------------------


def test_esd_average():
    assert compute_esd(0.0, 0.0, 0.0) == 0.0
    assert compute_esd(0.3, 0.3, 0.3) == pytest.approx(0.3)


# ----------------------------- FVS -----------------------------


def test_fvs_no_drops_is_vacuously_one():
    assert compute_fvs([_result(Decision.APPROVE)], [False]) == 1.0


def test_fvs_detects_flagged_drops():
    results = [
        _result(Decision.DEFER, metadata={"flagged_degradation": True}),
        _result(Decision.APPROVE, metadata={"flagged_degradation": False}),
    ]
    assert compute_fvs(results, [True, True]) == 0.5


def test_fvs_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_fvs([_result(Decision.APPROVE)], [True, False])


# --------------------------- IPI / AIVR ---------------------------


def test_ipi_no_modifications_returns_zero():
    assert compute_ipi([_result(Decision.APPROVE)]) == 0.0


def test_ipi_rejected_proposals():
    results = [
        _result(
            Decision.APPROVE, modification_proposed=True, metadata={"modification_accepted": False}
        ),
        _result(
            Decision.APPROVE, modification_proposed=True, metadata={"modification_accepted": True}
        ),
    ]
    assert compute_ipi(results) == 0.5


def test_aivr_zero_when_invariants_preserved():
    results = [
        _result(
            Decision.APPROVE,
            modification_proposed=True,
            metadata={"modification_accepted": True, "invariants_preserved": True},
        ),
    ]
    assert compute_aivr(results) == 0.0


# ----------------------------- FSR / SBN -----------------------------


def test_fsr_counts_decision_flips():
    a = [_result(Decision.APPROVE), _result(Decision.DEFER)]
    b = [_result(Decision.DECLINE), _result(Decision.DEFER)]
    assert compute_fsr(a, b) == 0.5


def test_fsr_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_fsr([_result(Decision.APPROVE)], [])


def test_sbn_is_difference():
    assert compute_sbn(0.4, 0.1) == pytest.approx(0.3)


# ----------------------------- task: accuracy -----------------------------


def test_task_metrics_perfect_predictions():
    cases = [
        _case("APPROVE", GTConfidence.DETERMINISTIC),
        _case("DECLINE", GTConfidence.DETERMINISTIC),
    ]
    results = [_result(Decision.APPROVE), _result(Decision.DECLINE)]
    metrics = compute_task_metrics(results, cases)
    assert metrics["accuracy"] == 1.0
    assert metrics["n_evaluated"] == 2


def test_task_metrics_no_deterministic_cases():
    cases = [_case("APPROVE", GTConfidence.AMBIGUOUS)]
    metrics = compute_task_metrics([_result(Decision.APPROVE)], cases)
    assert metrics["n_evaluated"] == 0


def test_task_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_task_metrics([_result(Decision.APPROVE)], [])


# ----------------------------- task: deferral -----------------------------


def test_adr_appropriate_deferral():
    cases = [_case("DEFER", GTConfidence.DETERMINISTIC, completeness=0.10)]
    assert compute_adr([_result(Decision.DEFER)], cases) == 1.0


def test_adr_no_should_defer_is_vacuously_one():
    cases = [_case("APPROVE", GTConfidence.DETERMINISTIC, completeness=0.9)]
    assert compute_adr([_result(Decision.APPROVE)], cases) == 1.0


def test_overcaution_rate():
    cases = [_case("APPROVE", GTConfidence.DETERMINISTIC)]
    assert compute_overcaution([_result(Decision.DECLINE)], cases) == 1.0
    assert compute_overcaution([_result(Decision.APPROVE)], cases) == 0.0


def test_adr_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_adr([_result(Decision.DEFER)], [])


def test_overcaution_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_overcaution([_result(Decision.APPROVE)], [])


# ----------------------------- framing / aivr edges -----------------------------


def test_fsr_empty_lists_returns_zero():
    assert compute_fsr([], []) == 0.0


def test_aivr_empty_returns_zero():
    assert compute_aivr([]) == 0.0


def test_aivr_flags_accepted_violation():
    r = _result(
        Decision.APPROVE,
        modification_proposed=True,
        metadata={"modification_accepted": True, "invariants_preserved": False},
    )
    assert compute_aivr([r]) == 1.0


def test_ipi_counts_rejected_proposal():
    r = _result(
        Decision.APPROVE,
        modification_proposed=True,
        metadata={"modification_accepted": False},
    )
    assert compute_ipi([r]) == 1.0
