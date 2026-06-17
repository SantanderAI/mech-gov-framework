# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke tests using the offline mock LLM (no credentials/network)."""

from mech_gov.data.banking_case import BankingCase, Decision, TransactionType
from mech_gov.data.generator import generate_dataset_from_config
from mech_gov.data.ground_truth import assign_ground_truth
from mech_gov.experiment.runner import _DEFAULT_DIST_CONFIG, ExperimentRunner, RunConfig
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.llm.registry import create_llm

_VALID = {d.value for d in Decision}


def _case(**overrides) -> BankingCase:
    base = dict(
        case_id="t-0001",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.60,
        completeness=0.50,
        regulatory_flags=["KYC"],
        amount_usd=100_000.0,
        jurisdiction="US",
        customer_tenure_years=2.0,
        counterparty_risk=0.30,
    )
    base.update(overrides)
    return BankingCase(**base)


def test_r1_runs_with_mock():
    result = R1TextOnly().process_case(_case(), create_llm({"provider": "mock"}))
    assert result.regime == "R1"
    assert result.decision.value in _VALID


def test_r2_runs_with_mock():
    result = R2Mechanical().process_case(_case(), create_llm({"provider": "mock"}))
    assert result.regime == "R2"
    assert result.decision.value in _VALID


def test_r2_hard_gate_declines_high_risk_sanctions():
    # K0_6: risk > 0.9 AND SANCTIONS -> mechanical DECLINE, LLM not consulted.
    result = R2Mechanical().process_case(
        _case(case_id="t-gate", risk_score=0.95, regulatory_flags=["SANCTIONS"]),
        create_llm({"provider": "mock"}),
    )
    assert result.decision == Decision.DECLINE
    assert "K0_6" in result.gates_triggered
    assert result.metadata.get("hard_gate_override") is True
    assert result.tokens_used == 0  # gate fired before any LLM call


def test_generator_uses_packaged_config():
    cases = generate_dataset_from_config(
        seed=1,
        n_cases_per_condition=10,
        stress_conditions=["S0"],
        distributions_path=_DEFAULT_DIST_CONFIG,
    )
    assert len(cases) == 10
    assign_ground_truth(cases)
    assert all(c.gt_confidence is not None for c in cases)
    assert all(c.gt_decision_set for c in cases)


def test_experiment_runner_end_to_end(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    config = RunConfig(
        model_name="mock",
        regime_name="R2",
        condition="S0",
        seed=42,
        cases_per_condition=10,
    )
    result = runner.run_single(config, create_llm({"provider": "mock"}), skip_completed=False)
    assert result is not None
    assert "governance" in result.metrics
    assert "task" in result.metrics
    assert "CDL" in result.metrics["governance"]
