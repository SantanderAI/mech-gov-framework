# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the experiment runner orchestration (offline mock LLM)."""

import json

import pytest

from mech_gov.experiment.runner import (
    ExperimentResult,
    ExperimentRunner,
    RunConfig,
    create_regime,
)
from mech_gov.llm.registry import create_llm


def _llm():
    return create_llm({"provider": "mock"})


def test_create_regime_variants():
    assert create_regime("R1").regime_name == "R1"
    assert create_regime("R2").regime_name == "R2"
    assert create_regime("R3").regime_name == "R3"


def test_create_regime_unknown():
    with pytest.raises(ValueError):
        create_regime("R9")


def test_experiment_result_to_dict_roundtrip():
    res = ExperimentResult(
        run_id="r",
        model="m",
        regime="R2",
        condition="S0",
        seed=1,
        n_cases=2,
        metrics={"governance": {"CDL": 0.0}},
        model_id="mock",
        timestamp="t",
        code_version="abc",
        hyperparameters={"k": 1},
    )
    d = res.to_dict()
    assert d["run_id"] == "r"
    assert d["model_id"] == "mock"
    assert d["hyperparameters"] == {"k": 1}


def test_run_full_experiment_with_stress_and_manifest(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    results = runner.run_full_experiment(
        models={"mock": _llm()},
        regimes=["R2"],
        conditions=["S0", "S1"],
        seeds=[1],
        cases_per_condition=6,
    )
    assert len(results) == 2
    manifest = tmp_path / "mock" / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["config"]["runs_completed"] == 2


def test_run_single_skips_completed(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    cfg = RunConfig(
        model_name="mock", regime_name="R2", condition="S0", seed=1, cases_per_condition=5
    )
    first = runner.run_single(cfg, _llm(), skip_completed=True)
    assert first is not None
    second = runner.run_single(cfg, _llm(), skip_completed=True)
    assert second is None  # already completed → skipped


def test_run_condition_direct(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    from mech_gov.data.banking_case import BankingCase, TransactionType

    cases = [
        BankingCase(
            case_id=f"c{i}",
            transaction_type=TransactionType.CREDIT_APPROVAL,
            risk_score=0.4,
            completeness=0.6,
        )
        for i in range(3)
    ]
    results = runner.run_condition(create_regime("R1"), cases, _llm(), seed=7)
    assert len(results) == 3


def test_on_run_complete_callback(tmp_path):
    seen = []
    runner = ExperimentRunner(results_dir=str(tmp_path), on_run_complete=seen.append)
    cfg = RunConfig(
        model_name="mock", regime_name="R2", condition="S0", seed=2, cases_per_condition=6
    )
    runner.run_single(cfg, _llm(), skip_completed=False)
    assert len(seen) == 1
