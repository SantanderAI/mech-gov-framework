# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Additional runner branch coverage: R3 metrics, manifest merge, fallbacks, callbacks."""

from mech_gov.experiment.runner import ExperimentRunner, RunConfig
from mech_gov.llm.base import LLMResponse
from mech_gov.llm.registry import create_llm


def _llm():
    return create_llm({"provider": "mock"})


def test_run_single_r3_emits_ipi_aivr(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    cfg = RunConfig(
        model_name="mock", regime_name="R3", condition="S0", seed=1, cases_per_condition=6
    )
    result = runner.run_single(cfg, _llm(), skip_completed=False)
    assert "IPI" in result.metrics["governance"]
    assert "AIVR" in result.metrics["governance"]


def test_run_single_condition_by_enum_name(tmp_path):
    # Passing the enum *name* (not value) exercises the KeyError-fallback branch.
    runner = ExperimentRunner(results_dir=str(tmp_path))
    cfg = RunConfig(
        model_name="mock",
        regime_name="R2",
        condition="S1_HIGH_RISK",
        seed=2,
        cases_per_condition=6,
    )
    result = runner.run_single(cfg, _llm(), skip_completed=False)
    assert result.condition == "S1_HIGH_RISK"


def test_manifest_merge_on_second_run(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    common = dict(regimes=["R2"], conditions=["S0"], seeds=[1], cases_per_condition=6)
    runner.run_full_experiment(models={"mock": _llm()}, **common)
    # Second run merges into the existing manifest.json.
    runner.run_full_experiment(
        models={"mock": _llm()},
        regimes=["R2"],
        conditions=["S0"],
        seeds=[2],
        cases_per_condition=6,
    )
    assert (tmp_path / "mock" / "manifest.json").exists()


def test_full_experiment_skips_already_completed(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    args = dict(
        models={"mock": _llm()},
        regimes=["R2"],
        conditions=["S0"],
        seeds=[1],
        cases_per_condition=6,
    )
    first = runner.run_full_experiment(**args)
    assert len(first) == 1
    second = runner.run_full_experiment(**args)  # all skipped
    assert second == []


class _BadModelIdLLM:
    provider = "bad"

    @property
    def model_id(self):
        raise RuntimeError("no id")

    def invoke(self, system_prompt, user_message, temperature=0.0, max_tokens=2048):
        return LLMResponse(
            content='{"decision": "ESCALATE"}',
            model_id="x",
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.0,
            stop_reason="stop",
        )


def test_run_single_tolerates_model_id_failure(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    cfg = RunConfig(
        model_name="bad", regime_name="R1", condition="S0", seed=3, cases_per_condition=6
    )
    result = runner.run_single(cfg, _BadModelIdLLM(), skip_completed=False)
    assert result.model_id == ""


def test_run_single_tolerates_callback_failure(tmp_path):
    def _boom(_result):
        raise RuntimeError("callback failed")

    runner = ExperimentRunner(results_dir=str(tmp_path), on_run_complete=_boom)
    cfg = RunConfig(
        model_name="mock", regime_name="R2", condition="S0", seed=4, cases_per_condition=6
    )
    result = runner.run_single(cfg, _llm(), skip_completed=False)
    assert result is not None  # callback error swallowed
