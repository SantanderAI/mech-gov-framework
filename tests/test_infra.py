# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for logging config, the factory shim, registry helpers and ground-truth coverage."""

import logging

import pytest
import yaml

from mech_gov.data.generator import generate_dataset_from_config
from mech_gov.data.ground_truth import assign_ground_truth, validate_coverage
from mech_gov.experiment.runner import _DEFAULT_DIST_CONFIG
from mech_gov.llm import factory
from mech_gov.llm.base import LLMResponse
from mech_gov.llm.registry import (
    available_providers,
    create_all_llms,
    load_models_config,
    register_provider,
)
from mech_gov.logging_config import setup_logging

# ----------------------------- logging -----------------------------


def test_setup_logging_console_and_file(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging("DEBUG", log_file=str(log_file))
    setup_logging("INFO")  # second call exercises the handler-reset branch
    logging.getLogger("mech_gov").info("hello")
    assert log_file.exists()


# ----------------------------- factory shim -----------------------------


def test_factory_create_llm_shim():
    llm = factory.create_llm({"provider": "mock"})
    assert llm.provider == "mock"
    assert "mock" in factory.available_providers()


def test_factory_create_all_llms(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump({"models": {"m1": {"provider": "mock"}}}))
    llms = factory.create_all_llms(str(cfg))
    assert "m1" in llms


# ----------------------------- registry helpers -----------------------------


def test_load_models_config_flat_and_nested(tmp_path):
    nested = tmp_path / "n.yaml"
    nested.write_text(yaml.safe_dump({"models": {"a": {"provider": "mock"}}}))
    flat = tmp_path / "f.yaml"
    flat.write_text(yaml.safe_dump({"a": {"provider": "mock"}}))
    assert "a" in load_models_config(nested)
    assert "a" in load_models_config(flat)


def test_load_models_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models_config(tmp_path / "nope.yaml")


def test_create_all_llms_filters_by_name(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump({"a": {"provider": "mock"}, "b": {"provider": "mock"}}))
    only = create_all_llms(str(cfg), model_names=["a"])
    assert set(only) == {"a"}


def test_register_custom_provider():
    class _Dummy:
        provider = "dummy"
        model_id = "dummy"

        def invoke(self, system_prompt, user_message, temperature=0.0, max_tokens=2048):
            return LLMResponse(content="x", model_id="dummy")

    register_provider("dummy", lambda cfg: _Dummy())
    assert "dummy" in available_providers()


# ----------------------------- ground truth -----------------------------


def test_validate_coverage_reports_breakdown():
    cases = generate_dataset_from_config(
        seed=3,
        n_cases_per_condition=40,
        stress_conditions=["S0"],
        distributions_path=_DEFAULT_DIST_CONFIG,
    )
    assign_ground_truth(cases)
    report = validate_coverage(cases)
    assert report["total_cases"] == 40
    assert "deterministic" in report
    assert "pct_deterministic" in report


def test_validate_coverage_empty():
    assert "error" in validate_coverage([])
