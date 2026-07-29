# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Resume behaviour of the experiment runner under versioned run identity."""

from __future__ import annotations

import json
import logging

import pytest
import yaml

import mech_gov
from mech_gov.experiment.identity import IdentityWarnings, canonical_bytes
from mech_gov.experiment.runner import (
    _DEFAULT_DIST_CONFIG,
    ExperimentRunner,
    RunConfig,
    _CompletionIndex,
    _decide_skip,
    _resolve_identity,
)
from mech_gov.llm.providers.mock import MockLLM
from mech_gov.llm.providers.openai_compatible import OpenAICompatibleLLM
from mech_gov.llm.registry import create_llm

N = 5  # smallest count the generator accepts: one case per transaction type
M = 10  # a materially different count


def _llm(model_id: str = "mock"):
    return MockLLM(model_id=model_id)


def _cfg(n: int = N, dist: str | None = None, **overrides):
    kwargs = dict(
        model_name="mock", regime_name="R2", condition="S0", seed=42, cases_per_condition=n
    )
    kwargs.update(overrides)
    if dist is not None:
        kwargs["distributions_config_path"] = dist
    return RunConfig(**kwargs)


def _identity_warnings(caplog):
    return [r for r in caplog.records if r.name == "mech_gov.experiment.identity"]


def _codes(caplog):
    return sorted({r.message.split("]")[0].lstrip("[") for r in _identity_warnings(caplog)})


def _write_dist(tmp_path, mutate=None, name="dist.yaml"):
    """Copy the bundled distributions config, optionally changing one number."""
    with open(_DEFAULT_DIST_CONFIG, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if mutate is not None:
        mutate(loaded)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    return str(path)


def _records(results_dir):
    path = results_dir / "experiment_results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------
# The reported defect
# --------------------------------------------------------------------------


def test_changed_case_count_executes_instead_of_skipping(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    first = runner.run_single(_cfg(N), _llm(), skip_completed=True)
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        second = runner.run_single(_cfg(M), _llm(), skip_completed=True)

    assert first is not None and second is not None
    assert first.n_cases == N and second.n_cases == M
    assert first.run_label == second.run_label  # same readable label
    assert first.identity_sha256 != second.identity_sha256  # different identity
    assert _codes(caplog) == ["MGI002"]
    assert "fields=cases_per_condition" in _identity_warnings(caplog)[0].message


def test_exact_repeat_still_skips_without_warning(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is None
    assert _identity_warnings(caplog) == []


def test_changed_model_id_behind_same_alias_executes(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    first = runner.run_single(_cfg(), _llm("model-v1"), skip_completed=True)
    second = runner.run_single(_cfg(), _llm("model-v2"), skip_completed=True)
    assert second is not None
    assert first.identity_sha256 != second.identity_sha256


def test_changed_distribution_content_executes(tmp_path):
    def bump(loaded):
        loaded["credit_approval"]["risk_score"]["params"]["a"] += 1

    original = _write_dist(tmp_path, name="a.yaml")
    changed = _write_dist(tmp_path, mutate=bump, name="b.yaml")
    runner = ExperimentRunner(results_dir=str(tmp_path / "results"))
    assert runner.run_single(_cfg(dist=original), _llm(), skip_completed=True) is not None
    assert runner.run_single(_cfg(dist=changed), _llm(), skip_completed=True) is not None


def test_same_distribution_content_at_another_path_still_skips(tmp_path):
    first_path = _write_dist(tmp_path, name="one.yaml")
    second_path = _write_dist(tmp_path, name="two.yaml")
    runner = ExperimentRunner(results_dir=str(tmp_path / "results"))
    assert runner.run_single(_cfg(dist=first_path), _llm(), skip_completed=True) is not None
    assert runner.run_single(_cfg(dist=second_path), _llm(), skip_completed=True) is None


def test_changed_provider_parameters_execute(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    assert runner.run_single(_cfg(), MockLLM(response='{"decision": "APPROVE"}'), False) is not None
    second = runner.run_single(_cfg(), MockLLM(response='{"decision": "DEFER"}'), True)
    assert second is not None


def test_mock_response_fingerprint_is_length_unambiguous():
    left = MockLLM(responses=["a\0b", "c"]).resume_identity_parameters()
    right = MockLLM(responses=["a", "b\0c"]).resume_identity_parameters()
    assert left != right


def test_openai_compatible_endpoint_changes_identity():
    first = OpenAICompatibleLLM(base_url="https://one.example/v1", model="shared-model")
    second = OpenAICompatibleLLM(base_url="https://two.example/v1", model="shared-model")
    warnings = IdentityWarnings()
    first_identity = _resolve_identity(_cfg(), first, "label", {}, warnings)
    second_identity = _resolve_identity(_cfg(), second, "label", {}, warnings)
    assert first_identity is not None and second_identity is not None
    assert first_identity.digest != second_identity.digest
    assert "one.example" not in json.dumps(first_identity.payload)
    assert "two.example" not in json.dumps(second_identity.payload)


def test_openai_compatible_default_port_is_semantically_stable():
    implicit = OpenAICompatibleLLM(base_url="https://example.test/v1", model="model")
    explicit = OpenAICompatibleLLM(base_url="https://example.test:443/v1", model="model")
    assert implicit.resume_identity_parameters() == explicit.resume_identity_parameters()


def test_openai_compatible_top_p_changes_identity():
    first = OpenAICompatibleLLM(base_url="https://example.test/v1", model="model", top_p=0.8)
    second = OpenAICompatibleLLM(base_url="https://example.test/v1", model="model", top_p=0.9)
    warnings = IdentityWarnings()
    first_identity = _resolve_identity(_cfg(), first, "label", {}, warnings)
    second_identity = _resolve_identity(_cfg(), second, "label", {}, warnings)
    assert first_identity is not None and second_identity is not None
    assert first_identity.digest != second_identity.digest


def test_bedrock_top_p_changes_identity(monkeypatch):
    boto3 = pytest.importorskip("boto3")

    class _Session:
        def client(self, *_args, **_kwargs):
            return object()

    monkeypatch.setattr(boto3, "Session", lambda **_kwargs: _Session())
    from mech_gov.llm.providers.bedrock import BedrockLLM

    first = BedrockLLM("model", top_p=0.8)
    second = BedrockLLM("model", top_p=0.9)
    warnings = IdentityWarnings()
    first_identity = _resolve_identity(_cfg(), first, "label", {}, warnings)
    second_identity = _resolve_identity(_cfg(), second, "label", {}, warnings)
    assert first_identity is not None and second_identity is not None
    assert first_identity.digest != second_identity.digest


def test_openai_compatible_credentialed_endpoint_is_non_resumable():
    llm = OpenAICompatibleLLM(
        base_url="https://user:secret@example.test/v1",  # pragma: allowlist secret
        api_key="sk-secret",  # pragma: allowlist secret
        model="model",
    )
    assert llm.resume_identity_parameters() is None


# --------------------------------------------------------------------------
# Non-resumable inputs
# --------------------------------------------------------------------------


def test_opaque_provider_is_non_resumable(tmp_path, caplog):
    def responder(_system, _user):
        return '{"decision": "ESCALATE"}'

    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        first = runner.run_single(_cfg(), MockLLM(responder=responder), skip_completed=True)
        second = runner.run_single(_cfg(), MockLLM(responder=responder), skip_completed=True)

    assert first is not None and second is not None  # never skipped
    assert first.identity == {} and first.identity_sha256 == ""
    assert first.run_id == first.run_label
    assert "MGI006" in _codes(caplog)


def test_nonresumable_provider_is_quiet_when_resume_is_disabled(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(
            _cfg(),
            MockLLM(responder=lambda _system, _user: '{"decision": "ESCALATE"}'),
            skip_completed=False,
        )
    assert result is not None and result.identity_sha256 == ""
    assert _identity_warnings(caplog) == []


def test_unavailable_model_id_is_non_resumable(tmp_path, caplog):
    class _NoModelId(MockLLM):
        @property
        def model_id(self):
            raise RuntimeError("unavailable")

    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _NoModelId(), skip_completed=True)
    assert result is not None and result.identity_sha256 == ""
    assert "MGI006" in _codes(caplog)


def test_unresolvable_provider_parameters_are_non_resumable(tmp_path, caplog):
    class _BadParams(MockLLM):
        def resume_identity_parameters(self):
            return {"temperature": float("nan")}

    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _BadParams(), skip_completed=True)
    assert result is not None and result.identity_sha256 == ""
    assert "MGI006" in _codes(caplog)


@pytest.mark.parametrize("bad_parameters", [["not-a-pair"], "not-a-mapping", 42])
def test_invalid_provider_parameter_types_are_non_resumable(tmp_path, caplog, bad_parameters):
    class _BadParams(MockLLM):
        def resume_identity_parameters(self):
            return bad_parameters

    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _BadParams(), skip_completed=True)
    assert result is not None and result.identity_sha256 == ""
    assert "MGI006" in _codes(caplog)


def test_provider_identity_hook_exception_is_non_resumable(tmp_path, caplog):
    class _RaisingParams(MockLLM):
        def resume_identity_parameters(self):
            raise RuntimeError("secret exception detail")

    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _RaisingParams(), skip_completed=True)
    assert result is not None and result.identity_sha256 == ""
    assert "MGI006" in _codes(caplog)
    assert "secret exception detail" not in caplog.text


def test_missing_package_version_is_non_resumable(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(mech_gov, "__version__", "", raising=False)
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _llm(), skip_completed=True)
    assert result is not None and result.identity_sha256 == ""
    assert "MGI007" in _codes(caplog)


def test_a_provider_that_never_overrides_the_contract_is_non_resumable(tmp_path):
    warnings = IdentityWarnings()
    llm = create_llm({"provider": "callable", "callable": lambda s, u: '{"decision": "DEFER"}'})
    identity = _resolve_identity(_cfg(), llm, "label", {}, warnings)
    assert identity is None


# --------------------------------------------------------------------------
# Legacy and malformed stored records
# --------------------------------------------------------------------------


def _seed_legacy(tmp_path, **extra):
    record = {"run_id": "mock-R2-S0-seed42", "model": "mock", "n_cases": N}
    record.update(extra)
    (tmp_path / "experiment_results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_legacy_label_warns_and_executes(tmp_path, caplog):
    _seed_legacy(tmp_path)
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _llm(), skip_completed=True)
    assert result is not None
    assert _codes(caplog) == ["MGI001"]


def test_unknown_schema_warns_and_executes(tmp_path, caplog):
    _seed_legacy(tmp_path, identity_schema=99, identity_sha256="a" * 64, identity={})
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    assert "MGI003" in _codes(caplog)


def test_invalid_digest_warns_and_executes(tmp_path, caplog):
    _seed_legacy(tmp_path, identity_schema=1, identity_sha256="not-a-digest", identity={})
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    assert "MGI004" in _codes(caplog)


def test_tampered_payload_warns_and_executes(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    runner.run_single(_cfg(), _llm(), skip_completed=False)
    path = tmp_path / "experiment_results.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    record["identity"]["cases_per_condition"] = 999
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "mock" / "baseline.jsonl").unlink()

    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    assert "MGI004" in _codes(caplog)


@pytest.mark.parametrize(
    "line",
    [
        "{not json",  # unparseable
        "",  # blank
        "   ",  # whitespace only
        "[1, 2, 3]",  # valid JSON, not an object
        '"a string"',  # valid JSON, not an object
        "42",  # valid JSON, not an object
        '{"run_id": 123}',  # object with a non-string label
        '{"run_id": null}',  # object with an empty label
        '{"run_id":"x","seed":' + "1" * 5000 + "}",  # oversized integer
    ],
)
def test_unusable_jsonl_lines_are_tolerated(tmp_path, line, caplog):
    (tmp_path / "experiment_results.jsonl").write_text(line + "\n", encoding="utf-8")
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    assert _identity_warnings(caplog) == []


def test_invalid_utf8_result_file_warns_and_executes(tmp_path, caplog):
    (tmp_path / "experiment_results.jsonl").write_bytes(b"\xff\xfe")
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _llm(), skip_completed=True)
    assert result is not None
    assert "MGI008" in _codes(caplog)
    assert str(tmp_path) not in caplog.text


def test_jsonl_named_directory_warns_and_executes(tmp_path, caplog):
    (tmp_path / "unreadable.jsonl").mkdir()
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        result = runner.run_single(_cfg(), _llm(), skip_completed=True)
    assert result is not None
    assert "MGI008" in _codes(caplog)


def test_scan_failure_disables_skip_even_with_a_valid_copy(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    assert runner.run_single(_cfg(), _llm(), skip_completed=False) is not None
    (tmp_path / "unreadable.jsonl").write_bytes(b"\xff")
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    assert "MGI008" in _codes(caplog)


def test_invalid_matching_record_disables_skip_even_with_a_valid_copy(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    assert runner.run_single(_cfg(), _llm(), skip_completed=False) is not None
    path = tmp_path / "experiment_results.jsonl"
    invalid = {
        "run_id": "mock-R2-S0-seed42",
        "identity_schema": 1,
        "identity_sha256": "not-a-digest",
        "identity": {},
    }
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(invalid) + "\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    assert "MGI004" in _codes(caplog)


def test_a_junk_line_does_not_hide_a_real_completed_identity(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    runner.run_single(_cfg(), _llm(), skip_completed=False)
    path = tmp_path / "experiment_results.jsonl"
    path.write_text("{not json\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert runner.run_single(_cfg(), _llm(), skip_completed=True) is None


def test_record_without_run_id_creates_no_completed_identity(tmp_path, caplog):
    (tmp_path / "experiment_results.jsonl").write_text('{"model": "mock"}\n', encoding="utf-8")
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(), _llm(), skip_completed=True) is not None
    assert _identity_warnings(caplog) == []


def test_valid_identity_without_any_label_cannot_skip(tmp_path):
    producer = ExperimentRunner(results_dir=str(tmp_path / "producer"))
    result = producer.run_single(_cfg(), _llm(), skip_completed=False)
    record = {
        "identity_schema": result.identity["identity_schema"],
        "identity_sha256": result.identity_sha256,
        "identity": result.identity,
    }
    results_dir = tmp_path / "consumer"
    results_dir.mkdir()
    (results_dir / "experiment_results.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    consumer = ExperimentRunner(results_dir=str(results_dir))
    assert consumer.run_single(_cfg(), _llm(), skip_completed=True) is not None


def test_conflicting_payloads_for_one_digest_disable_skipping(caplog):
    """Defensive guard: verified digests bind their payload, so this needs seeding."""
    runner_warnings = IdentityWarnings()
    identity = _resolve_identity(_cfg(), _llm(), "mock-R2-S0-seed42", {}, runner_warnings)
    record = {
        "run_label": identity.label,
        "identity_schema": 1,
        "identity_sha256": identity.digest,
        "identity": identity.payload,
    }
    index = _CompletionIndex()
    index.digests.add(identity.digest)
    index.canonical[identity.digest] = b"{}"  # a different canonical payload

    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        index.add(record, IdentityWarnings())
        assert identity.digest in index.conflicts
        assert _decide_skip(identity, identity.label, index, IdentityWarnings()) is False
    assert "MGI005" in _codes(caplog)


def test_duplicate_output_copies_count_as_one_identity(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    runner.run_single(_cfg(), _llm(), skip_completed=False)
    # The same result is written to both the flat file and the model subdir.
    index = runner._completion_index(IdentityWarnings())
    assert len(index.digests) == 1
    assert len(index.payloads_by_label["mock-R2-S0-seed42"]) == 1


def test_warning_is_emitted_once_per_scan(tmp_path, caplog):
    _seed_legacy(tmp_path)
    (tmp_path / "mock").mkdir()
    (tmp_path / "mock" / "baseline.jsonl").write_text(
        json.dumps({"run_id": "mock-R2-S0-seed42"}) + "\n", encoding="utf-8"
    )
    runner = ExperimentRunner(results_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        runner.run_single(_cfg(), _llm(), skip_completed=True)
    assert len(_identity_warnings(caplog)) == 1


def test_multiple_prior_identities_report_the_union_of_changed_fields(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    runner.run_single(_cfg(N), _llm(), skip_completed=False)
    runner.run_single(_cfg(N, seed=42, condition="S0"), _llm("model-v2"), skip_completed=False)
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        assert runner.run_single(_cfg(M), _llm(), skip_completed=True) is not None
    message = _identity_warnings(caplog)[0].message
    assert "cases_per_condition" in message and "model_id_sha256" in message


# --------------------------------------------------------------------------
# Compatibility and privacy
# --------------------------------------------------------------------------


def test_historical_records_are_never_rewritten(tmp_path):
    _seed_legacy(tmp_path)
    path = tmp_path / "experiment_results.jsonl"
    before = path.read_text(encoding="utf-8")
    ExperimentRunner(results_dir=str(tmp_path)).run_single(_cfg(), _llm(), skip_completed=True)
    assert path.read_text(encoding="utf-8").startswith(before)


def test_result_shape_and_output_paths_are_preserved(tmp_path):
    seen: list = []
    runner = ExperimentRunner(results_dir=str(tmp_path), on_run_complete=seen.append)
    result = runner.run_single(_cfg(), _llm(), skip_completed=False)

    assert (tmp_path / "experiment_results.jsonl").exists()
    assert (tmp_path / "mock" / "baseline.jsonl").exists()
    assert len(seen) == 1 and seen[0] is result

    record = _records(tmp_path)[0]
    for key in ("run_id", "model", "regime", "condition", "seed", "n_cases", "metrics"):
        assert key in record
    assert record["model_id"] == "mock"  # frozen: raw id still on the result
    assert record["run_label"] == "mock-R2-S0-seed42"
    assert record["run_id"] == "mock-R2-S0-seed42"
    assert record["identity_schema"] == 1
    assert len(record["identity_sha256"]) == 64


def test_manifest_still_records_the_raw_model_id(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    runner.run_full_experiment(
        models={"mock": _llm()}, regimes=["R2"], conditions=["S0"], seeds=[1], cases_per_condition=N
    )
    manifest = json.loads((tmp_path / "mock" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "mock"
    assert "identity" not in manifest


def test_full_experiment_still_skips_an_identical_repeat(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    args = dict(
        models={"mock": _llm()},
        regimes=["R2"],
        conditions=["S0"],
        seeds=[1],
        cases_per_condition=N,
    )
    assert len(runner.run_full_experiment(**args)) == 1
    assert runner.run_full_experiment(**args) == []


def test_full_experiment_reruns_when_the_case_count_changes(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    args = dict(models={"mock": _llm()}, regimes=["R2"], conditions=["S0"], seeds=[1])
    assert len(runner.run_full_experiment(cases_per_condition=N, **args)) == 1
    assert len(runner.run_full_experiment(cases_per_condition=M, **args)) == 1


def test_identity_metadata_excludes_paths_and_raw_identifiers(tmp_path):
    dist = _write_dist(tmp_path, name="secret-looking-path.yaml")
    runner = ExperimentRunner(results_dir=str(tmp_path / "results"))
    runner.run_single(_cfg(dist=dist), _llm("arn:aws:bedrock:eu-west-1::deployment"), False)

    identity = json.dumps(_records(tmp_path / "results")[0]["identity"])
    assert "secret-looking-path" not in identity
    assert str(tmp_path) not in identity
    assert "arn:aws" not in identity
    assert "deployment" not in identity


def test_collision_diagnostics_never_print_values(tmp_path, caplog):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    runner.run_single(_cfg(N), _llm("arn:aws:bedrock:eu-west-1::deployment"), skip_completed=False)
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        runner.run_single(_cfg(M), _llm("arn:aws:bedrock:eu-west-1::deployment"), True)
    message = _identity_warnings(caplog)[0].message
    assert "arn:aws" not in message
    assert f"cases_per_condition={N}" not in message
    assert f"cases_per_condition={M}" not in message
    assert "mock-R2-S0-seed42" not in message
    assert "reference_sha256=" in message
    assert "cases_per_condition" in message


@pytest.mark.parametrize("provider", ["openai_compatible", "bedrock"])
def test_resumable_cloud_provider_identity_excludes_credentials(provider, monkeypatch):
    monkeypatch.setenv("MECH_GOV_LLM_BASE_URL", "https://internal.example/v1")
    monkeypatch.setenv("MECH_GOV_LLM_API_KEY", "sk-super-secret")
    monkeypatch.setenv("MECH_GOV_LLM_MODEL", "gpt-test")
    if provider == "bedrock":
        pytest.importorskip("boto3")
    config = {
        "openai_compatible": {"provider": "openai_compatible"},
        "bedrock": {"provider": "bedrock", "model_id": "anthropic.test"},
    }[provider]
    parameters = create_llm(config).resume_identity_parameters()
    assert parameters["provider"] == provider
    assert parameters["top_p"] == 1.0
    assert any(key.endswith("_sha256") for key in parameters)
    assert set(parameters) == {
        "provider",
        "top_p",
        "region_sha256" if provider == "bedrock" else "endpoint_sha256",
    }
    serialized = json.dumps(parameters)
    for secret in ("secret", "sk-super-secret", "internal", "us-east-1"):
        assert secret not in serialized


def test_sagemaker_is_non_resumable_without_account_scoped_deployment_identity(monkeypatch):
    pytest.importorskip("boto3")
    llm = create_llm(
        {"provider": "sagemaker", "endpoint_name": "internal-endpoint", "region": "eu-west-1"}
    )
    assert llm.resume_identity_parameters() is None


def test_canonical_bytes_of_a_stored_payload_match_the_live_one(tmp_path):
    runner = ExperimentRunner(results_dir=str(tmp_path))
    result = runner.run_single(_cfg(), _llm(), skip_completed=False)
    stored = _records(tmp_path)[0]["identity"]
    assert canonical_bytes(stored) == canonical_bytes(result.identity)
