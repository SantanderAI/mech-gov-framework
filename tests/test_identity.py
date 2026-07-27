# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for deterministic, versioned run identity."""

from __future__ import annotations

import datetime
import json
import logging

import pytest

from mech_gov.experiment.identity import (
    PAYLOAD_FIELDS,
    IdentityError,
    IdentityWarnings,
    build_run_identity,
    canonical_bytes,
    changed_fields,
    digest_payload,
    distributions_digest,
    verified_stored_digest,
)

BASE = {
    "label": "mock-R2-S0-seed42",
    "model_name": "mock",
    "model_id": "vendor/model-v1",
    "regime_name": "R2",
    "condition": "S0",
    "seed": 42,
    "cases_per_condition": 10,
    "distributions_config": {"credit_approval": {"amount": {"mu": 1.0}}},
    "provider_parameters": {"top_p": 1.0},
    "package_version": "0.1.0",
}


def identity(**overrides):
    return build_run_identity(**{**BASE, **overrides})


# --------------------------------------------------------------------------
# Label / digest separation
# --------------------------------------------------------------------------


def test_label_and_authoritative_digest_are_separate():
    ident = identity()
    assert ident.label == "mock-R2-S0-seed42"
    assert len(ident.digest) == 64


def test_identity_is_deterministic_across_instances():
    assert identity().digest == identity().digest


def test_schema_one_has_a_pinned_canonical_vector():
    expected = (
        '{"cases_per_condition":"n:1E+1","condition":"s:S0",'
        '"distributions_sha256":"s:e0c57a20148c66a32df79a41b725c4fbf2525e30e585890b2bf9b65ed449367f",'
        '"identity_schema":"n:1",'
        '"model_id_sha256":"s:f0cb9b49bc405836585ef9d6785501b0685241788111424e03fba0f5d55e06b6",'
        '"model_name":"s:mock","package_version":"s:0.1.0",'
        '"provider_parameters":{"top_p":"n:1"},"regime_name":"s:R2","seed":"n:42"}'
    )
    ident = identity()
    assert canonical_bytes(ident.payload) == expected.encode("utf-8")
    expected_digest = (
        "a5e7a917d45ead0077a43eacd286a96d"  # pragma: allowlist secret
        "2c30110d97dee330e1bb12dcc76108cd"  # pragma: allowlist secret
    )
    assert ident.digest == expected_digest


def test_payload_carries_exactly_the_declared_fields():
    assert set(identity().payload) == set(PAYLOAD_FIELDS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("cases_per_condition", 1000),
        ("model_id", "vendor/model-v2"),
        ("model_name", "other"),
        ("regime_name", "R1"),
        ("condition", "S1"),
        ("seed", 43),
        ("distributions_config", {"credit_approval": {"amount": {"mu": 2.0}}}),
        ("provider_parameters", {"top_p": 0.9}),
        ("package_version", "0.2.0"),
    ],
)
def test_every_result_affecting_input_changes_identity(field, value):
    assert identity(**{field: value}).digest != identity().digest


def test_label_alone_does_not_change_identity():
    # The readable label is metadata, not part of the equivalence domain.
    assert identity(label="something-else").digest == identity().digest


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------


def test_mapping_insertion_order_is_irrelevant():
    first = distributions_digest({"a": 1, "b": {"x": 1, "y": 2}})
    second = distributions_digest({"b": {"y": 2, "x": 1}, "a": 1})
    assert first == second


def test_list_order_is_significant():
    assert distributions_digest({"k": [1, 2]}) != distributions_digest({"k": [2, 1]})


def test_equivalent_numbers_normalize_identically():
    assert distributions_digest({"k": 1}) == distributions_digest({"k": 1.0})
    assert distributions_digest({"k": 0}) == distributions_digest({"k": -0.0})
    assert distributions_digest({"k": 100}) == distributions_digest({"k": 100.0})


def test_scalar_types_never_collide():
    digests = {distributions_digest({"k": v}) for v in (None, False, True, 0, "z:", "b:false")}
    assert len(digests) == 6


def test_identical_content_at_another_path_is_identical():
    # distributions_digest consumes loaded content, so the source path is absent.
    loaded = {"credit_approval": {"amount": {"mu": 1.0}}}
    assert distributions_digest(dict(loaded)) == distributions_digest(loaded)


def test_utf8_content_is_preserved_not_escaped():
    payload = {"k": "café-Ω"}
    assert "café-Ω".encode() in canonical_bytes(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(bad):
    with pytest.raises(IdentityError, match="non-finite"):
        distributions_digest({"k": bad})


@pytest.mark.parametrize(
    "bad", [datetime.date(2026, 1, 1), datetime.datetime(2026, 1, 1), b"bytes", {1, 2}, object()]
)
def test_unsupported_types_are_rejected(bad):
    with pytest.raises(IdentityError, match="unsupported"):
        distributions_digest({"k": bad})


def test_non_string_mapping_keys_are_rejected():
    with pytest.raises(IdentityError, match="non-string mapping key"):
        distributions_digest({1: "x"})


def test_rejection_reports_the_field_path_without_the_value():
    with pytest.raises(IdentityError) as excinfo:
        distributions_digest({"outer": [{"inner": float("nan")}]})
    assert ".outer[0].inner" in str(excinfo.value)
    assert "nan" not in str(excinfo.value).replace("non-finite", "")


def test_empty_configuration_still_fingerprints():
    assert len(distributions_digest({})) == 64


# --------------------------------------------------------------------------
# Privacy
# --------------------------------------------------------------------------


def test_raw_model_id_is_hashed_never_stored_in_identity():
    ident = identity(model_id="arn:aws:bedrock:eu-west-1::secret-deployment")
    serialized = json.dumps(ident.payload)
    assert "arn:aws" not in serialized
    assert "secret-deployment" not in serialized
    assert len(ident.payload["model_id_sha256"]) == 64


def test_missing_model_or_package_identity_is_refused():
    with pytest.raises(IdentityError, match="model_id"):
        identity(model_id="")
    with pytest.raises(IdentityError, match="package_version"):
        identity(package_version="")


# --------------------------------------------------------------------------
# Stored-record validation
# --------------------------------------------------------------------------


def stored_record(ident=None):
    ident = ident or identity()
    return json.loads(
        json.dumps(
            {
                "run_id": ident.label,
                "run_label": ident.label,
                "identity_schema": 1,
                "identity_sha256": ident.digest,
                "identity": ident.payload,
            }
        )
    )


def test_valid_record_is_accepted():
    ident = identity()
    assert verified_stored_digest(stored_record(ident)) == (ident.digest, None)


def test_legacy_record_reports_no_identity_and_no_warning():
    assert verified_stored_digest({"run_id": "mock-R2-S0-seed42"}) == (None, None)


@pytest.mark.parametrize(
    "schema, expected_code",
    [(99, "MGI003"), (0, "MGI003"), ("1", "MGI003"), (True, "MGI003"), (None, "MGI003")],
)
def test_unsupported_schema_is_refused(schema, expected_code):
    record = stored_record()
    record["identity_schema"] = schema
    digest, code = verified_stored_digest(record)
    assert digest is None and code == expected_code


@pytest.mark.parametrize("digest", ["", "xyz", "A" * 64, "a" * 63, 12345])
def test_malformed_digest_is_refused(digest):
    record = stored_record()
    record["identity_sha256"] = digest
    assert verified_stored_digest(record) == (None, "MGI004")


def test_tampered_payload_is_refused():
    record = stored_record()
    record["identity"]["cases_per_condition"] = 999
    assert verified_stored_digest(record) == (None, "MGI004")


def test_incomplete_payload_is_refused():
    record = stored_record()
    del record["identity"]["seed"]
    assert verified_stored_digest(record) == (None, "MGI004")


def test_extra_payload_field_is_refused():
    record = stored_record()
    record["identity"]["injected"] = "x"
    assert verified_stored_digest(record) == (None, "MGI004")


def test_payload_schema_must_match_record_schema():
    record = stored_record()
    record["identity"]["identity_schema"] = 2
    assert verified_stored_digest(record) == (None, "MGI004")


def test_non_mapping_payload_is_refused():
    record = stored_record()
    record["identity"] = ["not", "a", "mapping"]
    assert verified_stored_digest(record) == (None, "MGI004")


def test_uncanonicalizable_payload_is_refused():
    ident = identity()
    record = stored_record(ident)
    record["identity"]["provider_parameters"] = {"top_p": float("nan")}
    assert verified_stored_digest(record) == (None, "MGI004")


def test_digest_survives_json_round_trip():
    ident = identity()
    assert digest_payload(json.loads(json.dumps(ident.payload))) == ident.digest


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def test_changed_fields_reports_sorted_names_only():
    left = identity().payload
    right = identity(cases_per_condition=1000, seed=1).payload
    assert changed_fields(left, right) == ["cases_per_condition", "seed"]


def test_changed_fields_is_empty_for_equal_payloads():
    assert changed_fields(identity().payload, identity().payload) == []


def test_changed_fields_uses_canonical_not_python_equality():
    assert changed_fields({"value": True}, {"value": 1}) == ["value"]
    assert changed_fields({"value": (1, 2)}, {"value": [1.0, 2.0]}) == []


def test_warning_names_fields_but_never_values(caplog):
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        IdentityWarnings().emit("MGI002", "mock-R2-S0-seed42", ["cases_per_condition"])
    record = caplog.records[0]
    assert record.name == "mech_gov.experiment.identity"
    assert record.levelno == logging.WARNING
    assert "[MGI002]" in record.message
    assert "fields=cases_per_condition" in record.message
    assert "action=execute" in record.message
    assert "1000" not in record.message


def test_warnings_are_deduplicated_per_code_and_label(caplog):
    warnings = IdentityWarnings()
    with caplog.at_level(logging.WARNING, logger="mech_gov.experiment.identity"):
        warnings.emit("MGI001", "label-a")
        warnings.emit("MGI001", "label-a")
        warnings.emit("MGI001", "label-b")
        warnings.emit("MGI003", "label-a")
    assert len(caplog.records) == 3
