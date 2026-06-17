# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for distribution loaders and samplers (offline, seeded)."""

import numpy as np
import pytest
import yaml

from mech_gov.data.distributions import (
    load_distributions,
    perturb_params,
    sample_beta,
    sample_categorical,
    sample_exponential,
    sample_field,
    sample_flags,
    sample_flags_correlated,
    sample_lognormal,
)
from mech_gov.data.generator import make_rng
from mech_gov.experiment.runner import _DEFAULT_DIST_CONFIG


def test_load_distributions_packaged_config():
    cfg = load_distributions(_DEFAULT_DIST_CONFIG)
    assert isinstance(cfg, dict) and cfg


def test_load_distributions_missing_file():
    with pytest.raises(FileNotFoundError):
        load_distributions("/no/such/file.yaml")


def test_load_distributions_missing_type(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"only_one_type": {}}))
    with pytest.raises(ValueError):
        load_distributions(bad)


def test_samplers_return_expected_shape():
    rng = make_rng(0)
    assert sample_beta(rng, {"a": 2, "b": 5}, size=4).shape == (4,)
    assert sample_lognormal(rng, {"mu": 0.0, "sigma": 1.0}, size=3).shape == (3,)
    assert sample_exponential(rng, {"lambda": 1.5}, size=2).shape == (2,)


def test_sample_categorical_normalizes_weights():
    rng = make_rng(1)
    out = sample_categorical(rng, ["a", "b"], [3.0, 1.0], size=20)
    assert set(out.tolist()) <= {"a", "b"}


def test_sample_flags_independent():
    rng = make_rng(2)
    out = sample_flags(rng, {"KYC": 1.0, "AML": 0.0}, size=5)
    assert all("KYC" in flags and "AML" not in flags for flags in out)


def test_sample_flags_correlated_monotone_with_risk():
    rng = make_rng(3)
    low = sample_flags_correlated(rng, {"AML": 0.5}, np.array([0.0] * 200))
    high = sample_flags_correlated(rng, {"AML": 0.5}, np.array([1.0] * 200))
    assert sum(len(f) for f in high) >= sum(len(f) for f in low)


def test_sample_field_beta_with_range_clip():
    rng = make_rng(4)
    vals = sample_field(
        rng, {"distribution": "beta", "params": {"a": 2, "b": 2}, "range": [0.2, 0.8]}, size=50
    )
    assert vals.min() >= 0.2 and vals.max() <= 0.8


def test_sample_field_categorical():
    rng = make_rng(5)
    vals = sample_field(
        rng, {"distribution": "categorical", "values": ["x"], "weights": [1.0]}, size=3
    )
    assert list(vals) == ["x", "x", "x"]


def test_sample_field_unknown_distribution():
    with pytest.raises(ValueError):
        sample_field(make_rng(6), {"distribution": "weibull", "params": {}})


def test_perturb_params():
    assert perturb_params({"a": 2.0, "b": 4.0}, 0.5) == {"a": 1.0, "b": 2.0}
