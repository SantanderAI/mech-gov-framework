# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Distribution definitions loader and sampler for mech_gov v2 synthetic dataset.

Loads distribution parameters from config/distributions.yaml and provides
sampling functions. All randomness goes through numpy.random.Generator
for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mech_gov.data.banking_case import TransactionType

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_distributions(config_path: str | Path) -> dict[str, dict]:
    """Load distribution definitions from YAML config.

    Args:
        config_path: Path to distributions.yaml

    Returns:
        Dict mapping transaction_type name -> field distributions
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Distribution config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate that all transaction types are present
    for tt in TransactionType:
        if tt.value not in config:
            raise ValueError(
                f"Missing distribution config for transaction type '{tt.value}'. "
                f"Found: {list(config.keys())}"
            )

    return config


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def sample_beta(rng: np.random.Generator, params: dict[str, float], size: int = 1) -> np.ndarray:
    """Sample from Beta distribution."""
    a = params["a"]
    b = params["b"]
    return rng.beta(a, b, size=size)


def sample_lognormal(
    rng: np.random.Generator, params: dict[str, float], size: int = 1
) -> np.ndarray:
    """Sample from LogNormal distribution."""
    mu = params["mu"]
    sigma = params["sigma"]
    return rng.lognormal(mu, sigma, size=size)


def sample_exponential(
    rng: np.random.Generator, params: dict[str, float], size: int = 1
) -> np.ndarray:
    """Sample from Exponential distribution."""
    lam = params["lambda"]
    return rng.exponential(1.0 / lam, size=size)


def sample_categorical(
    rng: np.random.Generator, values: list[str], weights: list[float], size: int = 1
) -> np.ndarray:
    """Sample from Categorical distribution."""
    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()  # Normalize
    return rng.choice(values, size=size, p=weights_arr)


def sample_flags(
    rng: np.random.Generator, probabilities: dict[str, float], size: int = 1
) -> list[list[str]]:
    """Sample regulatory flags as independent Bernoulli per flag.

    Returns:
        List of lists — each inner list contains the flags present for that case.
    """
    results = []
    for _ in range(size):
        flags = []
        for flag_name, prob in probabilities.items():
            if rng.random() < prob:
                flags.append(flag_name)
        results.append(sorted(flags))
    return results


# Flags whose base probability should be strongly modulated by risk_score
_SEVERE_FLAGS = {"AML", "SANCTIONS", "INSIDER"}


def sample_flags_correlated(
    rng: np.random.Generator,
    probabilities: dict[str, float],
    risk_scores: np.ndarray,
    correlation_strength: float = 0.6,
) -> list[list[str]]:
    """Sample flags with probability conditioned on risk_score.

    For each case *i* the effective probability of flag *f* is:

        p_eff(f, i) = clip( p_base(f) * (1 + strength * (risk_i - 0.5)), 0, 1 )

    where *strength* is larger for severe flags (AML, SANCTIONS, INSIDER)
    and halved for minor flags (KYC, CONCENTRATION).  This gives a monotone
    relationship: higher risk → more severe flags, lower risk → fewer.

    Args:
        rng: Numpy random generator.
        probabilities: Base (marginal) probabilities from distributions.yaml.
        risk_scores: Array of risk_score values, one per case.
        correlation_strength: Controls how much risk modulates flag probs.
            0.0 = independent (degrades to sample_flags).
            0.6 = moderate positive correlation (default).

    Returns:
        List of flag-lists, one per case.
    """
    size = len(risk_scores)
    results: list[list[str]] = []

    for i in range(size):
        risk = risk_scores[i]
        flags: list[str] = []
        for flag_name, base_prob in probabilities.items():
            # Severe flags get full strength; minor flags get half
            s = correlation_strength if flag_name in _SEVERE_FLAGS else correlation_strength * 0.5
            p_eff = base_prob * (1.0 + s * (risk - 0.5))
            p_eff = float(np.clip(p_eff, 0.0, 1.0))
            if rng.random() < p_eff:
                flags.append(flag_name)
        results.append(sorted(flags))

    return results


# ---------------------------------------------------------------------------
# Field sampler dispatch
# ---------------------------------------------------------------------------

SAMPLER_REGISTRY = {
    "beta": sample_beta,
    "lognormal": sample_lognormal,
    "exponential": sample_exponential,
}


def sample_field(
    rng: np.random.Generator, field_config: dict[str, Any], size: int = 1
) -> np.ndarray:
    """Sample a single numeric field based on its distribution config.

    Args:
        rng: Numpy random generator
        field_config: Dict with 'distribution' and 'params' keys
        size: Number of samples

    Returns:
        Array of sampled values
    """
    dist_type = field_config["distribution"]

    if dist_type == "categorical":
        return sample_categorical(
            rng,
            values=field_config["values"],
            weights=field_config["weights"],
            size=size,
        )

    if dist_type not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown distribution type '{dist_type}'. "
            f"Available: {list(SAMPLER_REGISTRY.keys()) + ['categorical']}"
        )

    sampler = SAMPLER_REGISTRY[dist_type]
    values = sampler(rng, field_config["params"], size=size)

    # Clip to range if specified
    if "range" in field_config:
        lo, hi = field_config["range"]
        values = np.clip(values, lo, hi)

    return values


def perturb_params(params: dict[str, float], factor: float) -> dict[str, float]:
    """Perturb distribution parameters by a factor for sensitivity analysis.

    Args:
        params: Original distribution parameters
        factor: Perturbation factor, e.g. 0.8 for -20%, 1.2 for +20%

    Returns:
        New params dict with perturbed values
    """
    return {k: v * factor for k, v in params.items()}
