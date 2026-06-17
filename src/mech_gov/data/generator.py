# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Case generator for mech_gov v2 synthetic dataset.

Generates BankingCase instances from distribution configs with full
seed control. All randomness flows through numpy.random.Generator(PCG64(seed)).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mech_gov.data.banking_case import (
    BankingCase,
    StressCondition,
    TransactionType,
)
from mech_gov.data.distributions import (
    load_distributions,
    sample_field,
    sample_flags_correlated,
)

logger = logging.getLogger("mech_gov.data.generator")


def make_rng(seed: int) -> np.random.Generator:
    """Create a reproducible random generator from seed."""
    return np.random.Generator(np.random.PCG64(seed))


def generate_cases_for_type(
    rng: np.random.Generator,
    transaction_type: TransactionType,
    type_config: dict,
    n_cases: int,
    stress_condition: StressCondition = StressCondition.S0_BASELINE,
    start_index: int = 0,
    seed: int | None = None,
) -> list[BankingCase]:
    """Generate N cases for a single transaction type and condition.

    Args:
        rng: Numpy random generator (controls all randomness)
        transaction_type: Which type to generate
        type_config: Distribution config for this type (from distributions.yaml)
        n_cases: Number of cases to generate
        stress_condition: Which stress condition to tag (actual transforms applied separately)
        start_index: Starting index for case IDs
        seed: Seed value for metadata tracking

    Returns:
        List of BankingCase instances (ground truth NOT yet assigned)
    """
    # Sample all fields in bulk for efficiency
    risk_scores = sample_field(rng, type_config["risk_score"], size=n_cases)
    completeness_vals = sample_field(rng, type_config["completeness"], size=n_cases)
    flags_list = sample_flags_correlated(
        rng,
        type_config["flags"]["probabilities"],
        risk_scores,
        correlation_strength=type_config.get("flags", {}).get("correlation_strength", 0.6),
    )
    amounts = sample_field(rng, type_config["amount_usd"], size=n_cases)
    jurisdictions = sample_field(rng, type_config["jurisdiction"], size=n_cases)
    tenures = sample_field(rng, type_config["customer_tenure_years"], size=n_cases)
    cp_risks = sample_field(rng, type_config["counterparty_risk"], size=n_cases)

    cases = []
    for i in range(n_cases):
        idx = start_index + i
        case_id = f"{transaction_type.value}-{stress_condition.value}-{idx:04d}"

        case = BankingCase(
            case_id=case_id,
            transaction_type=transaction_type,
            risk_score=float(np.clip(risk_scores[i], 0.0, 1.0)),
            completeness=float(np.clip(completeness_vals[i], 0.0, 1.0)),
            regulatory_flags=flags_list[i],
            amount_usd=float(max(0.0, amounts[i])),
            jurisdiction=str(jurisdictions[i]),
            customer_tenure_years=float(max(0.0, tenures[i])),
            counterparty_risk=float(np.clip(cp_risks[i], 0.0, 1.0)),
            stress_condition=stress_condition,
            seed=seed,
        )
        cases.append(case)

    return cases


def generate_dataset(
    seed: int,
    n_cases_per_condition: int,
    stress_conditions: list[StressCondition],
    distributions_config: dict[str, dict],
    transaction_types: list[TransactionType] | None = None,
) -> list[BankingCase]:
    """Generate a full dataset: all types × all conditions.

    Each transaction type gets n_cases_per_condition / len(transaction_types)
    cases per condition (rounded). Total = n_per_type × n_types × n_conditions.

    Args:
        seed: Master seed for reproducibility
        n_cases_per_condition: Total cases per condition (split across types)
        stress_conditions: Which conditions to generate
        distributions_config: Loaded from distributions.yaml
        transaction_types: Which types to generate (default: all)

    Returns:
        List of all generated BankingCase instances
    """
    if transaction_types is None:
        transaction_types = list(TransactionType)

    n_types = len(transaction_types)
    n_per_type = n_cases_per_condition // n_types

    if n_per_type < 1:
        raise ValueError(
            f"n_cases_per_condition ({n_cases_per_condition}) too small for "
            f"{n_types} transaction types. Need at least {n_types}."
        )

    all_cases: list[BankingCase] = []
    # Use master seed to derive sub-seeds per (type, condition) pair
    master_rng = make_rng(seed)

    logger.info(
        "Generating dataset: seed=%d  N/cond=%d  conditions=%s  types=%d  total=%d",
        seed,
        n_cases_per_condition,
        [c.value for c in stress_conditions],
        n_types,
        n_per_type * n_types * len(stress_conditions),
    )

    for condition in stress_conditions:
        for tt in transaction_types:
            # Derive a deterministic sub-seed for this (type, condition) pair
            sub_seed = int(master_rng.integers(0, 2**31))
            sub_rng = make_rng(sub_seed)

            type_config = distributions_config[tt.value]
            cases = generate_cases_for_type(
                rng=sub_rng,
                transaction_type=tt,
                type_config=type_config,
                n_cases=n_per_type,
                stress_condition=condition,
                start_index=len(all_cases),
                seed=seed,
            )
            all_cases.extend(cases)

    return all_cases


def generate_dataset_from_config(
    seed: int,
    n_cases_per_condition: int,
    stress_conditions: list[str],
    distributions_path: str | Path,
    transaction_types: list[str] | None = None,
) -> list[BankingCase]:
    """Convenience wrapper: load config and generate dataset.

    Args:
        seed: Master seed
        n_cases_per_condition: Cases per condition
        stress_conditions: Condition names as strings (e.g. "S0", "S1")
        distributions_path: Path to distributions.yaml
        transaction_types: Type names as strings (default: all)

    Returns:
        List of BankingCase instances
    """
    dist_config = load_distributions(distributions_path)

    conditions = [StressCondition(c) for c in stress_conditions]

    types = None
    if transaction_types is not None:
        types = [TransactionType(t) for t in transaction_types]

    return generate_dataset(
        seed=seed,
        n_cases_per_condition=n_cases_per_condition,
        stress_conditions=conditions,
        distributions_config=dist_config,
        transaction_types=types,
    )
