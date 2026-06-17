# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Task Metrics — Decision Accuracy, F1 Macro, MCC.

Computed ONLY on deterministic ground-truth cases (~40% of dataset).
These are the cases where a single correct answer exists.

the design spec §4.4.
"""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from mech_gov.data.banking_case import BankingCase, GTConfidence
from mech_gov.governance.regime import DecisionResult


def compute_task_metrics(
    results: list[DecisionResult],
    cases: list[BankingCase],
) -> dict[str, float]:
    """Compute task-level metrics on deterministic ground-truth cases.

    Filters to cases where gt_confidence == DETERMINISTIC and gt_decision
    is set. Ambiguous cases are excluded (no single correct answer).

    Args:
        results: Decision results from a regime
        cases: Corresponding banking cases (same order)

    Returns:
        Dict with 'accuracy', 'f1_macro', 'mcc', and 'n_evaluated'.
        Returns zeros and n_evaluated=0 if no deterministic cases.
    """
    if len(results) != len(cases):
        raise ValueError(
            f"results and cases must have same length. " f"Got {len(results)} and {len(cases)}."
        )

    # Filter to deterministic GT cases only
    gt_pairs = [
        (r, c)
        for r, c in zip(results, cases, strict=False)
        if c.gt_confidence == GTConfidence.DETERMINISTIC and c.gt_decision
    ]

    if not gt_pairs:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "mcc": 0.0,
            "n_evaluated": 0,
        }

    y_true = [str(c.gt_decision) for _, c in gt_pairs]
    y_pred = [r.decision.value for r, _ in gt_pairs]

    # Get all possible labels for consistent computation
    all_labels = sorted(set(y_true) | set(y_pred))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0.0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "n_evaluated": len(gt_pairs),
    }
