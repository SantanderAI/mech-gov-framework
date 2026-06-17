# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Generate the synthetic banking example dataset to JSONL.

Example:
    python scripts/generate_dataset.py --n 100 --seed 42 --out dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mech_gov.data as _data_pkg
from mech_gov.data.generator import generate_dataset_from_config
from mech_gov.data.ground_truth import assign_ground_truth

_DEFAULT_DIST = str(Path(_data_pkg.__file__).resolve().parent / "banking_distributions.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the bundled synthetic banking dataset.")
    parser.add_argument("--n", type=int, default=100, help="cases per condition (>= 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", default="S0", choices=["S0", "S1", "S2", "S3"])
    parser.add_argument(
        "--distributions",
        default=_DEFAULT_DIST,
        help="path to a distributions YAML (defaults to the bundled spec)",
    )
    parser.add_argument("--out", default="dataset.jsonl")
    parser.add_argument(
        "--no-ground-truth", action="store_true", help="skip ground-truth assignment"
    )
    args = parser.parse_args()

    cases = generate_dataset_from_config(
        seed=args.seed,
        n_cases_per_condition=args.n,
        stress_conditions=[args.condition],
        distributions_path=args.distributions,
    )
    if not args.no_ground_truth:
        assign_ground_truth(cases)

    with open(args.out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict()) + "\n")

    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
