# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Run a governance regime over the synthetic banking dataset and print metrics.

Examples:
    python scripts/run_governance.py --regime R2 --provider mock --n 20
    python scripts/run_governance.py --regime R1 \
        --models-config configs/models.example.yaml --model local --n 50

Writes per-run JSONL to the results directory and prints the governance + task
metrics (CDL, DIU, accuracy, macro-F1, MCC, ...).
"""

from __future__ import annotations

import argparse
import json

from mech_gov.experiment.runner import ExperimentRunner, RunConfig
from mech_gov.llm.registry import create_llm, load_models_config


def _build_llm(args):
    if args.models_config:
        models = load_models_config(args.models_config)
        if args.model not in models:
            raise SystemExit(
                f"Model '{args.model}' not found in {args.models_config}. "
                f"Available: {list(models)}"
            )
        return create_llm(models[args.model])
    # No config file: build directly from the chosen provider name.
    return create_llm({"provider": args.provider, "model_id": args.provider})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a governance regime over the banking example dataset."
    )
    parser.add_argument("--regime", default="R2", choices=["R1", "R2", "R3"])
    parser.add_argument("--condition", default="S0", choices=["S0", "S1", "S2", "S3"])
    parser.add_argument("--n", type=int, default=20, help="cases per condition (>= 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider",
        default="mock",
        help="provider name when no --models-config is given (default: mock)",
    )
    parser.add_argument(
        "--models-config",
        default=None,
        help="path to a models YAML, e.g. configs/models.example.yaml",
    )
    parser.add_argument("--model", default="mock", help="model name within --models-config")
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    llm = _build_llm(args)
    runner = ExperimentRunner(results_dir=args.results_dir)
    config = RunConfig(
        model_name=args.model if args.models_config else args.provider,
        regime_name=args.regime,
        condition=args.condition,
        seed=args.seed,
        cases_per_condition=args.n,
    )
    result = runner.run_single(config, llm, skip_completed=False)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
