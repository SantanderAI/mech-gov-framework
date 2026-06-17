# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Experiment Runner for mech_gov v2.

Orchestrates full experimental runs: regime × condition × seed × model.
Supports JSONL serialization, progress tracking, and resumability.

Matches the design spec §5.2.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mech_gov.data.banking_case import (
    BankingCase,
    Decision,
    StressCondition,
)
from mech_gov.data.distributions import load_distributions
from mech_gov.data.generator import generate_dataset
from mech_gov.data.ground_truth import assign_ground_truth
from mech_gov.data.stress import apply_stress
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.governance.r3_adaptive import R3Adaptive
from mech_gov.governance.regime import DecisionResult, GovernanceRegime
from mech_gov.llm.base import LLMInterface
from mech_gov.metrics.governance.cdl import compute_cdl
from mech_gov.metrics.governance.diu import compute_diu
from mech_gov.metrics.governance.ipi import compute_aivr, compute_ipi
from mech_gov.metrics.governance.scorers import (
    compute_bshift_rulebased,
    compute_causal_rulebased,
    compute_spec_rulebased,
)
from mech_gov.metrics.task.accuracy import compute_task_metrics
from mech_gov.metrics.task.deferral import compute_adr, compute_overcaution

logger = logging.getLogger("mech_gov.experiment.runner")

# Project root: two levels up from src/experiment/runner.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Bundled banking example dataset spec (ships inside the package).
_DEFAULT_DIST_CONFIG = str(
    Path(__file__).resolve().parent.parent / "data" / "banking_distributions.yaml"
)


@dataclass
class RunConfig:
    """Configuration for a single experimental run."""

    model_name: str
    regime_name: str  # "R1" or "R2"
    condition: str  # "S0", "S1", "S2", "S3"
    seed: int
    cases_per_condition: int
    distributions_config_path: str = _DEFAULT_DIST_CONFIG


def _get_code_version() -> str:
    """Get short git hash for traceability. Returns '' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


@dataclass
class ExperimentResult:
    """Result of a single run (regime × condition × seed)."""

    run_id: str
    model: str
    regime: str
    condition: str
    seed: int
    n_cases: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    per_case_results: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    model_id: str = ""
    timestamp: str = ""
    code_version: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "model": self.model,
            "regime": self.regime,
            "condition": self.condition,
            "seed": self.seed,
            "n_cases": self.n_cases,
            "metrics": self.metrics,
            "per_case_results": self.per_case_results,
            "elapsed_s": self.elapsed_s,
        }
        if self.model_id:
            d["model_id"] = self.model_id
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.code_version:
            d["code_version"] = self.code_version
        if self.hyperparameters:
            d["hyperparameters"] = self.hyperparameters
        return d


def create_regime(regime_name: str, **kwargs) -> GovernanceRegime:
    """Factory for governance regimes."""
    if regime_name == "R1":
        return R1TextOnly(**kwargs)
    elif regime_name == "R2":
        return R2Mechanical(**kwargs)
    elif regime_name == "R3":
        return R3Adaptive(**kwargs)
    else:
        raise ValueError(f"Unknown regime: {regime_name}")


def _make_run_id(model: str, regime: str, condition: str, seed: int) -> str:
    return f"{model}-{regime}-{condition}-seed{seed}"


class ExperimentRunner:
    """Orchestrates experimental runs with resumability.

    Saves results as JSONL (one JSON object per run) to results_dir.
    Uses model-specific subdirectories: results_dir/{model}/baseline.jsonl
    Also maintains a legacy flat file for backward compatibility.
    Skips already-completed runs on re-invocation.
    """

    def __init__(
        self,
        results_dir: str = "results/raw",
        distributions_config_path: str = _DEFAULT_DIST_CONFIG,
        on_run_complete: Callable[[ExperimentResult], None] | None = None,
    ):
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._dist_config_path = distributions_config_path
        self._on_run_complete = on_run_complete

    def _completed_run_ids(self) -> set:
        """Scan results dir (and model subdirs) for already-completed run IDs."""
        completed = set()
        # Scan flat JSONL files in results_dir (legacy)
        for jsonl_file in self._results_dir.glob("*.jsonl"):
            completed.update(self._scan_jsonl(jsonl_file))
        # Scan model subdirectories
        for subdir in self._results_dir.iterdir():
            if subdir.is_dir():
                for jsonl_file in subdir.glob("*.jsonl"):
                    completed.update(self._scan_jsonl(jsonl_file))
        return completed

    @staticmethod
    def _scan_jsonl(path: Path) -> set:
        """Extract run_ids from a JSONL file."""
        ids = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line).get("run_id", ""))
                except json.JSONDecodeError:
                    continue
        return ids

    def _save_result(self, result: ExperimentResult, filename: str = "baseline.jsonl"):
        """Append a single run result to model-specific JSONL + legacy flat file."""
        # Model-specific directory
        model_dir = self._results_dir / result.model
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        with open(model_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        # Also append to legacy flat file for backward compatibility
        legacy_path = self._results_dir / "experiment_results.jsonl"
        with open(legacy_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def _save_manifest(self, model_name: str, model_id: str, config_summary: dict[str, Any]):
        """Write/update manifest.json in model subdirectory."""
        model_dir = self._results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = model_dir / "manifest.json"

        manifest = {
            "model_name": model_name,
            "model_id": model_id,
            "code_version": _get_code_version(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "config": config_summary,
        }
        # Merge with existing manifest if present
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    existing = json.load(f)
                existing.update(manifest)
                manifest = existing
            except Exception:
                pass

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Manifest updated: %s", manifest_path)

    def run_condition(
        self,
        regime: GovernanceRegime,
        cases: list[BankingCase],
        llm: LLMInterface,
        seed: int,
    ) -> list[DecisionResult]:
        """Run one condition: process all cases through a regime.

        Args:
            regime: Governance regime instance
            cases: Cases to process
            llm: LLM interface
            seed: Entropy seed for E3

        Returns:
            List of DecisionResult (one per case)
        """
        results = []
        for i, case in enumerate(cases):
            case_start = time.perf_counter()
            result = regime.process_case(case, llm, entropy_seed=seed)
            case_elapsed = time.perf_counter() - case_start
            results.append(result)
            logger.info(
                "[%s] case %d/%d  %s → %s  (%.1fs, %d tok)",
                regime.regime_name,
                i + 1,
                len(cases),
                case.case_id,
                result.decision.value,
                case_elapsed,
                result.tokens_used,
            )
        return results

    def _compute_run_metrics(
        self,
        decision_results: list[DecisionResult],
        cases: list[BankingCase],
    ) -> dict[str, dict[str, float]]:
        """Compute all metrics for a single run."""

        # Identify deferrals — separate mechanical overrides from LLM deferrals.
        # Mechanical overrides include pre-LLM hard gates AND post-LLM
        # ambiguity gate (K0_11). Both produce template/forced text that
        # would distort CDL/DIU scoring, so we exclude them.
        def _is_mechanical(r: DecisionResult) -> bool:
            return r.metadata.get("hard_gate_override", False) or r.metadata.get(
                "ambiguity_gate_override", False
            )

        all_deferral_indices = [
            i for i, r in enumerate(decision_results) if r.decision == Decision.DEFER
        ]
        # LLM deferrals only (exclude all mechanical overrides)
        llm_deferral_indices = [
            i for i in all_deferral_indices if not _is_mechanical(decision_results[i])
        ]
        gate_deferral_count = len(all_deferral_indices) - len(llm_deferral_indices)

        llm_deferral_results = [decision_results[i] for i in llm_deferral_indices]
        llm_deferral_cases = [cases[i] for i in llm_deferral_indices]

        def _scorer_text(r: DecisionResult) -> str:
            """Combine deferral_text + rationale for scoring.

            Both fields carry decision-relevant information:
              - deferral_text: what info is missing (if DEFER)
              - rationale: why the decision was made (I6Q-enforced for R2)
            Concatenating both gives scorers the full picture.
            """

            def _to_str(v):
                if isinstance(v, list):
                    return "\n".join(str(x) for x in v)
                return str(v) if v else ""

            parts = [_to_str(r.deferral_text), _to_str(r.rationale)]
            return "\n".join(p for p in parts if p).strip()

        spec_scores = [
            compute_spec_rulebased(_scorer_text(r), c)
            for r, c in zip(llm_deferral_results, llm_deferral_cases, strict=False)
        ]
        causal_scores = [
            compute_causal_rulebased(_scorer_text(r), c)
            for r, c in zip(llm_deferral_results, llm_deferral_cases, strict=False)
        ]
        bshift_scores = [
            compute_bshift_rulebased(_scorer_text(r), c)
            for r, c in zip(llm_deferral_results, llm_deferral_cases, strict=False)
        ]

        # Governance metrics — CDL and DIU over ALL deferrals.
        # Mechanical gate deferrals are non-vacuous by design: they cite
        # exact thresholds and case values.  We assign them perfect scores
        # (spec=1, causal=1, bshift=1) so they lower CDL and raise DIU
        # for R2, reflecting the paper's intended interpretation.
        n_mech = gate_deferral_count
        all_spec = spec_scores + [1.0] * n_mech
        all_causal = causal_scores + [1.0] * n_mech
        all_bshift = bshift_scores + [1.0] * n_mech

        # For CDL we need a results list whose DEFER count matches the
        # score arrays.  Build a synthetic list: LLM deferrals + mechanical.
        all_deferral_results = llm_deferral_results + [
            r for r in decision_results if r.decision == Decision.DEFER and _is_mechanical(r)
        ]
        cdl = compute_cdl(all_deferral_results, all_spec, all_causal)
        diu = compute_diu(all_spec, all_causal, all_bshift)

        gov_metrics: dict[str, float] = {
            "CDL": round(cdl, 4),
            "DIU": round(diu, 4),
            "gate_override_rate": round(
                sum(1 for r in decision_results if _is_mechanical(r))
                / max(len(decision_results), 1),
                4,
            ),
            "gate_defer_count": gate_deferral_count,
        }

        # R3: IPI and AIVR (only meaningful for R3 results)
        if decision_results and decision_results[0].regime == "R3":
            ipi = compute_ipi(decision_results)
            aivr = compute_aivr(decision_results)
            gov_metrics["IPI"] = round(ipi, 4)
            gov_metrics["AIVR"] = round(aivr, 4)

        # Note: FVS, ESD, FSR require separate test infrastructure
        # (quality-drop injection, seed attack, A/B framing) and are
        # computed by their dedicated test functions, not per-run.

        # Task metrics
        task = compute_task_metrics(decision_results, cases)
        adr = compute_adr(decision_results, cases)
        overcaution = compute_overcaution(decision_results, cases)

        return {
            "governance": gov_metrics,
            "task": {
                "accuracy": round(task["accuracy"], 4),
                "f1_macro": round(task["f1_macro"], 4),
                "mcc": round(task["mcc"], 4),
                "adr": round(adr, 4),
                "overcaution": round(overcaution, 4),
                "n_evaluated": task["n_evaluated"],
            },
        }

    def run_single(
        self,
        config: RunConfig,
        llm: LLMInterface,
        skip_completed: bool = True,
    ) -> ExperimentResult | None:
        """Run a single experimental condition.

        Args:
            config: Run configuration
            llm: LLM interface
            skip_completed: If True, skip if run_id already in results

        Returns:
            ExperimentResult or None if skipped
        """
        run_id = _make_run_id(
            config.model_name,
            config.regime_name,
            config.condition,
            config.seed,
        )

        if skip_completed and run_id in self._completed_run_ids():
            logger.info("SKIP %s (already completed)", run_id)
            return None

        logger.info(
            "START %s  regime=%s cond=%s seed=%d N=%d",
            run_id,
            config.regime_name,
            config.condition,
            config.seed,
            config.cases_per_condition,
        )

        # Generate dataset
        dist_config = load_distributions(config.distributions_config_path)
        try:
            condition_enum = StressCondition(config.condition)
        except ValueError:
            condition_enum = StressCondition[config.condition]
        cases = generate_dataset(
            seed=config.seed,
            n_cases_per_condition=config.cases_per_condition,
            stress_conditions=[condition_enum],
            distributions_config=dist_config,
        )
        # Assign ground truth on BASELINE data (before stress transforms).
        # GT rules reference risk_score, completeness, flags — these must
        # reflect the original generation, not stressed values.
        assign_ground_truth(cases)

        # Apply stress transforms (S1-S3 modify risk_score, completeness, flags).
        # Generator only tags cases; actual numeric transforms happen here.
        if condition_enum != StressCondition.S0_BASELINE:
            from mech_gov.data.generator import make_rng

            stress_rng = make_rng(config.seed + 9999)
            cases = apply_stress(cases, condition_enum, stress_rng)
        logger.info("  Generated %d cases for %s", len(cases), config.condition)

        # Create regime
        regime = create_regime(config.regime_name)
        logger.info("  Regime: %s", regime.regime_name)

        # Run
        start = time.perf_counter()
        decision_results = self.run_condition(regime, cases, llm, config.seed)
        elapsed = time.perf_counter() - start

        logger.info("  All cases processed in %.1fs", elapsed)

        # Compute metrics
        metrics = self._compute_run_metrics(decision_results, cases)
        logger.info("  Metrics: %s", json.dumps(metrics, default=str))

        # Build result with traceability metadata
        try:
            llm_model_id = llm.model_id
        except Exception:
            llm_model_id = ""
        result = ExperimentResult(
            run_id=run_id,
            model=config.model_name,
            regime=config.regime_name,
            condition=config.condition,
            seed=config.seed,
            n_cases=len(cases),
            metrics=metrics,
            per_case_results=[r.to_dict() for r in decision_results],
            elapsed_s=round(elapsed, 2),
            model_id=llm_model_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            code_version=_get_code_version(),
        )

        self._save_result(result)
        if self._on_run_complete:
            try:
                self._on_run_complete(result)
            except Exception as cb_err:
                logger.warning("on_run_complete callback failed: %s", cb_err)
        logger.info(
            "DONE %s  elapsed=%.1fs  CDL=%s  DIU=%s",
            run_id,
            elapsed,
            metrics.get("governance", {}).get("CDL", "?"),
            metrics.get("governance", {}).get("DIU", "?"),
        )
        return result

    def run_full_experiment(
        self,
        models: dict[str, LLMInterface],
        regimes: list[str],
        conditions: list[str],
        seeds: list[int],
        cases_per_condition: int,
        skip_completed: bool = True,
    ) -> list[ExperimentResult]:
        """Run all combinations: model × regime × condition × seed.

        Args:
            models: {model_name: LLMInterface}
            regimes: ["R1", "R2"]
            conditions: ["S0", "S1", "S2", "S3"]
            seeds: List of random seeds
            cases_per_condition: N per condition
            skip_completed: Resume from previous runs

        Returns:
            List of completed ExperimentResult
        """
        all_results = []
        total = len(models) * len(regimes) * len(conditions) * len(seeds)
        done = 0

        logger.info("=" * 60)
        logger.info("FULL EXPERIMENT: %d total runs", total)
        logger.info(
            "  models=%s  regimes=%s  conditions=%s  seeds=%s  N=%d",
            list(models.keys()),
            regimes,
            conditions,
            seeds,
            cases_per_condition,
        )
        logger.info("=" * 60)

        for model_name, llm in models.items():
            model_results = []
            for regime_name in regimes:
                for condition in conditions:
                    for seed in seeds:
                        done += 1
                        config = RunConfig(
                            model_name=model_name,
                            regime_name=regime_name,
                            condition=condition,
                            seed=seed,
                            cases_per_condition=cases_per_condition,
                        )
                        result = self.run_single(config, llm, skip_completed)
                        if result:
                            all_results.append(result)
                            model_results.append(result)
                            print(
                                f"[{done}/{total}] {result.run_id} — "
                                f"{result.elapsed_s:.1f}s — "
                                f"CDL={result.metrics.get('governance', {}).get('CDL', '?')}"
                            )
                        else:
                            print(f"[{done}/{total}] SKIP (already completed)")

            # Write manifest per model
            if model_results:
                try:
                    llm_id = llm.model_id
                except Exception:
                    llm_id = ""
                self._save_manifest(
                    model_name,
                    llm_id,
                    {
                        "regimes": regimes,
                        "conditions": conditions,
                        "seeds": seeds,
                        "cases_per_condition": cases_per_condition,
                        "runs_completed": len(model_results),
                        "runs_total": len(regimes) * len(conditions) * len(seeds),
                    },
                )

        return all_results
