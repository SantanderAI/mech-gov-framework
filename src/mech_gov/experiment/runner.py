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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mech_gov
from mech_gov.data.banking_case import (
    BankingCase,
    Decision,
    StressCondition,
)
from mech_gov.data.distributions import load_distributions
from mech_gov.data.generator import generate_dataset
from mech_gov.data.ground_truth import assign_ground_truth
from mech_gov.data.stress import apply_stress
from mech_gov.experiment.identity import (
    IdentityWarnings,
    RunIdentity,
    build_run_identity,
    canonical_bytes,
    changed_fields,
    verified_stored_digest,
)
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
    run_label: str = ""
    identity_sha256: str = ""
    identity: dict[str, Any] = field(default_factory=dict)

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
        if self.run_label:
            d["run_label"] = self.run_label
        if self.identity and self.identity_sha256:
            d["identity_schema"] = self.identity.get("identity_schema")
            d["identity_sha256"] = self.identity_sha256
            d["identity"] = self.identity
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
    """Build the human-readable run label.

    This string is intentionally coarse: it does **not** decide whether a run
    has already completed. See :mod:`mech_gov.experiment.identity`.
    """
    return f"{model}-{regime}-{condition}-seed{seed}"


def _package_version() -> str:
    """Package version used as code provenance inside run identity."""
    return getattr(mech_gov, "__version__", "") or ""


@dataclass
class _CompletionIndex:
    """Verified identities and legacy labels found in stored result files."""

    digests: set[str] = field(default_factory=set)
    canonical: dict[str, bytes] = field(default_factory=dict)
    conflicts: set[str] = field(default_factory=set)
    payloads_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    legacy_labels: set[str] = field(default_factory=set)
    invalid_labels: set[str] = field(default_factory=set)
    scan_failed: bool = False

    def add(self, record: Mapping[str, Any], warnings: IdentityWarnings) -> None:
        """Index one stored record, warning instead of trusting bad identities."""
        label = record.get("run_label") or record.get("run_id") or ""
        if not isinstance(label, str) or not label:
            return
        digest, code = verified_stored_digest(record)
        if code is not None:
            self.invalid_labels.add(label)
            warnings.emit(code, label)
            return
        if digest is None:
            self.legacy_labels.add(label)
            return
        blob = canonical_bytes(record["identity"])
        if digest in self.digests:
            # Defensive: verified digests bind their payload, so a mismatch here
            # means validation was bypassed. Duplicate copies are not conflicts.
            if self.canonical[digest] != blob:
                self.conflicts.add(digest)
                warnings.emit("MGI005", label)
            return
        self.digests.add(digest)
        self.canonical[digest] = blob
        self.payloads_by_label.setdefault(label, []).append(dict(record["identity"]))


def _result_files(results_dir: Path) -> Iterator[Path]:
    """Yield the legacy flat JSONL files and the per-model ones."""
    yield from results_dir.glob("*.jsonl")
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            yield from subdir.glob("*.jsonl")


def _scan_completion_index(results_dir: Path, warnings: IdentityWarnings) -> _CompletionIndex:
    """Build the completion index, tolerating malformed JSONL lines."""
    index = _CompletionIndex()
    try:
        paths = list(_result_files(results_dir))
    except OSError:
        index.scan_failed = True
        warnings.emit("MGI008", "<results-dir>")
        return index
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, RecursionError):
                        continue
                    if not isinstance(record, Mapping):
                        continue
                    try:
                        index.add(record, warnings)
                    except Exception:
                        label = record.get("run_label") or record.get("run_id")
                        if isinstance(label, str) and label:
                            index.invalid_labels.add(label)
                            warnings.emit("MGI004", label)
                        else:
                            index.scan_failed = True
                            warnings.emit("MGI004", str(path))
        except (OSError, UnicodeError):
            index.scan_failed = True
            warnings.emit("MGI008", str(path))
    return index


def _resolve_identity(
    config: RunConfig,
    llm: LLMInterface,
    label: str,
    dist_config: Mapping[str, Any],
    warnings: IdentityWarnings,
    warn_non_resumable: bool = True,
) -> RunIdentity | None:
    """Build the versioned identity, or return None when the run is non-resumable."""
    try:
        model_id = llm.model_id
        getter = getattr(llm, "resume_identity_parameters", None)
        parameters = getter() if callable(getter) else None
    except Exception:
        model_id = ""
        parameters = None
    if not isinstance(model_id, str) or not model_id or not isinstance(parameters, Mapping):
        if warn_non_resumable:
            warnings.emit("MGI006", label)
        return None
    version = _package_version()
    if not version:
        if warn_non_resumable:
            warnings.emit("MGI007", label)
        return None
    try:
        return build_run_identity(
            label=label,
            model_name=config.model_name,
            model_id=model_id,
            regime_name=config.regime_name,
            condition=config.condition,
            seed=config.seed,
            cases_per_condition=config.cases_per_condition,
            distributions_config=dist_config,
            provider_parameters=parameters,
            package_version=version,
        )
    except Exception:
        if warn_non_resumable:
            warnings.emit("MGI006", label)
        return None


def _decide_skip(
    identity: RunIdentity | None,
    label: str,
    index: _CompletionIndex,
    warnings: IdentityWarnings,
) -> bool:
    """Decide whether a requested run may be treated as already completed."""
    if identity is None:
        return False
    if index.scan_failed or label in index.invalid_labels:
        return False
    if identity.digest in index.conflicts:
        warnings.emit("MGI005", label)
        return False
    if identity.digest in index.digests:
        return True
    stored = index.payloads_by_label.get(label)
    if stored:
        differing = {name for prior in stored for name in changed_fields(identity.payload, prior)}
        warnings.emit("MGI002", label, sorted(differing))
    elif label in index.legacy_labels:
        warnings.emit("MGI001", label)
    return False


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

    def _completion_index(self, warnings: IdentityWarnings) -> _CompletionIndex:
        """Scan results dir (and model subdirs) for verified completed identities."""
        return _scan_completion_index(self._results_dir, warnings)

    def _prepare_run(
        self,
        config: RunConfig,
        llm: LLMInterface,
        label: str,
        skip_completed: bool,
    ) -> tuple[dict[str, Any], RunIdentity | None, bool]:
        """Resolve identity inputs before the completion check and decide on skipping."""
        dist_config = load_distributions(config.distributions_config_path)
        warnings = IdentityWarnings()
        identity = _resolve_identity(
            config,
            llm,
            label,
            dist_config,
            warnings,
            warn_non_resumable=skip_completed,
        )
        skip = skip_completed and _decide_skip(
            identity, label, self._completion_index(warnings), warnings
        )
        return dist_config, identity, skip

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
            skip_completed: If True, skip only when a stored, verified machine
                identity exactly matches the current run. Legacy run labels do
                not establish completion.

        Returns:
            ExperimentResult or None if skipped
        """
        label = _make_run_id(
            config.model_name,
            config.regime_name,
            config.condition,
            config.seed,
        )

        dist_config, identity, skip = self._prepare_run(config, llm, label, skip_completed)
        run_id = label
        if skip:
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
            run_label=label,
            identity_sha256=identity.digest if identity is not None else "",
            identity=dict(identity.payload) if identity is not None else {},
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
