# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source readiness scaffolding:
  - Apache 2.0 `NOTICE`, expanded `CONTRIBUTING.md` (CLA + issue/PR flow),
    `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CODEOWNERS`
  - `CITATION.cff` and a README citation block
  - Issue templates (bug, feature) and PR template
  - `pyproject.toml` tooling config (ruff, black, mypy, pytest, coverage) and
    real project URLs
  - SPDX headers on Python sources, scripts, examples and tests
  - GitHub Actions workflows (third-party actions pinned to SHA digests):
    - `ci.yml` — ruff + black + mypy + pytest matrix (3.10/3.11/3.12) with Codecov
    - `codeql.yml` — CodeQL SAST (push, PR, weekly cron)
    - `dep-scan.yml` — `pip-audit` (push, PR, daily cron)
    - `license-check.yml` — SPDX header verification + dependency-license allowlist (`pip-licenses`)
    - `pattern-check.yml` — internal-pattern scan with allowlist
    - `scorecard.yml` — OpenSSF Scorecard supply-chain analysis
    - `cla.yml` — CLA Assistant Lite
    - `stale.yml` — stale issues/PRs automation
    - `release.yml` — versioned source archive attached to GitHub Releases
  - `.github/dependabot.yml` — monthly Python and GitHub Actions updates
  - README badges, attribution and Citation sections
- Privacy gate (R2): a pre-LLM governance primitive
  (`mech_gov.governance.primitives.privacy_gate`) that reversibly tokenizes
  direct identifiers (EMAIL, PHONE, SSN, PAN, IBAN, IP) before the model is
  consulted, and mechanically DEFERs a case when residual identifiers exceed a
  configurable budget or detection fails (fail-closed). Stdlib-only, vendor-
  neutral, configurable via `PrivacyConfig`; records `privacy_entities_found`
  and `privacy_residual_pii` counts in `DecisionResult.metadata` (the reversible
  token map is never persisted). Supports a pluggable `PiiRecognizer`. Ships an
  offline `examples/privacy_demo.py`.

## [0.1.0] - 2026-06-12

### Added
- `mech_gov` framework: model-agnostic governance for LLM decisions in
  high-stakes settings
- Governance regimes: `R1` (text-only), `R2` (mechanical enforcement — hard
  gates, candidate freezing, argument-quality / I6Q checks, ambiguity gate,
  commit–reveal entropy step) and `R3` (adaptive)
- Vendor-neutral LLM interface (`mech_gov.llm.base.LLMInterface`) with a
  registry and `mock`, `callable`, `openai_compatible` providers plus optional
  `bedrock`/`sagemaker` backends behind an extra
- Governance metrics (CDL, DIU, FVS, ESD, FSR, IPI) and task metrics
  (accuracy, macro-F1, MCC, deferral rate)
- Synthetic banking decision dataset generator and an experiment runner with
  ablation, framing/FVS and seed tests
- CLI scripts `generate_dataset.py` and `run_governance.py`, plus offline
  examples and a mock-backed regression test suite

[Unreleased]: https://github.com/SantanderAI/mech-gov-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SantanderAI/mech-gov-framework/releases/tag/v0.1.0
