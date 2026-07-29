# mech_gov — Mechanical Governance for LLM Decisions

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/SantanderAI/mech-gov-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/SantanderAI/mech-gov-framework/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SantanderAI/mech-gov-framework/actions/workflows/codeql.yml/badge.svg)](https://github.com/SantanderAI/mech-gov-framework/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://github.com/SantanderAI/mech-gov-framework/actions/workflows/scorecard.yml/badge.svg)](https://github.com/SantanderAI/mech-gov-framework/actions/workflows/scorecard.yml)
[![codecov](https://codecov.io/gh/SantanderAI/mech-gov-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/SantanderAI/mech-gov-framework)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://www.conventionalcommits.org)

A model-agnostic Python **framework** for enforcing and measuring governance on
LLM decisions in high-stakes settings — mechanical gates, governance metrics and
a synthetic decision dataset.

Open source by **Santander AI Lab**. It contrasts a text-only governance regime
(**R1**) with **mechanical enforcement** (**R2**) — hard gates, candidate
freezing, argument-quality checks, an ambiguity gate, and a commit–reveal
entropy step — plus an adaptive regime (**R3**).

> **Vendor-neutral by design.** Nothing in the core depends on a specific cloud
> or model provider. Bring your own LLM backend via a small adapter; the
> framework never needs to know which one you use.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e .
# optional extras:
#   pip install -e ".[dev]"      # tests
#   pip install -e ".[viz]"      # plotting helpers
#   pip install -e ".[bedrock]"  # AWS Bedrock/SageMaker backends
```

Requires Python 3.10+.

## Quickstart (offline, no credentials)

```python
from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.llm.registry import create_llm

llm = create_llm({"provider": "mock"})  # deterministic, offline
case = BankingCase(
    case_id="demo-1",
    transaction_type=TransactionType.CREDIT_APPROVAL,
    risk_score=0.62, completeness=0.55, regulatory_flags=["KYC"],
)
result = R2Mechanical().process_case(case, llm)
print(result.decision.value, "|", result.gates_triggered)
```

Or run the bundled examples / CLI:

```bash
python examples/quickstart_mock.py
python scripts/run_governance.py --regime R2 --provider mock --n 20
```

## Bring your own LLM

The only contract is `mech_gov.llm.base.LLMInterface.invoke(...)`. Three
dependency-free ways to supply a backend:

**1. Wrap any function (`callable`)** — the recommended way to use a proprietary
or internal backend:

```python
from mech_gov.llm.registry import create_llm

def my_backend(system_prompt, user_message, temperature=0.0, max_tokens=2048):
    # call your own SDK / gateway / local model; return the raw text
    ...

llm = create_llm({"provider": "callable", "callable": my_backend})
```

**2. Any OpenAI-compatible HTTP endpoint (`openai_compatible`)** — OpenAI, Azure
OpenAI, vLLM, Ollama, Together, LM Studio, or an internal gateway. Uses only the
standard library:

```bash
export MECH_GOV_LLM_BASE_URL=http://localhost:11434/v1
export MECH_GOV_LLM_MODEL=llama3.1
# export MECH_GOV_LLM_API_KEY=...    # if your endpoint needs one
```
```python
llm = create_llm({"provider": "openai_compatible"})
```

**3. Optional cloud backends (`bedrock`, `sagemaker`)** — only available after
`pip install -e ".[bedrock]"`. The core install never imports a cloud SDK.

To add your own provider, implement `LLMInterface`, expose a
`build(config) -> LLMInterface`, and register it in
`mech_gov.llm.registry` (see `CONTRIBUTING.md`).

## Governance regimes

| Regime | Module | Behaviour |
| --- | --- | --- |
| **R1** | `mech_gov.governance.r1_text_only` | Text-only: the LLM interprets the policy with no mechanical enforcement. |
| **R2** | `mech_gov.governance.r2_mechanical` | Mechanical pipeline: hard gates → entropy commit → candidate freezing → argument-quality (I6Q) → ambiguity gate → reveal. |
| **R3** | `mech_gov.governance.r3_adaptive` | Adaptive/exploratory regime. |

All regimes implement `process_case(case, llm, entropy_seed=None) -> DecisionResult`.

## Metrics

`mech_gov.metrics.governance` provides **CDL** (cosmetic-deadlock rate),
**DIU** (deferral information utilisation), **FVS**, **ESD**, **FSR**, and
**IPI**; `mech_gov.metrics.task` provides accuracy, macro-F1, MCC, and
deferral-rate metrics.

## Project layout

```
mech_gov_framework/
├── pyproject.toml          # packaging; boto3 is an optional [bedrock] extra
├── README.md  LICENSE  CONTRIBUTING.md
├── src/mech_gov/           # the importable package (vendor-neutral core)
│   ├── llm/                # base interface, registry, providers/
│   ├── governance/         # R1, R2, R3, primitives, policy templates
│   ├── metrics/            # governance + task metrics
│   ├── data/               # synthetic banking dataset + bundled config
│   └── experiment/         # runner, ablation, framing/FVS/seed tests
├── scripts/                # generate_dataset.py, run_governance.py
├── examples/               # quickstart_mock.py, custom_provider.py
├── configs/                # models.example.yaml
└── tests/                  # offline tests (mock provider)
```

## Experiment resume and run identity

`ExperimentRunner` is resumable: re-running an experiment skips work that is
already on disk. Each result therefore carries two different things.

| Field | Purpose |
|---|---|
| `run_label` | Readable name, e.g. `mock-R2-S0-seed42`. Never decides resume. |
| `identity_sha256` | Full SHA-256 over the canonical identity payload. **Decides resume.** |
| `identity` | The canonical payload itself, so the digest is reproducible. |
| `identity_schema` | Version of the canonical identity contract. |
| `run_id` | The previous readable identifier, unchanged for compatibility. |

A run is skipped **only** when a stored record carries the exact same verified
`identity_sha256`. The identity covers the case count, regime, condition, seed,
a digest of the resolved model ID, a digest of the *loaded* distributions
configuration, the provider's declared adapter/deployment fingerprint and
generation parameters, and the package version. Changing any of them re-runs
the experiment instead of silently reusing an older result.

`run_label` is present on every new result. `identity_schema`,
`identity_sha256`, and `identity` are present only when all required identity
producers are available; a non-resumable result retains the legacy `run_id`
without claiming a machine identity.

The distributions fingerprint is taken from the parsed configuration, not the
file bytes, so reformatting or editing comments does not invalidate previous
runs, while a changed parameter does. Equivalent content at a different path
fingerprints identically.

Canonical numbers are type-tagged and derived exactly from integers or the
round-trip representation of finite floats. Decimal normalization uses explicit
precision equal to the operand's digit count, so a caller's mutable
`decimal` context cannot round two configurations into one identity.
Non-finite floats are rejected.

### Non-resumable runs

Some runs cannot prove equivalence. They always execute, and log why:

- the provider cannot expose a stable, non-empty `model_id`;
- the provider cannot declare its adapter/deployment identity and generation
  parameters (see below);
- the package version is unavailable.

Failing open is deliberate: recomputing a run costs time, whereas wrongly
skipping one silently corrupts a result set.

Code provenance is bounded to the package version. Source edits made without a
version change are intentionally outside this identity contract; use
`skip_completed=False` while iterating on an editable checkout.

### Declaring provider parameters

`LLMInterface.resume_identity_parameters()` returns an allowlisted adapter
identifier, privacy-safe deployment fingerprint, and generation-affecting
values, or `None`. The base implementation returns `None`, so existing custom
providers remain API-compatible but always execute until they declare a stable
identity. The bundled callable and SageMaker providers are non-resumable:
arbitrary callables are opaque, and an endpoint name plus region does not prove
which account-scoped SageMaker deployment is selected.

The bundled OpenAI-compatible and Bedrock adapters currently expose exactly one
constructor-configured generation parameter, `top_p`; both declare it alongside
their privacy-safe target fingerprint. Per-call `temperature` and `max_tokens`
come from governance-regime code and are therefore covered by the package
version. Connection, credential/profile, timeout, and retry settings affect
transport behavior rather than the successful response contract and are
deliberately excluded.

Return only public values and privacy-safe fingerprints — never an API key,
credential, raw private URL, profile name, endpoint, timeout, or path.

```python
class MyProvider(LLMInterface):
    def resume_identity_parameters(self):
        return {
            "provider": "my_provider",
            "deployment_sha256": self._safe_deployment_fingerprint,
            "top_p": self._top_p,
        }
```

SHA-256 fingerprints prevent raw identifiers from appearing in the identity
payload; they are pseudonymous identifiers, not encryption. Use non-secret,
stable model and deployment identifiers. For backward compatibility, the
existing raw `model_id` fields in results and manifests are unchanged.
The readable model alias remains in `run_id`, `run_label`, and `identity` as
public metadata; do not put credentials or private identifiers in aliases.

### Legacy result files

Records written before versioned identity carry a `run_id` and nothing else.
They are never treated as equivalent to a versioned identity: a matching
readable label logs `MGI001` and the run executes again. Existing files are read
but never rewritten or deleted. The first post-upgrade sweep can therefore
repeat legacy provider calls and incur their normal cost.

Identity and resume warnings use the `mech_gov.experiment.identity` logger with
stable codes `MGI001`–`MGI008`. They name the configuration fields that differ
and include a 12-hex SHA-256 reference for correlation, but never print the raw
run label, result path, or configuration values. `MGI008` distinguishes
unreadable completion state from a stored payload that failed validation.

Distribution configuration is loaded before any resume decision because its
parsed content is part of identity. A missing or invalid distributions file
therefore raises even when an older readable label exists; silently skipping
without validating an identity input would be unsafe.

## CLI

```bash
# Generate the synthetic banking dataset to JSONL
python scripts/generate_dataset.py --n 100 --seed 42 --out dataset.jsonl

# Run a regime and print metrics (uses the offline mock by default)
python scripts/run_governance.py --regime R2 --provider mock --n 50

# Use a configured backend
python scripts/run_governance.py --regime R2 \
    --models-config configs/models.example.yaml --model local --n 50
```

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
issue/PR workflow and the Contributor License Agreement (CLA). Please also read
our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). To report a vulnerability, follow
[`SECURITY.md`](SECURITY.md).

## Citation

If you use mech_gov in your research, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{mech_gov_2026,
  title        = {mech\_gov: Mechanical Governance for LLM Decisions},
  author       = {{Santander AI Lab}},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/SantanderAI/mech-gov-framework},
  license      = {Apache-2.0}
}
```

## Disclaimer

This software is an open source project from the **Santander AI Lab**, provided **"as is"** under its [license](LICENSE), without warranties or conditions of any kind. It is **not an official Banco Santander product or service**, carries no commitment of production support, and does not constitute financial, legal or professional advice.

"Santander" and its logo are registered trademarks of **Banco Santander, S.A.** The project license does not grant any right to use them beyond factual attribution.

If you believe you have found a security vulnerability, follow our [security policy](https://github.com/SantanderAI/.github/blob/main/SECURITY.md) — do not open a public issue. You are responsible for assessing the suitability of this software for your use case and for keeping your own deployments up to date.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Open source by Santander AI Lab.
