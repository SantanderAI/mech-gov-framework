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

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Open source by Santander AI Lab.
