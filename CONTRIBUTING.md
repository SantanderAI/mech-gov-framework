# Contributing to mech_gov

Thanks for your interest in improving the framework. This guide covers how to
report issues, submit changes, and the license your contributions fall under.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Contributor License Agreement (CLA)

Before we can merge your contribution, you must sign the Santander Open Source
**Contributor License Agreement**. The first time you open a pull request, the
**CLA Assistant** bot will comment with a link and instructions; signing is a
one-time action that covers all your future contributions to this repository.
All contributions are made under the project's [Apache License 2.0](LICENSE).

## Reporting issues

- Search the [existing issues](../../issues) first to avoid duplicates.
- Open a **Bug report** or **Feature request** using the issue templates.
- For **security vulnerabilities**, do **not** open a public issue — follow
  [SECURITY.md](SECURITY.md) (private email or GitHub Security Advisory).
- Never include secrets, API keys, internal URLs, or proprietary content.

## Submitting a pull request

1. Fork the repository and create a topic branch from `main`.
2. Make your change in a focused commit set; follow
   [Conventional Commits](https://www.conventionalcommits.org/) for messages.
3. Run the checks locally (`ruff check .`, `black --check .`, `mypy src`,
   `pytest`) and add or update tests for new behaviour.
4. Open a PR against `main`, fill in the PR template, and link any related
   issue (e.g. `Closes #123`).
5. Ensure the CLA check is green and CI passes. A maintainer
   (`@SantanderAI/mech-gov-framework-maintainers`) will review.

## Development setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

The test suite uses the offline `mock` LLM provider, so it runs with no network
access and no credentials.

## Guidelines

- **Keep the core vendor-neutral.** Nothing under `src/mech_gov/` (outside
  `llm/providers/aws.py` and the `bedrock_*` / `sagemaker_*` modules) may import
  a cloud SDK. New backends belong in `src/mech_gov/llm/providers/` and must be
  registered in `src/mech_gov/llm/registry.py`.
- **No secrets in the repo.** Pass credentials via environment variables.
- **Add tests** for new behaviour and keep them runnable offline.
- **Style:** follow the existing code style; keep public APIs documented.

## Adding a new LLM provider

1. Implement `mech_gov.llm.base.LLMInterface` in a new module under
   `src/mech_gov/llm/providers/`.
2. Expose a `build(config: dict) -> LLMInterface` entry point.
3. Register it in `registry.py` (eagerly if dependency-free, lazily if it needs
   an optional dependency).
4. Add a test using a fake/stubbed transport where possible.
