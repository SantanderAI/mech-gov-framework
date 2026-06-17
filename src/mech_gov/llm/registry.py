# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Provider registry for mech_gov LLM clients.

Maps a provider name (from config) to a builder function. Dependency-free
providers (``mock``, ``callable``, ``openai_compatible``) are registered
eagerly; cloud providers that need optional dependencies (``bedrock``,
``sagemaker``) are registered with lazy builders, so importing this module
never pulls in a cloud SDK.

Usage:
    from mech_gov.llm.registry import create_llm, load_models_config

    models = load_models_config("configs/models.example.yaml")
    llm = create_llm(models["local"])
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from mech_gov.llm.base import LLMInterface

logger = logging.getLogger("mech_gov.llm.registry")

ProviderBuilder = Callable[[dict[str, Any]], LLMInterface]

_PROVIDERS: dict[str, ProviderBuilder] = {}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Register (or override) a provider builder under ``name``."""
    _PROVIDERS[name] = builder


def available_providers() -> list[str]:
    """Return the sorted list of registered provider names."""
    return sorted(_PROVIDERS)


def create_llm(model_config: dict[str, Any], **overrides: Any) -> LLMInterface:
    """Create an LLM client from a model config dict.

    Args:
        model_config: A dict with at least a ``provider`` key plus
            provider-specific fields.
        **overrides: Optional values that override config entries when not None
            (e.g. ``profile_name=...``).

    Returns:
        An :class:`LLMInterface` implementation.
    """
    cfg = dict(model_config)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    provider = cfg.get("provider")
    if not provider:
        raise ValueError("model config must include a 'provider' key.")

    builder = _PROVIDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Registered: {available_providers()}. "
            "AWS Bedrock/SageMaker require: pip install mech-gov-framework[bedrock]"
        )

    logger.info("Creating LLM via provider '%s'", provider)
    return builder(cfg)


def load_models_config(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a models config YAML file.

    Accepts either a top-level ``models:`` mapping or a flat mapping of
    ``name -> config``.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Models config not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("models", raw)


def create_all_llms(
    config_path: str | Path,
    model_names: list[str] | None = None,
    **overrides: Any,
) -> dict[str, LLMInterface]:
    """Create every LLM client defined in a models config file."""
    models_cfg = load_models_config(config_path)
    result: dict[str, LLMInterface] = {}
    for name, entry in models_cfg.items():
        if model_names and name not in model_names:
            continue
        result[name] = create_llm(entry, **overrides)
    return result


def _register_builtins() -> None:
    # Dependency-free providers (registered eagerly).
    from mech_gov.llm.providers import callable_provider, mock, openai_compatible

    register_provider("mock", mock.build)
    register_provider("callable", callable_provider.build)
    register_provider("openai", openai_compatible.build)
    register_provider("openai_compatible", openai_compatible.build)

    # Optional cloud providers — lazy: boto3 is only imported if actually used.
    def _bedrock(cfg: dict[str, Any]) -> LLMInterface:
        from mech_gov.llm.providers.bedrock import build

        return build(cfg)

    def _sagemaker(cfg: dict[str, Any]) -> LLMInterface:
        from mech_gov.llm.providers.sagemaker import build

        return build(cfg)

    register_provider("bedrock", _bedrock)
    register_provider("sagemaker", _sagemaker)


_register_builtins()
