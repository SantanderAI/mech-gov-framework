# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Backwards-compatible factory API.

This module is a thin shim over :mod:`mech_gov.llm.registry`, kept so that code
written against the older ``create_llm`` / ``create_all_llms`` /
``load_models_config`` API keeps working. New code should prefer the registry
directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mech_gov.llm.base import LLMInterface
from mech_gov.llm.registry import available_providers, load_models_config, register_provider
from mech_gov.llm.registry import create_all_llms as _create_all_llms
from mech_gov.llm.registry import create_llm as _create_llm

__all__ = [
    "create_llm",
    "create_all_llms",
    "load_models_config",
    "register_provider",
    "available_providers",
]


def create_llm(
    model_config: dict[str, Any],
    profile_name: str | None = None,
) -> LLMInterface:
    """Create an LLM client from a model config dict (see the registry)."""
    return _create_llm(model_config, profile_name=profile_name)


def create_all_llms(
    config_path: str | Path,
    model_names: list[str] | None = None,
    profile_name: str | None = None,
) -> dict[str, LLMInterface]:
    """Create every LLM client defined in a models config file."""
    return _create_all_llms(config_path, model_names=model_names, profile_name=profile_name)
