# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Abstract LLM interface for mech_gov v2.

Defines the contract that all LLM providers (Bedrock Claude, SageMaker Llama)
must implement. Matches the design spec §3.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str


class LLMInterface(ABC):
    """Abstract interface for LLM providers.

    All governance regimes interact with LLMs exclusively through this
    interface, ensuring model-agnostic evaluation.
    """

    @abstractmethod
    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return a structured response.

        Args:
            system_prompt: System-level instructions (governance policy)
            user_message: User-level content (case description)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content, usage stats, and latency
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier string."""
        ...

    @property
    def provider(self) -> str:
        """Return the provider name (e.g. 'bedrock', 'sagemaker')."""
        return "unknown"
