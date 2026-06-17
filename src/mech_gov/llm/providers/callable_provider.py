# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Bring-your-own-LLM provider.

Wrap ANY callable as a vendor-neutral LLM client. This is the recommended way
to plug in a proprietary or internal inference backend: the framework never
needs to know — or reveal — which backend you use.

Example:
    def my_backend(system_prompt, user_message, temperature=0.0, max_tokens=2048):
        # call your own SDK / gateway / local model here
        return '{"decision": "APPROVE", "rationale": "...", ...}'

    from mech_gov.llm.providers.callable_provider import CallableLLM
    llm = CallableLLM(my_backend)
"""

from __future__ import annotations

import time
from collections.abc import Callable

from mech_gov.llm.base import LLMInterface, LLMResponse


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


class CallableLLM(LLMInterface):
    """Adapt a user-supplied function to the :class:`LLMInterface`.

    The function may accept either the keyword signature
    ``(system_prompt, user_message, temperature, max_tokens)`` or a simple
    ``(system_prompt, user_message)`` positional signature. It must return
    either a plain string (the model output) or a fully-formed
    :class:`LLMResponse`.
    """

    def __init__(
        self,
        fn: Callable[..., str | LLMResponse],
        model_id: str = "callable",
    ):
        if not callable(fn):
            raise TypeError("CallableLLM requires a callable `fn`.")
        self._fn = fn
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "callable"

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        start_ms = time.perf_counter() * 1000
        try:
            out = self._fn(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            # Fall back to a simple two-argument positional signature.
            out = self._fn(system_prompt, user_message)
        latency_ms = time.perf_counter() * 1000 - start_ms

        if isinstance(out, LLMResponse):
            return out

        content = str(out)
        return LLMResponse(
            content=content,
            model_id=self._model_id,
            input_tokens=_estimate_tokens(system_prompt) + _estimate_tokens(user_message),
            output_tokens=_estimate_tokens(content),
            latency_ms=latency_ms,
            stop_reason="stop",
        )


def build(config: dict) -> CallableLLM:
    """Build a :class:`CallableLLM` from a config dict (registry entry point)."""
    fn = config.get("callable")
    if fn is None:
        raise ValueError(
            "The 'callable' provider requires a 'callable' entry in the config: "
            "a function (system_prompt, user_message, temperature, max_tokens) "
            "-> str | LLMResponse."
        )
    return CallableLLM(fn=fn, model_id=config.get("model_id", "callable"))
