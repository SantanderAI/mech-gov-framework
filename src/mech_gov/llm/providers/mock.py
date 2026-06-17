# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Offline mock LLM provider — deterministic, no network, no credentials.

Useful for tests, examples, and CI. Returns a canned JSON decision in the
schema the governance regimes expect, or delegates to a user ``responder``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from mech_gov.llm.base import LLMInterface, LLMResponse

# A valid decision payload in the schema R1/R2 parse (decision + rationale +
# pro/con arguments). Arguments are long enough to pass the default I6Q check.
_DEFAULT_RESPONSE = json.dumps(
    {
        "decision": "ESCALATE",
        "rationale": "Deterministic mock response used for offline testing and demos.",
        "pro_arguments": [
            "The transaction shows characteristics that could justify approval "
            "under normal review conditions.",
        ],
        "con_arguments": [
            "Insufficient verified information is available to fully rule out "
            "elevated regulatory risk here.",
        ],
    }
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


class MockLLM(LLMInterface):
    """Deterministic in-process LLM for tests and demos.

    Args:
        response: Fixed raw string returned for every invoke. Defaults to a
            valid JSON decision payload.
        responses: Optional list of raw strings cycled across calls. Overrides
            ``response`` when provided.
        responder: Optional callable ``(system_prompt, user_message) -> str``.
            Overrides both other modes when provided.
        model_id: Identifier reported by the client.
    """

    def __init__(
        self,
        response: str | None = None,
        responses: list[str] | None = None,
        responder: Callable[[str, str], str] | None = None,
        model_id: str = "mock",
    ):
        self._response = response if response is not None else _DEFAULT_RESPONSE
        self._responses = list(responses) if responses else None
        self._responder = responder
        self._model_id = model_id
        self._call_index = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "mock"

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        start_ms = time.perf_counter() * 1000
        if self._responder is not None:
            content = self._responder(system_prompt, user_message)
        elif self._responses:
            content = self._responses[self._call_index % len(self._responses)]
            self._call_index += 1
        else:
            content = self._response
        latency_ms = time.perf_counter() * 1000 - start_ms

        return LLMResponse(
            content=content,
            model_id=self._model_id,
            input_tokens=_estimate_tokens(system_prompt) + _estimate_tokens(user_message),
            output_tokens=_estimate_tokens(content),
            latency_ms=latency_ms,
            stop_reason="stop",
        )


def build(config: dict) -> MockLLM:
    """Build a :class:`MockLLM` from a config dict (registry entry point)."""
    return MockLLM(
        response=config.get("response"),
        responses=config.get("responses"),
        responder=config.get("responder"),
        model_id=config.get("model_id", "mock"),
    )
