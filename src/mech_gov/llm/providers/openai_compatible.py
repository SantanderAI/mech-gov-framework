# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Generic OpenAI-compatible HTTP provider (vendor-neutral).

Talks to any endpoint implementing the OpenAI Chat Completions API:
OpenAI, Azure OpenAI, vLLM, Ollama, Together, LM Studio, or an internal
gateway. Uses only the Python standard library — no vendor SDK and no
cloud-specific code.

Configuration (config dict keys, with environment-variable fallbacks):
    base_url   MECH_GOV_LLM_BASE_URL   e.g. http://localhost:11434/v1
    api_key    MECH_GOV_LLM_API_KEY    bearer token (optional for local servers)
    model      MECH_GOV_LLM_MODEL      model name to request
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from mech_gov.llm.base import LLMInterface, LLMResponse


class OpenAICompatibleLLM(LLMInterface):
    """LLM client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        top_p: float = 1.0,
        timeout_s: int = 120,
        path: str = "/chat/completions",
    ):
        base_url = base_url or os.environ.get("MECH_GOV_LLM_BASE_URL")
        if not base_url:
            raise ValueError(
                "openai_compatible provider requires 'base_url' "
                "(or the MECH_GOV_LLM_BASE_URL environment variable)."
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("MECH_GOV_LLM_API_KEY")
        self._model = model or os.environ.get("MECH_GOV_LLM_MODEL")
        if not self._model:
            raise ValueError(
                "openai_compatible provider requires 'model' "
                "(or the MECH_GOV_LLM_MODEL environment variable)."
            )
        self._top_p = top_p
        self._timeout_s = timeout_s
        self._path = path if path.startswith("/") else "/" + path

    @property
    def model_id(self) -> str:
        return self._model or ""

    @property
    def provider(self) -> str:
        return "openai_compatible"

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        url = self._base_url + self._path
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self._top_p,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        start_ms = time.perf_counter() * 1000
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(f"openai_compatible request failed ({exc.code}): {detail}") from exc
        latency_ms = time.perf_counter() * 1000 - start_ms

        choice = (body.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content", "") or ""
        usage = body.get("usage") or {}

        return LLMResponse(
            content=content,
            model_id=self._model or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            stop_reason=choice.get("finish_reason", "stop"),
        )


def build(config: dict) -> OpenAICompatibleLLM:
    """Build an :class:`OpenAICompatibleLLM` from a config dict (registry entry point)."""
    return OpenAICompatibleLLM(
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        model=config.get("model") or config.get("model_id"),
        top_p=config.get("top_p", 1.0),
        timeout_s=config.get("timeout_s", 120),
        path=config.get("path", "/chat/completions"),
    )
