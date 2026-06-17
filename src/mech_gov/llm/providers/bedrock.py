# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Amazon Bedrock client (model-agnostic) for mech_gov.

Implements LLMInterface via the Bedrock **Converse API**, which exposes a
single, unified request/response format across model families — Anthropic
Claude, Meta Llama, Mistral, Amazon Titan, Cohere, and others. There is no need
for per-family clients: pass any Bedrock model id (or inference-profile id) as
``model_id``.

Optional dependency — requires ``pip install mech-gov-framework[bedrock]``.
"""

from __future__ import annotations

import logging
import random
import time

from mech_gov.llm.base import LLMInterface, LLMResponse

logger = logging.getLogger("mech_gov.llm.bedrock")


class BedrockLLM(LLMInterface):
    """Any Amazon Bedrock model via the Converse API.

    Works across model families (Claude, Llama, Mistral, Titan, Cohere, ...)
    because the Converse API normalises chat formatting and token accounting.
    Bedrock hosts the model as a managed service — pay per token, nothing to
    deploy.
    """

    # Bedrock error codes that warrant a retry
    _RETRYABLE_ERRORS = {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "ModelNotReadyException",
        "InternalServerException",
    }

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        profile_name: str | None = None,
        top_p: float = 1.0,
        max_retries: int = 5,
        read_timeout_s: int = 120,
    ):
        self._model_id = model_id
        self._region = region
        self._top_p = top_p
        self._max_retries = max_retries

        try:
            import boto3
            from botocore.config import Config as BotoConfig
            from botocore.exceptions import ClientError
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'bedrock' provider requires boto3. "
                "Install it with: pip install mech-gov-framework[bedrock]"
            ) from exc
        self._ClientError = ClientError

        session_kwargs = {"region_name": region}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        boto_config = BotoConfig(
            read_timeout=read_timeout_s,
            connect_timeout=30,
            retries={"max_attempts": 0},  # we handle retries ourselves
        )
        session = boto3.Session(**session_kwargs)
        self._client = session.client(
            "bedrock-runtime",
            region_name=region,
            config=boto_config,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "bedrock"

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Invoke the model via the Bedrock Converse API.

        The Converse API handles chat formatting automatically across model
        families, so no model-specific chat template is required.
        """
        logger.debug(
            "Bedrock invoke: model=%s temp=%.1f max_tok=%d", self._model_id, temperature, max_tokens
        )

        start_ms = time.perf_counter() * 1000

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.converse(
                    modelId=self._model_id,
                    system=[{"text": system_prompt}],
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": user_message}],
                        },
                    ],
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature,
                        "topP": self._top_p,
                    },
                )
                break  # success
            except self._ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")
                if error_code in self._RETRYABLE_ERRORS and attempt < self._max_retries:
                    backoff = min(2**attempt + random.random(), 60)
                    logger.warning(
                        "Bedrock retryable error %s (attempt %d/%d), " "backing off %.1fs: %s",
                        error_code,
                        attempt,
                        self._max_retries,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)
                    last_exc = exc
                    continue
                logger.error("Bedrock invoke FAILED (non-retryable): %s", exc)
                raise
            except Exception as exc:
                if attempt < self._max_retries:
                    backoff = min(2**attempt + random.random(), 60)
                    logger.warning(
                        "Bedrock unexpected error (attempt %d/%d), " "backing off %.1fs: %s",
                        attempt,
                        self._max_retries,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)
                    last_exc = exc
                    continue
                logger.error("Bedrock invoke FAILED after %d attempts: %s", self._max_retries, exc)
                raise
        else:
            # All retries exhausted
            logger.error("Bedrock invoke FAILED: all %d retries exhausted", self._max_retries)
            raise last_exc  # type: ignore[misc]

        elapsed_ms = time.perf_counter() * 1000 - start_ms

        # Extract content from Converse response
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        content = content_blocks[0].get("text", "") if content_blocks else ""

        # Token usage from Converse API
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        stop_reason = response.get("stopReason", "unknown")

        logger.debug(
            "Bedrock response: %.0fms  in=%d out=%d stop=%s",
            elapsed_ms,
            input_tokens,
            output_tokens,
            stop_reason,
        )
        if stop_reason == "max_tokens":
            logger.warning("Bedrock: response truncated (max_tokens)")

        return LLMResponse(
            content=content,
            model_id=self._model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            stop_reason=stop_reason,
        )


def build(config: dict) -> BedrockLLM:
    """Build a :class:`BedrockLLM` from a config dict (registry entry point)."""
    model_id = config.get("model_id")
    if not model_id:
        raise ValueError("bedrock provider requires 'model_id'.")
    return BedrockLLM(
        model_id=model_id,
        region=config.get("region", "us-east-1"),
        profile_name=config.get("profile_name"),
        top_p=config.get("top_p", 1.0),
    )
