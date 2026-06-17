# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Amazon SageMaker endpoint client for mech_gov.

Implements LLMInterface for a text-generation model hosted on a SageMaker
real-time endpoint (for example a JumpStart deployment). The request payload
uses the common text-generation contract (``inputs`` + ``parameters``) and a
Llama-style chat template; adjust the template constants below for other model
families.

Optional dependency — requires ``pip install mech-gov-framework[bedrock]``.
"""

from __future__ import annotations

import json
import logging
import time

from mech_gov.llm.base import LLMInterface, LLMResponse

logger = logging.getLogger("mech_gov.llm.sagemaker")


# Llama 3.1 Instruct chat template
LLAMA_SYSTEM_TAG = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}\n<|eot_id|>"
)
LLAMA_USER_TAG = "<|start_header_id|>user<|end_header_id|>\n\n{user}\n<|eot_id|>"
LLAMA_ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _format_llama_prompt(system_prompt: str, user_message: str) -> str:
    """Format a prompt in Llama 3.1 Instruct chat template."""
    parts = [
        LLAMA_SYSTEM_TAG.format(system=system_prompt),
        LLAMA_USER_TAG.format(user=user_message),
        LLAMA_ASSISTANT_TAG,
    ]
    return "".join(parts)


class SageMakerLLM(LLMInterface):
    """A text-generation model hosted on a SageMaker real-time endpoint.

    The endpoint must already be deployed; pass its name via ``endpoint_name``.
    Uses a Llama-style chat template by default (see the constants above).
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
        profile_name: str | None = None,
        top_p: float = 1.0,
    ):
        self._endpoint_name = endpoint_name
        self._region = region
        self._top_p = top_p

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'sagemaker' provider requires boto3. "
                "Install it with: pip install mech-gov-framework[bedrock]"
            ) from exc

        session_kwargs = {"region_name": region}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)
        self._client = session.client("sagemaker-runtime")

    @property
    def model_id(self) -> str:
        return f"sagemaker@{self._endpoint_name}"

    @property
    def provider(self) -> str:
        return "sagemaker"

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Invoke Llama via SageMaker JumpStart endpoint.

        JumpStart payload format:
        {
            "inputs": "<formatted prompt>",
            "parameters": {"max_new_tokens": ..., "temperature": ..., ...}
        }
        """
        prompt = _format_llama_prompt(system_prompt, user_message)

        logger.debug(
            "SageMaker invoke: endpoint=%s temp=%.1f max_tok=%d",
            self._endpoint_name,
            temperature,
            max_tokens,
        )

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),  # JumpStart requires > 0
                "top_p": self._top_p,
                "do_sample": temperature > 0.0,
                "stop": ["<|eot_id|>"],
            },
        }

        start_ms = time.perf_counter() * 1000

        try:
            response = self._client.invoke_endpoint(
                EndpointName=self._endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
        except Exception as exc:
            logger.error("SageMaker invoke FAILED on %s: %s", self._endpoint_name, exc)
            raise

        latency_ms = time.perf_counter() * 1000 - start_ms

        body = json.loads(response["Body"].read().decode("utf-8"))

        # JumpStart returns a list of generated texts
        if isinstance(body, list):
            generated_text = body[0].get("generated_text", "")
        elif isinstance(body, dict):
            generated_text = body.get("generated_text", "")
        else:
            generated_text = str(body)

        # Strip the prompt from the response if echoed back
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        # Remove stop token if present
        generated_text = generated_text.replace("<|eot_id|>", "").strip()

        # Approximate token counts (JumpStart doesn't always return these)
        input_tokens = len(prompt.split()) * 4 // 3  # rough estimate
        output_tokens = len(generated_text.split()) * 4 // 3

        stop_reason = "stop" if "<|eot_id|>" in str(body) else "length"
        logger.debug(
            "SageMaker response: %.0fms  ~in=%d ~out=%d stop=%s",
            latency_ms,
            input_tokens,
            output_tokens,
            stop_reason,
        )
        if stop_reason == "length":
            logger.warning("SageMaker: response may be truncated (no stop token)")

        return LLMResponse(
            content=generated_text,
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
        )


def build(config: dict) -> SageMakerLLM:
    """Build a :class:`SageMakerLLM` from a config dict (registry entry point)."""
    endpoint = config.get("endpoint_name")
    if not endpoint:
        raise ValueError("sagemaker provider requires 'endpoint_name'.")
    return SageMakerLLM(
        endpoint_name=endpoint,
        region=config.get("region", "us-east-1"),
        profile_name=config.get("profile_name"),
        top_p=config.get("top_p", 1.0),
    )
