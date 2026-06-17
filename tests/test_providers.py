# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendor-neutral LLM provider layer (offline)."""

import pytest

from mech_gov.llm.base import LLMResponse
from mech_gov.llm.registry import available_providers, create_llm


def test_builtin_providers_registered():
    names = available_providers()
    for expected in ("mock", "callable", "openai", "openai_compatible", "bedrock", "sagemaker"):
        assert expected in names


def test_mock_provider_returns_llmresponse():
    llm = create_llm({"provider": "mock"})
    resp = llm.invoke("system", "user")
    assert isinstance(resp, LLMResponse)
    assert resp.content
    assert llm.provider == "mock"


def test_callable_provider_wraps_function():
    def fn(system_prompt, user_message, temperature=0.0, max_tokens=2048):
        return "hello world"

    llm = create_llm({"provider": "callable", "callable": fn})
    assert llm.invoke("s", "u").content == "hello world"
    assert llm.provider == "callable"


def test_callable_provider_two_arg_signature():
    def fn(system_prompt, user_message):
        return "ok"

    llm = create_llm({"provider": "callable", "callable": fn})
    assert llm.invoke("s", "u").content == "ok"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        create_llm({"provider": "does-not-exist"})


def test_missing_provider_key_raises():
    with pytest.raises(ValueError):
        create_llm({"model_id": "x"})


def test_bedrock_validates_before_importing_boto3():
    # build_bedrock checks required fields before importing boto3, so this
    # raises ValueError (not ImportError) regardless of whether boto3 exists.
    with pytest.raises(ValueError):
        create_llm({"provider": "bedrock"})  # missing model_id
