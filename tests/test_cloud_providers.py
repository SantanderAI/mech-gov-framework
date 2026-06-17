# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for cloud/HTTP providers using injected fakes (no network, no boto3)."""

import io
import json
import sys
import types

import pytest

from mech_gov.llm.base import LLMResponse
from mech_gov.llm.providers import openai_compatible

# ----------------------- openai_compatible (urllib mocked) -----------------------


class _FakeHTTPResponse:
    def __init__(self, body: dict):
        self._data = json.dumps(body).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_openai_compatible_invoke(monkeypatch):
    body = {
        "choices": [{"message": {"content": '{"decision": "APPROVE"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }
    monkeypatch.setattr(
        openai_compatible.urllib.request,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(body),
    )
    llm = openai_compatible.build({"base_url": "http://x/v1", "model": "m", "api_key": "k"})
    resp = llm.invoke("sys", "user")
    assert isinstance(resp, LLMResponse)
    assert resp.input_tokens == 12
    assert llm.provider == "openai_compatible"
    assert llm.model_id == "m"


def test_openai_compatible_http_error(monkeypatch):
    import urllib.error

    def _raise(req, timeout=0):
        raise urllib.error.HTTPError("http://x", 500, "boom", {}, io.BytesIO(b"err"))

    monkeypatch.setattr(openai_compatible.urllib.request, "urlopen", _raise)
    llm = openai_compatible.build({"base_url": "http://x/v1", "model": "m"})
    with pytest.raises(RuntimeError):
        llm.invoke("s", "u")


def test_openai_compatible_requires_base_url(monkeypatch):
    monkeypatch.delenv("MECH_GOV_LLM_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        openai_compatible.build({"model": "m"})


def test_openai_compatible_requires_model(monkeypatch):
    monkeypatch.delenv("MECH_GOV_LLM_MODEL", raising=False)
    with pytest.raises(ValueError):
        openai_compatible.build({"base_url": "http://x"})


# ----------------------- bedrock (boto3 faked via sys.modules) -----------------------


class _FakeClientError(Exception):
    def __init__(self, code="ThrottlingException"):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def _install_fake_boto3(monkeypatch, converse_response=None, invoke_response=None, client=None):
    boto3 = types.ModuleType("boto3")

    class _Client:
        def converse(self, **kwargs):
            return converse_response

        def invoke_endpoint(self, **kwargs):
            return invoke_response

    chosen = client or _Client()

    class _Session:
        def __init__(self, **kwargs):
            pass

        def client(self, *args, **kwargs):
            return chosen

    boto3.Session = _Session

    botocore = types.ModuleType("botocore")
    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = lambda **kwargs: object()
    botocore_exc = types.ModuleType("botocore.exceptions")
    botocore_exc.ClientError = _FakeClientError

    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", botocore_exc)


def test_bedrock_build_requires_model_id():
    from mech_gov.llm.providers import bedrock

    with pytest.raises(ValueError):
        bedrock.build({})


def test_bedrock_invoke_parses_converse_response(monkeypatch):
    _install_fake_boto3(
        monkeypatch,
        converse_response={
            "output": {"message": {"content": [{"text": '{"decision": "DECLINE"}'}]}},
            "usage": {"inputTokens": 5, "outputTokens": 9},
            "stopReason": "end_turn",
        },
    )
    from mech_gov.llm.providers import bedrock

    llm = bedrock.build({"model_id": "anthropic.claude", "region": "us-east-1"})
    resp = llm.invoke("sys", "user")
    assert resp.content == '{"decision": "DECLINE"}'
    assert resp.input_tokens == 5
    assert llm.model_id == "anthropic.claude"
    assert llm.provider == "bedrock"


# ----------------------- sagemaker (boto3 faked) -----------------------


def test_sagemaker_build_requires_endpoint():
    from mech_gov.llm.providers import sagemaker

    with pytest.raises(ValueError):
        sagemaker.build({})


def test_sagemaker_invoke_parses_response(monkeypatch):
    body = json.dumps([{"generated_text": "APPROVE <|eot_id|>"}]).encode("utf-8")
    _install_fake_boto3(monkeypatch, invoke_response={"Body": io.BytesIO(body)})
    from mech_gov.llm.providers import sagemaker

    llm = sagemaker.build({"endpoint_name": "my-endpoint"})
    resp = llm.invoke("sys", "user")
    assert "APPROVE" in resp.content
    assert llm.provider == "sagemaker"
    assert llm.model_id == "sagemaker@my-endpoint"


def test_sagemaker_invoke_dict_body_no_stop_token(monkeypatch):
    body = json.dumps({"generated_text": "DECLINE response without stop"}).encode("utf-8")
    _install_fake_boto3(monkeypatch, invoke_response={"Body": io.BytesIO(body)})
    from mech_gov.llm.providers import sagemaker

    llm = sagemaker.build({"endpoint_name": "ep"})
    resp = llm.invoke("sys", "user")
    assert resp.stop_reason == "length"  # no <|eot_id|> present
    assert "DECLINE" in resp.content


def test_bedrock_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    class _FlakyClient:
        def __init__(self):
            self.calls = 0

        def converse(self, **kwargs):
            self.calls += 1
            if self.calls < 2:
                raise _FakeClientError("ThrottlingException")
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
                "stopReason": "end_turn",
            }

    _install_fake_boto3(monkeypatch, client=_FlakyClient())
    from mech_gov.llm.providers import bedrock

    llm = bedrock.build({"model_id": "m", "profile_name": "p"})
    assert llm.invoke("s", "u").content == "ok"


def test_bedrock_non_retryable_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    class _BadClient:
        def converse(self, **kwargs):
            raise _FakeClientError("ValidationException")

    _install_fake_boto3(monkeypatch, client=_BadClient())
    from mech_gov.llm.providers import bedrock

    llm = bedrock.build({"model_id": "m"})
    with pytest.raises(_FakeClientError):
        llm.invoke("s", "u")


def test_bedrock_generic_exception_exhausts_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    class _BoomClient:
        def converse(self, **kwargs):
            raise RuntimeError("network down")

    _install_fake_boto3(monkeypatch, client=_BoomClient())
    from mech_gov.llm.providers import bedrock

    llm = bedrock.build({"model_id": "m"})
    with pytest.raises(RuntimeError):
        llm.invoke("s", "u")
