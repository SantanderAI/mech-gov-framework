# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the R2 pre-LLM privacy gate (offline, no LLM)."""

import pytest

from mech_gov.data.banking_case import Decision
from mech_gov.governance.primitives.privacy_gate import (
    PrivacyConfig,
    RegexRecognizer,
    detokenize,
    privacy_gate,
    scan_and_tokenize,
)


@pytest.mark.parametrize(
    "text,etype",
    [
        ("contact jane@bank.com today", "EMAIL"),
        ("ssn 123-45-6789 on file", "SSN"),
        ("call 555-123-4567 now", "PHONE"),
        ("card 4111 1111 1111 1111 used", "PAN"),
        ("iban DE89370400440532013000 set", "IBAN"),
        ("host 192.168.0.1 logged", "IP"),
    ],
)
def test_regex_recognizer_detects_each_type(text, etype):
    ents = RegexRecognizer().recognize(text)
    assert len(ents) == 1
    assert ents[0].etype == etype
    assert text[ents[0].start : ents[0].end] == ents[0].text


def test_regex_recognizer_no_pii_returns_empty():
    assert RegexRecognizer().recognize("Risk Score: 0.620, Flags: AML") == []


def test_tokenize_replaces_pii_and_is_reversible():
    cfg = PrivacyConfig()
    text = "Email jane@bank.com or call 555-123-4567"
    res = scan_and_tokenize(text, cfg, RegexRecognizer())
    assert "jane@bank.com" not in res.redacted_text
    assert "555-123-4567" not in res.redacted_text
    assert "{{EMAIL_1}}" in res.redacted_text
    assert "{{PHONE_1}}" in res.redacted_text
    assert res.entities_found == 2
    assert detokenize(res.redacted_text, res.token_map) == text


def test_tokenize_dedupes_repeated_value_and_numbers_by_first_appearance():
    cfg = PrivacyConfig()
    text = "a@x.com and c@y.com and a@x.com"
    res = scan_and_tokenize(text, cfg, RegexRecognizer())
    assert res.entities_found == 3
    assert len(res.token_map) == 2  # a@x.com reused
    assert res.redacted_text == "{{EMAIL_1}} and {{EMAIL_2}} and {{EMAIL_1}}"
    assert detokenize(res.redacted_text, res.token_map) == text


def test_tokenize_deterministic():
    cfg = PrivacyConfig()
    text = "Email jane@bank.com or call 555-123-4567"
    a = scan_and_tokenize(text, cfg, RegexRecognizer())
    b = scan_and_tokenize(text, cfg, RegexRecognizer())
    assert a.redacted_text == b.redacted_text
    assert a.token_map == b.token_map


def test_no_pii_passthrough_unchanged():
    cfg = PrivacyConfig()
    text = "Risk Score: 0.620\nRegulatory Flags: AML, KYC"
    res = scan_and_tokenize(text, cfg, RegexRecognizer())
    assert res.redacted_text == text
    assert res.token_map == {}
    assert res.entities_found == 0


class _BoomRecognizer:
    def recognize(self, text):
        raise RuntimeError("detector unavailable")


def test_gate_passes_clean_prompt():
    res = privacy_gate("Email jane@bank.com", PrivacyConfig(), RegexRecognizer())
    assert res.forced_decision is None
    assert res.residual_pii == 0
    assert "{{EMAIL_1}}" in res.redacted_text


def test_gate_defers_on_residual_over_budget():
    # 8-digit account number is not a PAN (>=13) / phone / SSN, so the precise
    # pass misses it; the high-recall residual scan catches it.
    res = privacy_gate("account 12345678 flagged", PrivacyConfig(), RegexRecognizer())
    assert res.residual_pii >= 1
    assert res.forced_decision == Decision.DEFER


def test_gate_budget_allows_some_residual():
    cfg = PrivacyConfig(max_residual_pii=1)
    res = privacy_gate("account 12345678 flagged", cfg, RegexRecognizer())
    assert res.forced_decision is None


def test_gate_fails_closed_on_recognizer_error():
    res = privacy_gate("anything", PrivacyConfig(), _BoomRecognizer())
    assert res.forced_decision == Decision.DEFER
    assert res.redacted_text == ""
    assert res.token_map == {}


def test_gate_disabled_is_noop():
    cfg = PrivacyConfig(enabled=False)
    res = privacy_gate("Email jane@bank.com", cfg, RegexRecognizer())
    assert res.forced_decision is None
    assert res.redacted_text == "Email jane@bank.com"
    assert res.token_map == {}
