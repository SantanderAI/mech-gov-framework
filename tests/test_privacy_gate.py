# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the R2 pre-LLM privacy gate (offline, no LLM)."""

import pytest

from mech_gov.governance.primitives.privacy_gate import RegexRecognizer


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
