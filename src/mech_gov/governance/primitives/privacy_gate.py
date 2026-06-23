# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Privacy Gate — pre-LLM PII minimization for the R2 Mechanical regime.

Reversibly tokenizes direct identifiers in the prompt before the LLM is called,
so the model never sees raw personal data. If residual (untokenizable) PII
remains above the configured budget, the gate forces a mechanical DEFER — the
case is never sent to the model in the clear.

Stdlib-only by default (regex recognizers). An optional NER backend can be
supplied behind the ``PiiRecognizer`` protocol. Maps to data minimization
(GDPR Art. 5(1)(c)) and OWASP LLM06 (Sensitive Information Disclosure).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

PRIVACY_GATE_ID = "PRIV_0"


@dataclass(frozen=True)
class PiiEntity:
    """A single detected identifier span."""

    etype: str
    start: int
    end: int
    text: str


class PiiRecognizer(Protocol):
    """Detect identifier spans in text.

    Implementations may raise on failure; the gate treats any exception as a
    fail-closed signal.
    """

    def recognize(self, text: str) -> list[PiiEntity]: ...


# High-precision patterns. Order = priority when spans overlap (first wins).
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("PAN", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)


class RegexRecognizer:
    """Default stdlib recognizer for high-precision identifier types."""

    def recognize(self, text: str) -> list[PiiEntity]:
        found: list[PiiEntity] = []
        for etype, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                found.append(PiiEntity(etype, m.start(), m.end(), m.group()))
        # Resolve overlaps: earliest start first, then longest span; drop overlaps.
        found.sort(key=lambda e: (e.start, -(e.end - e.start)))
        resolved: list[PiiEntity] = []
        last_end = -1
        for e in found:
            if e.start >= last_end:
                resolved.append(e)
                last_end = e.end
        return resolved
