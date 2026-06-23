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
from dataclasses import dataclass, field
from typing import Protocol

from mech_gov.data.banking_case import Decision

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


@dataclass
class PrivacyConfig:
    """Privacy gate thresholds and token formatting."""

    enabled: bool = True
    max_residual_pii: int = 0  # residual > budget => fail-safe DEFER
    token_prefix: str = "{{"
    token_suffix: str = "}}"


@dataclass
class PrivacyResult:
    """Outcome of running the privacy gate over a prompt body."""

    redacted_text: str
    token_map: dict[str, str] = field(default_factory=dict)
    entities_found: int = 0
    residual_pii: int = 0
    forced_decision: Decision | None = None


def _make_token(config: PrivacyConfig, etype: str, n: int) -> str:
    return f"{config.token_prefix}{etype}_{n}{config.token_suffix}"


def scan_and_tokenize(
    text: str,
    config: PrivacyConfig,
    recognizer: PiiRecognizer,
) -> PrivacyResult:
    """Replace every detected identifier with a stable, reversible token.

    Tokens are numbered per type by order of first appearance; identical values
    of the same type reuse one token. ``residual_pii`` / ``forced_decision`` are
    populated by :func:`privacy_gate`.
    """
    entities = recognizer.recognize(text)

    counters: dict[str, int] = {}
    token_for_value: dict[tuple[str, str], str] = {}
    token_map: dict[str, str] = {}

    # Pass 1 (left -> right): assign stable token numbers.
    for ent in sorted(entities, key=lambda e: e.start):
        key = (ent.etype, ent.text)
        if key not in token_for_value:
            counters[ent.etype] = counters.get(ent.etype, 0) + 1
            token = _make_token(config, ent.etype, counters[ent.etype])
            token_for_value[key] = token
            token_map[token] = ent.text

    # Pass 2 (right -> left): splice tokens in without shifting earlier offsets.
    redacted = text
    for ent in sorted(entities, key=lambda e: e.start, reverse=True):
        token = token_for_value[(ent.etype, ent.text)]
        redacted = redacted[: ent.start] + token + redacted[ent.end :]

    return PrivacyResult(
        redacted_text=redacted,
        token_map=token_map,
        entities_found=len(entities),
    )


def detokenize(text: str, token_map: dict[str, str]) -> str:
    """Restore original values from a token map.

    Only known tokens are replaced; unknown/hallucinated tokens are left
    untouched.
    """
    for token, original in token_map.items():
        text = text.replace(token, original)
    return text


# High-recall residual patterns: deliberately noisy, run over the *redacted*
# text to catch identifier-shaped leftovers the precise pass missed. Privacy
# tokens ({{TYPE_N}}) contain no "@" and no >=7-digit run, so they are not
# counted.
_RESIDUAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d{7,}"),
    re.compile(r"[^\s@]+@[^\s@]+"),
)


def _count_residual(text: str) -> int:
    return sum(len(p.findall(text)) for p in _RESIDUAL_PATTERNS)


def privacy_gate(
    text: str,
    config: PrivacyConfig,
    recognizer: PiiRecognizer | None = None,
) -> PrivacyResult:
    """Minimize PII in ``text`` before the model is consulted.

    Returns a :class:`PrivacyResult`. ``forced_decision`` is ``Decision.DEFER``
    when residual PII exceeds ``config.max_residual_pii`` or the recognizer
    fails (fail-closed); otherwise ``None``.
    """
    if not config.enabled:
        return PrivacyResult(redacted_text=text)

    recognizer = recognizer or RegexRecognizer()
    try:
        result = scan_and_tokenize(text, config, recognizer)
    except Exception:
        # Fail closed: cannot verify minimization -> defer, expose nothing.
        return PrivacyResult(redacted_text="", forced_decision=Decision.DEFER)

    result.residual_pii = _count_residual(result.redacted_text)
    if result.residual_pii > config.max_residual_pii:
        result.forced_decision = Decision.DEFER
    return result
