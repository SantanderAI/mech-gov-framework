# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, versioned identity for experiment runs.

The readable run label (``mock-R2-S0-seed42``) is convenient for humans but too
coarse to decide whether a run has already completed: it omits the case count,
the resolved model identifier, the distribution configuration, and provider
generation parameters. Two materially different experiments can therefore share
a label, and the second one can be skipped silently.

This module separates the two concerns: ``label`` stays the legacy readable
string, while ``digest`` is a SHA-256 over a versioned, allowlisted canonical
payload. Only the full digest decides equality — the truncated digest embedded
in ``run_id`` exists for display and filesystem convenience.

The module is deliberately inert. It never reads environment variables, never
touches the filesystem, never invokes a model, and never accepts raw provider
configuration. Callers pass already-resolved, allowlisted values, so no secret,
endpoint, prompt, case, or path can reach an identity payload or a log line.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

logger = logging.getLogger("mech_gov.experiment.identity")

#: Version of the canonical payload layout. Bump when fields change meaning.
IDENTITY_SCHEMA = 1

#: Payload versions this build is able to verify.
SUPPORTED_IDENTITY_SCHEMAS = frozenset({IDENTITY_SCHEMA})

#: Hex characters of the digest shown in ``run_id``. Display only.
DISPLAY_DIGEST_LENGTH = 12

#: Exact set of keys a schema-1 payload must contain.
PAYLOAD_FIELDS = frozenset(
    "identity_schema model_name model_id_sha256 regime_name condition seed "
    "cases_per_condition distributions_sha256 provider_parameters package_version".split()
)

#: Stable warning codes. Every condition below resolves to "execute".
WARNING_SUMMARIES = {
    "MGI001": "legacy result with the same run label carries no versioned identity",
    "MGI002": "run label matches a different verified identity",
    "MGI003": "stored identity uses an unsupported schema",
    "MGI004": "stored identity payload or digest failed validation",
    "MGI005": "identity digest is claimed by conflicting payloads",
    "MGI006": "stable model or provider identity is unavailable",
    "MGI007": "package provenance is unavailable",
}

_FULL_DIGEST_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class IdentityError(ValueError):
    """A configuration value cannot take part in a deterministic identity."""


def _canonical_number(value: int | float, path: str) -> str:
    """Render a finite number so that ``1``, ``1.0`` and ``-0.0`` behave sanely."""
    if isinstance(value, float) and not math.isfinite(value):
        raise IdentityError(f"non-finite numeric value at {path or '<root>'}")
    as_decimal = Decimal(value) if isinstance(value, int) else Decimal(repr(value))
    if as_decimal == 0:
        return "0"
    return str(as_decimal.normalize())


def _normalize(value: Any, path: str = "") -> Any:
    """Recursively rewrite a loaded value into a type-tagged canonical form.

    Scalars are tagged so that ``None``, ``False`` and the strings ``"z:"`` or
    ``"b:false"`` can never collide. Mapping keys must be strings and are
    ordered by the serializer; list order stays significant.
    """
    if value is None:
        return "z:"
    if isinstance(value, bool):  # bool is a subclass of int — check it first
        return "b:true" if value else "b:false"
    if isinstance(value, str):
        return "s:" + value
    if isinstance(value, (int, float)):
        return "n:" + _canonical_number(value, path)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityError(f"non-string mapping key at {path or '<root>'}")
            normalized[key] = _normalize(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise IdentityError(f"unsupported {type(value).__name__} value at {path or '<root>'}")


def _canonical_text(value: Any) -> str:
    """Serialize a normalized value deterministically."""
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the exact bytes a payload digest is computed over."""
    return _canonical_text(dict(payload)).encode("utf-8")


def digest_payload(payload: Mapping[str, Any]) -> str:
    """Return the authoritative full SHA-256 digest of a payload."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def distributions_digest(distributions_config: Mapping[str, Any]) -> str:
    """Fingerprint the *semantics* of a loaded distributions configuration.

    The input is the mapping returned by ``load_distributions``, so comments,
    formatting, key order and the source path are all excluded: identical
    content at another path fingerprints identically, and changed content at
    the same path does not.
    """
    return _sha256_text(_canonical_text(dict(distributions_config)))


@dataclass(frozen=True)
class RunIdentity:
    """A readable label paired with the digest that actually decides equality."""

    label: str
    payload: dict[str, Any]
    digest: str

    @property
    def run_id(self) -> str:
        """Readable label plus a truncated digest. Never used for equality."""
        return f"{self.label}--{self.digest[:DISPLAY_DIGEST_LENGTH]}"


def build_run_identity(
    *,
    label: str,
    model_name: str,
    model_id: str,
    regime_name: str,
    condition: str,
    seed: int,
    cases_per_condition: int,
    distributions_config: Mapping[str, Any],
    provider_parameters: Mapping[str, Any],
    package_version: str,
) -> RunIdentity:
    """Build a versioned identity from already-resolved, allowlisted inputs.

    ``model_id`` is hashed in memory and never persisted in the payload. The
    caller is responsible for establishing that it is stable and non-empty.
    """
    if not model_id:
        raise IdentityError("model_id must be a stable non-empty string")
    if not package_version:
        raise IdentityError("package_version must be a non-empty string")
    payload: dict[str, Any] = {
        "identity_schema": IDENTITY_SCHEMA,
        "model_name": model_name,
        "model_id_sha256": _sha256_text(model_id),
        "regime_name": regime_name,
        "condition": condition,
        "seed": seed,
        "cases_per_condition": cases_per_condition,
        "distributions_sha256": distributions_digest(distributions_config),
        "provider_parameters": dict(provider_parameters),
        "package_version": package_version,
    }
    return RunIdentity(label=label, payload=payload, digest=digest_payload(payload))


def verified_stored_digest(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Validate one stored result record before it may cause a skip.

    Returns ``(digest, warning_code)``. A digest is returned only when the
    schema is supported, the payload is complete, the stored digest is a
    lowercase SHA-256, and re-canonicalizing the payload reproduces it.
    ``(None, None)`` means the record is legacy and carries no identity at all.
    """
    schema = record.get("identity_schema")
    payload = record.get("identity")
    stored = record.get("identity_sha256")
    if schema is None and payload is None and stored is None:
        return None, None
    if isinstance(schema, bool) or not isinstance(schema, int):
        return None, "MGI003"
    if schema not in SUPPORTED_IDENTITY_SCHEMAS:
        return None, "MGI003"
    if not isinstance(stored, str) or not _FULL_DIGEST_RE.match(stored):
        return None, "MGI004"
    if not isinstance(payload, Mapping) or set(payload) != set(PAYLOAD_FIELDS):
        return None, "MGI004"
    if payload.get("identity_schema") != schema:
        return None, "MGI004"
    try:
        recomputed = digest_payload(payload)
    except IdentityError:
        return None, "MGI004"
    return (stored, None) if recomputed == stored else (None, "MGI004")


def changed_fields(current: Mapping[str, Any], stored: Mapping[str, Any]) -> list[str]:
    """Return the sorted names of payload fields that differ. Never the values."""
    names = set(current) | set(stored)
    return sorted(name for name in names if current.get(name) != stored.get(name))


class IdentityWarnings:
    """Emits each identity warning at most once per completion scan."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()

    def emit(self, code: str, label: str, fields: Sequence[str] = ()) -> None:
        """Log one deduplicated warning naming changed fields but no values."""
        if (code, label) in self._seen:
            return
        self._seen.add((code, label))
        logger.warning(
            "[%s] %s; fields=%s; action=execute",
            code,
            WARNING_SUMMARIES[code],
            ",".join(fields),
        )
