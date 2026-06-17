# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Entropy E3 — Commit-Reveal mechanism for R2 Mechanical regime.

Prevents seed-conditioning attacks where the LLM manipulates entropy.
the design spec §2.4:
  1. COMMIT: Generate nonce N, hash H(N), store H(N) before LLM call
  2. GENERATE: Use N as part of random seed for candidate generation
  3. SELECT: Use independent entropy source for candidate selection
  4. REVEAL: Publish N after selection; verify H(N) matches commit
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass


@dataclass
class E3Commit:
    """Commit phase result — created BEFORE the LLM call."""

    nonce: str  # Random nonce N
    nonce_hash: str  # H(N) = SHA-256(N)
    selection_seed: int  # Independent seed for candidate selection


@dataclass
class E3Reveal:
    """Reveal phase result — created AFTER candidate selection."""

    nonce: str
    nonce_hash: str
    verified: bool  # H(revealed_N) == committed H(N)


def e3_commit(entropy_seed: int | None = None) -> E3Commit:
    """Generate a commit for the E3 protocol.

    Creates a random nonce and its hash. The hash is "published" (stored)
    before the LLM call. The nonce is used to seed candidate generation.

    Args:
        entropy_seed: Optional deterministic seed for reproducibility in tests.
                      In production, uses secrets.token_hex for true randomness.

    Returns:
        E3Commit with nonce, hash, and independent selection seed
    """
    if entropy_seed is not None:
        # Deterministic mode for reproducibility
        import numpy as np

        rng = np.random.Generator(np.random.PCG64(entropy_seed))
        nonce = rng.integers(0, 2**63).item()
        nonce_str = f"{nonce:016x}"
        selection_seed = int(rng.integers(0, 2**31 - 1))
    else:
        # Production mode: cryptographically random
        nonce_str = secrets.token_hex(16)
        selection_seed = secrets.randbelow(2**31)

    nonce_hash = hashlib.sha256(nonce_str.encode("utf-8")).hexdigest()

    return E3Commit(
        nonce=nonce_str,
        nonce_hash=nonce_hash,
        selection_seed=selection_seed,
    )


def e3_reveal(commit: E3Commit) -> E3Reveal:
    """Verify the commit-reveal protocol.

    Recomputes the hash from the stored nonce and verifies it matches
    the committed hash. In a real adversarial setting, the nonce would
    be revealed only after candidate selection is final.

    Args:
        commit: The E3Commit from before the LLM call

    Returns:
        E3Reveal with verification result
    """
    recomputed_hash = hashlib.sha256(commit.nonce.encode("utf-8")).hexdigest()

    verified = recomputed_hash == commit.nonce_hash

    return E3Reveal(
        nonce=commit.nonce,
        nonce_hash=commit.nonce_hash,
        verified=verified,
    )
