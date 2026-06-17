# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""mech_gov — Mechanical Governance for LLM Decisions.

A model-agnostic framework for enforcing governance on LLM decisions:
text-only (R1) vs mechanical enforcement (R2), plus governance metrics
(CDL, DIU, FVS, ESD, FSR) and a synthetic decision dataset.

LLM access is fully pluggable via :mod:`mech_gov.llm.registry`. Nothing in
the core install depends on a specific cloud vendor — bring your own backend
(see :mod:`mech_gov.llm.providers`).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
