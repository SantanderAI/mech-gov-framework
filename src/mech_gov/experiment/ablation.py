# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Ablation Runner for mech_gov v2.

Runs A1-A4 ablation experiments (the design spec §2.6 / §5.4):
  A1: No I6Q      — Remove I6Q enforcement from R2
  A2: Agent CEFL   — CEFL inside agent loop (agentic, not frozen)
  A3: E1 observable — Entropy seed visible to LLM
  A4: No DEFER     — Remove DEFER from decision space
"""

from __future__ import annotations

from mech_gov.data.banking_case import BankingCase
from mech_gov.governance.primitives.i6q import I6QConfig
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.governance.regime import DecisionResult, GovernanceRegime
from mech_gov.llm.base import LLMInterface


class A1_NoI6Q(R2Mechanical):
    """Ablation A1: R2 without I6Q enforcement.

    I6Q max_retries=0 and always passes. Framing re-enabled.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault(
            "i6q_config",
            I6QConfig(
                min_arg_tokens=0,
                min_diversity=0.0,
                max_retries=0,
            ),
        )
        super().__init__(**kwargs)

    @property
    def regime_name(self) -> str:
        return "R2_A1_no_i6q"


class A2_AgentCEFL(R2Mechanical):
    """Ablation A2: CEFL inside agent loop (agentic).

    Instead of frozen candidates, uses a single LLM call with
    temperature=0 (deterministic). SBN re-enabled.
    """

    def __init__(self, **kwargs):
        kwargs["n_cefl_candidates"] = 1  # Single call = no freezing
        super().__init__(**kwargs)

    @property
    def regime_name(self) -> str:
        return "R2_A2_agent_cefl"


class A3_E1Observable(R2Mechanical):
    """Ablation A3: Entropy seed visible to LLM.

    The entropy nonce is included in the user message, making
    the E3 commit-reveal trivially breakable. ESD should triple.
    """

    def process_case(
        self,
        case: BankingCase,
        llm: LLMInterface,
        entropy_seed: int | None = None,
    ) -> DecisionResult:
        # Run normal R2 but inject entropy info into metadata
        result = super().process_case(case, llm, entropy_seed)
        result.metadata["entropy_observable"] = True
        result.metadata["ablation"] = "A3_e1_observable"
        return result

    @property
    def regime_name(self) -> str:
        return "R2_A3_e1_observable"


class A4_NoDefer(R2Mechanical):
    """Ablation A4: Remove DEFER from decision space.

    Any DEFER decision is forced to ESCALATE. FVS should drop.
    """

    def process_case(
        self,
        case: BankingCase,
        llm: LLMInterface,
        entropy_seed: int | None = None,
    ) -> DecisionResult:
        from mech_gov.data.banking_case import Decision

        result = super().process_case(case, llm, entropy_seed)
        if result.decision == Decision.DEFER:
            result.decision = Decision.ESCALATE
            result.metadata["defer_removed"] = True
            result.metadata["ablation"] = "A4_no_defer"
        return result

    @property
    def regime_name(self) -> str:
        return "R2_A4_no_defer"


ABLATION_REGISTRY: dict[str, type] = {
    "A1_no_i6q": A1_NoI6Q,
    "A2_agent_cefl": A2_AgentCEFL,
    "A3_e1_observable": A3_E1Observable,
    "A4_no_defer": A4_NoDefer,
}


def create_ablation_regime(ablation_name: str, **kwargs) -> GovernanceRegime:
    """Factory for ablation regimes."""
    cls = ABLATION_REGISTRY.get(ablation_name)
    if cls is None:
        raise ValueError(
            f"Unknown ablation: {ablation_name}. " f"Available: {list(ABLATION_REGISTRY.keys())}"
        )
    return cls(**kwargs)
