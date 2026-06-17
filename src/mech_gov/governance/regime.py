# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Base governance regime class and DecisionResult dataclass.

Matches the design spec §2.1. All regimes (R1, R2, R3) inherit from GovernanceRegime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mech_gov.data.banking_case import BankingCase, Decision
from mech_gov.llm.base import LLMInterface


@dataclass
class DecisionResult:
    """Structured result from processing a case through a governance regime.

    Contains the decision, rationale, metadata, and regime-specific fields.
    Matches the design spec §2.1 DecisionResult schema exactly.
    """

    case_id: str
    regime: str  # "R1", "R2", "R3"
    decision: Decision
    rationale: str
    pro_arguments: list[str] = field(default_factory=list)
    con_arguments: list[str] = field(default_factory=list)
    deferral_text: str | None = None
    conditions_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_raw_response: str = ""
    processing_time_ms: float = 0.0
    tokens_used: int = 0

    # R2-specific (populated only for R2)
    gates_triggered: list[str] = field(default_factory=list)
    cefl_candidates: int | None = None
    cefl_candidate_scores: list[dict[str, Any]] | None = None
    i6q_passed: bool | None = None
    entropy_nonce: str | None = None

    # R3-specific (populated only for R3)
    modification_proposed: bool | None = None
    modification_accepted: bool | None = None
    drift_budget_remaining: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL output."""
        return {
            "case_id": self.case_id,
            "regime": self.regime,
            "decision": (
                self.decision.value if isinstance(self.decision, Decision) else self.decision
            ),
            "rationale": self.rationale,
            "pro_arguments": self.pro_arguments,
            "con_arguments": self.con_arguments,
            "deferral_text": self.deferral_text,
            "conditions_text": self.conditions_text,
            "metadata": self.metadata,
            "llm_raw_response": self.llm_raw_response,
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used,
            "gates_triggered": self.gates_triggered,
            "cefl_candidates": self.cefl_candidates,
            "cefl_candidate_scores": self.cefl_candidate_scores,
            "i6q_passed": self.i6q_passed,
            "entropy_nonce": self.entropy_nonce,
            "modification_proposed": self.modification_proposed,
            "modification_accepted": self.modification_accepted,
            "drift_budget_remaining": self.drift_budget_remaining,
        }


class GovernanceRegime(ABC):
    """Base class for all governance regimes.

    Subclasses: R1TextOnly, R2Mechanical, (R3Adaptive — stub/exploratory).
    """

    @property
    @abstractmethod
    def regime_name(self) -> str:
        """Return regime identifier: 'R1', 'R2', or 'R3'."""
        ...

    @abstractmethod
    def process_case(
        self,
        case: BankingCase,
        llm: LLMInterface,
        entropy_seed: int | None = None,
    ) -> DecisionResult:
        """Process a single case through the governance pipeline.

        Args:
            case: The banking case to evaluate
            llm: LLM interface to use for generation
            entropy_seed: Optional seed for E3 commit-reveal (R2 only)

        Returns:
            DecisionResult with decision, rationale, and metadata
        """
        ...
