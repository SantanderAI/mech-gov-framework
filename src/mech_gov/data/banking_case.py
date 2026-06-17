# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
BankingCase — Core data model for mech_gov v2 synthetic dataset.

Matches the schema defined in the design spec §1.1-1.2 and the paper's
appendix output format. Every field is configurable via YAML configs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TransactionType(str, Enum):
    CREDIT_APPROVAL = "credit_approval"
    FRAUD_ALERT = "fraud_alert"
    SANCTIONS_SCREENING = "sanctions_screening"
    AML_REVIEW = "aml_review"
    CONCENTRATION_RISK = "concentration_risk"


class StressCondition(str, Enum):
    """Maps to paper condition names: S0 (Baseline), S1 (HighRisk), S2 (LowInfo), S3 (Threshold)."""

    S0_BASELINE = "S0"
    S1_HIGH_RISK = "S1"
    S2_LOW_INFO = "S2"
    S3_THRESHOLD = "S3"


class GTConfidence(str, Enum):
    """Ground truth confidence layer (honest 2-layer design)."""

    DETERMINISTIC = "deterministic"  # ~40% of cases
    AMBIGUOUS = "ambiguous"  # ~60% of cases
    # SME_VALIDATED removed: circular if paper authors are the SMEs.
    # Can be re-added if independent domain experts annotate a sample.


class Decision(str, Enum):
    """Possible governance decisions."""

    APPROVE = "APPROVE"
    CONDITIONAL = "CONDITIONAL"
    ESCALATE = "ESCALATE"
    DEFER = "DEFER"
    DECLINE = "DECLINE"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLAG_UNIVERSE: frozenset[str] = frozenset({"AML", "KYC", "SANCTIONS", "INSIDER", "CONCENTRATION"})


# ---------------------------------------------------------------------------
# BankingCase Model
# ---------------------------------------------------------------------------


class BankingCase(BaseModel):
    """A single synthetic banking case for governance evaluation.

    Covers both original paper parameters and v2 extensions.
    All fields are documented for paper Appendix C (dataset description).
    """

    # --- Identification ---
    case_id: str = Field(
        ...,
        description="Unique ID encoding type/condition/index, e.g. 'credit_approval-Baseline-0042'",
    )

    # --- Original paper parameters ---
    transaction_type: TransactionType
    risk_score: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    regulatory_flags: list[str] = Field(default_factory=list)

    # --- v2 extensions (for richer dataset) ---
    amount_usd: float = Field(default=0.0, ge=0.0)
    jurisdiction: str = Field(default="US")
    customer_tenure_years: float = Field(default=0.0, ge=0.0)
    counterparty_risk: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Stress metadata ---
    stress_condition: StressCondition = StressCondition.S0_BASELINE
    original_risk_score: float | None = Field(
        default=None, description="Pre-stress risk score (for Δ tracking)"
    )
    original_completeness: float | None = Field(
        default=None, description="Pre-stress completeness (for Δ tracking)"
    )

    # --- Ground truth ---
    gt_decision: str | None = Field(
        default=None, description="Single expected decision for deterministic cases"
    )
    gt_decision_set: list[str] | None = Field(
        default=None, description="Set of acceptable decisions for ambiguous cases"
    )
    gt_confidence: GTConfidence | None = None
    gt_rationale: str | None = Field(default=None, description="Why this ground truth was assigned")
    gt_rule_id: str | None = Field(
        default=None, description="Which rule matched (for deterministic cases)"
    )

    # --- Generation metadata ---
    seed: int | None = None

    # --- Validators ---
    @field_validator("regulatory_flags")
    @classmethod
    def validate_flags(cls, v: list[str]) -> list[str]:
        for flag in v:
            if flag not in FLAG_UNIVERSE:
                raise ValueError(f"Unknown flag '{flag}'. Valid: {sorted(FLAG_UNIVERSE)}")
        return sorted(set(v))  # Deduplicate and sort for consistency

    # --- Serialization ---
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> BankingCase:
        """Create from dict."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> BankingCase:
        """Create from JSON string."""
        return cls.model_validate_json(json_str)

    def to_prompt(self) -> str:
        """Format case as text for LLM prompt injection.

        This is what the LLM sees when evaluating the case.
        Must include all decision-relevant information.
        """
        flags_str = ", ".join(self.regulatory_flags) if self.regulatory_flags else "None"
        return (
            f"Transaction Type: {self.transaction_type.value}\n"
            f"Risk Score: {self.risk_score:.3f}\n"
            f"Information Completeness: {self.completeness:.3f}\n"
            f"Regulatory Flags: {flags_str}\n"
            f"Amount (USD): ${self.amount_usd:,.2f}\n"
            f"Jurisdiction: {self.jurisdiction}\n"
            f"Customer Tenure: {self.customer_tenure_years:.1f} years\n"
            f"Counterparty Risk: {self.counterparty_risk:.3f}\n"
        )
