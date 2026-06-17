# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Quickstart: run R1 and R2 governance on one case using the offline mock LLM.

No credentials or network access required.

Run:
    python examples/quickstart_mock.py
"""

from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.governance.r1_text_only import R1TextOnly
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.llm.registry import create_llm


def main() -> None:
    llm = create_llm({"provider": "mock"})

    case = BankingCase(
        case_id="demo-0001",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.62,
        completeness=0.55,
        regulatory_flags=["KYC"],
        amount_usd=250_000.0,
        jurisdiction="US",
        customer_tenure_years=3.0,
        counterparty_risk=0.40,
    )

    for regime in (R1TextOnly(), R2Mechanical()):
        result = regime.process_case(case, llm)
        print(f"[{regime.regime_name}] {case.case_id} -> {result.decision.value}")
        print(f"    rationale: {result.rationale}")
        if result.gates_triggered:
            print(f"    gates triggered: {result.gates_triggered}")


if __name__ == "__main__":
    main()
