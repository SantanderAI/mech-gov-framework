# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Privacy gate: minimize PII before the model is consulted (R2).

The privacy gate reversibly tokenizes direct identifiers so the LLM never sees
raw personal data, and mechanically DEFERs a case whose identifiers cannot be
safely minimized (fail-closed). Decisions in mech_gov are driven by risk and
regulatory flags — not identities — so tokenization does not affect the outcome.

Run:
    python examples/privacy_demo.py
"""

import json

from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.governance.primitives.privacy_gate import (
    PrivacyConfig,
    RegexRecognizer,
    detokenize,
    privacy_gate,
)
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.llm.registry import create_llm


def _echo_backend(system_prompt, user_message, temperature=0.0, max_tokens=2048):
    """Offline backend that echoes the (already tokenized) prompt it received,
    proving the model only ever sees tokens — never raw identifiers."""
    return json.dumps(
        {
            "decision": "ESCALATE",
            "rationale": f"Reviewed the following material: {user_message}",
            "pro_arguments": [
                "The counterparty profile and documented controls could support "
                "approval under standard review conditions.",
            ],
            "con_arguments": [
                "Verified information remains insufficient to rule out elevated "
                "regulatory risk for this counterparty at this time.",
            ],
        }
    )


def demo_tokenization() -> None:
    print("=" * 70)
    print("1. Tokenization — the model never sees raw identifiers")
    print("=" * 70)
    text = "Analyst note: contact jane.doe@bank.example or call 555-123-4567."
    result = privacy_gate(text, PrivacyConfig(), RegexRecognizer())
    print(f"original : {text}")
    print(f"to model : {result.redacted_text}")
    print(f"entities : {result.entities_found}, residual: {result.residual_pii}")
    print(f"restored : {detokenize(result.redacted_text, result.token_map)}")


def demo_r2_tokenized_prompt() -> None:
    print()
    print("=" * 70)
    print("2. In R2 — the prompt is tokenized before the model is consulted")
    print("=" * 70)
    llm = create_llm({"provider": "callable", "callable": _echo_backend, "model_id": "echo"})
    case = BankingCase(
        case_id="privacy-demo-1",
        transaction_type=TransactionType.CREDIT_APPROVAL,
        risk_score=0.40,
        completeness=0.70,
        regulatory_flags=["KYC"],
        jurisdiction="reach jane.doe@bank.example",  # free-text carries PII
    )
    result = R2Mechanical().process_case(case, llm)
    leaked = "jane.doe@bank.example" in result.llm_raw_response
    print(f"decision           : {result.decision.value}")
    print(f"entities tokenized : {result.metadata['privacy_entities_found']}")
    print(f"raw email reached the model? {leaked}")


def demo_fail_safe_defer() -> None:
    print()
    print("=" * 70)
    print("3. Fail-safe — residual PII forces a mechanical DEFER (no model call)")
    print("=" * 70)
    calls = {"n": 0}

    def _counting_backend(system_prompt, user_message, temperature=0.0, max_tokens=2048):
        calls["n"] += 1
        return _echo_backend(system_prompt, user_message)

    llm = create_llm({"provider": "callable", "callable": _counting_backend, "model_id": "echo"})
    case = BankingCase(
        case_id="privacy-demo-2",
        transaction_type=TransactionType.AML_REVIEW,
        risk_score=0.40,
        completeness=0.70,
        regulatory_flags=["KYC"],
        jurisdiction="ref acct 12345678",  # residual identifier, not tokenizable
    )
    result = R2Mechanical().process_case(case, llm)
    print(f"decision    : {result.decision.value}")
    print(f"gates fired : {result.gates_triggered}")
    print(f"model calls : {calls['n']}")


def main() -> None:
    demo_tokenization()
    demo_r2_tokenized_prompt()
    demo_fail_safe_defer()


if __name__ == "__main__":
    main()
