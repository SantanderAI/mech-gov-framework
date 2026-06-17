# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Bring your own LLM: wrap any function as a vendor-neutral client.

This is how you plug in a proprietary or internal inference backend without the
framework depending on — or revealing — which backend you use. Replace the body
of ``my_backend`` with a call to your own SDK, gateway, or local model.

Run:
    python examples/custom_provider.py
"""

import json

from mech_gov.data.banking_case import BankingCase, TransactionType
from mech_gov.governance.r2_mechanical import R2Mechanical
from mech_gov.llm.registry import create_llm


def my_backend(system_prompt, user_message, temperature=0.0, max_tokens=2048):
    """Return the model's raw text output.

    The governance regimes expect a JSON object with keys: ``decision``,
    ``rationale``, ``pro_arguments``, ``con_arguments`` (and optionally
    ``deferral_info_needed`` / ``conditions``). Here we return a fixed payload
    so the example runs offline; swap in a real backend call in practice.
    """
    return json.dumps(
        {
            "decision": "DEFER",
            "rationale": "Customer identity documents are incomplete for this jurisdiction.",
            "pro_arguments": [
                "The counterparty has a multi-year relationship with no prior "
                "adverse findings on record.",
            ],
            "con_arguments": [
                "Required identity verification documents for this jurisdiction "
                "are missing entirely here.",
            ],
            "deferral_info_needed": (
                "Provide certified identity documents valid in the customer's jurisdiction."
            ),
        }
    )


def main() -> None:
    # Via the registry (provider="callable"). Equivalent to:
    #   from mech_gov.llm.providers.callable_provider import CallableLLM
    #   llm = CallableLLM(my_backend)
    llm = create_llm({"provider": "callable", "callable": my_backend, "model_id": "my-backend"})

    case = BankingCase(
        case_id="demo-0002",
        transaction_type=TransactionType.SANCTIONS_SCREENING,
        risk_score=0.50,
        completeness=0.45,
        regulatory_flags=["KYC"],
    )
    result = R2Mechanical().process_case(case, llm)
    print(f"[{result.regime}] {case.case_id} -> {result.decision.value}")
    print(f"    rationale: {result.rationale}")


if __name__ == "__main__":
    main()
