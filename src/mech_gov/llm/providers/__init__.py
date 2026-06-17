# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""LLM provider implementations for mech_gov.

Providers translate the vendor-neutral :class:`mech_gov.llm.base.LLMInterface`
into concrete backends. The framework ships with dependency-free providers and
treats every cloud/vendor backend as optional and pluggable:

    mock                in-process, deterministic (tests, demos, CI)
    callable            wrap ANY user function -> bring your own backend
    openai_compatible   generic OpenAI-style HTTP endpoint (OpenAI, Azure,
                        vLLM, Ollama, Together, or an internal gateway)
    bedrock / sagemaker OPTIONAL — require `pip install mech-gov-framework[bedrock]`

Nothing in the core install imports a cloud SDK. Use the provider registry
(:mod:`mech_gov.llm.registry`) to build clients from config.
"""
