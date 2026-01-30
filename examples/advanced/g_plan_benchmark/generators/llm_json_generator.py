"""
LLMJsonGenerator: Generic LLM-backed JSON generation.

Generates structured JSON output from an LLM, validated by a type-specific
guard. Used for pre-planning steps like problem analysis, reconnaissance,
and critique — any step where the LLM produces structured JSON rather than
a workflow plan.

Used in the classify-then-plan and full decomposed pipelines for the
g_analysis, g_recon, and g_strategy action pairs.

Shares all LLM infrastructure with LLMPlanGenerator (same backends,
same JSON extraction, same prompt rendering).
"""

from __future__ import annotations

from typing import Any

from .llm_plan_generator import LLMPlanGenerator, LLMPlanGeneratorConfig


class LLMJsonGeneratorConfig(LLMPlanGeneratorConfig):
    """Configuration for LLMJsonGenerator (same as LLMPlanGenerator)."""


class LLMJsonGenerator(LLMPlanGenerator):
    """
    Generic LLM-backed JSON generator.

    Calls an LLM with a structured prompt (role/constraints/task) and
    extracts JSON from the response. The prompt template and guard
    determine what kind of JSON is produced and validated.

    Identical in mechanics to LLMPlanGenerator — the difference is
    semantic: LLMJsonGenerator is for auxiliary JSON outputs (analysis,
    recon, critique) while LLMPlanGenerator is for workflow plans.

    Supports "ollama" and "huggingface" backends.
    """

    config_class = LLMJsonGeneratorConfig

    def __init__(
        self,
        config: LLMJsonGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            # Upcast to parent config type
            super().__init__(config=config)
        else:
            super().__init__(config=LLMJsonGeneratorConfig(**kwargs))
