"""
PlanGenerator: Deterministic generator that returns plan artifacts.

This is a deterministic generator (like RulesExtractorGenerator in sdlc_v2).
It loads pre-defined plans from the plan catalog and optionally injects defects.
No LLM is needed — the generator is deterministic, the guard is the variable under test.

For the benchmark, the CLI iterates over plan variants and defect types,
invoking this generator for each combination.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, Context, ContextSnapshot
from atomicguard.domain.prompts import PromptTemplate

from ..defects import DefectType, inject_defect
from ..models import PlanDefinition

logger = logging.getLogger("g_plan_benchmark")

PLANS_DIR = Path(__file__).parent.parent / "plans"


@dataclass
class PlanGeneratorConfig:
    """Configuration for PlanGenerator."""

    plan_source: str = "sdlc_v2"  # Plan variant to load from catalog
    defect: str | None = None  # Defect type to inject (None = valid plan)


class PlanGenerator(GeneratorInterface):
    """
    Deterministic generator that returns plan artifacts from a catalog.

    Follows the same pattern as RulesExtractorGenerator: no LLM call,
    deterministic output. The plan artifact is loaded from a JSON file
    in the plans/ directory.

    Optionally injects a named defect into the plan before returning it,
    enabling controlled benchmark runs.
    """

    config_class = PlanGeneratorConfig

    def __init__(
        self,
        config: PlanGeneratorConfig | None = None,
        plan_source: str = "sdlc_v2",
        defect: str | None = None,
        **_kwargs: Any,
    ):
        if config is None:
            config = PlanGeneratorConfig(plan_source=plan_source, defect=defect)
        self._plan_source = config.plan_source
        self._defect = config.defect
        self._version_counter = 0

    def set_plan_source(self, plan_source: str) -> None:
        """Update the plan source (for benchmark iteration)."""
        self._plan_source = plan_source

    def set_defect(self, defect: str | None) -> None:
        """Update the defect type (for benchmark iteration)."""
        self._defect = defect

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,  # noqa: ARG002
        action_pair_id: str = "g_plan",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,  # noqa: ARG002
    ) -> Artifact:
        """
        Generate a plan artifact from the catalog.

        The plan_source config selects which plan to load. If a defect
        is configured, it is injected before wrapping as an Artifact.
        """
        logger.debug(
            f"[PlanGenerator] Loading plan '{self._plan_source}' "
            f"(defect={self._defect})"
        )

        # Load plan from catalog
        plan_dict = self._load_plan(self._plan_source)

        # Optionally inject defect
        if self._defect:
            try:
                defect_type = DefectType(self._defect)
                plan_dict = inject_defect(plan_dict, defect_type)
                logger.debug(f"[PlanGenerator] Injected defect: {self._defect}")
            except ValueError:
                logger.warning(f"[PlanGenerator] Unknown defect type: {self._defect}")

        self._version_counter += 1
        content = json.dumps(plan_dict, indent=2)

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=self._version_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification,
                constraints=context.ambient.constraints,
                feedback_history=(),
                dependency_artifacts=context.dependency_artifacts,
            ),
        )

    def _load_plan(self, plan_source: str) -> dict[str, Any]:
        """Load a plan definition from the catalog."""
        # Try loading from plans/ directory
        plan_file = PLANS_DIR / f"{plan_source}.json"
        if plan_file.exists():
            with open(plan_file) as f:
                return json.load(f)

        # Try loading from a workflow.json path
        workflow_path = Path(plan_source)
        if workflow_path.exists() and workflow_path.name == "workflow.json":
            plan = PlanDefinition.from_workflow_json(workflow_path)
            return plan.to_dict()

        raise FileNotFoundError(
            f"Plan source not found: tried '{plan_file}' and '{plan_source}'"
        )
