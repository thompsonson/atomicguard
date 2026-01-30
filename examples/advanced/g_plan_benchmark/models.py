"""
Plan representation for G_plan validation.

Bridges workflow.json structure to PDDL-style precondition/effect model
used by the G_plan guards. Each workflow step maps to a PlanStep with:
- preconditions: state tokens required before execution
- effects: state tokens produced on success
- dependencies: step IDs that must complete first (from workflow.json 'requires')
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlanStep:
    """Single step in a workflow plan (PDDL-style representation)."""

    step_id: str
    name: str
    generator: str
    guard: str
    retry_budget: int
    preconditions: set[str] = field(default_factory=set)
    effects: set[str] = field(default_factory=set)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "generator": self.generator,
            "guard": self.guard,
            "retry_budget": self.retry_budget,
            "preconditions": sorted(self.preconditions),
            "effects": sorted(self.effects),
            "dependencies": self.dependencies,
        }


@dataclass
class PlanDefinition:
    """
    Workflow plan as a DAG of steps.

    This is the artifact content that G_plan guards validate.
    It bridges workflow.json (generator/guard/requires) to the
    PDDL-style precondition/effect model used for semantic validation.
    """

    plan_id: str
    initial_state: set[str]
    goal_state: set[str]
    steps: list[PlanStep]
    total_retry_budget: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "initial_state": sorted(self.initial_state),
            "goal_state": sorted(self.goal_state),
            "total_retry_budget": self.total_retry_budget,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanDefinition:
        steps = []
        for s in data.get("steps", []):
            steps.append(
                PlanStep(
                    step_id=s["step_id"],
                    name=s.get("name", s["step_id"]),
                    generator=s.get("generator", ""),
                    guard=s.get("guard", ""),
                    retry_budget=s.get("retry_budget", 3),
                    preconditions=set(s.get("preconditions", [])),
                    effects=set(s.get("effects", [])),
                    dependencies=s.get("dependencies", []),
                )
            )
        return cls(
            plan_id=data.get("plan_id", "unknown"),
            initial_state=set(data.get("initial_state", [])),
            goal_state=set(data.get("goal_state", [])),
            steps=steps,
            total_retry_budget=data.get("total_retry_budget", 10),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PlanDefinition:
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_workflow_json(cls, workflow_path: Path | str) -> PlanDefinition:
        """
        Load a PlanDefinition from a workflow.json file.

        Bridges the workflow.json format to precondition/effect model:
        - Each step's guard_id becomes its effect token
        - Each step's 'requires' entries become precondition tokens
        - initial_state = {'intent_received'} (workflow entry point)
        - goal_state = effect tokens of terminal steps
        """
        path = Path(workflow_path)
        with open(path) as f:
            config = json.load(f)

        action_pairs = config.get("action_pairs", {})
        rmax = config.get("rmax", 3)

        steps = []
        all_step_ids = set(action_pairs.keys())
        # Track which steps are required by others (non-terminal)
        required_by_others: set[str] = set()

        for step_id, ap_config in action_pairs.items():
            requires = ap_config.get("requires", [])
            for r in requires:
                required_by_others.add(r)

            # Preconditions: if no requires, needs intent_received (entry point)
            if requires:
                preconditions = set(requires)
            else:
                preconditions = {"intent_received"}

            steps.append(
                PlanStep(
                    step_id=step_id,
                    name=ap_config.get("description", step_id),
                    generator=ap_config.get("generator", ""),
                    guard=ap_config.get("guard", ""),
                    retry_budget=ap_config.get("rmax", rmax),
                    preconditions=preconditions,
                    effects={step_id},  # Each step produces its own token
                    dependencies=requires,
                )
            )

        # Terminal steps = those not required by any other step
        terminal_steps = all_step_ids - required_by_others
        goal_state = terminal_steps if terminal_steps else all_step_ids

        total_budget = sum(s.retry_budget for s in steps)

        return cls(
            plan_id=config.get("name", path.stem),
            initial_state={"intent_received"},
            goal_state=goal_state,
            steps=steps,
            total_retry_budget=total_budget,
        )
