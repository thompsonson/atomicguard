"""Experiment discovery — scan artifact_dags/ for instances and arms."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InstanceInfo:
    """Summary info for one instance."""

    instance_id: str
    short_name: str
    arms: list[str]


@dataclass
class ExperimentSummary:
    """Top-level experiment statistics."""

    instance_count: int
    arm_names: list[str]
    total_dags: int
    completed_count: int


def parse_short_name(instance_id: str) -> str:
    """Extract a human-readable short name from a SWE-bench instance ID.

    ``instance_org__repo-commitsha-vhash``
    → ``org/repo @ commitsha[:8]``
    """
    # Strip leading "instance_" prefix if present
    name = instance_id
    if name.startswith("instance_"):
        name = name[len("instance_") :]

    # Try to match org__repo-commit pattern
    m = re.match(r"^([^_]+)__([^-]+)-([0-9a-f]{8})", name)
    if m:
        org, repo, commit_prefix = m.group(1), m.group(2), m.group(3)
        return f"{org}/{repo} @ {commit_prefix}"

    # Fallback: first 40 chars
    return instance_id[:40]


class ExperimentDiscovery:
    """Scan an artifact_dags directory to discover instances and arms."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def list_instances(self) -> list[str]:
        """Return sorted instance directory names."""
        if not self.root.is_dir():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def list_arms(self, instance: str) -> list[str]:
        """Return sorted arm names for an instance."""
        inst_dir = self.root / instance
        if not inst_dir.is_dir():
            return []
        return sorted(
            d.name
            for d in inst_dir.iterdir()
            if d.is_dir() and (d / "index.json").exists()
        )

    def get_dag_path(self, instance: str, arm: str) -> Path:
        return self.root / instance / arm

    def get_instance_infos(self) -> list[InstanceInfo]:
        """Return info objects for all instances."""
        result = []
        for inst_id in self.list_instances():
            arms = self.list_arms(inst_id)
            if arms:
                result.append(
                    InstanceInfo(
                        instance_id=inst_id,
                        short_name=parse_short_name(inst_id),
                        arms=arms,
                    )
                )
        return result

    def get_summary(self) -> ExperimentSummary:
        """Compute top-level experiment statistics."""
        instances = self.get_instance_infos()
        all_arms: set[str] = set()
        total_dags = 0
        completed = 0

        for inst in instances:
            for arm in inst.arms:
                all_arms.add(arm)
                total_dags += 1
                # Check if workflow succeeded
                index_path = self.root / inst.instance_id / arm / "index.json"
                if index_path.exists():
                    try:
                        idx = json.loads(index_path.read_text())
                        statuses = [
                            a.get("status") for a in idx.get("artifacts", {}).values()
                        ]
                        # Has at least one accepted artifact in each action pair
                        if statuses and any(s == "accepted" for s in statuses):
                            completed += 1
                    except (json.JSONDecodeError, OSError):
                        pass

        return ExperimentSummary(
            instance_count=len(instances),
            arm_names=sorted(all_arms),
            total_dags=total_dags,
            completed_count=completed,
        )
