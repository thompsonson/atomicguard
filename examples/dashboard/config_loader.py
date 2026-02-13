"""Load workflow configs and prompts for the config viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atomicguard.visualization.workflow_config_exporter import (
    _extract_graph,
    _extract_prompts,
)


class ConfigLoader:
    """Load workflow JSON files and merge with prompt data."""

    def __init__(
        self,
        workflows_dir: Path | None,
        prompts_path: Path | None,
    ) -> None:
        self.workflows_dir = workflows_dir
        self._prompts: dict[str, Any] = {}
        if prompts_path and prompts_path.exists():
            self._prompts = json.loads(prompts_path.read_text())

    def list_variants(self) -> list[str]:
        """Return sorted workflow variant names (without .json)."""
        if not self.workflows_dir or not self.workflows_dir.is_dir():
            return []
        return sorted(p.stem for p in self.workflows_dir.glob("*.json"))

    def get_config(self, variant: str) -> dict[str, Any] | None:
        """Load a workflow config by variant name."""
        if not self.workflows_dir:
            return None
        path = self.workflows_dir / f"{variant}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def get_config_graph(
        self, variant: str
    ) -> tuple[list[dict], list[dict], dict[str, dict[str, str]]] | None:
        """Extract Cytoscape nodes, edges, and prompt data for a variant.

        Returns (nodes, edges, prompt_data) or None if not found.
        """
        config = self.get_config(variant)
        if config is None:
            return None

        nodes, edges = _extract_graph(config)
        prompt_data = _extract_prompts(config, self._prompts)
        return nodes, edges, prompt_data
