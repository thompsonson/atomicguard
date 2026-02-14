"""Discover experiments under an output directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class ExperimentEntry:
    """A single discovered experiment."""

    slug: str
    display_name: str
    artifact_dags_path: Path
    instance_count: int = 0
    arm_count: int = 0
    total_dags: int = 0
    completed_count: int = 0
    elapsed_seconds: float | None = None
    modified_at: datetime | None = None
    notes_path: Path | None = None


class ExperimentLocator:
    """Scan an output directory to discover experiments.

    Directory conventions:
      - ``output/{name}/artifact_dags/``           → single-run experiment
      - ``output/{name}/{model}/artifact_dags/``    → multi-model experiment
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def discover(self) -> list[ExperimentEntry]:
        """Return all discovered experiments sorted by slug."""
        entries: list[ExperimentEntry] = []
        if not self.output_dir.is_dir():
            return entries

        for child in sorted(self.output_dir.iterdir()):
            if not child.is_dir():
                continue

            direct = child / "artifact_dags"
            if direct.is_dir():
                # Single-run: output/{name}/artifact_dags/
                entry = self._build_entry(
                    slug=child.name,
                    display_name=child.name,
                    artifact_dags_path=direct,
                    meta_dir=child,
                )
                entries.append(entry)
            else:
                # Multi-model: output/{name}/{model}/artifact_dags/
                for sub in sorted(child.iterdir()):
                    if not sub.is_dir():
                        continue
                    sub_dags = sub / "artifact_dags"
                    if sub_dags.is_dir():
                        slug = f"{child.name}--{sub.name}"
                        display_name = f"{child.name} / {sub.name}"
                        entry = self._build_entry(
                            slug=slug,
                            display_name=display_name,
                            artifact_dags_path=sub_dags,
                            meta_dir=sub,
                        )
                        entries.append(entry)

        return entries

    def _build_entry(
        self,
        slug: str,
        display_name: str,
        artifact_dags_path: Path,
        meta_dir: Path,
    ) -> ExperimentEntry:
        entry = ExperimentEntry(
            slug=slug,
            display_name=display_name,
            artifact_dags_path=artifact_dags_path,
        )
        try:
            mtime = meta_dir.stat().st_mtime
            entry.modified_at = datetime.fromtimestamp(mtime, tz=UTC)
        except OSError:
            pass
        self._load_metadata(entry, meta_dir)
        if entry.instance_count == 0:
            self._count_from_filesystem(entry)

        # Discover NOTES.md — check meta_dir first, then parent (for
        # multi-model experiments where meta_dir is output/{name}/{model}
        # and the notes live at output/{name}/NOTES.md).
        for candidate in (meta_dir / "NOTES.md", meta_dir.parent / "NOTES.md"):
            if candidate.is_file():
                entry.notes_path = candidate
                break

        return entry

    def _load_metadata(self, entry: ExperimentEntry, meta_dir: Path) -> None:
        """Try loading pre-computed stats from summary JSON files."""
        for name in ("experiment_summary.json", "summary.json"):
            path = meta_dir / name
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    entry.instance_count = data.get(
                        "instance_count", entry.instance_count
                    )
                    entry.arm_count = data.get("arm_count", entry.arm_count)
                    entry.total_dags = data.get("total_dags", entry.total_dags)
                    entry.completed_count = data.get(
                        "completed_count", entry.completed_count
                    )
                    entry.elapsed_seconds = data.get("elapsed_seconds")
                    return
                except (json.JSONDecodeError, OSError):
                    pass

    def _count_from_filesystem(self, entry: ExperimentEntry) -> None:
        """Derive basic counts by scanning artifact_dags directory."""
        root = entry.artifact_dags_path
        if not root.is_dir():
            return
        arms: set[str] = set()
        instances = 0
        total_dags = 0
        for inst_dir in sorted(root.iterdir()):
            if not inst_dir.is_dir():
                continue
            inst_arms = [
                d.name
                for d in inst_dir.iterdir()
                if d.is_dir() and (d / "index.json").exists()
            ]
            if inst_arms:
                instances += 1
                total_dags += len(inst_arms)
                arms.update(inst_arms)
        entry.instance_count = instances
        entry.arm_count = len(arms)
        entry.total_dags = total_dags
