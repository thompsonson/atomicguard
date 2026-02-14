# Plan B: AtomicGuardViewer — Separate TUI Package

## Overview

A standalone TUI application for exploring AtomicGuard experiment results. Reads from the filesystem output directory (results.jsonl, traces/, artifact_dags/) — fully decoupled from the experiment runner. Can be started before, during, or after an experiment.

**No dependency on atomicguard at runtime** — the viewer parses raw JSONL and JSON files directly.

---

## Package Location

Lives in the same repository as a separate package:

```
atomicguard/
├── src/atomicguard/              # existing core library
├── examples/                     # existing experiment runners
├── viewer/                       # NEW: separate package
│   ├── pyproject.toml
│   ├── README.md
│   └── src/
│       └── atomicguard_viewer/
│           ├── __init__.py
│           ├── __main__.py       # python -m atomicguard_viewer
│           ├── app.py            # Main Textual app
│           ├── data.py           # Data loading (no atomicguard import)
│           └── widgets/
│               ├── __init__.py
│               ├── arm_list.py
│               ├── instance_list.py
│               └── detail_panel.py
└── pyproject.toml                # existing
```

---

## Package Configuration

### `viewer/pyproject.toml`

```toml
[project]
name = "atomicguard-viewer"
version = "0.1.0"
description = "TUI explorer for AtomicGuard experiment results"
requires-python = ">=3.11"
dependencies = [
    "textual>=0.50.0",
    "rich>=13.0.0",
]

[project.scripts]
agviewer = "atomicguard_viewer.__main__:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]
```

### Installation

```bash
# Development install from repo root
cd viewer && pip install -e .

# Or directly
pip install -e ./viewer

# Then run
agviewer /path/to/output/swe_bench_pro
# or
python -m atomicguard_viewer /path/to/output/swe_bench_pro
```

---

## Data Layer

### `viewer/src/atomicguard_viewer/data.py`

Parses experiment output with NO atomicguard imports. All data comes from files.

#### Expected directory structure

```
output_dir/
├── results.jsonl                          # One JSON object per line (ArmResult)
├── summary.json                           # Optional summary stats
├── traces/<instance_id>/<arm>/*.jsonl     # Workflow events (Extension 10)
└── artifact_dags/<instance_id>/<arm>/     # Artifact storage
    ├── index.json
    └── artifacts/
```

#### Data models (plain dataclasses, no atomicguard dependency)

```python
@dataclass
class RunResult:
    """Parsed from one line of results.jsonl."""
    instance_id: str
    arm: str
    patch_content: str = ""
    total_tokens: int = 0
    per_step_tokens: dict[str, int] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    init_time_seconds: float = 0.0
    workflow_time_seconds: float = 0.0
    error: str | None = None
    resolved: bool | None = None
    failed_step: str | None = None
    failed_guard: str | None = None
    retry_count: int = 0

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.failed_step:
            return "failed"
        if self.resolved is True:
            return "resolved"
        if self.resolved is False:
            return "eval_failed"
        return "success"


@dataclass
class TraceEvent:
    """Parsed from workflow event JSONL."""
    event_id: str
    event_type: str  # STEP_START, STEP_PASS, STEP_FAIL, STAGNATION, ESCALATE, CASCADE_INVALIDATE
    action_pair_id: str
    workflow_id: str
    guard_name: str | None = None
    verdict: str | None = None
    attempt: int | None = None
    summary: str = ""
    created_at: str = ""
    # Escalation details (only for ESCALATE events)
    escalation_targets: tuple[str, ...] = ()
    escalation_invalidated: tuple[str, ...] = ()
    escalation_e_count: int = 0
    escalation_e_max: int = 0
    escalation_stagnant_guard: str | None = None


@dataclass
class ArmSummary:
    """Aggregated stats for one arm."""
    name: str
    total: int
    resolved: int
    failed: int
    errors: int
    avg_wall_time: float
    total_tokens: int
    failed_by_guard: dict[str, int]
    total_retries: int
```

#### Key functions

```python
def load_results(output_dir: Path) -> list[RunResult]:
    """Load all results from results.jsonl."""

def load_trace(output_dir: Path, instance_id: str, arm: str) -> list[TraceEvent]:
    """Load workflow trace events for a specific run."""

def load_artifact_content(output_dir: Path, instance_id: str, arm: str, step_id: str) -> str:
    """Load artifact content for a specific step (from artifact_dags)."""

def compute_arm_summaries(results: list[RunResult]) -> list[ArmSummary]:
    """Aggregate results by arm."""

def watch_results(output_dir: Path, callback: Callable, interval: float = 2.0) -> None:
    """Poll results.jsonl for changes and call callback with new results.
    Uses mtime comparison to detect changes."""
```

---

## TUI Layout

### `viewer/src/atomicguard_viewer/app.py`

```
┌─ AtomicGuard Viewer ─ output/swe_bench_pro ─────────────────┐
│                                                              │
│ ARMS                        │ INSTANCES for: s1_tdd          │
│ ────────────────────────────│────────────────────────────────│
│ ▶ singleshot  12/50 ████░░  │ ✓ django__django-15347         │
│   s1_direct    8/50 ███░░░  │ ⏳ flask__flask-4992            │
│   s1_tdd       5/50 ██░░░░  │ ✗ requests__requests-1234      │
│   s1_tdd_v     3/50 █░░░░░  │ ! sympy__sympy-20442 (error)   │
│                             │                                │
├─────────────────────────────┴────────────────────────────────┤
│ django__django-15347 | s1_tdd | 42.3s | 15,230 tokens       │
│                                                              │
│ [Trace] [Patch] [Error] [Artifact]                           │
│ ─────────────────────────────────────────────────────────── │
│ #  Event              Action Pair    Guard         Attempt   │
│ 1  STEP_START         ap_analysis    -             1         │
│ 2  STEP_PASS          ap_analysis    AnalysisGuard 1         │
│ 3  STEP_START         ap_gen_test    -             1         │
│ 4  STEP_FAIL          ap_gen_test    TestRedGuard  1         │
│ 5  STEP_START         ap_gen_test    -             2         │
│ 6  STAGNATION         ap_gen_test    TestRedGuard  -         │
│ 7  ESCALATE           ap_gen_test    -             e1/e1     │
│ 8  CASCADE_INVALIDATE ap_analysis    -             -         │
│ ...                                                          │
├──────────────────────────────────────────────────────────────┤
│ q:quit  ↑↓:navigate  Enter:select  Esc:back  Tab:panel      │
│ f:filter  r:refresh  /:search                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Widgets

### `widgets/arm_list.py` — ARM Panel

- Selectable list of arms
- Each row shows: name, progress (completed/total), mini progress bar, resolve rate
- Selecting an arm filters the instance list
- Color coding: green (>50% resolved), yellow (>0%), red (0%)

### `widgets/instance_list.py` — Instance Panel

- Filtered by selected arm
- Status icons: ✓ resolved, ✗ failed (guard name), ! error, ⏳ pending
- Sortable by: status, wall time, tokens, instance ID
- Search/filter by instance ID substring

### `widgets/detail_panel.py` — Detail Panel

Tabbed view for the selected instance+arm:

**Trace tab** (default):

- Chronological table of workflow events
- Color-coded by event type (green=pass, red=fail, yellow=stagnation, magenta=escalate)
- Expandable escalation details (targets, invalidated steps, failure summary)

**Patch tab**:

- Syntax-highlighted unified diff of the generated patch
- Shows "No patch" if workflow failed before patch generation

**Error tab**:

- Full guard feedback from the last failed attempt
- All retry attempts with guard feedback
- Escalation history if applicable

**Artifact tab**:

- Raw artifact content for each completed step
- JSON viewer for structured artifacts (analysis, localization)

---

## Entry Point

### `viewer/src/atomicguard_viewer/__main__.py`

```python
"""AtomicGuard Viewer — TUI for exploring experiment results.

Usage:
    agviewer /path/to/output/swe_bench_pro
    python -m atomicguard_viewer /path/to/output/swe_bench_pro
    agviewer /path/to/output --watch  # Live updates during experiment
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="TUI explorer for AtomicGuard experiment results"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Experiment output directory (containing results.jsonl)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Enable live file watching for updates during experiment",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="File watch refresh interval in seconds (default: 2.0)",
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Directory not found: {args.output_dir}")
        raise SystemExit(1)

    results_file = args.output_dir / "results.jsonl"
    if not results_file.exists():
        print(f"Error: No results.jsonl found in {args.output_dir}")
        print("Run an experiment first, or check the path.")
        raise SystemExit(1)

    from .app import AtomicGuardViewerApp

    app = AtomicGuardViewerApp(
        output_dir=args.output_dir,
        watch=args.watch,
        refresh_interval=args.refresh,
    )
    app.run()


if __name__ == "__main__":
    main()
```

---

## Implementation Phases

### Phase 1: Package scaffold + data layer

1. Create `viewer/` directory structure and `pyproject.toml`
2. Implement `data.py` with `load_results()`, `load_trace()`, `compute_arm_summaries()`
3. Implement `__main__.py` entry point
4. Verify: `pip install -e ./viewer && agviewer --help`

### Phase 2: Basic TUI with arm + instance panels

1. Create `app.py` with 3-panel layout using Textual
2. Implement `arm_list.py` — selectable list with progress bars
3. Implement `instance_list.py` — filtered by arm, status icons
4. Detail panel shows basic info (instance ID, status, timing)
5. Verify: navigate between arms and instances

### Phase 3: Detail panel tabs

1. Add Trace tab — workflow event table with color coding
2. Add Patch tab — syntax-highlighted diff display
3. Add Error tab — guard feedback and retry history
4. Add Artifact tab — raw artifact content viewer

### Phase 4: File watching + polish

1. Add `watch_results()` to data layer
2. Wire file watching into Textual app (periodic refresh)
3. Add keyboard shortcuts, search/filter
4. Add `--watch` flag support

---

## Testing

Unit tests for the data layer (no TUI testing needed initially):

```bash
cd viewer && python -m pytest tests/ -v
```

Tests should cover:

- `load_results()` parsing of results.jsonl
- `load_trace()` parsing of event JSONL
- `compute_arm_summaries()` aggregation
- Edge cases: empty files, missing directories, malformed JSON lines

Create test fixtures from actual experiment output (anonymized if needed).

---

## Dependencies

**atomicguard-viewer only:**

- `textual>=0.50.0` — TUI framework
- `rich>=13.0.0` — Terminal rendering (Textual dependency, also used standalone)

**NOT required:**

- `atomicguard` — viewer reads raw files, no library import needed
- `click` — uses argparse for simplicity
- `matplotlib`, `pydantic-ai`, etc. — none of the core deps

---

## File Summary

| File | Purpose |
|------|---------|
| `viewer/pyproject.toml` | Package config with textual + rich deps |
| `viewer/README.md` | Usage docs |
| `viewer/src/atomicguard_viewer/__init__.py` | Package init |
| `viewer/src/atomicguard_viewer/__main__.py` | CLI entry point |
| `viewer/src/atomicguard_viewer/app.py` | Main Textual app |
| `viewer/src/atomicguard_viewer/data.py` | Data loading from filesystem |
| `viewer/src/atomicguard_viewer/widgets/__init__.py` | Widget exports |
| `viewer/src/atomicguard_viewer/widgets/arm_list.py` | ARM list panel |
| `viewer/src/atomicguard_viewer/widgets/instance_list.py` | Instance list panel |
| `viewer/src/atomicguard_viewer/widgets/detail_panel.py` | Tabbed detail view |
