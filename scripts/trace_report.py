#!/usr/bin/env python3
"""Generate trace report from workflow execution (Extension 10).

Usage:
    python scripts/trace_report.py <dag_path> --workflow-id <id> [--format table|timeline]

Example:
    python scripts/trace_report.py output/artifact_dag/ --workflow-id abc-123 --format table
"""

import argparse
from pathlib import Path

from atomicguard.infrastructure.persistence.workflow_events import (
    FilesystemWorkflowEventStore,
)


def format_table(events: list) -> None:
    """Format events as a table."""
    print(
        f"{'#':>3} {'Event':20} {'Action Pair':20} {'Guard':15} "
        f"{'Verdict':8} {'Attempt':7} {'Details'}"
    )
    print("-" * 100)

    for i, event in enumerate(events, 1):
        guard = event.guard_name or "-"
        verdict = event.verdict or "-"
        attempt = str(event.attempt) if event.attempt else "-"
        summary = event.summary[:30] if event.summary else ""

        print(
            f"{i:>3} {event.event_type.value:20} {event.action_pair_id:20} "
            f"{guard:15} {verdict:8} {attempt:>7} {summary}"
        )


def format_timeline(events: list) -> None:
    """Format events as a timeline."""
    for event in events:
        timestamp = event.created_at[:19] if event.created_at else "?"
        symbol = {
            "STEP_START": "[>]",
            "STEP_PASS": "[+]",
            "STEP_FAIL": "[-]",
            "STAGNATION": "[!]",
            "ESCALATE": "[^]",
            "CASCADE_INVALIDATE": "[x]",
        }.get(event.event_type.value, "[?]")

        line = f"{timestamp} {symbol} {event.action_pair_id}"

        if event.guard_name:
            line += f" ({event.guard_name})"
        if event.verdict:
            line += f" -> {event.verdict}"
        if event.summary:
            line += f": {event.summary[:50]}"

        print(line)

        # Show escalation details
        if event.escalation:
            esc = event.escalation
            print(f"              Targets: {', '.join(esc.targets)}")
            print(f"              Invalidated: {', '.join(esc.invalidated)}")
            print(f"              e_count: {esc.e_count}/{esc.e_max}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate workflow trace report")
    parser.add_argument("dag_path", type=Path, help="Path to artifact DAG directory")
    parser.add_argument("--workflow-id", required=True, help="Workflow ID to report")
    parser.add_argument(
        "--format",
        choices=["table", "timeline"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    store = FilesystemWorkflowEventStore(args.dag_path)
    events = store.get_events(args.workflow_id)

    if not events:
        print(f"No events found for workflow {args.workflow_id}")
        return

    print(f"Workflow: {args.workflow_id}")
    print(f"Events: {len(events)}")
    print()

    if args.format == "table":
        format_table(events)
    else:
        format_timeline(events)

    # Summary statistics
    print()
    print("Summary:")
    print(f"  Total events: {len(events)}")
    print(
        f"  Steps started: {sum(1 for e in events if e.event_type.value == 'STEP_START')}"
    )
    print(
        f"  Steps passed: {sum(1 for e in events if e.event_type.value == 'STEP_PASS')}"
    )
    print(
        f"  Steps failed: {sum(1 for e in events if e.event_type.value == 'STEP_FAIL')}"
    )
    print(
        f"  Escalations: {sum(1 for e in events if e.event_type.value == 'ESCALATE')}"
    )


if __name__ == "__main__":
    main()
