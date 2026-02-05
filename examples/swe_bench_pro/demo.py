"""CLI for SWE-Bench Pro evaluation.

Usage:
    # Generate patches only (fast, no Docker needed)
    uv run python -m examples.swe_bench_pro.demo experiment --arms singleshot --max-instances 5 --provider openai

    # Full pipeline: patches → evaluation → visualization (requires Docker)
    uv run python -m examples.swe_bench_pro.demo experiment --arms singleshot --max-instances 5 --provider openai --evaluate

    # Re-run evaluation on existing predictions
    uv run python -m examples.swe_bench_pro.demo evaluate --predictions-dir output/swe_bench_pro/predictions

    # Re-generate visualizations
    uv run python -m examples.swe_bench_pro.demo visualize --results output/swe_bench_pro/results.jsonl
"""

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger("swe_bench_pro")

_DEFAULT_LOGGING_CONFIG = Path(__file__).parent / "logging.json"


def _configure_logging(debug: bool, log_file: str | None = None) -> None:
    """Set up logging, suppressing noisy third-party loggers.

    When *log_file* is set, detailed logs go to the file **and** a
    concise summary stream is kept on stderr so the user sees progress.

    The suppression list is loaded from ``logging.json`` next to this
    file.  The file should contain a JSON object with a
    ``"suppress_loggers"`` key listing logger names to set to WARNING::

        {
            "suppress_loggers": ["urllib3", "httpx", ...]
        }

    If the file is missing, a built-in default list is used.
    """
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    fmt_detailed = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fmt_concise = logging.Formatter("%(levelname)s - %(message)s")

    if log_file:
        # File gets everything.
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(level)
        fh.setFormatter(fmt_detailed)
        root.addHandler(fh)
        # Terminal gets INFO+ with a shorter format.
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt_concise)
        root.addHandler(sh)
    else:
        # No file — detailed format to stderr.
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt_detailed)
        root.addHandler(sh)

    suppress: list[str] = []
    if _DEFAULT_LOGGING_CONFIG.exists():
        try:
            data = json.loads(_DEFAULT_LOGGING_CONFIG.read_text())
            suppress = data.get("suppress_loggers", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse %s: %s", _DEFAULT_LOGGING_CONFIG, e)

    if not suppress:
        # Built-in fallback — only used if config file is missing.
        suppress = [
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "filelock",
            "datasets",
            "huggingface_hub",
            "fsspec",
        ]

    for name in suppress:
        logging.getLogger(name).setLevel(logging.WARNING)


def _report(msg: str, *, warn: bool = False, fg: str | None = None) -> None:
    """Log a message and echo it to the terminal.

    Use for key user-facing messages that should always appear on the
    terminal (with optional colour) *and* be recorded in the log file.
    """
    if warn:
        logger.warning(msg)
    else:
        logger.info(msg)
    styled = click.style(msg, fg=fg) if fg else msg
    click.echo(styled)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option(
    "--log-file",
    default=None,
    type=click.Path(),
    help="Write logs to file instead of stderr",
)
def cli(debug: bool, log_file: str | None) -> None:
    """SWE-Bench Pro evaluation CLI."""
    _configure_logging(debug, log_file=log_file)


# =========================================================================
# Helper functions
# =========================================================================


def _rewrite_results_with_resolved(
    results: list, output_dir: str
) -> None:
    """Rewrite results.jsonl with resolved status from evaluation.

    This updates the single source of truth (results.jsonl) to include
    the resolved field from evaluation results.
    """
    from dataclasses import asdict

    results_path = Path(output_dir) / "results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Rewrote %s with resolved status", results_path)


def _print_summary_with_resolved(results: list) -> None:
    """Print summary with resolved counts by arm."""
    from collections import defaultdict

    by_arm: dict[str, list] = defaultdict(list)
    for r in results:
        by_arm[r.arm].append(r)

    click.echo("\nSummary by arm:")
    for arm in sorted(by_arm.keys()):
        arm_results = by_arm[arm]
        total = len(arm_results)
        with_patch = sum(1 for r in arm_results if r.patch_content)
        resolved = sum(1 for r in arm_results if r.resolved)

        if with_patch > 0:
            resolve_rate = resolved / with_patch * 100
            click.echo(
                click.style(
                    f"  {arm}: {with_patch}/{total} patches, "
                    f"{resolved}/{with_patch} resolved ({resolve_rate:.1f}%)",
                    fg="green" if resolved > 0 else "yellow",
                )
            )
        else:
            click.echo(click.style(f"  {arm}: 0/{total} patches", fg="yellow"))


# =========================================================================
# experiment
# =========================================================================

_ARM_MAP = {
    "singleshot": "02_singleshot",
    "s1_direct": "03_s1_direct",
    "s1_tdd": "04_s1_tdd",
    "s1_tdd_verified": "05_s1_tdd_verified",
    "s1_tdd_behavior": "06_s1_tdd_behavior",
}


@cli.command()
@click.option(
    "--model",
    default="moonshotai/kimi-k2-0905",
    help="Model ID",
)
@click.option(
    "--provider",
    required=True,
    help="LLM provider (ollama, openrouter, huggingface, openai)",
)
@click.option(
    "--base-url",
    default="",
    help="API base URL (e.g. https://openrouter.ai/api/v1)",
)
@click.option(
    "--api-key",
    default=None,
    help="API key (default: from LLM_API_KEY env var)",
)
@click.option(
    "--arms",
    default="singleshot,s1_direct,s1_tdd",
    help="Comma-separated arm names",
)
@click.option(
    "--language",
    default=None,
    help="Language filter (python, go, javascript, typescript). None = all.",
)
@click.option(
    "--output-dir",
    default="output/swe_bench_pro",
    help="Directory for experiment results",
)
@click.option("--split", default="test", help="Dataset split")
@click.option("--max-instances", default=0, type=int, help="Max instances (0=all)")
@click.option(
    "--instances",
    default=None,
    help="Comma-separated instance ID substrings to include (e.g., 'openlibrary-798055d1,qutebrowser-e64622cd')",
)
@click.option(
    "--max-workers", default=1, type=int, help="Parallel workers (1=sequential)"
)
@click.option("--resume", is_flag=True, help="Resume from existing results")
@click.option(
    "--evaluate",
    is_flag=True,
    help="Run evaluation and visualization after generating patches",
)
@click.option(
    "--eval-max-workers",
    default=4,
    type=int,
    help="Parallel workers for evaluation (requires --evaluate)",
)
def experiment(
    model: str,
    provider: str,
    base_url: str,
    api_key: str | None,
    arms: str,
    language: str | None,
    output_dir: str,
    split: str,
    max_instances: int,
    instances: str | None,
    max_workers: int,
    resume: bool,
    evaluate: bool,
    eval_max_workers: int,
) -> None:
    """Run workflow arms across SWE-Bench Pro instances."""
    from .experiment_runner import SWEBenchProRunner

    raw_arms = [a.strip() for a in arms.split(",") if a.strip()]
    invalid = [a for a in raw_arms if a not in _ARM_MAP]
    if invalid:
        click.echo(
            click.style(
                f"Unknown arm name(s): {', '.join(invalid)}. "
                f"Valid arms: {', '.join(sorted(_ARM_MAP))}",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)

    arm_list = [_ARM_MAP[a] for a in raw_arms]

    # Parse instance filter
    instance_filter: list[str] | None = None
    if instances:
        instance_filter = [i.strip() for i in instances.split(",") if i.strip()]

    click.echo(f"Running SWE-Bench Pro experiment with model={model}")
    click.echo(f"Arms: {arm_list}")
    if language:
        click.echo(f"Language: {language}")
    if instance_filter:
        click.echo(f"Instance filter: {instance_filter}")
    click.echo(f"Output: {output_dir}")
    if max_workers > 1:
        click.echo(f"Parallel workers: {max_workers}")

    runner = SWEBenchProRunner(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        output_dir=output_dir,
    )
    results = runner.run_all(
        arms=arm_list,
        split=split,
        language=language,
        max_instances=max_instances if max_instances > 0 else None,
        resume_from=output_dir if resume else None,
        max_workers=max_workers,
        instance_filter=instance_filter,
    )

    # --- Summary & predictions ---
    from .evaluation import evaluate_predictions_inline, prepare_predictions

    total = len(results)
    with_patch = sum(1 for r in results if r.patch_content)
    errors = sum(1 for r in results if r.workflow_status == "error")

    _report(
        f"Done: {total} runs, {with_patch} with patches, "
        f"{total - with_patch - errors} empty, {errors} errors",
        fg="green" if with_patch > 0 else "yellow",
    )

    pred_files = prepare_predictions(results, output_dir)
    pred_dir = Path(output_dir) / "predictions"

    # Full pipeline when --evaluate is set (no fallbacks - failures propagate)
    if evaluate and pred_files and with_patch > 0:
        # Step 2: Evaluation
        _report("Running SWE-Bench Pro evaluation...", fg="cyan")
        resolved_map = evaluate_predictions_inline(
            pred_dir,
            eval_max_workers=eval_max_workers,
        )

        # Merge resolved status into results
        for r in results:
            key = (r.instance_id, r.arm)
            if key in resolved_map:
                r.resolved = resolved_map[key]

        # Rewrite results.jsonl with resolved status
        _rewrite_results_with_resolved(results, output_dir)

        # Step 3: Visualization
        _report("Generating visualizations...", fg="cyan")
        from examples.swe_bench_ablation.analysis import (
            generate_visualizations,
            load_results,
        )

        arm_results = load_results(str(Path(output_dir) / "results.jsonl"))

        # Build resolved map from results for visualization
        resolved_map_for_viz = {
            r.instance_id: r.resolved for r in results if r.resolved is not None
        }

        paths = generate_visualizations(
            arm_results, resolved_map_for_viz or None, output_dir
        )
        _report(f"Generated {len(paths)} visualizations", fg="green")

        # Print summary with resolved counts
        _print_summary_with_resolved(results)

    elif evaluate and with_patch == 0:
        _report(
            "Skipping evaluation (no patches to evaluate).",
            warn=True,
            fg="yellow",
        )
    elif with_patch > 0 and pred_files:
        _report(f"Predictions written to {pred_dir}/")
        for arm, path in pred_files.items():
            click.echo(f"  {arm}: {path}")
        eval_cmd = (
            "uv run python -m examples.swe_bench_pro.demo evaluate "
            f"--predictions-dir {pred_dir}"
        )
        _report(f"To evaluate, run:\n  {eval_cmd}")
    else:
        _report(
            "No predictions to evaluate (all patches empty or errored).",
            warn=True,
            fg="yellow",
        )


# =========================================================================
# evaluate
# =========================================================================


@cli.command()
@click.option(
    "--predictions-dir",
    required=True,
    help="Directory containing prediction JSON files",
)
@click.option(
    "--eval-mode",
    default="local",
    type=click.Choice(["local", "modal"]),
    help="Execution mode",
)
@click.option("--max-workers", default=4, type=int, help="Parallel workers")
@click.option("--timeout", default=7200, type=int, help="Timeout in seconds")
@click.option(
    "--eval-repo",
    default=None,
    help="Path to SWE-bench_Pro-os clone (auto-cloned if omitted)",
)
def evaluate(
    predictions_dir: str,
    eval_mode: str,
    max_workers: int,
    timeout: int,
    eval_repo: str | None,
) -> None:
    """Run SWE-Bench Pro evaluation on generated predictions."""
    from .evaluation import (
        ensure_eval_repo,
        load_evaluation_results,
        run_evaluation,
    )

    pred_dir = Path(predictions_dir)
    if not pred_dir.exists():
        click.echo(
            click.style(f"Predictions dir not found: {pred_dir}", fg="red"),
            err=True,
        )
        raise SystemExit(1)

    # Ensure eval repo is available
    if eval_repo:
        eval_repo_path = Path(eval_repo)
    else:
        click.echo("Cloning / updating SWE-Bench Pro eval repo...")
        eval_repo_path = ensure_eval_repo()
    click.echo(f"Eval repo: {eval_repo_path}")

    # Evaluate each prediction file
    for pred_file in sorted(pred_dir.glob("*.json")):
        arm = pred_file.stem
        click.echo(f"\nEvaluating arm={arm} from {pred_file}")

        result = run_evaluation(
            predictions_path=pred_file,
            eval_repo_path=eval_repo_path,
            mode=eval_mode,
            max_workers=max_workers,
            timeout=timeout,
        )

        if result["status"] == "success":
            resolved = load_evaluation_results(
                str(pred_file.parent),
                run_id=arm,
            )
            total = len(resolved)
            passed = sum(1 for v in resolved.values() if v)
            click.echo(
                click.style(
                    f"  {arm}: {passed}/{total} resolved",
                    fg="green" if passed > 0 else "yellow",
                )
            )
        else:
            click.echo(click.style(f"  {arm}: evaluation {result['status']}", fg="red"))


# =========================================================================
# visualize
# =========================================================================


@cli.command()
@click.option(
    "--results",
    default="output/swe_bench_pro/results.jsonl",
    help="Path to results.jsonl",
)
@click.option(
    "--resolved",
    default=None,
    help="Path to resolved.json (instance_id -> bool)",
)
@click.option(
    "--output-dir",
    default="output/swe_bench_pro",
    help="Directory for visualizations",
)
def visualize(results: str, resolved: str | None, output_dir: str) -> None:
    """Generate visualizations from experiment results."""
    from examples.swe_bench_ablation.analysis import (
        generate_visualizations,
        load_results,
    )

    click.echo(f"Loading results from {results}")
    arm_results = load_results(results)

    if not arm_results:
        click.echo(click.style("No results found", fg="red"), err=True)
        raise SystemExit(1)

    click.echo(f"Loaded {len(arm_results)} results")

    resolved_map: dict[str, bool] | None = None
    if resolved:
        resolved_path = Path(resolved)
        if not resolved_path.exists():
            click.echo(
                click.style(f"Resolved file not found: {resolved}", fg="red"),
                err=True,
            )
            raise SystemExit(1)
        resolved_map = json.loads(resolved_path.read_text())
        click.echo(f"Loaded {len(resolved_map or {})} resolved entries")

    paths = generate_visualizations(arm_results, resolved_map, output_dir)

    click.echo(click.style(f"Generated {len(paths)} visualizations:", fg="green"))
    for p in paths:
        click.echo(f"  {p}")


# =========================================================================
# list-instances
# =========================================================================


@cli.command("list-instances")
@click.option("--split", default="test", help="Dataset split")
def list_instances(split: str) -> None:
    """Show SWE-Bench Pro instance statistics by language."""
    from .dataset import load_swe_bench_pro

    instances = load_swe_bench_pro(split=split)

    by_lang: dict[str, int] = {}
    by_repo: dict[str, int] = {}
    for inst in instances:
        by_lang[inst.repo_language] = by_lang.get(inst.repo_language, 0) + 1
        by_repo[inst.repo] = by_repo.get(inst.repo, 0) + 1

    click.echo(f"SWE-Bench Pro ({split} split): {len(instances)} instances\n")

    click.echo("By language:")
    for lang in sorted(by_lang, key=by_lang.get, reverse=True):  # type: ignore[arg-type]
        click.echo(f"  {lang:15s} {by_lang[lang]:4d}")

    click.echo("\nBy repository:")
    for repo in sorted(by_repo, key=by_repo.get, reverse=True):  # type: ignore[arg-type]
        click.echo(f"  {repo:40s} {by_repo[repo]:4d}")


if __name__ == "__main__":
    cli()
