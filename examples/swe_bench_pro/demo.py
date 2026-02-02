"""CLI for SWE-Bench Pro evaluation.

Usage:
    python -m examples.swe_bench_pro.demo experiment --arms singleshot --max-instances 5
    python -m examples.swe_bench_pro.demo evaluate --predictions-dir output/swe_bench_pro/predictions
    python -m examples.swe_bench_pro.demo visualize --results output/swe_bench_pro/results.jsonl
"""

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger("swe_bench_pro")


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """SWE-Bench Pro evaluation CLI."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# =========================================================================
# experiment
# =========================================================================

_ARM_MAP = {
    "singleshot": "02_singleshot",
    "s1_direct": "03_s1_direct",
    "s1_tdd": "04_s1_tdd",
}


@cli.command()
@click.option(
    "--model",
    default="Qwen/Qwen2.5-Coder-32B-Instruct",
    help="HuggingFace model ID",
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
@click.option("--max-workers", default=1, type=int, help="Parallel workers (1=sequential)")
@click.option("--resume", is_flag=True, help="Resume from existing results")
def experiment(
    model: str,
    arms: str,
    language: str | None,
    output_dir: str,
    split: str,
    max_instances: int,
    max_workers: int,
    resume: bool,
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

    click.echo(f"Running SWE-Bench Pro experiment with model={model}")
    click.echo(f"Arms: {arm_list}")
    if language:
        click.echo(f"Language: {language}")
    click.echo(f"Output: {output_dir}")
    if max_workers > 1:
        click.echo(f"Parallel workers: {max_workers}")

    runner = SWEBenchProRunner(model=model, output_dir=output_dir)
    runner.run_all(
        arms=arm_list,
        split=split,
        language=language,
        max_instances=max_instances if max_instances > 0 else None,
        resume_from=output_dir if resume else None,
        max_workers=max_workers,
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
            click.echo(
                click.style(f"  {arm}: evaluation {result['status']}", fg="red")
            )


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
    from examples.swe_bench_ablation.analysis import generate_visualizations, load_results

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
