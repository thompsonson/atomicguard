"""ISMIS 2026 multi-model experiment runner.

Orchestrates running 3 workflow arms across 13+ models on SWE-Bench Pro
for the ISMIS paper on decomposed backtracking.

Usage::

    from examples.swe_bench_pro.ismis_runner import ISMISExperimentRunner

    runner = ISMISExperimentRunner(output_dir="output/ismis_2026")
    runner.run_experiment()
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .experiment_runner import SWEBenchProRunner

logger = logging.getLogger("swe_bench_pro.ismis")


# =========================================================================
# Model Configuration
# =========================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model in the experiment."""

    model_id: str
    provider: str = "openrouter"
    output_mode: str = "tool"
    size_tier: str = ""
    model_family: str = ""
    label: str = ""

    @property
    def short_name(self) -> str:
        """Short name for file paths and display."""
        return self.label or self.model_id.split("/")[-1]


# ISMIS 2026 model roster — 13+ models across size tiers and families.
# Output mode: "tool" = function calling (default), "prompted" = text-based JSON.
ISMIS_MODELS: list[ModelConfig] = [
    # Tool Output models (function calling)
    ModelConfig(
        model_id="deepseek/deepseek-chat-v3-0324",
        provider="openrouter",
        output_mode="tool",
        size_tier="671B_MoE",
        model_family="deepseek",
        label="deepseek-v3",
    ),
    ModelConfig(
        model_id="qwen/qwen-2.5-coder-32b-instruct",
        provider="openrouter",
        output_mode="tool",
        size_tier="32B",
        model_family="qwen",
        label="qwen-coder-32b",
    ),
    ModelConfig(
        model_id="qwen/qwen-2.5-coder-7b-instruct",
        provider="openrouter",
        output_mode="tool",
        size_tier="7B",
        model_family="qwen",
        label="qwen-coder-7b",
    ),
    ModelConfig(
        model_id="openai/gpt-4o-mini",
        provider="openrouter",
        output_mode="tool",
        size_tier="proprietary",
        model_family="openai",
        label="gpt-4o-mini",
    ),
    ModelConfig(
        model_id="anthropic/claude-3.5-haiku",
        provider="openrouter",
        output_mode="tool",
        size_tier="proprietary",
        model_family="anthropic",
        label="claude-haiku",
    ),
    ModelConfig(
        model_id="google/gemini-2.0-flash-001",
        provider="openrouter",
        output_mode="tool",
        size_tier="proprietary",
        model_family="google",
        label="gemini-flash",
    ),
    ModelConfig(
        model_id="mistralai/mistral-large-latest",
        provider="openrouter",
        output_mode="tool",
        size_tier="123B",
        model_family="mistral",
        label="mistral-large",
    ),
    ModelConfig(
        model_id="meta-llama/llama-3.1-70b-instruct",
        provider="openrouter",
        output_mode="tool",
        size_tier="70B",
        model_family="meta",
        label="llama-70b",
    ),
    # Prompted Output models (text-based JSON)
    ModelConfig(
        model_id="meta-llama/llama-3.1-8b-instruct",
        provider="openrouter",
        output_mode="prompted",
        size_tier="8B",
        model_family="meta",
        label="llama-8b",
    ),
    ModelConfig(
        model_id="microsoft/phi-4",
        provider="openrouter",
        output_mode="prompted",
        size_tier="14B",
        model_family="microsoft",
        label="phi-4",
    ),
    ModelConfig(
        model_id="mistralai/codestral-latest",
        provider="openrouter",
        output_mode="prompted",
        size_tier="22B",
        model_family="mistral",
        label="codestral",
    ),
    ModelConfig(
        model_id="deepseek/deepseek-coder",
        provider="openrouter",
        output_mode="prompted",
        size_tier="33B",
        model_family="deepseek",
        label="deepseek-coder",
    ),
    ModelConfig(
        model_id="qwen/qwen-2.5-72b-instruct",
        provider="openrouter",
        output_mode="prompted",
        size_tier="72B",
        model_family="qwen",
        label="qwen-72b",
    ),
]


# =========================================================================
# Experiment arms — three levels of decomposition
# =========================================================================


ISMIS_ARMS: list[str] = [
    "02_singleshot",  # Baseline: 1 step (problem → patch)
    "04_s1_tdd",  # S1-TDD: 3 steps (analysis → test → patch)
    "08_s1_decomposed_lite",  # S1-Decomposed: 5 steps (analysis → localise → fix → test → patch)
]


# =========================================================================
# Runner
# =========================================================================


@dataclass
class ModelRunSummary:
    """Summary of results for one model."""

    model_id: str
    label: str
    output_mode: str
    size_tier: str
    model_family: str
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class ISMISExperimentRunner:
    """Multi-model experiment runner for ISMIS 2026.

    Iterates over models and arms, delegating each model's runs
    to :class:`SWEBenchProRunner`.
    """

    def __init__(
        self,
        output_dir: str = "output/ismis_2026",
        clone_dir: str | None = None,
        models: list[ModelConfig] | None = None,
        arms: list[str] | None = None,
        api_key: str | None = None,
    ):
        self._output_dir = Path(output_dir)
        self._clone_dir = clone_dir
        self._models = models or ISMIS_MODELS
        self._arms = arms or ISMIS_ARMS
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

        if not self._api_key:
            self._api_key = os.environ.get("LLM_API_KEY", "")

    @staticmethod
    def configure_logging(
        output_dir: str | Path,
        debug: bool = False,
    ) -> Path:
        """Set up dual logging: detailed file + concise stderr.

        Args:
            output_dir: Directory for the log file.
            debug: Enable DEBUG level (default INFO).

        Returns:
            Path to the log file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "experiment.log"

        level = logging.DEBUG if debug else logging.INFO
        root = logging.getLogger()
        root.setLevel(level)

        # File handler — detailed format, all levels
        fh = logging.FileHandler(str(log_file), mode="a")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(fh)

        # Stderr handler — concise format, INFO+
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        root.addHandler(sh)

        # Suppress noisy third-party loggers
        suppress_config = Path(__file__).parent / "logging.json"
        suppress: list[str] = []
        if suppress_config.exists():
            try:
                data = json.loads(suppress_config.read_text())
                suppress = data.get("suppress_loggers", [])
            except (json.JSONDecodeError, KeyError):
                pass
        if not suppress:
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

        logger.info("Logging to %s", log_file)
        return log_file

    def run_experiment(
        self,
        language: str = "python",
        max_instances: int | None = None,
        instance_filter: list[str] | None = None,
        max_workers: int = 1,
        models_filter: list[str] | None = None,
        arms_filter: list[str] | None = None,
    ) -> list[ModelRunSummary]:
        """Run the full ISMIS experiment across all models and arms.

        Args:
            language: Language filter for SWE-Bench Pro instances.
            max_instances: Cap on instances per model.
            instance_filter: Instance ID substrings to include.
            max_workers: Parallel workers per model run.
            models_filter: If provided, only run models whose label
                contains one of these substrings.
            arms_filter: If provided, only run these arm names.

        Returns:
            List of per-model summaries.
        """
        # Set up dual logging (file + stderr) if not already configured
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            self.configure_logging(self._output_dir)

        models = self._models
        arms = self._arms

        if models_filter:
            models = [
                m for m in models if any(f in m.short_name for f in models_filter)
            ]
        if arms_filter:
            arms = [a for a in arms if a in arms_filter]

        total_models = len(models)
        total_runs = total_models * len(arms) * (max_instances or 30)
        logger.info(
            "ISMIS Experiment: %d models x %d arms x %d instances = ~%d runs",
            total_models,
            len(arms),
            max_instances or 30,
            total_runs,
        )

        summaries: list[ModelRunSummary] = []
        experiment_start = time.time()

        for i, model_config in enumerate(models, 1):
            model_start = time.time()
            logger.info(
                "=" * 60 + "\nModel %d/%d: %s (%s, output_mode=%s)\n" + "=" * 60,
                i,
                total_models,
                model_config.model_id,
                model_config.size_tier,
                model_config.output_mode,
            )

            # Per-model output directory
            model_output = self._output_dir / model_config.short_name
            model_output.mkdir(parents=True, exist_ok=True)

            # Save model config for reproducibility
            config_path = model_output / "model_config.json"
            config_path.write_text(json.dumps(asdict(model_config), indent=2))

            runner = SWEBenchProRunner(
                model=model_config.model_id,
                provider=model_config.provider,
                base_url="",
                api_key=self._api_key,
                output_dir=str(model_output),
                clone_dir=self._clone_dir,
                output_mode=model_config.output_mode,
            )

            try:
                results = runner.run_all(
                    arms=arms,
                    language=language,
                    max_instances=max_instances,
                    instance_filter=instance_filter,
                    resume_from=str(model_output)
                    if (model_output / "results.jsonl").exists()
                    else None,
                    max_workers=max_workers,
                )
            except Exception:
                logger.exception("Model %s failed entirely", model_config.model_id)
                results = []

            # Evaluate patches against SWE-Bench Pro benchmark
            from .evaluation import evaluate_predictions_inline, prepare_predictions

            with_patch = sum(
                1
                for r in results
                if r.patch_content and not r.error and not r.failed_step
            )
            if with_patch > 0:
                pred_files = prepare_predictions(results, str(model_output))
                pred_dir = model_output / "predictions"
                if pred_files:
                    logger.info(
                        "Evaluating %d patches against SWE-Bench Pro...",
                        with_patch,
                    )
                    resolved_map = evaluate_predictions_inline(pred_dir)
                    for r in results:
                        key = (r.instance_id, r.arm)
                        if key in resolved_map:
                            r.resolved = resolved_map[key]
                    # Rewrite results.jsonl with resolved status
                    results_path = model_output / "results.jsonl"
                    with open(results_path, "w") as f:
                        for r in results:
                            f.write(json.dumps(asdict(r)) + "\n")
                    logger.info("Rewrote %s with resolved status", results_path)

            # Build per-model summary
            model_elapsed = time.time() - model_start
            summary = self._build_model_summary(model_config, results, model_elapsed)
            summaries.append(summary)

            # Save per-model summary
            summary_path = model_output / "model_summary.json"
            summary_path.write_text(json.dumps(asdict(summary), indent=2))

            logger.info(
                "Model %s complete in %.0fs. Results: %s",
                model_config.short_name,
                model_elapsed,
                {
                    arm: data.get("eval_resolved", 0)
                    for arm, data in summary.arms.items()
                },
            )

        # Save experiment-wide summary
        experiment_elapsed = time.time() - experiment_start
        self._save_experiment_summary(summaries, experiment_elapsed)

        return summaries

    def _build_model_summary(
        self,
        config: ModelConfig,
        results: list,
        elapsed: float,
    ) -> ModelRunSummary:
        """Build summary for a single model's results."""
        from collections import defaultdict

        summary = ModelRunSummary(
            model_id=config.model_id,
            label=config.short_name,
            output_mode=config.output_mode,
            size_tier=config.size_tier,
            model_family=config.model_family,
            elapsed_seconds=elapsed,
        )

        by_arm: dict[str, list] = defaultdict(list)
        for r in results:
            by_arm[r.arm].append(r)

        for arm, arm_results in by_arm.items():
            n = len(arm_results)
            patches_generated = sum(
                1
                for r in arm_results
                if r.patch_content and not r.error and not r.failed_step
            )
            resolved = sum(1 for r in arm_results if r.resolved is True)
            errors = sum(1 for r in arm_results if r.error)
            guard_failures = sum(
                1 for r in arm_results if r.failed_step and not r.error
            )
            total_tokens = sum(r.total_tokens for r in arm_results)
            total_wall = sum(r.wall_time_seconds for r in arm_results)

            summary.arms[arm] = {
                "n": n,
                "patches_generated": patches_generated,
                "eval_resolved": resolved,
                "resolution_rate": round(resolved / n, 4) if n else 0.0,
                "workflow_success_rate": round(patches_generated / n, 4) if n else 0.0,
                "errors": errors,
                "guard_failures": guard_failures,
                "total_tokens": total_tokens,
                "mean_tokens": round(total_tokens / n) if n else 0,
                "mean_wall_time": round(total_wall / n, 1) if n else 0.0,
            }

        return summary

    def _save_experiment_summary(
        self,
        summaries: list[ModelRunSummary],
        elapsed: float,
    ) -> None:
        """Save experiment-wide summary with cross-model comparison."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        experiment_data: dict[str, Any] = {
            "experiment": "ISMIS 2026 — 13-Model Evaluation",
            "arms": self._arms,
            "total_models": len(summaries),
            "elapsed_seconds": round(elapsed, 2),
            "models": [asdict(s) for s in summaries],
        }

        # Cross-model comparison table
        comparison: list[dict[str, Any]] = []
        for s in summaries:
            row: dict[str, Any] = {
                "model": s.label,
                "size_tier": s.size_tier,
                "output_mode": s.output_mode,
            }
            for arm in self._arms:
                arm_data = s.arms.get(arm, {})
                arm_short = arm.replace("_", "").replace("0", "")
                row[f"{arm_short}_resolved"] = arm_data.get("eval_resolved", 0)
                row[f"{arm_short}_rate"] = arm_data.get("resolution_rate", 0.0)
                row[f"{arm_short}_tokens"] = arm_data.get("mean_tokens", 0)
            comparison.append(row)

        experiment_data["comparison_table"] = comparison

        summary_path = self._output_dir / "experiment_summary.json"
        summary_path.write_text(json.dumps(experiment_data, indent=2))
        logger.info("Experiment summary written to %s", summary_path)
