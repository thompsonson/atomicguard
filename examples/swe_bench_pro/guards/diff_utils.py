"""Diff utilities for regenerating unified diffs from edits.

Provides functions to regenerate unified diffs from validated search/replace
edits, ensuring patches always apply cleanly.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("swe_bench_pro.guards.diff_utils")


def regenerate_diff_from_edits(
    edits: list[dict[str, str]],
    repo_root: str,
) -> str | None:
    """Regenerate unified diff from validated search/replace edits.

    This ensures the diff always applies cleanly by:
    1. Copying modified files to a temp directory
    2. Applying edits via search/replace
    3. Generating a fresh diff

    Args:
        edits: List of {"file": path, "search": str, "replace": str}
        repo_root: Path to the repository

    Returns:
        Unified diff string, or None if regeneration failed.
    """
    if not edits:
        return None

    repo_path = Path(repo_root)
    temp_dir = None

    try:
        # Create temp directory for modified files
        temp_dir = tempfile.mkdtemp(prefix="atomicguard_diff_")
        temp_path = Path(temp_dir)

        # Track which files we modify
        modified_files: set[str] = set()

        for edit in edits:
            file_path = edit.get("file", "")
            search = edit.get("search", "")
            replace = edit.get("replace", "")

            if not file_path or not search:
                continue

            src_file = repo_path / file_path
            if not src_file.exists():
                logger.warning(
                    "File not found for diff regeneration: %s", file_path
                )
                continue

            # Read original content
            try:
                original = src_file.read_text()
            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)
                continue

            # Check if search string exists
            if search not in original:
                logger.warning(
                    "Search string not found in %s during diff regeneration",
                    file_path,
                )
                continue

            # Check if search string is unique (multiple matches = ambiguous)
            match_count = original.count(search)
            if match_count > 1:
                logger.warning(
                    "Search string appears %d times in %s - skipping ambiguous edit",
                    match_count,
                    file_path,
                )
                continue

            # Apply the edit
            modified = original.replace(search, replace, 1)

            # Write to temp location (preserving directory structure)
            temp_file = temp_path / file_path
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(modified)

            # Also copy original for diff
            orig_dir = temp_path / "orig"
            orig_file = orig_dir / file_path
            orig_file.parent.mkdir(parents=True, exist_ok=True)
            orig_file.write_text(original)

            modified_files.add(file_path)

        if not modified_files:
            return None

        # Generate unified diff for each modified file
        diff_parts = []
        for file_path in sorted(modified_files):
            orig_file = temp_path / "orig" / file_path
            new_file = temp_path / file_path

            try:
                result = subprocess.run(
                    [
                        "diff",
                        "-u",
                        f"--label=a/{file_path}",
                        f"--label=b/{file_path}",
                        str(orig_file),
                        str(new_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                # diff returns 1 when files differ (which is expected)
                if result.returncode in (0, 1) and result.stdout:
                    diff_parts.append(result.stdout)
            except Exception as e:
                logger.warning("Failed to generate diff for %s: %s", file_path, e)

        if diff_parts:
            return "\n".join(diff_parts)
        return None

    except Exception as e:
        logger.warning("Failed to regenerate diff from edits: %s", e)
        return None

    finally:
        # Clean up temp directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
