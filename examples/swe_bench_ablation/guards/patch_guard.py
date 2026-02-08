"""PatchGuard: Validates patch format and application.

Ensures the patch:
1. Has valid format (unified diff or search-replace edits)
2. Applies cleanly with `git apply --check`
3. Results in syntactically valid Python
"""

import ast
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")


class PatchGuard(GuardInterface):
    """Validates patch output.

    Checks:
    - Valid JSON with patch or edits
    - Patch applies cleanly (git apply --check)
    - Modified Python files have valid syntax
    """

    def __init__(
        self,
        require_git_apply: bool = True,
        require_syntax_valid: bool = True,
        repo_root: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard.

        Args:
            require_git_apply: Check patch applies with git
            require_syntax_valid: Check Python syntax after patch
            repo_root: Repository root for validation
        """
        self._require_git_apply = require_git_apply
        self._require_syntax_valid = require_syntax_valid
        self._repo_root = repo_root

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the patch artifact.

        Args:
            artifact: The patch artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info("[PatchGuard] Validating artifact %s...", artifact.artifact_id[:8])

        # Parse JSON
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="PatchGuard",
            )

        # Check for error field
        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="PatchGuard",
            )

        # Get patch content
        patch_content = data.get("patch", "")
        edits = data.get("edits", [])

        if not patch_content and not edits:
            return GuardResult(
                passed=False,
                feedback="No patch or edits found in output",
                guard_name="PatchGuard",
            )

        # Track errors separately for better feedback
        format_errors: list[str] = []
        edit_errors: list[str] = []
        apply_errors: list[str] = []

        # Check for identical search/replace edits (no-op)
        if edits and not patch_content:
            if all(
                edit.get("search", "") == edit.get("replace", "")
                for edit in edits
            ):
                edit_errors.append(
                    "All edits have identical search and replace strings (no actual changes)"
                )

        # Check that all edited files exist in the repo
        if edits and self._repo_root:
            missing_files = []
            for edit in edits:
                file_path = edit.get("file", "")
                full_path = Path(self._repo_root) / file_path
                if not full_path.exists():
                    missing_files.append(file_path)
            if missing_files:
                if len(missing_files) <= 3:
                    edit_errors.append(
                        f"Files not found in repository: {', '.join(missing_files)}"
                    )
                else:
                    edit_errors.append(
                        f"Files not found in repository: {', '.join(missing_files[:3])} "
                        f"and {len(missing_files) - 3} more"
                    )

        # Validate unified diff format if present
        if patch_content:
            format_errors = self._validate_diff_format(patch_content)

        # Check Python syntax if we have edits and repo root (validates SEARCH/REPLACE)
        if self._require_syntax_valid and edits and self._repo_root and not edit_errors:
            syntax_errors = self._check_syntax(edits, self._repo_root)
            edit_errors.extend(syntax_errors)

        # Check git apply if we have a patch and repo root (validates unified diff)
        if self._require_git_apply and patch_content and self._repo_root and not format_errors:
            apply_errors = self._check_git_apply(patch_content, self._repo_root)

        # Build feedback based on what failed
        all_errors = format_errors + edit_errors + apply_errors

        if all_errors:
            feedback = self._build_failure_feedback(
                format_errors=format_errors,
                edit_errors=edit_errors,
                apply_errors=apply_errors,
                edits=edits,
            )
            logger.info("[PatchGuard] ✗ REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="PatchGuard",
            )

        edit_count = len(edits) if edits else 0
        patch_lines = len(patch_content.split("\n")) if patch_content else 0
        feedback = f"Patch is valid: {edit_count} edits, {patch_lines} diff lines"
        logger.info("[PatchGuard] ✓ PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="PatchGuard",
        )

    def _build_failure_feedback(
        self,
        format_errors: list[str],
        edit_errors: list[str],
        apply_errors: list[str],
        edits: list[dict[str, str]],
    ) -> str:
        """Build detailed feedback distinguishing edit vs diff application failures."""
        sections: list[str] = []

        # Case 1: Format errors (basic diff structure issues)
        if format_errors:
            sections.append("DIFF FORMAT ERRORS:\n- " + "\n- ".join(format_errors))

        # Case 2: Edit errors (search/replace issues)
        if edit_errors:
            sections.append("EDIT VALIDATION FAILED:\n- " + "\n- ".join(edit_errors))

        # Case 3: Edits valid but git apply failed (context mismatch)
        if apply_errors and not edit_errors:
            sections.append(
                "EDIT VALIDATION PASSED:\n"
                f"  ✓ {len(edits)} edit(s) have valid search strings and syntax\n\n"
                "DIFF APPLICATION FAILED:\n- " + "\n- ".join(apply_errors) + "\n\n"
                "The SEARCH/REPLACE edits are correct, but the unified diff has "
                "incorrect context lines.\n"
                "Regenerate the unified diff with accurate context (the unchanged "
                "lines surrounding your changes must match the actual file)."
            )
        elif apply_errors:
            # Both edit and apply errors
            sections.append("DIFF APPLICATION FAILED:\n- " + "\n- ".join(apply_errors))

        return "\n\n".join(sections)

    def _validate_diff_format(self, patch_content: str) -> list[str]:
        """Validate unified diff format."""
        errors: list[str] = []

        if not patch_content.strip():
            errors.append("Empty patch")
            return errors

        lines = patch_content.split("\n")

        # Check for required diff markers
        has_minus_file = any(line.startswith("---") for line in lines)
        has_plus_file = any(line.startswith("+++") for line in lines)
        has_hunk_header = any(line.startswith("@@") for line in lines)

        if not has_minus_file:
            errors.append("Missing '---' file marker in diff")
        if not has_plus_file:
            errors.append("Missing '+++' file marker in diff")
        if not has_hunk_header:
            errors.append("Missing '@@ ... @@' hunk header in diff")

        # Check for actual changes (not just headers)
        has_changes = any(
            (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
            for line in lines
        )
        if not has_changes and not errors:
            errors.append("Patch has no actual changes (no added or removed lines)")

        return errors

    def _check_git_apply(self, patch_content: str, repo_root: str) -> list[str]:
        """Check if patch applies cleanly with git."""
        errors: list[str] = []

        # Get current HEAD for debugging
        try:
            head_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            current_head = head_result.stdout.strip() if head_result.returncode == 0 else "unknown"
        except Exception:
            current_head = "unknown"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch_content)
            patch_file = f.name

        try:
            result = subprocess.run(
                ["git", "apply", "--check", patch_file],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                if stderr:
                    errors.append(
                        f"git apply --check failed (HEAD={current_head}): {stderr}"
                    )
                else:
                    errors.append(
                        f"git apply --check failed (HEAD={current_head}, no error message)"
                    )

        except subprocess.TimeoutExpired:
            errors.append("git apply --check timed out")
        except FileNotFoundError:
            errors.append("git command not found")
        except Exception as e:
            errors.append(f"git apply check error: {e}")
        finally:
            Path(patch_file).unlink(missing_ok=True)

        return errors

    def _check_syntax(
        self,
        edits: list[dict[str, str]],
        repo_root: str,
    ) -> list[str]:
        """Check Python syntax after applying edits."""
        errors: list[str] = []

        for edit in edits:
            file_path = edit.get("file", "")
            search = edit.get("search", "")
            replace = edit.get("replace", "")

            if not file_path.endswith(".py"):
                continue

            full_path = Path(repo_root) / file_path
            if not full_path.exists():
                continue

            try:
                original = full_path.read_text()

                # Apply the edit
                if search not in original:
                    preview = search[:200].replace('\n', '\\n')
                    # Find similar content to help the generator
                    similar = self._find_similar_content(original, search, file_path)
                    errors.append(
                        f"Search string not found in {file_path}.\n"
                        f"You searched for:\n{preview!r}\n\n"
                        f"{similar}"
                    )
                    continue

                modified = original.replace(search, replace, 1)

                # Check syntax
                try:
                    ast.parse(modified)
                except SyntaxError as e:
                    errors.append(
                        f"Syntax error in {file_path} after patch: "
                        f"line {e.lineno}: {e.msg}"
                    )

            except Exception as e:
                logger.warning(f"Error checking syntax for {file_path}: {e}")

        return errors

    def _find_similar_content(
        self,
        file_content: str,
        search_string: str,
        file_path: str,
    ) -> str:
        """Find similar content in file to help generator correct its search string.

        Uses the first line of the search string to locate approximate position,
        then shows surrounding context.
        """
        lines = file_content.split("\n")
        search_lines = search_string.split("\n")

        if not search_lines:
            return f"File {file_path} has {len(lines)} lines."

        # Get first non-empty line of search to use as anchor
        first_search_line = ""
        for line in search_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_search_line = stripped
                break

        if not first_search_line:
            # All comment lines or empty - just show file structure
            return self._show_file_structure(lines, file_path)

        # Find best matching line in file
        best_match_idx = -1
        best_match_score = 0

        for i, line in enumerate(lines):
            score = self._line_similarity(line.strip(), first_search_line)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i

        if best_match_score < 0.3:
            # No good match - show file structure instead
            return self._show_file_structure(lines, file_path)

        # Show context around the best match
        start = max(0, best_match_idx - 3)
        end = min(len(lines), best_match_idx + len(search_lines) + 3)

        context_lines = []
        for i in range(start, end):
            prefix = ">>>" if i == best_match_idx else "   "
            context_lines.append(f"{prefix} {i + 1:4d}: {lines[i]}")

        match_quality = "exact" if best_match_score > 0.9 else "similar"
        return (
            f"Found {match_quality} content at line {best_match_idx + 1}:\n"
            + "\n".join(context_lines)
            + "\n\nUse the EXACT content shown above as your search string."
        )

    def _line_similarity(self, line1: str, line2: str) -> float:
        """Similarity score between two lines (0-1), weighted for code patterns."""
        if not line1 or not line2:
            return 0.0
        if line1 == line2:
            return 1.0

        # Extract function/class name if present - check this FIRST
        def extract_name(line: str) -> str | None:
            for prefix in ("def ", "async def ", "class "):
                if line.startswith(prefix):
                    rest = line[len(prefix):]
                    # Get the name before ( or :
                    name = rest.split("(")[0].split(":")[0].strip()
                    return name
            return None

        name1 = extract_name(line1)
        name2 = extract_name(line2)

        # If both are function/class definitions, compare names
        if name1 and name2:
            if name1 == name2:
                return 0.95  # Same name, different signature - highest priority
            # Partial match only if one is a prefix of the other
            if name2.startswith(name1) or name1.startswith(name2):
                return 0.6  # e.g., validate vs validate_patch
            return 0.1  # Different function names

        # If searching for a def/class, only match other defs/classes
        if name2 and not name1:
            return 0.0  # Searching for function but this line isn't one

        # Check for substring match - weight by length of match
        if line1 in line2:
            return 0.5 + 0.3 * (len(line1) / len(line2))
        if line2 in line1:
            return 0.5 + 0.3 * (len(line2) / len(line1))

        # Count common words
        words1 = set(line1.split())
        words2 = set(line2.split())
        if not words1 or not words2:
            return 0.0

        common = len(words1 & words2)
        total = len(words1 | words2)
        return common / total if total > 0 else 0.0

    def _show_file_structure(self, lines: list[str], file_path: str) -> str:
        """Show file structure when no similar content found."""
        # Find function/class definitions
        definitions = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "async def ")):
                definitions.append(f"  {i + 1:4d}: {stripped[:60]}")

        if definitions:
            return (
                f"No similar content found. File structure of {file_path}:\n"
                + "\n".join(definitions[:15])
                + (f"\n  ... and {len(definitions) - 15} more" if len(definitions) > 15 else "")
            )
        return f"No similar content found in {file_path} ({len(lines)} lines)."
