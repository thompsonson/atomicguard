"""Edit Grounder: Resolves search string grounding issues.

Handles common LLM mistakes when generating search strings:
- Incorrect leading whitespace/indentation
- Trailing whitespace differences
- Minor formatting variations

The grounder attempts to "ground" a search string to the actual file content
by finding the best match and adjusting indentation/whitespace.
"""

import re
from dataclasses import dataclass


@dataclass
class GroundingResult:
    """Result of grounding a search string to file content."""

    success: bool
    grounded_search: str
    original_search: str
    adjustment_made: str  # Description of what was changed
    match_start: int  # Character offset in file where match starts
    match_end: int  # Character offset where match ends


def _strip_uniform_leading(text: str) -> tuple[str, str]:
    """Strip uniform leading whitespace from all lines.

    Returns (stripped_text, removed_prefix) where removed_prefix is the
    common leading whitespace that was removed from each line.

    Example:
        "    def foo():\\n        pass" -> ("def foo():\\n    pass", "    ")
    """
    lines = text.split("\n")
    if not lines:
        return text, ""

    # Find minimum indentation (ignoring empty lines)
    min_indent = float("inf")
    for line in lines:
        if line.strip():  # Non-empty line
            stripped = line.lstrip()
            indent_len = len(line) - len(stripped)
            min_indent = min(min_indent, indent_len)

    if min_indent == float("inf") or min_indent == 0:
        return text, ""

    # Get the actual prefix characters (could be tabs or spaces)
    prefix = ""
    for line in lines:
        if line.strip():
            prefix = line[:min_indent]
            break

    # Strip the common prefix from all lines
    stripped_lines = []
    for line in lines:
        if line.startswith(prefix):
            stripped_lines.append(line[len(prefix) :])
        elif not line.strip():  # Empty or whitespace-only line
            stripped_lines.append(line.lstrip())
        else:
            stripped_lines.append(line)

    return "\n".join(stripped_lines), prefix


def _reindent_to_match(search: str, target_indent: str) -> str:
    """Reindent search string to match target indentation.

    Args:
        search: The search string to reindent
        target_indent: The indentation prefix to apply

    Returns:
        Search string with adjusted indentation.
    """
    lines = search.split("\n")
    reindented = []

    for line in lines:
        if line.strip():  # Non-empty line
            # Strip existing leading whitespace and add target indent
            reindented.append(target_indent + line.lstrip())
        else:
            reindented.append(line)

    return "\n".join(reindented)


def _find_pattern_in_content(pattern: str, content: str) -> list[tuple[int, int]]:
    """Find all occurrences of a pattern in content.

    Returns list of (start, end) character offsets.
    """
    matches = []
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(pattern)))
        start = pos + 1
    return matches


def _get_line_indent(content: str, char_offset: int) -> str:
    """Get the indentation of the line containing char_offset.

    Returns the whitespace prefix of that line.
    """
    # Find start of line
    line_start = content.rfind("\n", 0, char_offset) + 1
    # Find first non-whitespace on line
    match = re.match(r"^(\s*)", content[line_start:])
    return match.group(1) if match else ""


def ground_search_string(
    search: str,
    file_content: str,
) -> GroundingResult:
    """Ground a search string to actual file content.

    Attempts to find the search string in the file content, adjusting for
    common LLM mistakes like incorrect indentation or trailing whitespace.

    Args:
        search: The search string from the LLM's edit
        file_content: The actual file content

    Returns:
        GroundingResult with success=True if grounded, False otherwise.
    """
    original_search = search

    # 1. Exact match - no grounding needed
    if search in file_content:
        matches = _find_pattern_in_content(search, file_content)
        return GroundingResult(
            success=True,
            grounded_search=search,
            original_search=original_search,
            adjustment_made="none (exact match)",
            match_start=matches[0][0] if matches else 0,
            match_end=matches[0][1] if matches else len(search),
        )

    # 2. Try stripping trailing whitespace from each line
    search_stripped = "\n".join(line.rstrip() for line in search.split("\n"))
    if search_stripped in file_content:
        matches = _find_pattern_in_content(search_stripped, file_content)
        return GroundingResult(
            success=True,
            grounded_search=search_stripped,
            original_search=original_search,
            adjustment_made="stripped trailing whitespace",
            match_start=matches[0][0] if matches else 0,
            match_end=matches[0][1] if matches else len(search_stripped),
        )

    # 3. Try normalizing to file's line endings
    content_stripped = "\n".join(line.rstrip() for line in file_content.split("\n"))
    if search_stripped in content_stripped:
        # Found in normalized content - find in original
        matches = _find_pattern_in_content(search_stripped, content_stripped)
        if matches:
            return GroundingResult(
                success=True,
                grounded_search=search_stripped,
                original_search=original_search,
                adjustment_made="stripped trailing whitespace (both sides)",
                match_start=matches[0][0],
                match_end=matches[0][1],
            )

    # 4. Try with uniform leading whitespace stripped
    stripped_search, removed_prefix = _strip_uniform_leading(search)
    if stripped_search in file_content:
        matches = _find_pattern_in_content(stripped_search, file_content)
        return GroundingResult(
            success=True,
            grounded_search=stripped_search,
            original_search=original_search,
            adjustment_made=f"stripped {len(removed_prefix)} chars of leading whitespace",
            match_start=matches[0][0] if matches else 0,
            match_end=matches[0][1] if matches else len(stripped_search),
        )

    # 5. Try finding with different indentation levels
    # Strip all leading whitespace and search for core content
    stripped_search, _ = _strip_uniform_leading(search)
    stripped_search = "\n".join(line.rstrip() for line in stripped_search.split("\n"))

    # Look for the first significant line in the file
    first_line = ""
    for line in stripped_search.split("\n"):
        if line.strip():
            first_line = line.strip()
            break

    if first_line and first_line in file_content:
        # Find where this line appears in the file
        pos = file_content.find(first_line)
        if pos != -1:
            # Get the indentation at that position
            target_indent = _get_line_indent(file_content, pos)
            # Reindent the search string
            reindented = _reindent_to_match(search, target_indent)
            reindented = "\n".join(line.rstrip() for line in reindented.split("\n"))

            if reindented in file_content:
                matches = _find_pattern_in_content(reindented, file_content)
                return GroundingResult(
                    success=True,
                    grounded_search=reindented,
                    original_search=original_search,
                    adjustment_made=f"reindented to '{target_indent}'",
                    match_start=matches[0][0] if matches else 0,
                    match_end=matches[0][1] if matches else len(reindented),
                )

    # 6. Failed to ground
    return GroundingResult(
        success=False,
        grounded_search=search,
        original_search=original_search,
        adjustment_made="failed to ground",
        match_start=-1,
        match_end=-1,
    )


def ground_edits(
    edits: list[dict[str, str]],
    repo_root: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """Ground all edits in a patch to their actual file content.

    Args:
        edits: List of {"file": path, "search": str, "replace": str} dicts
        repo_root: Path to the repository root

    Returns:
        Tuple of (grounded_edits, warnings) where grounded_edits has adjusted
        search strings and warnings contains messages about adjustments made.
    """
    from pathlib import Path

    grounded_edits = []
    warnings = []

    for edit in edits:
        file_path = edit.get("file", "")
        search = edit.get("search", "")
        replace = edit.get("replace", "")

        if not file_path or not search:
            grounded_edits.append(edit)
            continue

        full_path = Path(repo_root) / file_path
        if not full_path.exists():
            grounded_edits.append(edit)
            continue

        try:
            file_content = full_path.read_text()
        except Exception:
            grounded_edits.append(edit)
            continue

        result = ground_search_string(search, file_content)

        if result.success:
            if result.adjustment_made != "none (exact match)":
                warnings.append(f"{file_path}: {result.adjustment_made}")
            grounded_edits.append(
                {
                    "file": file_path,
                    "search": result.grounded_search,
                    "replace": replace,
                }
            )
        else:
            # Keep original - let downstream validation handle it
            grounded_edits.append(edit)
            warnings.append(f"{file_path}: failed to ground search string")

    return grounded_edits, warnings
