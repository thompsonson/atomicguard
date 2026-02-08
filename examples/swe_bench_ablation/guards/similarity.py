"""Similarity utilities for finding similar content in files.

Provides functions for comparing code lines and finding similar content
to help generators correct their search strings.
"""


def line_similarity(line1: str, line2: str) -> float:
    """Similarity score between two lines (0-1), weighted for code patterns.

    Args:
        line1: First line to compare
        line2: Second line to compare

    Returns:
        Similarity score between 0.0 (no match) and 1.0 (exact match).
    """
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


def show_file_structure(lines: list[str], file_path: str) -> str:
    """Show file structure when no similar content found.

    Extracts function and class definitions to give an overview of the file.

    Args:
        lines: Lines of the file content
        file_path: Path to the file (for display)

    Returns:
        Formatted string showing file structure.
    """
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


def find_similar_content(
    file_content: str,
    search_string: str,
    file_path: str,
) -> str:
    """Find similar content in file to help generator correct its search string.

    Uses the first line of the search string to locate approximate position,
    then shows surrounding context.

    Args:
        file_content: The full content of the file
        search_string: The search string that wasn't found
        file_path: Path to the file (for display)

    Returns:
        Formatted string with similar content or file structure.
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
        return show_file_structure(lines, file_path)

    # Find best matching line in file
    best_match_idx = -1
    best_match_score = 0.0

    for i, line in enumerate(lines):
        score = line_similarity(line.strip(), first_search_line)
        if score > best_match_score:
            best_match_score = score
            best_match_idx = i

    if best_match_score < 0.3:
        # No good match - show file structure instead
        return show_file_structure(lines, file_path)

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
