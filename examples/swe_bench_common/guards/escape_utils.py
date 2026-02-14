"""Shared escape-detection utilities for guards.

Detects common JSON-level escaping that leaks into Python code when LLMs
(especially Gemini) produce incorrectly escaped tool-call arguments.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------
# Each check is a simple string/regex test against the artifact content.
# The content is the Python string *after* JSON deserialization, so we are
# looking for literal characters that should not be present in well-formed
# Python source code.
#
# Example: Gemini sends  \\\"  in JSON → after parsing the artifact
# content contains the two-character sequence  \"  (backslash + quote).
# ---------------------------------------------------------------------------

# Regex for a literal backslash followed by a double-quote that is NOT
# preceded by another backslash (negative look-behind avoids matching
# valid \\\" inside Python strings where the intent is a literal backslash
# before a closing quote).
_RE_BACKSLASH_DQUOTE = re.compile(r'(?<!\\)\\"')

# Same for single-quotes.
_RE_BACKSLASH_SQUOTE = re.compile(r"(?<!\\)\\'")

# Literal backslash-n / backslash-t as two characters (not actual
# whitespace).  We look for them only when the code has real newlines
# nearby — the "single-line" heuristic below handles the fully-flat case.
_RE_ESCAPED_NEWLINE = re.compile(r"(?<!\\)\\n")
_RE_ESCAPED_TAB = re.compile(r"(?<!\\)\\t")

# Header / footer included in every feedback message.
_FEEDBACK_HEADER = (
    "Your code contains incorrectly escaped characters. This typically happens "
    "when JSON-level escaping leaks into Python code."
)

_FEEDBACK_FOOTER = (
    "When providing Python code in a tool call argument, write the code directly "
    "with normal Python syntax. Do NOT add JSON-level escaping yourself — the tool "
    "framework handles serialization automatically."
)


def detect_escape_issues(code: str) -> str | None:
    """Scan *code* for common JSON-escaping artefacts.

    Returns a formatted feedback string if issues are found, or ``None``
    if the code looks clean.
    """
    issues: list[str] = []

    # ── Escaped quotes ────────────────────────────────────────────────
    # Count occurrences — a single \" could be legitimate Python, but
    # 2+ is almost certainly JSON escaping leaking through.
    dquote_hits = len(_RE_BACKSLASH_DQUOTE.findall(code))
    if dquote_hits >= 2:
        issues.append('- Found \\" (escaped double-quotes) — use plain " instead')

    squote_hits = len(_RE_BACKSLASH_SQUOTE.findall(code))
    if squote_hits >= 2:
        issues.append("- Found \\' (escaped single-quotes) — use plain ' instead")

    # ── Escaped f-string delimiters ───────────────────────────────────
    if 'f\\"' in code or "f\\'" in code:
        issues.append(
            '- Found escaped f-string delimiter (f\\") — use plain f" instead'
        )

    # ── Single-line heuristic ─────────────────────────────────────────
    # If the content has no actual newlines but contains literal \n
    # sequences, the whole string was likely serialised with JSON escaping.
    if "\n" not in code and "\\n" in code and len(code) > 40:
        issues.append(
            "- Code appears to be a single line with \\n literals instead of "
            "actual newlines — write multi-line code normally"
        )

    if not issues:
        return None

    return (
        f"{_FEEDBACK_HEADER}\n\n"
        "Detected issues:\n" + "\n".join(issues) + f"\n\n{_FEEDBACK_FOOTER}"
    )
