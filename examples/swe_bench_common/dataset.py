"""Shared dataset utilities for SWE-bench experiment runners.

Contains shared utilities for parsing dataset fields.
"""

import json


def _parse_test_list(value: str | list[str]) -> list[str]:
    """Parse test list from string or list.

    SWE-bench datasets may store test lists as JSON strings or Python lists.
    This function normalizes both formats.

    Args:
        value: Either a list of test names or a JSON string containing a list.

    Returns:
        List of test names.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        if value.strip():
            return [value]
    return []
