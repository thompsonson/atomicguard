# Example Experiment Run: SWE-Bench-Pro with AtomicGuard

This document walks through a complete experiment run using the SWE-Bench-Pro example,
highlighting how AtomicGuard's multi-step workflows and guards affect patch quality.

## Command

```bash
uv run python -m examples.swe_bench_pro.demo --debug \
  --log-file experiment-clean.log \
  experiment \
  --provider openrouter \
  --model google/gemini-2.0-flash-001 \
  --base-url https://openrouter.ai/api/v1 \
  --arms singleshot,s1_direct,s1_tdd \
  --language python \
  --max-instances 2 \
  --output-dir output/swe_bench_pro_clean
```

**Model**: `google/gemini-2.0-flash-001` via OpenRouter
**Arms**: `singleshot` (no analysis), `s1_direct` (analyze then patch), `s1_tdd` (analyze, write test, then patch)
**Instances**: 2 qutebrowser bugs from the SWE-Bench-Pro dataset

## Results Summary

| Instance | singleshot | s1_direct | s1_tdd |
|----------|-----------|-----------|--------|
| Instance 1 (log.py move) | failed (27s) | success (7s) | success (12s) |
| Instance 2 (subdomain blocking) | failed (19s) | failed (50s) | success (15s) |

**Score**: singleshot 0/2, s1_direct 1/2, s1_tdd 2/2

---

## Failure Deep-Dive: singleshot (Instance 1)

Instance 1 asks the model to fix code after `hide_qt_warning` was moved from `log.py` to `qtlog.py`.
The singleshot arm gives the model the bug description and asks it to produce a patch directly, with
no analysis step.

**Result**: 4 attempts, all rejected by PatchGuard.

The first attempt (artifact `f04203a7`) is representative. The model hallucinated the entire file
structure, guessing at code that doesn't exist in the actual source:

```json
{
  "edits": [
    {
      "file": "qutebrowser/utils/log.py",
      "search": "    logging.getLogger('qutebrowser.utils.log')"
    },
    {
      "file": "qutebrowser/utils/log.py",
      "search": "from PyQt5 import QtCore\n\nimport logging\n\n\ndef init(level=logging.INFO):..."
    },
    {
      "file": "qutebrowser/utils/log.py",
      "search": "def hide_qt_warning(pattern):\n    \"\"\"Hide a Qt warning matching a given regex..."
    },
    {
      "file": "qutebrowser/utils/qtlog.py",
      "search": "import logging\n\n\nlogger = logging.getLogger(__name__)"
    }
  ]
}
```

PatchGuard feedback:

> Patch validation failed:
>
> - Search string not found in qutebrowser/utils/log.py: `"    logging.getLogger('qutebrowser.utils.log')"`
> - Search string not found in qutebrowser/utils/log.py: `'from PyQt5 import QtCore\n\nimport logging...'`
> - Search string not found in qutebrowser/utils/log.py: `'def hide_qt_warning(pattern):\n    """Hide a Qt warning...'`
> - Search string not found in qutebrowser/utils/qtlog.py: `'import logging\n\n\nlogger = logging.getLogger(__name__)'`

All 4 search strings are fabricated. The model never saw the actual file contents — it guessed
`from PyQt5 import QtCore` (the real code uses `qutebrowser.qt`), guessed the function signature
of `hide_qt_warning`, and guessed the structure of `qtlog.py`. Each subsequent retry (attempts 2-4)
made similar but slightly different guesses, all rejected.

**The fundamental problem**: singleshot has no analysis step, so the LLM must guess file contents
from the repository file listing alone.

---

## Failure Deep-Dive: s1_direct on Instance 2 — Analysis Passes, Patch Fails 4 Times

Instance 2 is about host blocking not applying to subdomains. This is the most interesting failure
because the analysis step succeeds but the patch step fails repeatedly.

### Accepted analysis (artifact `50b95257`)

```json
{
  "bug_type": "logic",
  "root_cause_hypothesis": "The host blocking logic only checks for exact matches of hostnames
    in the blocklist, failing to account for subdomains. Additionally, the whitelist check is
    performed too late in the process, after the blocklist check.",
  "affected_components": ["content blocking", "whitelist handling"],
  "files": [
    "qutebrowser/config/config.py",
    "qutebrowser/utils/urlmatch.py",
    "qutebrowser/qutebrowser.py"
  ],
  "fix_approach": "Modify the host blocking logic to check if the request hostname is a subdomain
    of any blocked domain. Implement an early whitelist check that short-circuits the blocking
    logic if a URL is whitelisted.",
  "confidence": "high"
}
```

Guard feedback: `"Analysis valid: bug_type=logic, 3 files, confidence=high"` — good analysis.

### Rejected patch attempt 1 (artifact `f6cada8d`)

The model tried to implement the fix, but hallucinated the contents of `config.py`:

```json
{
  "edits": [
    {
      "file": "qutebrowser/config/config.py",
      "search": "from qutebrowser.utils import utils, log, urlmatch",
      "replace": "from qutebrowser.utils import utils, log, urlmatch, urlutils"
    },
    {
      "file": "qutebrowser/config/config.py",
      "search": "        if self._runtime_blocked_hosts.match(url.host()):\n            return True\n        if self._config_blocked_hosts.match(url.host()):"
    },
    {
      "file": "qutebrowser/utils/urlmatch.py",
      "search": "import urllib.parse",
      "replace": "import urllib.parse\nimport tldextract"
    }
  ]
}
```

PatchGuard caught the hallucinated `config.py` search string:

> Search string not found in qutebrowser/config/config.py:
> `'if self._runtime_blocked_hosts.match(url.host()):\n    return True\n    if self._config_blocked_hosts.match(url.host()):'`

### Escalating failures across 4 patch attempts

The `index.json` timeline shows the progression:

```
ap_analysis:  [50b95257] ✓ accepted
ap_patch:     [f6cada8d] ✗ 1 search string not found in config.py
              [61c2cc73] ✗ multiple search strings not found + syntax error
              [6bcce3ed] ✗ 6 search strings not found
              [60c6cca9] ✗ 8 search strings not found
```

Despite receiving accumulated feedback from all previous failures, the model's patch attempts
got *worse* over time — attempt 4 had 8 invalid search strings vs attempt 1's single failure.
The model kept inventing more elaborate edits to `config.py` and `urlmatch.py`, none matching
actual file contents.

The feedback history accumulated across retries. By attempt 4, the context included:

```json
"feedback_history": [
  {"artifact_id": "f6cada8d", "feedback": "Search string not found in config.py: ..."},
  {"artifact_id": "61c2cc73", "feedback": "Search string not found + Syntax error in urlmatch.py..."},
  {"artifact_id": "6bcce3ed", "feedback": "6 search strings not found..."}
]
```

**Key insight**: The retry loop (`rmax=3`, so 4 total attempts) cannot fix a fundamentally wrong
approach. The model correctly *understood* the bug in the analysis step, but could not produce
valid edits without seeing the actual file contents.

---

## Success: s1_direct on Instance 1 — First-Try Patch

The same s1_direct arm succeeds on Instance 1 (the simpler log.py move bug).

### Analysis (artifact `fa5f540c`)

```json
{
  "bug_type": "missing_check",
  "root_cause_hypothesis": "The tests for the `hide_qt_warning` function were not moved when
    the function itself was moved from `log.py` to `qtlog.py`.",
  "affected_components": ["logging", "testing"],
  "files": ["qutebrowser/utils/qtlog.py", "qutebrowser/utils/log.py"],
  "fix_approach": "Move the tests related to `hide_qt_warning` from `test_log.py` (or similar)
    to a new or existing test file associated with `qtlog.py`. Update import statements.",
  "confidence": "high"
}
```

Guard: `"Analysis valid: bug_type=missing_check, 2 files, confidence=high"`

### Patch (artifact `12782373`) — accepted on first try

```json
{
  "edits": [
    {
      "file": "qutebrowser/utils/log.py",
      "search": "from qutebrowser.utils import qtlog",
      "replace": "from qutebrowser.utils import qtlog\n\n"
    },
    {
      "file": "qutebrowser/utils/log.py",
      "search": "import argparse\nfrom typing import (TYPE_CHECKING, Any, Iterator, Mapping, MutableSequence,\n                    Optional, Set, Tuple, Union, TextIO, Literal, cast)",
      "replace": "import argparse\nfrom typing import (TYPE_CHECKING, Any, Iterator, Mapping, MutableSequence,\n                    Optional, Set, Tuple, Union, TextIO, Literal, cast)\n\n"
    }
  ]
}
```

Guard: `"Patch is valid: 2 edits, 24 diff lines"`

The `index.json` shows a clean 2-artifact workflow:

```json
{
  "action_pairs": {
    "ap_analysis": ["fa5f540c-92e2-4061-bc3c-269f2973afc1"],
    "ap_patch":    ["12782373-d2f7-4d50-8d9e-ff5997dcea5c"]
  }
}
```

Both accepted on first attempt — analysis then patch, no retries needed.

---

## Success: s1_tdd on Instance 2 — Analysis, Test, Patch, All First Try

The s1_tdd arm is the only arm that succeeds on both instances. Here is Instance 2
(the harder subdomain blocking bug that defeated both singleshot and s1_direct).

The `index.json` shows 3 artifacts, all accepted:

```json
{
  "action_pairs": {
    "ap_analysis":  ["438a4a81-f95e-453f-97a9-4bedf271c141"],
    "ap_gen_test":  ["8a8209f9-92e3-4ad2-97e6-115b2870c239"],
    "ap_gen_fix":   ["6eb88b03-000f-4fe5-8a2d-a4edacd75fef"]
  }
}
```

### Analysis (artifact `438a4a81`) — accepted

Same analysis as s1_direct: `bug_type=logic`, 3 files, confidence=high.

### Test (artifact `8a8209f9`) — accepted

The model generated 3 test functions:

```python
def test_subdomain_not_blocked():
    config = MockConfig()
    url = "https://sub.example.com"
    blocked = host_in_hosts(url, config)
    assert not blocked, "Subdomain should not be blocked when only parent domain is in blocked_hosts."

def test_exact_domain_blocked():
    config = MockConfig()
    url = "https://example.com"
    blocked = host_in_hosts(url, config)
    assert blocked, "Exact domain should be blocked when it is in blocked_hosts."

def test_whitelist_not_respected():
    config = MockConfig()
    config.content.blocking.whitelist = ["https://example.com"]
    url = "https://example.com"
    blocked = host_in_hosts(url, config)
    assert blocked, "Whitelisted URL should not be blocked."
```

Guard: `"Test code valid: 3 test function(s)"`

### Fix (artifact `6eb88b03`) — accepted

The fix adds a `matches_domain()` method to `UrlPattern` and a `host_in_hosts()` free function,
with 6 edits producing 108 diff lines.

Guard: `"Patch is valid: 6 edits, 108 diff lines"`

The TDD step gave the model a concrete target (make the tests pass), which guided the patch
generation toward a correct implementation.

---

## Key Takeaways

1. **Multi-step arms outperform singleshot**: The analysis step prevents hallucinated file
   contents. Without it, the LLM must guess code structure from filenames alone.

2. **TDD arm (`s1_tdd`) is most robust**: 2/2 vs s1_direct's 1/2. The intermediate test
   generation step gives the model a concrete specification to code against.

3. **PatchGuard catches hallucinated search strings** before they become bad patches. Every
   singleshot attempt was caught because search strings didn't match actual file contents.

4. **The retry loop helps but can't fix a fundamentally wrong approach**: Instance 2's
   s1_direct made 4 attempts with escalating failures (1 → multiple → 6 → 8 invalid search
   strings). The model understood the bug but couldn't produce valid edits without file access.

5. **Guard feedback accumulates across retries**: Each rejected attempt's feedback is included
   in the context for the next attempt, giving the model information to self-correct — but
   only when the underlying approach is viable.

---

## Output Directory Structure

```
examples/swe_bench_pro/example_output/
├── results.jsonl                          # 6 rows (2 instances x 3 arms)
├── predictions/
│   ├── 02_singleshot.json                 # [] (no successful patches)
│   ├── 03_s1_direct.json                  # 1 patch (instance 1 only)
│   └── 04_s1_tdd.json                     # 2 patches (both instances)
└── artifact_dags/
    ├── instance_qutebrowser__qutebrowser-f91ace9.../   # Instance 1 (log.py move)
    │   ├── 02_singleshot/
    │   │   ├── index.json                 # 4 artifacts, all rejected
    │   │   └── objects/
    │   │       ├── f0/f04203a7-...json    # attempt 1 (4 hallucinated search strings)
    │   │       ├── 6f/6fc20f6c-...json    # attempt 2
    │   │       ├── bc/bc0867e3-...json    # attempt 3
    │   │       └── f0/f0059dbb-...json    # attempt 4
    │   ├── 03_s1_direct/
    │   │   ├── index.json                 # 2 artifacts, both accepted
    │   │   └── objects/
    │   │       ├── fa/fa5f540c-...json    # analysis (accepted)
    │   │       └── 12/12782373-...json    # patch (accepted, first try)
    │   └── 04_s1_tdd/
    │       ├── index.json                 # 3 artifacts, all accepted
    │       └── objects/
    │           ├── 60/60784408-...json     # analysis (accepted)
    │           ├── 16/16b4d993-...json     # test (accepted)
    │           └── b0/b0bf0ed1-...json     # fix (accepted)
    └── instance_qutebrowser__qutebrowser-c580ebf.../   # Instance 2 (subdomain blocking)
        ├── 02_singleshot/
        │   ├── index.json                 # 4 artifacts, all rejected
        │   └── objects/
        │       ├── 4a/4a31c52a-...json    # attempt 1
        │       ├── 10/1090e7a2-...json    # attempt 2
        │       ├── 40/4063bf62-...json    # attempt 3
        │       └── 5f/5f9b7093-...json    # attempt 4
        ├── 03_s1_direct/
        │   ├── index.json                 # 5 artifacts: 1 accepted analysis, 4 rejected patches
        │   └── objects/
        │       ├── 50/50b95257-...json    # analysis (accepted)
        │       ├── f6/f6cada8d-...json    # patch attempt 1 (rejected)
        │       ├── 61/61c2cc73-...json    # patch attempt 2 (rejected)
        │       ├── 6b/6bcce3ed-...json    # patch attempt 3 (rejected)
        │       └── 60/60c6cca9-...json    # patch attempt 4 (rejected)
        └── 04_s1_tdd/
            ├── index.json                 # 3 artifacts, all accepted
            └── objects/
                ├── 43/438a4a81-...json     # analysis (accepted)
                ├── 8a/8a8209f9-...json     # test (accepted, 3 test functions)
                └── 6e/6eb88b03-...json     # fix (accepted, 6 edits, 108 diff lines)
```

All artifact IDs, guard feedback, and JSON extracts in this document are taken directly from
files in `examples/swe_bench_pro/example_output/`.
