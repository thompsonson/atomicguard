# ADD Tutorial Overview

Welcome to the Architecture-Driven Development (ADD) tutorial. This guide teaches you how to automatically generate architecture tests from documentation using AtomicGuard.

## What You'll Learn

By the end of this tutorial, you will be able to:

1. **Run the ADD example** to generate tests from sample documentation
2. **Write architecture documentation** that ADD can parse
3. **Understand the generated tests** and how they enforce architecture rules
4. **Customize ADD** for your own projects
5. **Integrate ADD programmatically** into your build pipeline

## What is ADD?

ADD (Architecture-Driven Development) is a workflow that:

1. Takes your **architecture documentation** (written in markdown)
2. **Extracts architecture gates** (rules about what can import what)
3. **Generates pytestarch tests** that enforce those gates
4. **Writes test files** to your project

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   architecture  │────▶│  ADD Generator   │────▶│  test_gates.py  │
│      .md        │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Why Use ADD?

- **Documentation as Code**: Your architecture rules are defined once, tests generated automatically
- **Self-Correcting**: If the LLM generates invalid code, guards catch it and retry
- **Consistent Tests**: Every gate follows the same testing pattern
- **Maintainable**: Update documentation, regenerate tests

## Prerequisites

Before starting, ensure you have:

- [ ] **Python 3.12+** installed
- [ ] **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] **Ollama** installed (`brew install ollama` on macOS)
- [ ] **Model pulled**: `ollama pull qwen2.5-coder:14b`
- [ ] **Ollama running**: `ollama serve`

## Tutorial Structure

| Part | Title | Time | Description |
|------|-------|------|-------------|
| 1 | [Quick Start](01-quickstart.md) | 5 min | Run ADD and see results |
| 2 | [Understanding Input](02-understanding-input.md) | 10 min | Architecture documentation format |
| 3 | [Understanding Output](03-understanding-output.md) | 10 min | Generated test structure |
| 4 | [The Pipeline](04-pipeline.md) | 15 min | How ADD works internally |
| 5 | [Customization](05-customization.md) | 15 min | CLI options and configuration |
| 6 | [Programmatic Usage](06-programmatic.md) | 10 min | Python API integration |
| 7 | [Troubleshooting](07-troubleshooting.md) | 5 min | Common issues and fixes |

**Total Time**: ~60 minutes

## Getting Started

Ready to begin? Start with [Quick Start](01-quickstart.md) to run your first ADD workflow.

---

**Next**: [01 - Quick Start](01-quickstart.md)
