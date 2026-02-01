# Basic Examples

Start here to learn the fundamentals of AtomicGuard.

## Progression

| Level | File | LLM | Description |
|-------|------|-----|-------------|
| 1 | [01_mock.py](01_mock.py) | Mock | Core concepts without external dependencies |
| 2 | [02_ollama.py](02_ollama.py) | Ollama | Real LLM with composite guards |
| 3 | [03_huggingface.py](03_huggingface.py) | HuggingFace | Real LLM via HuggingFace Inference API |

## 01_mock.py - Getting Started

**No LLM required** - perfect for understanding the core concepts.

Demonstrates:

- `DualStateAgent` retry loop
- `SyntaxGuard` validation
- Guard feedback → retry cycle
- Artifact provenance tracking

```bash
uv run python -m examples.basics.01_mock
```

The mock generator intentionally produces a syntax error on first attempt, then succeeds on retry - showing how guards drive refinement.

## 02_ollama.py - Real LLM

**Requires Ollama** with `qwen2.5-coder:7b` model.

Demonstrates:

- `OllamaGenerator` with real LLM
- `CompositeGuard` (SyntaxGuard + TestGuard)
- `RmaxExhausted` error handling
- Full provenance on failure

```bash
# First, ensure Ollama is running with the model:
ollama pull qwen2.5-coder:7b

# Then run:
uv run python -m examples.basics.02_ollama
```

## 03_huggingface.py - HuggingFace Inference API

**Requires `huggingface_hub`** and an API token.

Demonstrates:

- `HuggingFaceGenerator` with HuggingFace Inference Providers
- Same guard composition as 02 (SyntaxGuard + TestGuard)
- Configurable model, provider, temperature, and max_tokens

```bash
# Install the optional dependency:
uv pip install huggingface_hub

# Set your API token:
export HF_TOKEN="hf_your_token_here"

# Run:
uv run python -m examples.basics.03_huggingface
```

## Key Concepts

### DualStateAgent

The core executor that runs the generate → validate → retry loop:

```
Specification → Generator → Artifact → Guard → Pass/Fail
                    ↑__________________________|  (retry with feedback)
```

### Guards

Validators that return `⊤` (pass) or `⊥` (fail with feedback):

- **SyntaxGuard**: Validates Python syntax via `ast.parse()`
- **TestGuard**: Runs tests against generated code
- **CompositeGuard**: Chains multiple guards (AND composition)

### Artifacts

Immutable records of generated content with full provenance:

- `artifact_id`: Unique identifier
- `content`: The generated code
- `status`: pending → accepted/rejected
- `context`: Specification, constraints, feedback history

## Next Steps

After understanding these basics:

1. **tdd/** - Multi-step TDD workflows with guard composition
2. **checkpoint/** - Add human-in-the-loop intervention with checkpoint/resume
