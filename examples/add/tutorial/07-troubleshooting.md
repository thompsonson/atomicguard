# Part 7: Troubleshooting

Common issues and how to fix them.

## Quick Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| "Connection refused" | Ollama not running | `ollama serve` |
| "Model not found" | Model not pulled | `ollama pull qwen2.5-coder:14b` |
| "Exhausted retries" | Model struggling | Increase `--rmax` or simplify docs |
| "Invalid pytestarch API" | Hallucinated methods | Guard handles automatically |
| "No gates extracted" | Documentation unclear | Add more specific gate definitions |
| "Timeout" | Model too slow | Use a smaller model or faster hardware |

## Connection Issues

### "Connection refused"

**Symptom:**

```
requests.exceptions.ConnectionError: Connection refused
```

**Cause:** Ollama is not running.

**Solution:**

```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "Model not found"

**Symptom:**

```
ollama.error.ModelNotFound: model 'qwen2.5-coder:14b' not found
```

**Cause:** Model hasn't been pulled.

**Solution:**

```bash
# Pull the model
ollama pull qwen2.5-coder:14b

# Verify it's available
ollama list
```

### "Connection timeout"

**Symptom:**

```
requests.exceptions.ReadTimeout
```

**Cause:** Model is taking too long to respond.

**Solutions:**

1. Use a smaller model:

```bash
uv run python -m examples.add.run --model ollama:qwen2.5-coder:7b
```

1. Simplify your documentation (fewer gates)

1. Use faster hardware (GPU)

## Generation Issues

### "Exhausted retries"

**Symptom:**

```
RmaxExhausted: Action pair 'test_generation' exhausted 3 retries
```

**Cause:** Model can't generate valid output after all attempts.

**Solutions:**

1. Increase retries:

```bash
uv run python -m examples.add.run --rmax 5
```

1. Use a stronger model:

```bash
uv run python -m examples.add.run --model ollama:qwen2.5-coder:32b
```

1. Simplify your documentation:
   - Fewer gates
   - Clearer rule descriptions
   - More specific scope definitions

### "No gates extracted"

**Symptom:**

```
Expected at least 3 gates, got 0
```

**Cause:** Documentation doesn't match expected format.

**Solutions:**

1. Check your documentation has clear gate definitions:

```markdown
### Gate 1: Name

**Rule**: The actual rule text.

**Constraint Type**: dependency
```

1. Ensure you have the minimum required gates (`--min-gates`)

1. Check the run.log for extraction details:

```bash
cat examples/add/output/run.log | grep -i gate
```

### "Invalid pytestarch API"

**Symptom:**

```
Code execution error: AttributeError: 'Rule' object has no attribute 'modules_in'
```

**Cause:** Model hallucinated a non-existent API method.

**Solution:** This is handled automatically by `PytestArchAPIGuard`. The guard will:

1. Detect the invalid API
2. Provide feedback to the model
3. Retry with the feedback

If it keeps failing, the model may need stronger prompting. Check `prompts.json` for the test_generation constraints.

## Output Issues

### "Empty test file"

**Symptom:** Generated test file has no tests.

**Cause:** FileWriter received empty TestSuite.

**Solutions:**

1. Check if Stage 2 (TestCodeGen) succeeded:

```bash
grep "test_generation" examples/add/output/run.log
```

1. Lower the `--min-tests` threshold:

```bash
uv run python -m examples.add.run --min-tests 1
```

### "Tests don't match gates"

**Symptom:** Generated tests don't align with documentation gates.

**Cause:** Model misinterpreted the gates.

**Solutions:**

1. Make gate definitions more explicit:

```markdown
### Gate 1: Domain Independence

**Rule**: Modules in `src/domain/` MUST NOT import from `src/infrastructure/`.

**Scope**: All Python files in `src/domain/`

**Constraint Type**: dependency
```

1. Add examples in the prompts (prompts.json)

## Debugging

### Enable Verbose Logging

```bash
uv run python -m examples.add.run -v 2>&1 | tee debug.log
```

### Check Artifacts

Artifacts store the output of each stage:

```bash
# List artifacts
ls examples/add/output/artifacts/objects/

# View an artifact
cat examples/add/output/artifacts/objects/*/*
```

### Trace a Failure

1. Find the failed attempt in the log:

```bash
grep "Failed" examples/add/output/run.log
```

1. Find the artifact ID:

```bash
grep "Stored artifact" examples/add/output/run.log
```

1. Examine the artifact:

```bash
cat examples/add/output/artifacts/objects/XX/ARTIFACT_ID.json
```

### Check Model Response

If PydanticAI output validation fails, the raw response is logged:

```bash
grep "Raw model response" examples/add/output/run.log
```

## Performance Issues

### Slow Generation

**Symptoms:**

- Generation takes > 5 minutes
- Frequent timeouts

**Solutions:**

1. Use a faster model:

```bash
uv run python -m examples.add.run --model ollama:qwen2.5-coder:7b
```

1. Use GPU acceleration:

```bash
# On a GPU server
uv run python -m examples.add.run --host http://gpu-server:11434
```

1. Reduce documentation complexity

### High Memory Usage

**Symptoms:**

- System becomes unresponsive
- OOM errors

**Solutions:**

1. Use a smaller model (7B instead of 14B)
2. Close other applications
3. Use a remote Ollama server

## Getting Help

If you're still stuck:

1. **Check the logs**: `examples/add/output/run.log`
2. **Examine artifacts**: `examples/add/output/artifacts/`
3. **Open an issue**: Include logs and documentation

---

**Previous**: [06 - Programmatic Usage](06-programmatic.md) | **Back to**: [00 - Overview](00-overview.md)
