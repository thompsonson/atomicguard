#!/usr/bin/env python3
"""
Basic AtomicGuard example using OllamaGenerator.

This example demonstrates real LLM-powered generation with guard validation.
Requires Ollama running locally with a model installed.

Setup:
    1. Install Ollama: https://ollama.ai/
    2. Pull a model: ollama pull qwen2.5-coder:7b
    3. Run this script: python examples/basic_ollama.py

The example shows:
- Real LLM generation with OllamaGenerator
- Composite guards (SyntaxGuard + TestGuard)
- Automatic retry on guard failures
"""

from atomicguard import (
    ActionPair,
    CompositeGuard,
    DualStateAgent,
    InMemoryArtifactDAG,
    OllamaGenerator,
    RmaxExhausted,
    SyntaxGuard,
    TestGuard,
)


def main() -> None:
    """Demonstrate guard-validated generation with Ollama."""

    # OllamaGenerator connects to local Ollama instance
    # Change model name if using a different model
    generator = OllamaGenerator(model="qwen2.5-coder:7b")

    # CompositeGuard combines multiple guards
    # All guards must pass for the artifact to be accepted
    guard = CompositeGuard(
        [
            SyntaxGuard(),  # Validates Python syntax
            TestGuard("assert add(2, 3) == 5\nassert add(-1, 1) == 0"),  # Runs tests
        ]
    )

    # ActionPair couples generator with guard
    action_pair = ActionPair(generator=generator, guard=guard)

    # DualStateAgent executes with retry logic
    # rmax=3 means up to 3 attempts before giving up
    agent = DualStateAgent(
        action_pair=action_pair,
        artifact_dag=InMemoryArtifactDAG(),
        rmax=3,
    )

    prompt = """Write a Python function called 'add' that takes two numbers and returns their sum.
Return only the function definition, no explanations."""

    print("Executing guard-validated generation with Ollama...")
    print("Model: qwen2.5-coder:7b")
    print("Guards: SyntaxGuard + TestGuard")
    print("Max attempts: 3\n")

    try:
        artifact = agent.execute(prompt)

        print("=== SUCCESS ===")
        print(f"Artifact ID: {artifact.artifact_id}")
        print(f"Status: {artifact.status}")
        print(f"\nGenerated code:\n{artifact.content}")

    except RmaxExhausted as e:
        print("=== FAILED ===")
        print(f"Could not generate valid code after {e.rmax} attempts")
        print("\nAttempt history:")
        for i, (content, feedback) in enumerate(e.provenance, 1):
            print(f"\n--- Attempt {i} ---")
            print(f"Code:\n{content}")
            print(f"Feedback: {feedback}")


if __name__ == "__main__":
    main()
