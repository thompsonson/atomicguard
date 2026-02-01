#!/usr/bin/env python3
"""
Basic AtomicGuard example using HuggingFaceGenerator.

This example demonstrates real LLM-powered generation with guard validation
using HuggingFace Inference Providers.

Setup:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Set your token: export HF_TOKEN="hf_your_token_here"
    3. Run this script: uv run python -m examples.basics.03_huggingface

The example shows:
- Real LLM generation with HuggingFaceGenerator
- Composite guards (SyntaxGuard + TestGuard)
- Automatic retry on guard failures
"""

from atomicguard import (
    ActionPair,
    CompositeGuard,
    DualStateAgent,
    InMemoryArtifactDAG,
    RmaxExhausted,
    SyntaxGuard,
    TestGuard,
)
from atomicguard.infrastructure.llm import HuggingFaceGenerator
from atomicguard.infrastructure.llm.huggingface import HuggingFaceGeneratorConfig


def main() -> None:
    """Demonstrate guard-validated generation with HuggingFace."""

    # HuggingFaceGenerator connects to HuggingFace Inference API
    # Requires HF_TOKEN environment variable or explicit api_key
    generator = HuggingFaceGenerator(
        HuggingFaceGeneratorConfig(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
        )
    )

    # CompositeGuard combines multiple guards
    # All guards must pass for the artifact to be accepted
    guard = CompositeGuard(
        SyntaxGuard(),  # Validates Python syntax
        TestGuard("assert add(2, 3) == 5\nassert add(-1, 1) == 0"),  # Runs tests
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

    print("Executing guard-validated generation with HuggingFace...")
    print(f"Model: {generator._model}")
    print("Guards: SyntaxGuard + TestGuard")
    print("Max attempts: 3\n")

    try:
        artifact = agent.execute(prompt)

        print("=== SUCCESS ===")
        print(f"Artifact ID: {artifact.artifact_id}")
        print(f"Status: {artifact.status.value}")
        print(f"\nGenerated code:\n{artifact.content}")

    except RmaxExhausted as e:
        print("=== FAILED ===")
        print(f"Could not generate valid code: {e}")
        print("\nAttempt history:")
        for i, (artifact, feedback) in enumerate(e.provenance, 1):
            print(f"\n--- Attempt {i} ---")
            print(f"Code:\n{artifact.content}")
            print(f"Feedback: {feedback}")


if __name__ == "__main__":
    main()
