#!/usr/bin/env python3
"""
Basic AtomicGuard example using MockGenerator.

This example demonstrates the core concepts without requiring an LLM.
The MockGenerator returns predefined responses, making it ideal for:
- Testing and development
- Understanding the framework
- CI/CD pipelines

Run with: python examples/basic_mock.py
"""

from atomicguard import (
    ActionPair,
    DualStateAgent,
    InMemoryArtifactDAG,
    MockGenerator,
    PromptTemplate,
    SyntaxGuard,
)


def main() -> None:
    """Demonstrate guard-validated generation with mock responses."""

    # MockGenerator returns predefined responses in order
    # First response has a syntax error, second is valid
    generator = MockGenerator(
        responses=[
            # First attempt: invalid syntax (missing colon)
            "def add(a, b)\n    return a + b",
            # Second attempt: valid syntax
            "def add(a, b):\n    return a + b",
        ]
    )

    # SyntaxGuard validates Python syntax
    guard = SyntaxGuard()

    # ActionPair couples generator with guard
    template = PromptTemplate(
        role="code generator",
        constraints="write clean Python code",
        task="generate a function that adds two numbers",
    )
    action_pair = ActionPair(generator=generator, guard=guard, prompt_template=template)

    # DualStateAgent executes with retry logic (up to rmax attempts)
    agent = DualStateAgent(
        action_pair=action_pair,
        artifact_dag=InMemoryArtifactDAG(),
        rmax=3,
    )

    # Execute the generation
    print("Executing guard-validated generation...")
    print("(First attempt will fail syntax check, second will pass)\n")

    artifact = agent.execute("Write a function that adds two numbers")

    print("=== SUCCESS ===")
    print(f"Artifact ID: {artifact.artifact_id}")
    print(f"Status: {artifact.status}")
    print(f"\nGenerated code:\n{artifact.content}")


if __name__ == "__main__":
    main()
