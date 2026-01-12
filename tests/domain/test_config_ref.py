"""Tests for compute_config_ref - Extension 07 (Incremental Execution)."""

from atomicguard.domain.models import (
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    ContextSnapshot,
)
from atomicguard.domain.workflow import compute_config_ref


def make_artifact(
    action_pair_id: str,
    content: str,
    config_ref: str | None = None,
) -> Artifact:
    """Create a minimal Artifact for testing."""
    return Artifact(
        artifact_id=f"art-{action_pair_id}",
        workflow_id="wf-test",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id=action_pair_id,
        created_at="2024-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.ACCEPTED,
        guard_result=True,
        feedback="",
        context=ContextSnapshot(
            workflow_id="wf-test",
            specification="test",
            constraints="",
            feedback_history=(),
        ),
        source=ArtifactSource.GENERATED,
        config_ref=config_ref,
    )


class TestComputeConfigRefDeterministic:
    """Tests for deterministic hash computation."""

    def test_same_inputs_same_hash(self) -> None:
        """Same inputs produce same hash."""
        workflow_config = {
            "model": "qwen2.5-coder:7b",
            "action_pairs": {
                "g_test": {
                    "guard": "syntax",
                    "guard_config": {"strict": True},
                }
            },
        }
        prompt_config = {
            "g_test": {
                "role": "Test writer",
                "task": "Write tests",
            }
        }

        ref1 = compute_config_ref("g_test", workflow_config, prompt_config)
        ref2 = compute_config_ref("g_test", workflow_config, prompt_config)

        assert ref1 == ref2
        assert len(ref1) == 64  # SHA-256 hex

    def test_different_action_pair_different_hash(self) -> None:
        """Different action pair IDs produce different hashes."""
        workflow_config = {
            "model": "qwen2.5-coder:7b",
            "action_pairs": {
                "g_test": {"guard": "syntax"},
                "g_impl": {"guard": "dynamic_test"},
            },
        }
        prompt_config = {
            "g_test": {"role": "Test writer"},
            "g_impl": {"role": "Implementer"},
        }

        ref_test = compute_config_ref("g_test", workflow_config, prompt_config)
        ref_impl = compute_config_ref("g_impl", workflow_config, prompt_config)

        assert ref_test != ref_impl


class TestComputeConfigRefPromptChange:
    """Tests for prompt configuration changes."""

    def test_prompt_change_different_hash(self) -> None:
        """Changing prompt produces different hash."""
        workflow_config = {"action_pairs": {"g_test": {"guard": "syntax"}}}

        prompt_v1 = {"g_test": {"role": "Test writer"}}
        prompt_v2 = {"g_test": {"role": "Test writer v2"}}

        ref_v1 = compute_config_ref("g_test", workflow_config, prompt_v1)
        ref_v2 = compute_config_ref("g_test", workflow_config, prompt_v2)

        assert ref_v1 != ref_v2

    def test_prompt_field_change_different_hash(self) -> None:
        """Changing any prompt field produces different hash."""
        workflow_config = {"action_pairs": {"g_test": {"guard": "syntax"}}}

        prompt_v1 = {"g_test": {"role": "Writer", "task": "Write tests"}}
        prompt_v2 = {"g_test": {"role": "Writer", "task": "Write unit tests"}}

        ref_v1 = compute_config_ref("g_test", workflow_config, prompt_v1)
        ref_v2 = compute_config_ref("g_test", workflow_config, prompt_v2)

        assert ref_v1 != ref_v2


class TestComputeConfigRefModelChange:
    """Tests for model configuration changes."""

    def test_model_change_different_hash(self) -> None:
        """Changing model produces different hash."""
        workflow_v1 = {
            "model": "qwen2.5-coder:7b",
            "action_pairs": {"g_test": {"guard": "syntax"}},
        }
        workflow_v2 = {
            "model": "qwen2.5-coder:14b",
            "action_pairs": {"g_test": {"guard": "syntax"}},
        }
        prompt_config = {"g_test": {"role": "Writer"}}

        ref_v1 = compute_config_ref("g_test", workflow_v1, prompt_config)
        ref_v2 = compute_config_ref("g_test", workflow_v2, prompt_config)

        assert ref_v1 != ref_v2

    def test_action_pair_model_override(self) -> None:
        """Action pair model overrides workflow model."""
        workflow_v1 = {
            "model": "qwen2.5-coder:7b",
            "action_pairs": {"g_test": {"guard": "syntax"}},
        }
        workflow_v2 = {
            "model": "qwen2.5-coder:7b",
            "action_pairs": {
                "g_test": {"guard": "syntax", "model": "qwen2.5-coder:14b"}
            },
        }
        prompt_config = {"g_test": {"role": "Writer"}}

        ref_v1 = compute_config_ref("g_test", workflow_v1, prompt_config)
        ref_v2 = compute_config_ref("g_test", workflow_v2, prompt_config)

        assert ref_v1 != ref_v2


class TestComputeConfigRefGuardConfigChange:
    """Tests for guard configuration changes."""

    def test_guard_config_change_different_hash(self) -> None:
        """Changing guard_config produces different hash."""
        workflow_v1 = {
            "action_pairs": {
                "g_test": {"guard": "syntax", "guard_config": {"strict": True}}
            }
        }
        workflow_v2 = {
            "action_pairs": {
                "g_test": {"guard": "syntax", "guard_config": {"strict": False}}
            }
        }
        prompt_config = {"g_test": {"role": "Writer"}}

        ref_v1 = compute_config_ref("g_test", workflow_v1, prompt_config)
        ref_v2 = compute_config_ref("g_test", workflow_v2, prompt_config)

        assert ref_v1 != ref_v2

    def test_guard_type_change_different_hash(self) -> None:
        """Changing guard type produces different hash."""
        workflow_v1 = {"action_pairs": {"g_test": {"guard": "syntax"}}}
        workflow_v2 = {"action_pairs": {"g_test": {"guard": "import"}}}
        prompt_config = {"g_test": {"role": "Writer"}}

        ref_v1 = compute_config_ref("g_test", workflow_v1, prompt_config)
        ref_v2 = compute_config_ref("g_test", workflow_v2, prompt_config)

        assert ref_v1 != ref_v2


class TestComputeConfigRefRootActionPair:
    """Tests for root action pairs with no dependencies."""

    def test_root_action_pair_no_upstream(self) -> None:
        """Root action pair computes ref from own config only."""
        workflow_config = {
            "action_pairs": {
                "g_test": {"guard": "syntax"},
            }
        }
        prompt_config = {"g_test": {"role": "Writer"}}

        ref = compute_config_ref("g_test", workflow_config, prompt_config)

        assert len(ref) == 64

    def test_root_action_pair_empty_upstream(self) -> None:
        """Root action pair with explicit empty upstream_artifacts."""
        workflow_config = {
            "action_pairs": {
                "g_test": {"guard": "syntax", "requires": []},
            }
        }
        prompt_config = {"g_test": {"role": "Writer"}}

        ref1 = compute_config_ref("g_test", workflow_config, prompt_config)
        ref2 = compute_config_ref(
            "g_test", workflow_config, prompt_config, upstream_artifacts={}
        )

        assert ref1 == ref2


class TestComputeConfigRefUpstreamChange:
    """Tests for upstream artifact changes."""

    def test_upstream_artifact_content_change_different_hash(self) -> None:
        """Changing upstream artifact content produces different hash."""
        workflow_config = {
            "action_pairs": {
                "g_test": {"guard": "syntax"},
                "g_impl": {"guard": "dynamic_test", "requires": ["g_test"]},
            }
        }
        prompt_config = {
            "g_test": {"role": "Writer"},
            "g_impl": {"role": "Implementer"},
        }

        artifact_v1 = make_artifact("g_test", "test content v1", config_ref="ref1")
        artifact_v2 = make_artifact("g_test", "test content v2", config_ref="ref1")

        ref_v1 = compute_config_ref(
            "g_impl", workflow_config, prompt_config, {"g_test": artifact_v1}
        )
        ref_v2 = compute_config_ref(
            "g_impl", workflow_config, prompt_config, {"g_test": artifact_v2}
        )

        assert ref_v1 != ref_v2

    def test_upstream_config_ref_change_different_hash(self) -> None:
        """Changing upstream config_ref produces different hash."""
        workflow_config = {
            "action_pairs": {
                "g_test": {"guard": "syntax"},
                "g_impl": {"guard": "dynamic_test", "requires": ["g_test"]},
            }
        }
        prompt_config = {
            "g_test": {"role": "Writer"},
            "g_impl": {"role": "Implementer"},
        }

        artifact_v1 = make_artifact("g_test", "same content", config_ref="ref-v1")
        artifact_v2 = make_artifact("g_test", "same content", config_ref="ref-v2")

        ref_v1 = compute_config_ref(
            "g_impl", workflow_config, prompt_config, {"g_test": artifact_v1}
        )
        ref_v2 = compute_config_ref(
            "g_impl", workflow_config, prompt_config, {"g_test": artifact_v2}
        )

        assert ref_v1 != ref_v2


class TestComputeConfigRefMerklePropagation:
    """Tests for Merkle propagation through dependency chain."""

    def test_three_step_chain_propagation(self) -> None:
        """Changes propagate through g_test → g_impl → g_review chain."""
        workflow_config = {
            "action_pairs": {
                "g_test": {"guard": "syntax"},
                "g_impl": {"guard": "dynamic_test", "requires": ["g_test"]},
                "g_review": {"guard": "human", "requires": ["g_impl"]},
            }
        }
        prompt_config = {
            "g_test": {"role": "Writer"},
            "g_impl": {"role": "Implementer"},
            "g_review": {"role": "Reviewer"},
        }

        # Compute refs in topological order
        ref_test_v1 = compute_config_ref("g_test", workflow_config, prompt_config)
        artifact_test_v1 = make_artifact("g_test", "tests v1", config_ref=ref_test_v1)

        ref_impl_v1 = compute_config_ref(
            "g_impl", workflow_config, prompt_config, {"g_test": artifact_test_v1}
        )
        artifact_impl_v1 = make_artifact("g_impl", "impl v1", config_ref=ref_impl_v1)

        ref_review_v1 = compute_config_ref(
            "g_review", workflow_config, prompt_config, {"g_impl": artifact_impl_v1}
        )

        # Now change g_test prompt
        prompt_config_v2 = {
            "g_test": {"role": "Test Writer UPDATED"},  # Changed!
            "g_impl": {"role": "Implementer"},
            "g_review": {"role": "Reviewer"},
        }

        ref_test_v2 = compute_config_ref("g_test", workflow_config, prompt_config_v2)
        artifact_test_v2 = make_artifact("g_test", "tests v2", config_ref=ref_test_v2)

        ref_impl_v2 = compute_config_ref(
            "g_impl", workflow_config, prompt_config_v2, {"g_test": artifact_test_v2}
        )
        artifact_impl_v2 = make_artifact("g_impl", "impl v2", config_ref=ref_impl_v2)

        ref_review_v2 = compute_config_ref(
            "g_review", workflow_config, prompt_config_v2, {"g_impl": artifact_impl_v2}
        )

        # All refs should be different due to cascade
        assert ref_test_v1 != ref_test_v2
        assert ref_impl_v1 != ref_impl_v2
        assert ref_review_v1 != ref_review_v2

    def test_leaf_change_does_not_affect_root(self) -> None:
        """Changes to leaf don't propagate upstream."""
        workflow_config = {
            "action_pairs": {
                "g_test": {"guard": "syntax"},
                "g_impl": {"guard": "dynamic_test", "requires": ["g_test"]},
            }
        }
        prompt_v1 = {
            "g_test": {"role": "Writer"},
            "g_impl": {"role": "Implementer"},
        }
        prompt_v2 = {
            "g_test": {"role": "Writer"},
            "g_impl": {"role": "Implementer UPDATED"},  # Only leaf changed
        }

        ref_test_v1 = compute_config_ref("g_test", workflow_config, prompt_v1)
        ref_test_v2 = compute_config_ref("g_test", workflow_config, prompt_v2)

        # Root unchanged
        assert ref_test_v1 == ref_test_v2
