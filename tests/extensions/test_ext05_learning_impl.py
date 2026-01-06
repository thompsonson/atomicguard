"""
Extension 05: Learning Implementation Tests.

TDD tests for implementing:
- Dataset extraction (extract_training_data)
- Prompt formatters (ChatML, Alpaca)
- Incremental training support
- Evaluation metrics (first-attempt success rate)
"""

import pytest

from atomicguard.domain.models import ArtifactStatus


class TestDatasetExtraction:
    """Tests for extract_training_data()."""

    def test_extract_returns_list_of_dicts(self, populated_dag):
        """extract_training_data() returns list of training examples."""
        from atomicguard.domain.learning import extract_training_data

        data = extract_training_data(populated_dag)

        assert isinstance(data, list)
        assert all(isinstance(item, dict) for item in data)

    def test_each_example_has_prompt(self, populated_dag):
        """Each example has 'prompt' key."""
        from atomicguard.domain.learning import extract_training_data

        data = extract_training_data(populated_dag)

        for example in data:
            assert "prompt" in example
            assert isinstance(example["prompt"], str)
            assert len(example["prompt"]) > 0

    def test_each_example_has_completion(self, populated_dag):
        """Each example has 'completion' key."""
        from atomicguard.domain.learning import extract_training_data

        data = extract_training_data(populated_dag)

        for example in data:
            assert "completion" in example
            assert isinstance(example["completion"], str)

    def test_filter_by_predicate(self, populated_dag):
        """extract_training_data(filter_fn=...) applies filter."""
        from atomicguard.domain.extraction import StatusPredicate
        from atomicguard.domain.learning import extract_training_data

        # Only ACCEPTED artifacts
        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        data = extract_training_data(populated_dag, predicate=predicate)

        # Should have 5 examples (populated_dag has 5 ACCEPTED)
        assert len(data) == 5


class TestPromptFormatting:
    """Tests for prompt formatters."""

    def test_format_specification_prompt(self, sample_artifact):
        """format_prompt(artifact) includes specification."""
        from atomicguard.domain.learning import format_prompt

        prompt = format_prompt(sample_artifact)

        # Should include the specification from context
        assert sample_artifact.context.specification in prompt

    def test_format_includes_feedback_history(self, retry_chain_artifacts, memory_dag):
        """format_prompt includes H for refinement artifacts."""
        from atomicguard.domain.learning import format_prompt

        # Get artifact with feedback history (second or third in chain)
        artifact_with_history = retry_chain_artifacts[2]  # Final ACCEPTED

        prompt = format_prompt(
            artifact_with_history, include_history=True, dag=memory_dag
        )

        # Should include prior feedback
        assert "Test failed" in prompt or "Error" in prompt

    def test_chatml_format(self, sample_artifact):
        """ChatMLFormatter produces valid ChatML."""
        from atomicguard.domain.learning import ChatMLFormatter

        formatter = ChatMLFormatter()
        result = formatter.format(sample_artifact)

        # ChatML structure
        assert "<|im_start|>system" in result or "<|im_start|>user" in result
        assert "<|im_end|>" in result

    def test_alpaca_format(self, sample_artifact):
        """AlpacaFormatter produces valid Alpaca format."""
        from atomicguard.domain.learning import AlpacaFormatter

        formatter = AlpacaFormatter()
        result = formatter.format(sample_artifact)

        # Alpaca structure
        assert "instruction" in result.lower() or "### Instruction" in result
        assert "response" in result.lower() or "### Response" in result


class TestIncrementalTraining:
    """Tests for incremental training support."""

    def test_checkpoint_stores_last_item_id(self, tmp_path):
        """Training checkpoint tracks last processed item."""
        from atomicguard.domain.learning import TrainingCheckpoint

        checkpoint = TrainingCheckpoint(path=str(tmp_path / "checkpoint.json"))

        # Store last processed ID
        checkpoint.save(last_item_id="art-050")

        # Load and verify
        loaded = TrainingCheckpoint(path=str(tmp_path / "checkpoint.json"))
        assert loaded.last_item_id == "art-050"

    def test_resume_from_checkpoint(self, populated_dag, tmp_path):
        """Resume training from last_item_id."""
        from atomicguard.domain.learning import (
            TrainingCheckpoint,
            extract_training_data,
        )

        # First extraction
        all_data = extract_training_data(populated_dag)
        assert len(all_data) > 0

        # Save checkpoint at item 5
        checkpoint = TrainingCheckpoint(path=str(tmp_path / "checkpoint.json"))
        checkpoint.save(last_item_id=all_data[4]["artifact_id"])

        # Resume extraction
        new_data = extract_training_data(
            populated_dag, after_id=checkpoint.last_item_id
        )

        # Should only include items after checkpoint
        assert len(new_data) < len(all_data)

    def test_minimum_new_traces_threshold(self):
        """Skip training if fewer than N new traces."""
        from atomicguard.domain.learning import should_train

        # With 0 new traces - should not train
        assert should_train(new_trace_count=0, minimum=10) is False

        # With 5 new traces but minimum 10 - should not train
        assert should_train(new_trace_count=5, minimum=10) is False

        # With 15 new traces and minimum 10 - should train
        assert should_train(new_trace_count=15, minimum=10) is True


class TestEvaluationMetrics:
    """Tests for first-attempt success rate metrics."""

    def test_calculate_first_attempt_rate(self, populated_dag):
        """first_attempt_success_rate() returns 0-1 float."""
        from atomicguard.domain.learning import first_attempt_success_rate

        rate = first_attempt_success_rate(populated_dag)

        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_improvement_comparison(self):
        """compare_improvement(before, after) returns delta."""
        from atomicguard.domain.learning import compare_improvement

        before_rate = 0.3
        after_rate = 0.5

        delta = compare_improvement(before_rate, after_rate)

        # Improvement is positive delta
        assert delta == pytest.approx(0.2)

    def test_retry_count_distribution(self, memory_dag, retry_chain_artifacts):
        """retry_distribution() returns histogram of retries."""
        from atomicguard.domain.learning import retry_distribution

        # retry_chain_artifacts fixture populates memory_dag with the chain
        _ = retry_chain_artifacts  # Ensure fixture runs to populate dag
        dist = retry_distribution(memory_dag)

        # Should be a dict mapping attempt_number -> count
        assert isinstance(dist, dict)

        # retry_chain_artifacts has artifacts with attempts 1, 2, 3
        assert 1 in dist or 2 in dist or 3 in dist


class TestTrainingDataIntegrity:
    """Tests for training data quality and integrity."""

    def test_training_data_no_duplicates(self, populated_dag):
        """extract_training_data() returns unique examples."""
        from atomicguard.domain.learning import extract_training_data

        data = extract_training_data(populated_dag)

        # Check for duplicate artifact IDs
        ids = [item.get("artifact_id") for item in data if "artifact_id" in item]
        assert len(ids) == len(set(ids))

    def test_training_data_valid_json(self, populated_dag):
        """Training data can be serialized to JSON."""
        import json

        from atomicguard.domain.learning import extract_training_data

        data = extract_training_data(populated_dag)

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

        # And deserializable
        loaded = json.loads(json_str)
        assert len(loaded) == len(data)

    def test_training_data_has_metadata(self, populated_dag):
        """Training examples include metadata for provenance."""
        from atomicguard.domain.learning import extract_training_data

        data = extract_training_data(populated_dag, include_metadata=True)

        for example in data:
            # Should have artifact_id for traceability
            assert "artifact_id" in example or "metadata" in example
