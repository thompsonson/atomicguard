"""
DDD/Hexagonal Architecture Layer Rules.

Permanent tests enforcing dependency direction between layers:
- Domain must not access Application or Infrastructure
- Application must not access Infrastructure

These rules use PyTestArch's LayerRule API for declarative enforcement.
"""

import pytest
from pytestarch import LayerRule


class TestDDDLayerRules:
    """Permanent architecture rules enforcing Hexagonal architecture."""

    def test_domain_does_not_access_application(self, evaluable, layers):
        """Domain must be pure — no orchestration dependency."""
        rule = (
            LayerRule()
            .based_on(layers)
            .layers_that()
            .are_named("domain")
            .should_not()
            .access_layers_that()
            .are_named("application")
        )
        rule.assert_applies(evaluable)

    def test_domain_does_not_access_infrastructure(self, evaluable, layers):
        """Domain must not know about adapters."""
        rule = (
            LayerRule()
            .based_on(layers)
            .layers_that()
            .are_named("domain")
            .should_not()
            .access_layers_that()
            .are_named("infrastructure")
        )
        rule.assert_applies(evaluable)

    @pytest.mark.xfail(
        reason="Issue 3: application/workflow.py lazy-imports InMemoryArtifactDAG",
        strict=True,
    )
    def test_application_does_not_access_infrastructure(self, evaluable, layers):
        """Application depends on domain ports, not concrete adapters."""
        rule = (
            LayerRule()
            .based_on(layers)
            .layers_that()
            .are_named("application")
            .should_not()
            .access_layers_that()
            .are_named("infrastructure")
        )
        rule.assert_applies(evaluable)
