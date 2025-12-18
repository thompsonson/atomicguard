# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--next-version-placeholder-->

## [0.1.0] - 2025-12-16

### Added

- Initial release of AtomicGuard
- **Domain Models**: `Artifact`, `Context`, `GuardResult`, `WorkflowState`, `WorkflowResult`
- **Domain Interfaces**: `GeneratorInterface`, `GuardInterface`, `ArtifactDAGInterface`
- **Guards**: `SyntaxGuard`, `TestGuard`, `DynamicTestGuard`, `CompositeGuard`, `HumanReviewGuard`
- **Infrastructure**: `OllamaGenerator`, `MockGenerator`, `InMemoryArtifactDAG`, `FilesystemArtifactDAG`
- **Application**: `ActionPair`, `DualStateAgent`, `Workflow`, `WorkflowStep`
- Prompt templates and task definitions for TDD workflows
- Benchmark simulation tools

[0.1.0]: https://github.com/thompsonson/atomicguard/releases/tag/v0.1.0
