# CHANGELOG


## v1.0.0 (2025-12-26)

### Bug Fixes

- **schemas**: Resolve mypy type errors
  ([`9c5f6f4`](https://github.com/thompsonson/atomicguard/commit/9c5f6f48f5b826f154077637d0f7aa3d2331f50e))

- Remove backwards compatibility (not needed)
  ([`c971f49`](https://github.com/thompsonson/atomicguard/commit/c971f49c3bd2fb472129354892de82bd0100d014))

### Build System

- **deps**: Bump actions/setup-python from 5 to 6
  ([`1d91e43`](https://github.com/thompsonson/atomicguard/commit/1d91e435f0f5c7fa4178a229965d9d9d9ada3baa))

Bumps [actions/setup-python](https://github.com/actions/setup-python) from 5 to 6. - [Release
  notes](https://github.com/actions/setup-python/releases) -
  [Commits](https://github.com/actions/setup-python/compare/v5...v6)

--- updated-dependencies: - dependency-name: actions/setup-python dependency-version: '6'
  dependency-type: direct:production update-type: version-update:semver-major ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump python-semantic-release/publish-action
  ([`bb08fef`](https://github.com/thompsonson/atomicguard/commit/bb08fefaa2da4bf2c1d7d37dbc7d88a22b469320))

Bumps
  [python-semantic-release/publish-action](https://github.com/python-semantic-release/publish-action)
  from 9.14.0 to 10.5.3. - [Release
  notes](https://github.com/python-semantic-release/publish-action/releases) -
  [Changelog](https://github.com/python-semantic-release/publish-action/blob/main/releaserc.toml) -
  [Commits](https://github.com/python-semantic-release/publish-action/compare/v9.14.0...v10.5.3)

--- updated-dependencies: - dependency-name: python-semantic-release/publish-action
  dependency-version: 10.5.3 dependency-type: direct:production update-type:
  version-update:semver-major ...

Signed-off-by: dependabot[bot] <support@github.com>

### Features

- **schema**: Add JSON Schema definitions and dependency_artifacts refactor
  ([#12](https://github.com/thompsonson/atomicguard/pull/12),
  [`09ae1c1`](https://github.com/thompsonson/atomicguard/commit/09ae1c11f1d30de275324f03778f8d976d47ea2a))

- Add formal JSON Schema definitions for workflows, prompts, and artifacts - Refactor dependency_ids
  → dependency_artifacts to store (action_pair_id, artifact_id) tuples - Align domain models with
  schema format

- **schemas**: Add formal JSON Schema definitions aligned with paper framework
  ([`05b8d76`](https://github.com/thompsonson/atomicguard/commit/05b8d76801d2a7b90be977cd603c66995a0e50d9))

Add JSON Schema package for AtomicGuard configuration validation:

- workflow.schema.json: Workflow definition with action pairs, guards, preconditions -
  prompts.schema.json: Prompt templates for generators (a_gen) - artifact.schema.json: Artifact
  storage format in DAG

Schema aligns with paper's formal notation: - Ψ (specification) supports file/folder/service/inline
  sources - Ω (constraints) via workflow.constraints and guard_config - A = ⟨ρ, a_gen, G⟩ maps to
  action_pairs with requires/guard fields - G_θ (parameterized guards) via guard_config for
  thresholds - α (artifacts) with context hierarchy C = ⟨ℰ, C_local, H⟩

Includes validation utilities and README documentation.

### Refactoring

- **models**: Align dependency_artifacts with JSON schema format
  ([`f7406f6`](https://github.com/thompsonson/atomicguard/commit/f7406f6298f345c773e1d52f88c0f8870a9bfbe2))

Rename and retype context dependency fields to match the formal artifact.schema.json. This ensures
  semantic keys (action_pair_id) are preserved and artifacts are retrieved from ℛ (repository) by
  ID.

Changes: - ContextSnapshot.dependency_ids → dependency_artifacts: tuple[tuple[str, str], ...] -
  Context.dependencies → dependency_artifacts: tuple[tuple[str, str], ...] - Add
  Context.get_dependency(action_pair_id) helper method - Update filesystem serialization: dict for
  JSON, tuple for Python - Generators now retrieve full artifacts from ℛ via get_artifact() - Store
  artifact IDs in context, not full Artifact objects

Per paper Definition 5: Generators access prior artifacts via context.dependency_artifacts, then
  retrieve from ℛ when needed.


## v0.2.0 (2025-12-19)

### Bug Fixes

- Correcting the semantic release auth process
  ([`2360d1d`](https://github.com/thompsonson/atomicguard/commit/2360d1ddc56daddac697c113bc010b026e632020))

- Correcting the semantic release auth process
  ([`6e91806`](https://github.com/thompsonson/atomicguard/commit/6e918065b115823231233ffaac7f9661d9a4ac5e))

- Pushing the fixed tests
  ([`80cdf83`](https://github.com/thompsonson/atomicguard/commit/80cdf836be3af916e767c1c622a0a472b5c77ad4))

### Build System

- **deps**: Bump actions/upload-artifact from 4 to 6
  ([`59bcec4`](https://github.com/thompsonson/atomicguard/commit/59bcec471de74926fffbd93e27ff0bf65397ae68))

Bumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 4 to 6. - [Release
  notes](https://github.com/actions/upload-artifact/releases) -
  [Commits](https://github.com/actions/upload-artifact/compare/v4...v6)

--- updated-dependencies: - dependency-name: actions/upload-artifact dependency-version: '6'
  dependency-type: direct:production update-type: version-update:semver-major ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump astral-sh/setup-uv from 4 to 7
  ([`22648de`](https://github.com/thompsonson/atomicguard/commit/22648ded8db9610a0457f4ea1561e2430c9c3b97))

Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 4 to 7. - [Release
  notes](https://github.com/astral-sh/setup-uv/releases) -
  [Commits](https://github.com/astral-sh/setup-uv/compare/v4...v7)

--- updated-dependencies: - dependency-name: astral-sh/setup-uv dependency-version: '7'
  dependency-type: direct:production update-type: version-update:semver-major ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump codecov/codecov-action from 4 to 5
  ([`ef37b81`](https://github.com/thompsonson/atomicguard/commit/ef37b81179422f79832cf3c57f96997a067dc8a2))

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 4 to 5. - [Release
  notes](https://github.com/codecov/codecov-action/releases) -
  [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/codecov/codecov-action/compare/v4...v5)

--- updated-dependencies: - dependency-name: codecov/codecov-action dependency-version: '5'
  dependency-type: direct:production update-type: version-update:semver-major ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump actions/checkout from 4 to 6
  ([`d579361`](https://github.com/thompsonson/atomicguard/commit/d5793613d093322d12549587cac25b715901a9cd))

Bumps [actions/checkout](https://github.com/actions/checkout) from 4 to 6. - [Release
  notes](https://github.com/actions/checkout/releases) -
  [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/actions/checkout/compare/v4...v6)

--- updated-dependencies: - dependency-name: actions/checkout dependency-version: '6'
  dependency-type: direct:production update-type: version-update:semver-major ...

Signed-off-by: dependabot[bot] <support@github.com>

### Chores

- Removing paper from git tracking
  ([`8043bff`](https://github.com/thompsonson/atomicguard/commit/8043bffd04c87b8fc05f01e9c9e09e2b68b4274b))

### Documentation

- Add semantic agency remark to documentation and generator interface docs
  ([`06dba7c`](https://github.com/thompsonson/atomicguard/commit/06dba7c3c29e5d5fd9014deccbe1d5e1583f7482))

- Updated documentation to include the workflow status enum and handle the fatal response
  ([`a9ff9d8`](https://github.com/thompsonson/atomicguard/commit/a9ff9d8178bb2867731e6152e9710efc6f635f8d))

### Features

- **domain**: Add fatal escalation support to guard result
  ([`7eee0e5`](https://github.com/thompsonson/atomicguard/commit/7eee0e5b809308f95c91cdc9c40a00bf2d9d99f4))

Implement guard fatal state (⊥_fatal) to distinguish non-recoverable failures from retryable ones,
  aligning with paper Definition 6.

Changes: - Add 'fatal: bool' field to GuardResult (defaults False for backwards compat) - Add
  WorkflowStatus enum (SUCCESS, FAILED, ESCALATION) - Add EscalationRequired exception for fatal
  guard failures - Update DualStateAgent to raise EscalationRequired on fatal, skip retries - Update
  Workflow to catch EscalationRequired and return ESCALATION status - Add escalation_artifact and
  escalation_feedback to WorkflowResult - Update GeneratorInterface docstring with idempotency
  requirements

BREAKING CHANGE: WorkflowResult.success replaced with WorkflowResult.status

- **examples**: Add TDD workflow examples with guard composition
  ([`a491c74`](https://github.com/thompsonson/atomicguard/commit/a491c74659628d0c29182ce4eb3d4938f80774fb))

Add two examples demonstrating guard composition patterns:

tdd_human_review/ - CompositeGuard: SyntaxGuard + HumanReviewGuard - Human-in-loop validation for
  generated tests

tdd_import_guard/ - CompositeGuard: SyntaxGuard + ImportGuard + HumanReviewGuard - ImportGuard
  catches missing imports before human review

Both include CLI options, logging, and README docs. Exports ImportGuard from package __init__.py.

- Trigger initial PyPI release
  ([`f393daa`](https://github.com/thompsonson/atomicguard/commit/f393daa76ae95c8dc7730be9e8fdd6287c30a912))

### Refactoring

- **persistence**: Remove unused metadata param from store()
  ([`b340fd9`](https://github.com/thompsonson/atomicguard/commit/b340fd98574996784a9e74de822f1cd1f9d20fc6))

Artifact.feedback already stores guard feedback, making the metadata parameter redundant.
  Additionally, no get_metadata() method existed to retrieve stored values.

- **guards**: Reorganize into hybrid folder structure
  ([`8657435`](https://github.com/thompsonson/atomicguard/commit/865743578734c6e915d39101f0c7aa88d71d1ee9))

Restructure guards module by validation profile: - static/: Pure AST-based (SyntaxGuard,
  ImportGuard) - dynamic/: Subprocess execution (TestGuard, DynamicTestGuard) - interactive/:
  Human-in-loop (HumanReviewGuard) - composite/: Composition patterns (CompositeGuard)

Add ImportGuard - pure AST-based import validation that replaces TestCollectionGuard with a faster,
  atomic implementation.

Maintains backwards-compatible exports from guards/__init__.py.


## v0.1.0 (2025-12-18)

### Continuous Integration

- Correcting the publish action
  ([`8ad2ed3`](https://github.com/thompsonson/atomicguard/commit/8ad2ed32e1de1ac62d0cbd174735632e34a52f43))

### Features

- Initial release of AtomicGuard framework
  ([`ac8502e`](https://github.com/thompsonson/atomicguard/commit/ac8502ef916e4b4d011a44c9af7d3c659944e72c))

A guard-validated LLM code generation framework implementing the Dual-State Agent pattern for
  reliable artifact production.

Key components: - Domain models: Artifact, Context, GuardResult, PromptTemplate - Guards:
  SyntaxGuard, TestGuard, HumanGuard, CompositeGuard - Generators: MockGenerator, OllamaGenerator -
  Agents: DualStateAgent with retry logic up to rmax attempts - Persistence: InMemoryArtifactDAG,
  FilesystemArtifactDAG

Includes CI/CD pipeline with semantic versioning, comprehensive test suite, TDD workflow benchmarks,
  and full documentation.

- Initial release of AtomicGuard framework
  ([`db3076a`](https://github.com/thompsonson/atomicguard/commit/db3076a34fb761127215e0746776c5bee35c6ee3))

A guard-validated LLM code generation framework implementing the Dual-State Agent pattern for
  reliable artifact production.

Key components: - Domain models: Artifact, Context, GuardResult, PromptTemplate - Guards:
  SyntaxGuard, TestGuard, HumanGuard, CompositeGuard - Generators: MockGenerator, OllamaGenerator -
  Agents: DualStateAgent with retry logic up to rmax attempts - Persistence: InMemoryArtifactDAG,
  FilesystemArtifactDAG

Includes CI/CD pipeline with semantic versioning, comprehensive test suite, TDD workflow benchmarks,
  and full documentation.
