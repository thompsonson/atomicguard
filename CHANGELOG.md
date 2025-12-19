# CHANGELOG


## v0.2.0 (2025-12-19)

### Bug Fixes

- Correcting the semantic release auth process
  ([`2360d1d`](https://github.com/thompsonson/atomicguard/commit/2360d1ddc56daddac697c113bc010b026e632020))

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

### Features

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
