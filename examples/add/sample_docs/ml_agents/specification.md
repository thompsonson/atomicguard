# Architecture Gates

**Version:** 1.0
**Package Name**: `ml_agents_v2`
**Source Root**: `src/ml_agents_v2`

## Module Paths

| Layer | Python Module |
|-------|---------------|
| Domain | `ml_agents_v2.core.domain` |
| Application | `ml_agents_v2.core.application` |
| Infrastructure | `ml_agents_v2.infrastructure` |
| CLI | `ml_agents_v2.cli` |

## Layer Structure

```
┌─────────────────────────────────────────────┐
│              Infrastructure                 │
│  ┌───────────────────────────────────────┐  │
│  │            Application                │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │           Domain                │  │  │
│  │  │                                 │  │  │
│  │  │  Entities, Value Objects,       │  │  │
│  │  │  Domain Services, Repositories  │  │  │
│  │  └─────────────────────────────────┘  │  │
│  │                                       │  │
│  │  Orchestrators, DTOs                  │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Database, Providers, Parsers, Factories    │
└─────────────────────────────────────────────┘

         CLI (thin adapter layer)
```

---

## Architecture Gates

### Gate 1: Domain Independence

**Rule**: `ml_agents_v2.core.domain` MUST NOT import from `ml_agents_v2.infrastructure`

**Rationale**: Domain logic should be pure and framework-agnostic. It should not depend on databases, APIs, or any external systems.

**Scope**: All modules in `ml_agents_v2.core.domain/`

**Constraint Type**: dependency

---

### Gate 2: Application Layer Boundaries

**Rule**: `ml_agents_v2.core.application` MUST NOT import from `ml_agents_v2.infrastructure` directly.

**Rationale**: Use cases should depend on abstractions (ports defined in domain), not on concrete implementations. This enables testing without real infrastructure.

**Scope**: All modules in `ml_agents_v2.core.application/`

**Constraint Type**: dependency

---

### Gate 3: CLI Routes Through Application

**Rule**: `ml_agents_v2.cli` SHOULD import from `ml_agents_v2.core.application.services`

**Rationale**: CLI is a thin adapter layer. Business logic belongs in the application layer, not in CLI commands.

**Scope**: All modules in `ml_agents_v2.cli/`

**Constraint Type**: dependency

---

### Gate 4: Infrastructure Implements Domain Interfaces

**Rule**: `ml_agents_v2.infrastructure` SHOULD import from `ml_agents_v2.core.domain.repositories`

**Rationale**: Infrastructure provides concrete implementations of domain ports. Repository implementations must implement the interfaces defined in domain.

**Scope**: All modules in `ml_agents_v2.infrastructure/`

**Constraint Type**: dependency

---

## Layer Boundaries

1. Domain cannot import from application or infrastructure
2. Application cannot import from infrastructure
3. Infrastructure implements domain interfaces
4. CLI routes through application services
5. External frameworks should only appear in infrastructure
