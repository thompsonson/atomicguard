# Architecture Gates - AtomicGuard

**Version:** 1.0
**Package Name**: `atomicguard`
**Source Root**: `src/atomicguard`

## Module Paths

| Layer | Python Module |
|-------|---------------|
| Domain | `atomicguard.domain` |
| Application | `atomicguard.application` |
| Infrastructure | `atomicguard.infrastructure` |
| Guards | `atomicguard.guards` |
| Schemas | `atomicguard.schemas` |

## Layer Structure

```
┌─────────────────────────────────────────────┐
│              Infrastructure                 │
│  (LLM clients, persistence, I/O)            │
│  ┌───────────────────────────────────────┐  │
│  │            Application                │  │
│  │  (Agent, ActionPair, Workflow)        │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │           Domain                │  │  │
│  │  │  (Models, Interfaces, Prompts)  │  │  │
│  │  │  Pure business logic, no I/O    │  │  │
│  │  └─────────────────────────────────┘  │  │
│  │                                       │  │
│  └───────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘

    Guards (static/dynamic/interactive)
    Schemas (JSON Schema definitions)
```

---

## Architecture Gates

### Gate 1: Domain Independence

**Rule**: `atomicguard.domain` MUST NOT import from `atomicguard.infrastructure`

**Rationale**: Domain logic (models, interfaces) should be pure and have no I/O dependencies. This ensures the core business logic is testable without mocks.

**Scope**: All modules in `atomicguard.domain/`

**Constraint Type**: dependency

---

### Gate 2: Domain Purity

**Rule**: `atomicguard.domain` MUST NOT import from `atomicguard.application`

**Rationale**: The domain layer is the innermost layer. It should not depend on the application layer which orchestrates domain objects.

**Scope**: All modules in `atomicguard.domain/`

**Constraint Type**: dependency

---

### Gate 3: Application Uses Interfaces

**Rule**: `atomicguard.application` SHOULD import from `atomicguard.domain.interfaces`

**Rationale**: Application layer should depend on abstractions (interfaces defined in domain), not concrete implementations. This enables dependency injection.

**Scope**: All modules in `atomicguard.application/`

**Constraint Type**: dependency

---

### Gate 4: Static Guards Pure

**Rule**: `atomicguard.guards.static` MUST NOT import I/O libraries (subprocess, os.system, requests)

**Rationale**: Static guards perform pure validation (AST parsing, syntax checks). They should not have side effects or external dependencies.

**Scope**: All modules in `atomicguard.guards.static/`

**Constraint Type**: dependency

---

### Gate 5: Schemas Standalone

**Rule**: `atomicguard.schemas` MUST NOT import from other `atomicguard` modules

**Rationale**: Schema definitions are pure data (JSON Schema). They should be independently loadable without importing the runtime.

**Scope**: All modules in `atomicguard.schemas/`

**Constraint Type**: dependency

---

## Layer Boundaries

1. Domain cannot import from application or infrastructure
2. Application depends on domain interfaces, not infrastructure
3. Infrastructure implements domain interfaces
4. Static guards have no I/O dependencies
5. Schemas are standalone with no internal imports
