# Architecture Documentation

## Overview

This document defines the architecture gates for a Python application
following Clean Architecture / Hexagonal Architecture principles.

## Layer Structure

The system is organized into three concentric layers:

```
┌─────────────────────────────────────────────┐
│              Infrastructure                 │
│  ┌───────────────────────────────────────┐  │
│  │            Application                │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │           Domain                │  │  │
│  │  │                                 │  │  │
│  │  │  Entities, Value Objects,       │  │  │
│  │  │  Domain Services, Interfaces    │  │  │
│  │  └─────────────────────────────────┘  │  │
│  │                                       │  │
│  │  Use Cases, Application Services      │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Repositories, API Clients, CLI, Web        │
└─────────────────────────────────────────────┘
```

### Domain Layer (`src/myapp/domain/`)

- Core business logic
- Entities and Value Objects
- Domain Services
- Port interfaces (abstractions)

### Application Layer (`src/myapp/application/`)

- Use Cases (application services)
- Orchestrates domain objects
- Implements business workflows

### Infrastructure Layer (`src/myapp/infrastructure/`)

- Concrete implementations of ports
- Database repositories
- External API clients
- Framework-specific code

---

## Architecture Gates

### Gate 1: Domain Independence

**Rule**: The domain layer MUST NOT import from infrastructure.

**Rationale**: Domain logic should be pure and framework-agnostic.
It should not depend on databases, APIs, or any external systems.

**Scope**: All modules in `domain/`

**Constraint Type**: dependency

---

### Gate 2: Application Layer Boundaries

**Rule**: The application layer can import from domain but MUST NOT import from infrastructure directly.

**Rationale**: Use cases should depend on abstractions (ports defined in domain),
not on concrete implementations. This enables testing without real infrastructure.

**Scope**: All modules in `application/`

**Constraint Type**: dependency

---

### Gate 3: Dependency Direction

**Rule**: Dependencies MUST flow inward only: Infrastructure → Application → Domain.

**Rationale**: The Dependency Rule ensures inner layers know nothing about outer layers.
This makes the domain portable and testable.

**Scope**: Entire codebase

**Constraint Type**: dependency

---

### Gate 4: Entity Containment

**Rule**: All entity classes MUST be in the `domain.entities` package.

**Rationale**: Entities represent core business concepts with identity.
Keeping them in a dedicated package ensures clear organization.

**Scope**: Classes inheriting from `Entity` base class or decorated with `@entity`

**Constraint Type**: containment

---

### Gate 5: Repository Pattern

**Rule**:

- Repository interfaces (ports) MUST be in `domain.interfaces`
- Repository implementations MUST be in `infrastructure.persistence`

**Rationale**: Repositories abstract data access. The interface defines
what the domain needs; the implementation is an infrastructure concern.

**Scope**: Classes with names ending in `Repository`

**Constraint Type**: containment

---

### Gate 6: Value Object Immutability

**Rule**: Value objects in `domain.value_objects` MUST be immutable (frozen dataclasses or similar).

**Rationale**: Value objects are defined by their attributes, not identity.
Immutability prevents accidental mutation and ensures equality semantics.

**Scope**: All classes in `domain.value_objects/`

**Constraint Type**: naming

---

### Gate 7: Use Case Naming

**Rule**: Use case classes in `application/` MUST follow the pattern `*UseCase` or `*Handler`.

**Rationale**: Consistent naming makes the application layer's purpose clear.

**Scope**: Classes in `application/`

**Constraint Type**: naming

---

### Gate 8: No Direct Database Access

**Rule**: Only modules in `infrastructure.persistence` may import database libraries (sqlalchemy, pymongo, etc.).

**Rationale**: Database access is an infrastructure concern. Domain and application
layers should be database-agnostic.

**Scope**: Entire codebase except `infrastructure.persistence/`

**Constraint Type**: dependency

---

## Ubiquitous Language

| Term | Definition |
|------|------------|
| Entity | Domain object with a unique identity that persists over time |
| Value Object | Immutable object defined by its attributes, not identity |
| Aggregate | Cluster of entities treated as a single unit for data changes |
| Repository | Abstraction for collection-like access to aggregates |
| Port | Interface defining a boundary between layers |
| Adapter | Concrete implementation of a port |
| Use Case | Single application-level operation |
| Domain Service | Stateless service containing domain logic that doesn't fit in entities |

---

## Layer Boundaries

1. Domain cannot import from application or infrastructure
2. Application cannot import from infrastructure
3. Infrastructure can import from both application and domain
4. External frameworks should only appear in infrastructure
5. Tests may import from any layer but should prefer testing through ports
