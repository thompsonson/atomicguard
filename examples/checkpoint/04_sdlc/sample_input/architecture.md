# Architecture Documentation: Task Management System

## System Overview

A clean architecture implementation of a task management system following Domain-Driven Design principles.

## Layer Definitions

### Domain Layer (`src/taskmanager/domain/`)

The innermost layer containing business logic and rules.

#### Entities

**Task** - The core aggregate root

```python
@dataclass
class Task:
    id: TaskId
    title: str
    description: str | None
    priority: Priority
    status: Status
    due_date: datetime | None
    created_at: datetime
    updated_at: datetime
```

#### Value Objects

**TaskId** - Unique identifier (UUID-based)
**Priority** - Enum: LOW, MEDIUM, HIGH, CRITICAL
**Status** - Enum: PENDING, IN_PROGRESS, BLOCKED, COMPLETED

#### Domain Rules

1. Title validation: non-empty, max 200 characters
2. Description validation: max 2000 characters if present
3. Status transitions are explicitly validated
4. Due date must be in future when set

### Application Layer (`src/taskmanager/application/`)

Use cases and orchestration logic.

#### Ports (Interfaces)

**TaskRepository Protocol**

```python
class TaskRepository(Protocol):
    def save(self, task: Task) -> None: ...
    def get(self, task_id: TaskId) -> Task | None: ...
    def list_all(self) -> list[Task]: ...
    def delete(self, task_id: TaskId) -> bool: ...
```

**Clock Protocol**

```python
class Clock(Protocol):
    def now(self) -> datetime: ...
```

#### Use Cases

1. **CreateTaskUseCase** - Creates new task with validation
2. **UpdateStatusUseCase** - Validates and applies status transitions
3. **SetDueDateUseCase** - Sets or updates task due date
4. **ListTasksUseCase** - Returns tasks sorted by priority/due date

### Infrastructure Layer (`src/taskmanager/infrastructure/`)

Concrete implementations of ports.

#### Repositories

**InMemoryTaskRepository** - Dict-based storage for development/testing

#### Adapters

**SystemClock** - Wraps `datetime.now()` for testability

## Architecture Gates

### Gate 1: Domain Purity

```
domain/ MUST NOT import from application/ or infrastructure/
```

Rationale: Domain logic should be framework-agnostic and testable in isolation.

### Gate 2: Port Abstraction

```
application/ MUST depend on Protocol types, not concrete implementations
```

Rationale: Enables dependency injection and test doubles.

### Gate 3: Dependency Inversion

```
infrastructure/ → application/ → domain/
Never: domain/ → application/ or infrastructure/
```

Rationale: Outer layers can change without affecting inner layers.

## Testing Strategy

### Unit Tests (Domain)

- Test entities and value objects in isolation
- Verify domain rules and invariants
- No mocking required

### Integration Tests (Application)

- Test use cases with mock repositories
- Verify orchestration logic
- Use test doubles for Clock

### Architecture Tests (pytest-arch)

- Verify import constraints
- Enforce layer boundaries
- Run on every CI build

## Package Structure

```
src/
└── taskmanager/
    ├── __init__.py
    ├── domain/
    │   ├── __init__.py
    │   ├── entities.py
    │   ├── value_objects.py
    │   └── exceptions.py
    ├── application/
    │   ├── __init__.py
    │   ├── ports.py
    │   ├── use_cases.py
    │   └── services.py
    └── infrastructure/
        ├── __init__.py
        ├── repositories.py
        └── clock.py

tests/
├── unit/
│   ├── domain/
│   │   └── test_entities.py
│   └── application/
│       └── test_use_cases.py
├── integration/
│   └── test_task_service.py
└── architecture/
    └── test_layer_boundaries.py
```
