# Feature Requirements: Task Management System

## Overview

Implement a task management system that allows users to create, update, and track tasks with priorities and due dates.

## User Stories

### US-1: Create Task

As a user, I want to create a new task with a title, description, and priority so that I can track my work items.

**Acceptance Criteria:**

- Task must have a title (required, max 200 chars)
- Task can have an optional description (max 2000 chars)
- Priority must be one of: low, medium, high, critical
- Task is assigned a unique ID upon creation
- Task starts in "pending" status

### US-2: Update Task Status

As a user, I want to update the status of a task so that I can track its progress.

**Acceptance Criteria:**

- Valid status transitions:
  - pending → in_progress
  - in_progress → completed | blocked
  - blocked → in_progress
  - completed → (terminal, no transitions)
- Invalid transitions should raise an error
- Status change should record timestamp

### US-3: Set Due Date

As a user, I want to set a due date for a task so that I can manage deadlines.

**Acceptance Criteria:**

- Due date must be in the future when set
- System should flag tasks as "overdue" if past due date and not completed
- Due date can be updated or removed

### US-4: List Tasks by Priority

As a user, I want to list tasks sorted by priority so that I can focus on what's most important.

**Acceptance Criteria:**

- Tasks ordered: critical > high > medium > low
- Within same priority, order by due date (soonest first)
- Tasks without due dates come after those with due dates

## Architecture

### Layer Structure

```
src/taskmanager/
├── domain/
│   ├── entities.py      # Task entity
│   ├── value_objects.py # Priority, Status, TaskId
│   └── exceptions.py    # DomainError, InvalidTransition
├── application/
│   ├── use_cases.py     # CreateTask, UpdateStatus, SetDueDate
│   ├── ports.py         # TaskRepository (abstract)
│   └── services.py      # TaskService orchestration
└── infrastructure/
    ├── repositories.py  # InMemoryTaskRepository
    └── clock.py         # SystemClock for time operations
```

### Architecture Gates

#### Gate 1: Domain Independence

The domain layer MUST NOT import from application or infrastructure.

- `domain/` modules can only import from standard library and `domain/`
- Domain entities are pure Python dataclasses with no external dependencies

#### Gate 2: Application Abstraction

The application layer MUST depend on abstractions, not implementations.

- Use cases receive `TaskRepository` protocol, not `InMemoryTaskRepository`
- Time operations use `Clock` protocol, not `SystemClock`

#### Gate 3: Dependency Direction

Dependencies flow inward: Infrastructure → Application → Domain

- Infrastructure implements the ports defined in application
- Never import from outer layers into inner layers

## Technical Constraints

- Python 3.11+
- No external dependencies in domain layer
- Use Pydantic for validation at infrastructure boundaries only
- 100% test coverage for domain and application layers
