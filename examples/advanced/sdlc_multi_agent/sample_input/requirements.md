# Requirements: Simple Library Management System

Build a simple library management system using Domain-Driven Design principles.

## Functional Requirements

### Core Features
1. **Book Management**
   - Add new books to the library catalog
   - Track book availability (available, borrowed, reserved)
   - Each book has: ISBN, title, author, publication year

2. **Member Management**
   - Register library members
   - Each member has: member ID, name, email
   - Track member borrowing history

3. **Borrowing Operations**
   - Members can borrow available books
   - Maximum 3 books per member
   - Borrowing period: 14 days
   - Return books and update availability

### Business Rules
- A book cannot be borrowed if it's already borrowed or reserved
- A member cannot borrow more than 3 books at a time
- Overdue books (> 14 days) should be tracked
- ISBN must be unique
- Member email must be unique

## Non-Functional Requirements

### Architecture
- Follow Domain-Driven Design (DDD) patterns
- Use clean architecture with layers:
  - Domain layer (entities, value objects, repositories)
  - Application layer (use cases)
  - Infrastructure layer (implementations)

### Technical Constraints
- Python 3.10+
- Type hints required
- No external dependencies in domain layer
- Use factory patterns for entity creation
- Repository interfaces (no concrete implementations needed for PoC)

### Testing
- Include basic unit tests for domain entities
- Test state transitions (book availability, borrowing limits)
- Test business rule enforcement

## Expected Domain Model

### Entities
- **Book** (aggregate root): ISBN, title, author, year, status
- **Member** (aggregate root): member_id, name, email, borrowed_books
- **BorrowRecord**: book_id, member_id, borrow_date, due_date, return_date

### Value Objects
- **ISBN**: Validated ISBN-13 format
- **Email**: Validated email format
- **BorrowPeriod**: 14-day period with due date calculation

### Repositories (interfaces only)
- BookRepository
- MemberRepository
- BorrowRecordRepository

## Success Criteria
- All domain entities implemented with type hints
- Business rules enforced (borrowing limits, availability checks)
- State transitions implemented (book status changes)
- Basic unit tests pass
- Clean layer separation (domain, application, infrastructure)
