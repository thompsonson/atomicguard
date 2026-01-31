# Problem Statement: Code Review Agent

## Goal

Build an AI agent that assists developers by automatically reviewing code changes in pull requests and providing constructive feedback on code quality, potential bugs, security issues, and adherence to best practices.

## Context

The agent operates in a software development environment where:

- Developers submit pull requests with code changes on GitHub
- Code must adhere to team style guidelines and architectural patterns
- Security vulnerabilities should be detected before code reaches production
- Performance issues and anti-patterns should be flagged early
- The agent integrates with the existing CI/CD pipeline

## Requirements

1. Review code diffs from pull requests and identify issues
2. Check for style violations against configurable rule sets
3. Identify potential security vulnerabilities (OWASP Top 10)
4. Detect common anti-patterns and suggest improvements
5. Flag performance issues and inefficient code patterns
6. Generate actionable, constructive feedback comments
7. Prioritize issues by severity (critical, warning, suggestion)
8. Support multiple programming languages (Python, TypeScript, Go)
9. Learn from developer feedback to improve suggestions over time

## Constraints

- Reviews must complete within 60 seconds for typical PRs (< 500 lines)
- Must integrate with GitHub API for PR access and commenting
- Feedback should be constructive and specific, not vague
- False positive rate should be minimized to avoid alert fatigue
- Must handle rate limiting from external APIs gracefully
- Should operate within CI/CD resource constraints (memory, CPU)
- Must not expose sensitive code or credentials in logs
