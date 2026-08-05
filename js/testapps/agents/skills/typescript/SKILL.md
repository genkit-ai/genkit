---
name: typescript
description: TypeScript coding conventions, best practices, and patterns for writing clean, maintainable code.
---

# TypeScript Coding Conventions

## General Principles

- Use **TypeScript strict mode** (`"strict": true` in tsconfig.json).
- Prefer `const` over `let`; never use `var`.
- Use explicit return types on exported functions.
- Prefer interfaces over type aliases for object shapes.
- Use `unknown` instead of `any` where possible.

## Naming Conventions

- **Variables & functions**: `camelCase`
- **Classes & interfaces**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE` for true constants, `camelCase` for derived values
- **Files**: `kebab-case.ts`
- **Type parameters**: Single uppercase letter (`T`, `K`, `V`) or descriptive (`TResult`)

## File Structure

```
src/
  index.ts          # Entry point, exports
  types.ts          # Shared type definitions
  utils/            # Utility functions
  services/         # Business logic
  middleware/       # Express/framework middleware
```

## Error Handling

- Use typed errors (extend `Error` class).
- Always handle promise rejections.
- Prefer `try/catch` with specific error types over generic catches.

```typescript
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500
  ) {
    super(message);
    this.name = 'AppError';
  }
}
```

## Async Patterns

- Use `async/await` over raw promises.
- Use `Promise.all()` for concurrent independent operations.
- Avoid mixing callbacks and promises.

## Imports

- Use named imports: `import { thing } from './module'`
- Group imports: external packages first, then internal modules.
- Use `.js` extensions in import paths for ESM compatibility.

## Code Style

- Maximum line length: 100 characters.
- Use template literals for string interpolation.
- Prefer `Array.map/filter/reduce` over `for` loops for transformations.
- Use optional chaining (`?.`) and nullish coalescing (`??`).
- Destructure objects and arrays when accessing multiple properties.

## Testing

- Co-locate test files: `thing.ts` → `thing.test.ts`
- Use descriptive test names: `it('should return empty array when no items match')`
- Test edge cases: empty inputs, null values, error conditions.
