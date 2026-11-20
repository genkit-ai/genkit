---
name: genkit-tools-testing
description: Guide for writing and reviewing Vitest unit tests (*_test.ts) in genkit-tools/. Use when creating, modifying, fixing, or reviewing tests under genkit-tools/.
---

# Genkit Tools Testing Guide

## 1. Test Structure & Philosophy

- **One behavior per test**: Keep each `it(...)` block focused on a single behavior following `arrange -> act -> assert` with a clear, behavior-driven description. Split tests that assert multiple independent functions or behaviors.
- **Prefer single structured assertions**: Use one `toMatchObject` / `toEqual` assertion on structured outputs instead of scattering field-by-field `expect(...)` calls on the same object, and avoid asserting incidental ordering (e.g. `Object.keys(...)` insertion order) when order isn't part of the contract.
- **Parameterized variations (`it.each`)**: Use `it.each` when verifying the same behavior across multiple inputs:
  - Prefer **named-object rows** (`it.each([{ name: '...', ..., expected: ... }])` with `$name` in the test title) over positional tuples whenever rows have 3+ fields or complex objects.
  - Simple 2-element `[input, expected]` scalar tuples are fine for concise mappings; use separate `it(...)` blocks when precedence or setup differs noticeably.
- **Test our logic, not library internals**: Do not write unit tests that merely verify third-party library or Zod schema validation behavior. Focus on our domain logic, transformations, edge-case fallbacks, and error handling.

## 2. File Layout & Shared Fixtures

- **Mirror `src/` in `tests/`**: Place tests in `tests/<subpath>/<name>_test.ts` mirroring `src/<subpath>/<name>.ts`.
- **Keep fixtures separate from test suites**: Put reusable test builders and fixture constants in dedicated `testing/mock_data.ts` helpers rather than duplicating large inline objects across `*_test.ts` files.
