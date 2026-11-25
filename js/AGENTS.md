# Genkit JS - Developer Guide

This document defines module, documentation, and testing conventions for contributors and AI coding agents working under `js/` (`core`, `ai`, `genkit`, `plugins/*`, and `testapps/*`).

## 1. Imports & Documentation

- **Relative imports**: Use explicit `.js` extensions on all relative file imports in source files (e.g., `from './foo.js'`).
- **JSDocs & examples**: Document exported SDK members with JSDoc comments (`CONTRIBUTING.md`). In JSDoc `@example` blocks, READMEs, and samples, use `@genkit-ai/google-genai` (e.g., `googleAI.model('gemini-flash-latest')` or current models in `js/plugins/google-genai/src/googleai/gemini.ts`) rather than the legacy `@genkit-ai/googleai` package or deprecated model IDs.

## 2. Testing & Verification

- **Unit tests (`tests/*_test.ts`)**: Place unit tests in `tests/*_test.ts`.
- **Verification**: Run `pnpm build:js` and `pnpm test:js` from the repo root (or `pnpm build` and `pnpm test` in a package directory; rebuild via `pnpm build:js` when editing `core`, `ai`, or `genkit`).
