# Genkit Tools - Developer Guide

This document defines high-level architectural and testing conventions for contributors and AI coding agents working under `genkit-tools/` (`common`, `cli`, and `telemetry-server`).

## 1. Schemas, Types & Documentation

- **Reuse canonical types**: Check existing schemas and types in `common/src/types/` first and reuse them rather than introducing parallel definitions or overlapping variants.
- **Keep schema JSDocs domain-general**: Describe the domain semantics of types and fields rather than coupling schema documentation to a specific implementation detail or caller.

## 2. Testing & Skill Enforcement

- **CLI & MCP authoring**: When adding or modifying CLI commands (`cli/src/commands/`) or MCP tools (`cli/src/mcp/`), read and follow the `genkit-tools-cli` skill (`../.agents/skills/genkit-tools-cli/SKILL.md`).
- **Unit tests (`*.test.ts`)**: Whenever creating, modifying, fixing, or reviewing unit tests in `genkit-tools/`, read and follow the `genkit-tools-testing` skill (`../.agents/skills/genkit-tools-testing/SKILL.md`) first.
- **Verification**: Run `cd genkit-tools && pnpm lint && pnpm typecheck` and `pnpm test:genkit-tools` from the repo root (or `pnpm test` in a package directory).
