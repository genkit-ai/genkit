# Genkit Monorepo Guide

`js/`, `genkit-tools/`, and `go/` are separate workspaces with independent lockfiles and builds. Read the area guide before planning, editing, or reviewing changes:

- `js/` (JS SDK & plugins): `js/AGENTS.md`
- `genkit-tools/` (CLI, MCP, telemetry server, shared schemas): `genkit-tools/AGENTS.md`
- `go/` (Go SDK & plugins): `go/GEMINI.md`

## Repo-Wide Checks

- Run `pnpm format` from the repo root to apply Prettier formatting and add missing `Copyright <year> Google LLC` Apache-2.0 headers (`pnpm format:check` to verify; `go fmt -C go ./...` for Go).
- Use Conventional Commit and PR headers (`<= 120` chars, non-empty scope per `commitlint.config.js`) with scopes prefixed by the workspace directory, e.g. `fix(js/ai): ...`, `fix(genkit-tools/cli): ...`, or `feat(go/ai): ...`.
