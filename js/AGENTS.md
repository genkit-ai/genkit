# Genkit JS - Developer Guide

This document defines build, packaging, documentation, and testing conventions for contributors and AI coding agents working under `js/` (`core`, `ai`, `genkit`, `plugins/*`, and `testapps/*`).

## 1. Workspace, Build & Packaging

- **Isolated workspace**: `js/` is its own pnpm workspace (`js/pnpm-workspace.yaml`), separate from the repo root and `genkit-tools/`. Run `pnpm i` and `pnpm build` from `js/` or a package directory (`js/<pkg>`), and run tests via `pnpm test` in `js/<pkg>` (or `pnpm test:all` from `js/`). Build `@genkit-ai/core`, `@genkit-ai/ai`, and `genkit` first (`pnpm --dir js build:core` followed by `pnpm --dir js build:genkit`, or `pnpm --dir js build`) before building or testing downstream plugins that depend on local workspace types.
- **TypeScript & `NodeNext` imports**: Packages extend `js/tsconfig.json` (`module: "NodeNext"`, `moduleResolution: "NodeNext"`, `strict: true`, and `esModuleInterop: false` preserved so downstream consumers do not need `esModuleInterop` or `skipLibCheck`). Use explicit `.js` extensions on all relative file imports in source files (e.g., `from './foo.js'`); bare relative file imports fail the ESM rewrite plugin in `js/tsup.common.ts`.
- **Shared `tsup` config**: Packages emit dual CJS (`.js`, `.d.ts`) and ESM (`.mjs`, `.d.mts`) outputs to `lib/` via `defaultOptions` from `js/tsup.common.ts`. Keep per-package `tsup.config.ts` files as thin wrappers over `defaultOptions` rather than adding custom per-package DTS bundling or `onSuccess` hooks.
- **`package.json` `exports` ordering**: Place `"types"` first and `"default"` last within every condition block. When separating ESM and CJS declarations (required for packages with `export default` under `NodeNext`, and used across `js/plugins/*/package.json`), nest `"types"` inside `"import"` (`./lib/<entry>.d.mts`, `"default"`: `./lib/<entry>.mjs`) and `"require"` (`./lib/<entry>.d.ts`, `"default"`: `./lib/<entry>.js`) rather than placing a flat `.d.ts` `"types"` entry above `"import"`.

## 2. Documentation, Testing & Verification

- **JSDocs & model identifiers**: Document exported SDK members with JSDoc comments (`CONTRIBUTING.md`). In JSDoc `@example` blocks, READMEs, and samples, use `@genkit-ai/google-genai` (e.g., `googleAI.model('gemini-flash-latest')` or current models in `js/plugins/google-genai/src/googleai/gemini.ts`) rather than legacy `gemini-1.5-*`, `gemini-2.0-*`, or `gemini-2.5-*` model IDs or the legacy `@genkit-ai/googleai` package.
- **Unit tests (`tests/*_test.ts`)**: Place unit tests in `tests/*_test.ts` using the package's configured runner (`node --import tsx --test` / `tsx --test`, or `jest` where configured) and run `pnpm test` in each touched package.
- **License headers & formatting**: Include the `Copyright <year> Google LLC` Apache-2.0 header on new `.ts`/`.js` source files, and run `pnpm format` from the repo root alongside `pnpm build` in touched packages before opening a PR.
