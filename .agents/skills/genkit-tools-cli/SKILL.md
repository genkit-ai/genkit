---
name: genkit-tools-cli
description: Guide for writing and reviewing CLI commands and MCP server tools in genkit-tools/cli. Use when creating, modifying, or reviewing commands in cli/src/commands/ or MCP tools in cli/src/mcp/.
---

# Genkit Tools CLI & MCP Guide

## 1. CLI Commands (`cli/src/commands/`)

- **Naming**: Use `noun:verb` or `category:action` in kebab-case (e.g. `flow:run`, `eval:run`, `trace:get`). Use camelCase for code arguments and `<argName>` in help text.
- **Runtime lifecycle**: Use `runWithManager(projectRoot, async (manager) => ...)` from `../utils/manager-utils.js` (with `findProjectRoot()` from `@genkit-ai/tools-common/utils`) for commands that interact with a project runtime (see `cli/src/commands/flow-run.ts`).
- **Output & flags**: Log via `logger` from `@genkit-ai/tools-common/utils`; support `--output` (file/JSON output) and `--stream` where appropriate.
- **Registration**: Register new commands in `cli/src/index.ts`.

## 2. MCP Tools (`cli/src/mcp/`)

- **Naming**: Use `verb_noun` in snake_case (e.g. `list_flows`, `run_flow`, `get_trace`).
- **Static vs. runtime tools**:
  - **Static tools** (e.g. `defineDocsTool(server, projectRoot?)` in `cli/src/mcp/docs.ts`): Require no running Genkit runtime.
  - **Runtime tools** (`defineFlowTools`, `defineTraceTools`, `defineRuntimeTools` taking `(server, options: McpToolOptions)`): Interact with a project runtime via `options.manager.getManager(rootOrError)`.
- **Schema & project root**: Always wrap tool inputs with `getCommonSchema(options.explicitProjectRoot, { ... })` and resolve the root with `resolveProjectRoot(options.explicitProjectRoot, opts, options.projectRoot)` before accessing `options.manager` (see `cli/src/mcp/flows.ts`).
- **Analytics & errors**: Record invocations with `record(new McpRunToolEvent('<tool_name>'))`. Catch errors and return `` `{ isError: true, content: [{ type: 'text', text: `Error: ${formatMcpError(e)}` }] }` `` instead of throwing.
- **Registration**: Wire new tool groups into `startMcpServer()` (and update server `instructions` if user-facing) in `cli/src/mcp/server.ts`.
