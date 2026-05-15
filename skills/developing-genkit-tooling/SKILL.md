---
name: developing-genkit-tooling
description: Best practices for authoring Genkit tooling, including CLI commands and MCP server tools. Covers naming conventions, architectural patterns, and consistency guidelines.
---

# Developing Genkit Tooling

## Naming Conventions

Consistency in naming helps users and agents navigate the tooling.

### CLI Commands

Use **kebab-case** with colon separators for subcommands.

- **Format**: `noun:verb` or `category:action`
- **Examples**: `flow:run`, `eval:run`, `init`
- **Arguments**: Use camelCase in code (`flowName`) but standard format in help text (`<flowName>`).

### MCP Tools

Use **snake_case** for tool names to align with MCP standards.

- **Format**: `verb_noun`
- **Examples**: `list_flows`, `run_flow`, `list_genkit_docs`, `read_genkit_docs`

## CLI Command Architecture

Commands are implemented in `cli/src/commands/` using `commander`.

### Runtime Interaction

Most commands require interacting with the user's project runtime. Use the `runWithManager` utility to handle the lifecycle of the runtime process.

```typescript
import { runWithManager } from '../utils/manager-utils';

// ... command definition ...
.action(async (arg, options) => {
  await runWithManager(await findProjectRoot(), async (manager) => {
    // Interact with manager here
    const result = await manager.runAction({ key: arg });
  });
});
```

### Output Formatting

- **Logging**: Use `logger` from `@genkit-ai/tools-common/utils`.
- **Machine Readable**: Provide options for JSON output or file writing when the command produces data.
- **Streaming**: If the operation supports streaming (like `flow:run`), provide a `--stream` flag and pipe output to stdout.

## MCP Tool Architecture

MCP tools in `cli/src/mcp/` follow two distinct patterns: **Static** and **Runtime**.

### Static Tools (e.g., Docs)

These tools do not require a running Genkit project context.

- **Registration**: `defineDocsTool(server: McpServer)`
- **Dependencies**: Only the `server` instance.
- **Use Case**: Documentation, usage guides, global configuration.

### Runtime Tools (e.g., Flows, Runtime Control)

These tools interact with a specific Genkit project's runtime.

- **Registration**: `defineRuntimeTools(server: McpServer, options: McpToolOptions)`
- **Dependencies**: Requires `options` containing `manager` (process manager) and `projectRoot`.
- **Schema**: MUST use `getCommonSchema(options.explicitProjectRoot, ...)` to ensure the tool can accept a `projectRoot` argument when required (e.g., in multi-project environments).

```typescript
// Runtime tool definition pattern
server.registerTool(
  'my_runtime_tool',
  {
    inputSchema: getCommonSchema(options.explicitProjectRoot, {
      myArg: z.string(),
    }),
  },
  async (opts) => {
    // Resolve project root before action
    const rootOrError = resolveProjectRoot(
      options.explicitProjectRoot,
      opts,
      options.projectRoot
    );
    if (typeof rootOrError !== 'string') return rootOrError;

    // access manager via options.manager
  }
);
```

### Error Handling

MCP tools should generally catch errors and return them as content blocks with `isError: true` rather than throwing exceptions, which ensures the client receives a structured error response.

```typescript
try {
  // operation
} catch (err) {
  const message = err instanceof Error ? err.message : String(err);
  return {
    isError: true,
    content: [{ type: 'text', text: `Error: ${message}` }],
  };
}
```
