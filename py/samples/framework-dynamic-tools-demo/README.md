# Dynamic Tools Demo

Demonstrates Genkit's dynamic tool creation (`ai.dynamic_tool()`) and
sub-span tracing (`ai.run()`). These features let you create tools at
runtime and wrap arbitrary functions as traceable steps.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Dynamic Tool Creation | `dynamic_tool_demo` | Create tools at runtime with `ai.dynamic_tool()` |
| Sub-span Tracing | `run_step_demo` | Wrap functions as traceable steps with `ai.run()` |
| Combined Usage | `combined_demo` | Both `ai.run()` and `ai.dynamic_tool()` together |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Dynamic Tool** | A tool created on-the-fly, not registered globally — like hiring a temp worker for one job |
| **`ai.run()`** | Wrap any function as a named step in the trace — like adding a bookmark in a logbook |

## Quick Start

```bash
export GEMINI_API_KEY=your-api-key
./run.sh
```

Then open the Dev UI at http://localhost:4000.

## Flows

| Flow | What It Demonstrates |
|------|---------------------|
| `dynamic_tool_demo` | Creates a tool at runtime with `ai.dynamic_tool()` and calls it |
| `run_step_demo` | Wraps a plain function as a traceable step with `ai.run()` |
| `combined_demo` | Uses both `ai.run()` and `ai.dynamic_tool()` together |

## Key APIs

- **`ai.dynamic_tool(name, fn, description=...)`** -- Creates a tool that is NOT globally
  registered. Useful for one-off tools or tools generated from user input.
- **`ai.run(name, input, fn)`** -- Wraps a function call as a named step in the
  trace. The step appears in the Dev UI trace viewer.

## Development

The `run.sh` script uses `watchmedo` for hot reloading on file changes.
