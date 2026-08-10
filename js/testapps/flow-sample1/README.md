# Flow Fundamentals

Demonstrates core Genkit flow primitives — multi-step tracing, streaming,
context propagation, error handling, and long-running flows. No AI model
is used; these flows exercise the framework itself.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Basic Flow | `basic` | Multi-step flow with `ai.run()` traced steps |
| Context Propagation | `withContext` | Access flow context from within a flow |
| Streaming | `streamy` | Stream count objects at 1-second intervals |
| Error Handling | `throwy` | Flow that throws an error |
| Error in Steps | `throwy2` | Error thrown from a nested traced step |
| Multi-Step Tracing | `multiSteps` | Multiple traced steps with data transformation |
| Large Step Data | `largeSteps` | Steps with large payloads (~1MB strings) |
| Long-Running | `test-long-broadcast` | Multi-minute flow for broadcast testing |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Sample

```bash
pnpm build && pnpm start
```

Or via CLI:

```bash
genkit flow:run basic '"hello"'
genkit flow:run streamy 5 -s
```

## Testing This Demo

1. **Test basic flow**:
   ```bash
   genkit flow:run basic '"hello"'
   ```

2. **Test streaming**:
   ```bash
   genkit flow:run streamy 5 -s
   ```
   Should output `{count: 0}` through `{count: 4}` at 1-second intervals.

3. **Test error handling**:
   ```bash
   genkit flow:run throwy '"hello"'
   ```
   Should throw an error.

4. **Test multi-step**:
   ```bash
   genkit flow:run multiSteps '"world"'
   ```
   Should return a number after processing through multiple steps.

5. **Test long-running flow** (2-3 minutes):
   ```bash
   genkit flow:run test-long-broadcast '{"steps": 5, "stepDelay": 5000}'
   ```

6. **Expected behavior**:
   - `basic` returns a string after two traced steps
   - `streamy` streams objects at regular intervals
   - `throwy` / `throwy2` demonstrate error propagation in flows
   - `multiSteps` shows trace spans for each step in DevUI
