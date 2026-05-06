# Agent Conformance Testing

**Status**: Active  
**Spec Location**: `tests/specs/agent.yaml`

---

## 1. Overview

The agent conformance spec defines behavioral tests for the Genkit Agent
abstraction. Each language implementation (JS, Go, Dart, Python, …) provides a
**test harness** that reads the shared YAML spec and executes the tests against
its own agent implementation. This ensures all implementations behave
identically at the wire-protocol level.

The pattern mirrors `tests/specs/generate.yaml` for the Generate API.

---

## 2. Spec Format Reference

### Top-Level Structure

```yaml
tests:
  - name: <string>              # Human-readable test name
    description: <string>       # Optional description
    agent: <string>             # Name of the harness-provided agent
    invocations:                # Ordered sequence of operations
      - type: send | getSnapshotData | abort | waitUntilCompleted
        ...                     # Fields depend on invocation type
```

### Invocation Types

#### `send`

Sends inputs to the agent via its bidirectional streaming interface (e.g.
`streamBidi` in JS).

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"send"` | Required. |
| `init` | `AgentInit` | Initialization payload. May contain `snapshotId`, `state`, or be empty `{}`. |
| `inputs` | `AgentInput[]` | Ordered list of inputs to send. Each may contain `messages`, `toolRestarts`, and/or `detach`. |
| `modelResponses` | `GenerateResponseData[]` | Pre-programmed responses for the programmable model, one per `generate` call made by the agent. |
| `streamChunks` | `GenerateResponseChunkData[][]` | Optional. Pre-programmed streaming chunks, indexed by model call. Each inner array is emitted as a stream before the corresponding `modelResponses` entry. |
| `expectChunks` | `AgentStreamChunk[]` | **Strict ordered** list of expected stream chunks. |
| `expectOutput` | Object | Expected fields on the `AgentOutput`. See [Output Assertions](#output-assertions). |
| `captureSnapshotId` | `string` | Optional. Stores `output.snapshotId` under this name for use in later invocations via `{{name}}`. |
| `captureState` | `string` | Optional. Stores `output.state` under this name for use in later invocations via `{{name}}`. |

#### `getSnapshotData`

Fetches a snapshot by ID and asserts on its contents.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"getSnapshotData"` | Required. |
| `snapshotId` | `string` | The snapshot ID to fetch. Supports `{{name}}` references. |
| `expectSnapshot` | Object | See [Snapshot Assertions](#snapshot-assertions). |

#### `abort`

Aborts an agent by snapshot ID.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"abort"` | Required. |
| `snapshotId` | `string` | The snapshot ID to abort. Supports `{{name}}` references. |
| `expectPreviousStatus` | `string` | Expected previous status before abort (e.g. `"pending"`, `"done"`). |

#### `waitUntilCompleted`

Polls a snapshot until it reaches a terminal status (`done`, `failed`, or
`aborted`).

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"waitUntilCompleted"` | Required. |
| `snapshotId` | `string` | The snapshot ID to poll. Supports `{{name}}` references. |
| `timeoutMs` | `number` | Optional. Max time to wait in milliseconds. Default: `5000`. |
| `expectSnapshot` | Object | See [Snapshot Assertions](#snapshot-assertions). |

---

### Output Assertions

Used in `expectOutput` for `send` invocations.

| Field | Type | Description |
|-------|------|-------------|
| `message` | `MessageData` | If present, `output.message` must deep-equal this value. |
| `hasSnapshotId` | `boolean` | If `true`, asserts `output.snapshotId` is a non-empty string. |
| `stateContains` | `SessionState` (partial) | If present, asserts that `output.state` contains (at minimum) these fields. Uses "contains" / subset matching — the actual state may have additional fields. |
| `artifactsContain` | `Artifact[]` | If present, asserts that `output.artifacts` contains (at minimum) these entries. |

### Snapshot Assertions

Used in `expectSnapshot` for `getSnapshotData` and `waitUntilCompleted`
invocations.

| Field | Type | Description |
|-------|------|-------------|
| `parentId` | `string` | Expected `parentId`. Supports `{{name}}` references. |
| `status` | `string` | Expected `status` (e.g. `"done"`, `"pending"`, `"failed"`, `"aborted"`). |
| `stateContains` | `SessionState` (partial) | Subset match on `snapshot.state`. |
| `errorContains` | `object` (partial) | If present, asserts that `snapshot.error` contains (at minimum) these fields. Uses "contains" / subset matching. |

---

### Template References

Values of the form `{{name}}` are replaced at runtime with previously captured
values:

- `captureSnapshotId: snap1` → captures `output.snapshotId` as `snap1`
- `captureState: state1` → captures `output.state` as `state1`

These can be used anywhere a `snapshotId` or `state` is expected in subsequent
invocations:

```yaml
init: { snapshotId: '{{snap1}}' }
init: { state: '{{state1}}' }
snapshotId: '{{snap1}}'
```

Only simple `{{name}}` syntax is supported — no dot-paths or expressions.

---

### Assertion Semantics

| Assertion Type | Semantics |
|----------------|-----------|
| `expectChunks` | **Semi-strict**: the actual and expected chunk lists must have the same length and order. Individual chunks are matched with type-aware logic: `turnEnd` chunks only assert the key is present (the `snapshotId` is dynamic); `modelChunk` and `artifact` chunks use partial/contains matching on their payload. |
| `stateContains` | **Partial**: each specified field must be present and match. Additional fields in the actual state are ignored. For `messages`, the listed messages must appear in the same relative order but need not be contiguous (ordered subsequence matching). |
| `artifactsContain` | **Partial**: each specified artifact must be present (matched by name). |
| `message` | **Strict**: deep-equality on the message object. |
| `hasSnapshotId` | **Boolean**: asserts presence of a non-empty string. |

---

## 3. Harness Requirements

Each language must provide a test harness that:

1. **Parses** `tests/specs/agent.yaml`.
2. **Registers** the required harness-provided agents (see below).
3. **Runs** each test by executing its invocation sequence.
4. **Asserts** results according to the spec.

### Required Agents

The harness must register the following named agents.

#### Prompt-backed agents

These use a **programmable model** whose responses can be controlled per-test
via the `modelResponses` / `streamChunks` fields in `send` invocations.

| Agent Name | Description |
|------------|-------------|
| `promptAgent` | A prompt agent (equivalent to `defineAgent`) backed by the programmable model. **Client-managed** state (no store). |
| `promptAgentWithStore` | Same as `promptAgent` but with a **server-managed** in-memory session store. |
| `promptAgentWithTools` | A prompt agent with `testTool` registered. Client-managed state. |
| `promptAgentWithInterrupt` | A prompt agent with `interruptTool` registered and a server-managed store (for snapshot-based resume). |

#### Custom agents (hardcoded behavior)

These agents use `defineCustomAgent` with fixed, deterministic logic.
They do **not** use the programmable model — the `modelResponses` field
is not needed for tests targeting these agents.

| Agent Name | Description |
|------------|-------------|
| `customAgentBlocking` | Server-managed. Blocks indefinitely until its abort signal fires. Used for abort-while-pending tests. |
| `customAgentFailing` | Server-managed. Throws `Error('intentional failure')` during processing. Used for detach + background failure tests. |
| `customAgentWithArtifacts` | Client-managed. Adds artifact `doc1` (v1), updates it to `doc1` (v2), then adds `doc2`. Returns all artifacts. |
| `customAgentWithCustomState` | Client-managed. Reads `custom.counter`, increments it (default 0→1), and persists it. Returns `{ text: 'done' }`. |

### Required Tools

| Tool Name | Description | Input Schema | Output |
|-----------|-------------|--------------|--------|
| `testTool` | A simple tool | `{}` (empty) | `"tool called"` (string) |
| `interruptTool` | An interrupt tool | `{ query: string }` | `{ answer: string }` |

### Programmable Model

The harness must provide a model (named `programmableModel`) whose response
behavior can be programmed per model call within a test. For each `send`
invocation:

- `modelResponses[i]` is returned for the i-th `generate` call.
- `streamChunks[i]` (if present) is emitted as streaming chunks before the
  i-th response.

The programmable model must support tool definitions in requests (it receives
them but the harness controls responses).

---

## 4. Running Tests

### JavaScript ✅

The JS harness is the current reference implementation.

```bash
cd js/ai
npx tsx --test tests/agents_spec_test.ts
```

### Go ⏳

_(Coming soon — implement a Go harness that reads the same YAML spec.)_

---

## 5. Test Coverage

The spec currently covers the following categories (19 tests total):

| Category | Tests |
|----------|-------|
| Basic single-turn | Client-managed, server-managed |
| Streaming | Model chunk forwarding |
| Multi-turn | Multiple turns in one invocation |
| Tool calling | Automatic tool execution |
| Interrupt & resume | Snapshot-based tool interrupt resume |
| Snapshot chaining | Parent chain across invocations |
| Client-managed state | State seeding across invocations |
| Server-managed state | Init state ignored for server-managed agents |
| Detach | Background completion, background failure, pure detach without payload |
| Abort | Pending agent, completed agent, non-existent snapshot |
| Error details | Failed snapshot includes error message |
| Artifacts | Streamed chunks, deduplication by name |
| Custom state | Update during execution, persistence across invocations |

## 6. Future Extensions

The spec is designed to grow. Planned additions:

- **Client state transform** (verifying redaction)
- **Error cases** (detach without store, missing snapshot, etc.)
- **Multi-agent orchestration** (agent-to-agent delegation)
- **Concurrent turns** (parallel input processing)
