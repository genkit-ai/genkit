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
capabilities: [<capability>]    # Every name `requires` may use; see below
tests:
  - name: <string>              # Human-readable test name
    description: <string>       # Optional description
    agent: <string>             # Name of the harness-provided agent
    requires: [<capability>]    # Optional capability gate; see below
    steps:                      # Ordered sequence of operations
      - type: send | getSnapshotData | abort | waitUntilCompleted
        ...                     # Fields depend on step type
```

`requires` lists the capabilities a test depends on. Each harness declares
the capabilities its runtime implements and **skips** (not fails) any test
naming one it does not, so the shared spec can carry cases for features an
SDK has not adopted yet.

Because every harness skips what it does not recognize, a misspelled name
would otherwise skip the test in all of them and be reported by none. The
top-level `capabilities` list is the spec's own registry of valid names, and
a `requires` entry absent from it **fails** in every harness. Adding a
capability means adding it there and to the table below. Known capabilities:

| Capability | Meaning |
|------------|---------|
| `resumable-failures` | A failed turn commits what the generate call left at its last turn seam and persists it as a `failed` snapshot carrying the error; resume accepts that snapshot, and an input with no payload of its own re-attempts the turn. Gates model `error` entries, the empty input, a `failed` snapshot as a resume target, and the `promptAgentWithToolsAndStore` fixture. Implemented by: Go. |
| `resumable-aborts` | An aborted invocation persists the state through the last turn that committed, rolling back the one that did not finish, and resume accepts that snapshot. Gates an `aborted` snapshot carrying state, one as a resume target, and the `customAgentAbortable` fixture. Implemented by: Go. |

### Step Types

#### `send`

Sends inputs to the agent via its bidirectional streaming interface (e.g.
`streamBidi` in JS).

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"send"` | Required. |
| `init` | `AgentInit` | Initialization payload. May contain `snapshotId`, `state`, or be empty `{}`. |
| `inputs` | `AgentInput[]` | Ordered list of inputs to send. Each may contain `messages`, `resume` (with `respond` and/or `restart`), and/or `detach`. An input with none of those continues the conversation already in the session (requires `resumable-failures`). |
| `modelResponses` | `GenerateResponseData[]` | Pre-programmed turns for the programmable model, one per `generate` call made by the agent. An entry whose `error` is set (`{ status, message }`) fails that call with a classified error rather than returning (requires `resumable-failures`). |
| `streamChunks` | `GenerateResponseChunkData[][]` | Optional. Pre-programmed streaming chunks, indexed by model call. Each inner array is emitted as a stream before the corresponding `modelResponses` entry. |
| `expectChunks` | `AgentStreamChunk[]` | **Strict ordered** list of expected stream chunks. |
| `expectOutput` | Object | Expected fields on the `AgentOutput`. See [Output Assertions](#output-assertions). |
| `expectError` | Object | Optional. Asserts the turn *throws* (rather than resolving with a graceful `finishReason: 'failed'` output). Used for API-misuse cases (e.g. sending `state` to a server-managed agent). Fields: `status` (matched exactly) and `message` (matched as a substring). Mutually exclusive with `expectOutput`. |
| `captureSnapshotId` | `string` | Optional. Stores `output.snapshotId` under this name for use in later steps via `{{name}}`. |
| `captureState` | `string` | Optional. Stores `output.state` under this name for use in later steps via `{{name}}`. |
| `captureSessionId` | `string` | Optional. Stores `output.state.sessionId` under this name for use in later steps via `{{name}}`. |

#### `getSnapshotData`

Fetches a snapshot and asserts on its contents. Resolve the snapshot either by
an exact `snapshotId` or by a `sessionId` (which returns the session's latest
leaf snapshot). Exactly one of `snapshotId` / `sessionId` must be provided.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"getSnapshotData"` | Required. |
| `snapshotId` | `string` | Exact snapshot ID to fetch. Supports `{{name}}` references. Mutually exclusive with `sessionId`. |
| `sessionId` | `string` | Session ID (a UUID) whose latest (leaf) snapshot is resolved. Mutually exclusive with `snapshotId`. |
| `expectSnapshot` | Object | See [Snapshot Assertions](#snapshot-assertions). |
| `expectError` | `string` | If present, the lookup is expected to throw an error whose message contains this substring (e.g. resolving a branching session by `sessionId`). |


#### `abort`

Aborts an agent by snapshot ID.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"abort"` | Required. |
| `snapshotId` | `string` | The snapshot ID to abort. Supports `{{name}}` references. |
| `expectPreviousStatus` | `string` | Expected previous status before abort (e.g. `"pending"`, `"completed"`). YAML `~` means absent. |

#### `waitUntilCompleted`

Polls a snapshot until it reaches a terminal status (`completed`, `failed`, or
`aborted`).

An aborted snapshot reaches its status in two writes: the abort flips it, and
the finalize that follows stamps the finish reason and the state. This step
waits for the second one, so `expectSnapshot` never reads a row that is still
being written.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"waitUntilCompleted"` | Required. |
| `snapshotId` | `string` | The snapshot ID to poll. Supports `{{name}}` references. |
| `timeoutMs` | `number` | Optional. Max time to wait in milliseconds. Default: `5000`. |
| `expectSnapshot` | Object | See [Snapshot Assertions](#snapshot-assertions). |

---

### Output Assertions

Used in `expectOutput` for `send` steps.

| Field | Type | Description |
|-------|------|-------------|
| `message` | `MessageData` | If present, `output.message` must deep-equal this value. |
| `hasSnapshotId` | `boolean` | If `true`, asserts `output.snapshotId` is a non-empty string. |
| `hasSessionId` | `boolean` | If `true`, asserts `output.state.sessionId` is a non-empty string. |
| `stateContains` | `SessionState` (partial) | If present, asserts that `output.state` contains (at minimum) these fields. Uses "contains" / subset matching — the actual state may have additional fields. |
| `artifactsContain` | `Artifact[]` | If present, asserts that `output.artifacts` contains (at minimum) these entries. |
| `finishReason` | `string` | If present, `output.finishReason` must equal this value exactly (e.g. `stop` on a normal completion, `interrupted` on a tool pause, `failed` on a graceful failure). |
| `errorContains` | `object` (partial) | If present, asserts `output.error` contains these fields. `status` is matched exactly; `message` is matched as a substring. Set by the graceful-failure path when `finishReason` is `failed`. |


### Snapshot Assertions

Used in `expectSnapshot` for `getSnapshotData` and `waitUntilCompleted`
steps.

| Field | Type | Description |
|-------|------|-------------|
| `parentId` | `string` | Expected `parentId`. Supports `{{name}}` references. |
| `status` | `string` | Expected `status` (e.g. `"completed"`, `"pending"`, `"failed"`, `"aborted"`). |
| `finishReason` | `string` | Expected `snapshot.finishReason` (e.g. `failed`). Distinct from `status` — a failed run records `finishReason: failed` in addition to `status: failed`. |
| `hasSessionId` | `boolean` | If `true`, asserts `snapshot.state.sessionId` is a non-empty string. |
| `stateContains` | `SessionState` (partial) | Subset match on `snapshot.state`. |
| `errorContains` | `object` (partial) | If present, asserts that `snapshot.error` contains (at minimum) these fields. Uses "contains" / subset matching. |


---


### Template References

Values of the form `{{name}}` are replaced at runtime with previously captured
values:

- `captureSnapshotId: snap1` → captures `output.snapshotId` as `snap1`
- `captureState: state1` → captures `output.state` as `state1`
- `captureSessionId: sess1` → captures `output.state.sessionId` as `sess1`

These can be used anywhere a `snapshotId` or `state` is expected in subsequent
steps:

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
| `expectChunks` | **Semi-strict**: the actual and expected chunk lists must have the same length and order. Individual chunks are matched with type-aware logic: `turnEnd` chunks assert the key is present (the `snapshotId` is dynamic) and, when the spec specifies `turnEnd.finishReason`, assert that field matches exactly; `modelChunk`, `artifact`, and `customPatch` chunks use partial/contains matching on their payload. For `customPatch`, this means each expected JSON Patch operation must appear (in order) and match on the fields it specifies (e.g. `op` + `path`), so incremental diff ops can be asserted without pinning their exact `value` shape. |

| `stateContains` | **Partial**: each specified field must be present and match. Additional fields in the actual state are ignored. For `messages`, the listed messages must appear in the same relative order but need not be contiguous (ordered subsequence matching). |
| `artifactsContain` | **Partial**: each specified artifact must be present (matched by name). |
| `message` | **Strict**: deep-equality on the message object. |
| `hasSnapshotId` | **Boolean**: asserts presence of a non-empty string. |
| `hasSessionId` | **Boolean**: asserts `state.sessionId` is a non-empty string. |

---

## 3. Harness Requirements

Each language must provide a test harness that:

1. **Parses** `tests/specs/agent.yaml`.
2. **Registers** the required harness-provided agents (see below).
3. **Runs** each test by executing its step sequence.
4. **Asserts** results according to the spec.

### Required Agents

The harness must register the following named agents.

#### Prompt-backed agents

These use a **programmable model** whose responses can be controlled per-test
via the `modelResponses` / `streamChunks` fields in `send` steps.

| Agent Name | Description |
|------------|-------------|
| `promptAgent` | A prompt agent (equivalent to `defineAgent`) backed by the programmable model. **Client-managed** state (no store). |
| `promptAgentWithStore` | Same as `promptAgent` but with a **server-managed** in-memory session store. |
| `promptAgentWithTools` | A prompt agent with `testTool` registered. Client-managed state. |
| `promptAgentWithInterrupt` | A prompt agent with `interruptTool` registered and a server-managed store (for snapshot-based resume). |
| `promptAgentWithRestartTool` | A prompt agent with `restartTool` registered and a server-managed store. Used for `resume.restart` tests. |
| `promptAgentWithToolsAndStore` | A prompt agent with `testTool` and `flakyTool` registered and a server-managed store. Used for `resumable-failures` tests; only required by harnesses declaring that capability. |

#### Custom agents (hardcoded behavior)

These agents use `defineCustomAgent` with fixed, deterministic logic.
They do **not** use the programmable model — the `modelResponses` field
is not needed for tests targeting these agents.

| Agent Name | Description |
|------------|-------------|
| `customAgentAbortable` | Server-managed. Records each turn's input, replies `ack`, and commits. On the turn whose message is `block` it blocks until its context is cancelled and commits nothing. Used for `resumable-aborts` tests; only required by harnesses declaring that capability. |
| `customAgentBlocking` | Server-managed. Blocks indefinitely until its abort signal fires. Used for abort-while-pending tests. |
| `customAgentFailing` | Server-managed. Throws `Error('intentional failure')` during processing. Used for detach + background failure tests. |
| `customAgentWithArtifacts` | Client-managed. Adds artifact `doc1` (v1), updates it to `doc1` (v2), then adds `doc2`. Returns all artifacts. |
| `customAgentWithCustomState` | Client-managed. Reads `custom.counter`, increments it (default 0→1), and persists it. Returns `{ text: 'done' }`. |
| `customAgentWithMultiCustomState` | Client-managed. Performs three sequential custom-state updates in a single turn (`{ counter: 1, status: 'working' }`, then `counter: 2`, then `status: 'done'`). Used to verify the `customPatch` streaming contract (first patch is a whole-document replace, subsequent patches are incremental diffs). |
| `customAgentWithArtifactsStore` | Server-managed. Adds a numbered artifact (`doc1`, `doc2`, …) on each invocation based on existing artifact count. Returns all accumulated artifacts. Used for artifact persistence across snapshots. |
| `customAgentWithCustomStateStore` | Server-managed. Same counter logic as `customAgentWithCustomState` but with a server-managed store. Used for custom state persistence via snapshots. |

### Required Tools

| Tool Name | Description | Input Schema | Output |
|-----------|-------------|--------------|--------|
| `testTool` | A simple tool | `{}` (empty) | `"tool called"` (string) |
| `interruptTool` | An interrupt tool | `{ query: string }` | `{ answer: string }` |
| `restartTool` | A tool that requires confirmation; throws `ToolInterruptError` on first call, succeeds when `resumed` metadata is provided | `{ action: string }` | `{ result: string }` |
| `flakyTool` | Fails its first call in each test with an UNAVAILABLE error (`flaky tool failed`), succeeds after with `"tool recovered"`. Requires `resumable-failures`. | `{}` (empty) | `"tool recovered"` (string) |

### Programmable Model

The harness must provide a model (named `programmableModel`) whose response
behavior can be programmed per model call within a test. For each `send`
step:

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

### Python ✅

```bash
cd py
uv run pytest packages/genkit/tests/genkit/ai/agent_conformance_test.py
```

### Go ✅

```bash
cd go
go test ./ai/exp/ -run TestAgentConformance
```

---

## 5. Test Coverage

The spec currently covers the following categories (39 tests total):

| Category | Tests |
|----------|-------|
| Basic single-turn | Client-managed, server-managed |
| Streaming | Model chunk forwarding |
| Multi-turn | Multiple turns in one step |
| Tool calling | Automatic tool execution, multiple tool calls in one response |
| Interrupt & resume | Snapshot-based tool interrupt resume, multiple interrupt requests, state accumulation after resume |
| Interrupt & restart | Tool interrupt with `resume.restart` (re-execute with metadata) |
| Resume validation | Forged restart inputs rejected, non-existent tool respond rejected |
| Snapshot chaining | Parent chain across steps |
| Snapshot branching | Forking from a snapshot into independent histories |
| Server-managed sessions by sessionId | Resume by `sessionId`, fetch latest snapshot by `sessionId`, branching session lookup rejected, non-UUID `sessionId` accepted (any non-empty string), client-managed agent rejects `sessionId` (throws), `snapshotId`+`sessionId` mismatch rejected (throws) |
| Client-managed state | State seeding across steps |
| Server-managed state | Init state rejected for server-managed agents (throws) |
| Detach | Background completion, background failure, pure detach without payload |

| Abort | Pending agent, completed agent, non-existent snapshot, failed agent, already-aborted agent |
| Error details | Failed snapshot includes error message |
| Artifacts | Streamed chunks, deduplication by name, persistence across invocations (server-managed) |
| Custom state | Update during execution, persistence across steps (client-managed), persistence via snapshots (server-managed) |
| Custom state streaming (`customPatch`) | Single mutation streamed as a whole-document replace at root; multiple mutations stream a whole-document replace followed by incremental diffs; detached runs emit no `customPatch` chunks (mutation still persisted) |
| Finish reasons & graceful errors | `finishReason` `stop` on normal completions (output + `turnEnd`), `interrupted` on tool pauses, `failed` on rejected/failed turns; `failed` recorded on failure snapshots; structured `error` (`status` + `message`) surfaced on graceful failures instead of throwing |


## 6. Future Extensions

The spec is designed to grow. Planned additions:

- **Client state transform** (verifying redaction)
- **Error cases** (detach without store, missing snapshot, etc.)
- **Multi-agent orchestration** (agent-to-agent delegation)
- **Concurrent turns** (parallel input processing)
