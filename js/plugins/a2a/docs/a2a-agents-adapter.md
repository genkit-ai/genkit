# A2A ↔ Agent Flow Adapter Design

## Overview

This document captures the design for bidirectional integration between Genkit's
Agent Flow abstraction and the [A2A (Agent-to-Agent) protocol](https://github.com/a2aproject/a2a-js).

Two adapters are needed:

1. **Consume** (`defineA2AAgent`): Wrap a remote A2A agent as a local Genkit
   `Agent`, so it can be used with `streamBidi()`, `.run()`, as a sub-agent via
   the `agents()` middleware, etc.
2. **Expose** (`GenkitA2ARequestHandler`): Serve an existing Genkit `Agent` as
   an A2A-compliant endpoint so external A2A clients can interact with it.

**Design principle: start simple.** The initial implementation focuses on
synchronous request/response (`message/send` blocking mode). Streaming support
(`message/stream`) can be added later as an enhancement.

---

## Architecture Context

### Genkit Agent Flow

The agent flow architecture (`js/ai/src/session-flow.ts`) provides:

- **`Agent`** — a `BidiAction` that accepts `AgentInput`
  (messages / toolRestarts / detach) over a bidirectional stream, maintains
  session state via `SessionRunner`, and emits `AgentStreamChunk` (modelChunk,
  status, artifact, turnEnd) plus a final `AgentOutput`.
- **`defineCustomAgent`** — register an agent with a custom handler (`AgentFn`).
- **`defineAgent` / `definePromptAgent`** — prompt-backed convenience wrappers.
- **State management**: either _client-managed_ (state returned in
  `AgentOutput.state`, passed back in `AgentInit.state`) or _server-managed_
  (via a `SessionStore` + `snapshotId`).
- Features: detach (background execution), abort, multi-turn, interrupts,
  artifacts, streaming.

### A2A SDK (`@a2a-js/sdk` v0.3.x)

The SDK has a layered architecture:

```
Transport Layer
  ├── jsonRpcHandler (Express middleware)
  ├── restHandler    (Express middleware)
  ├── agentCardHandler (Express middleware)
  └── GrpcServer
         ↓
   A2ARequestHandler interface
     ├── sendMessage / sendMessageStream
     ├── getTask / cancelTask
     ├── getAgentCard
     └── push notification methods
         ↓
   DefaultRequestHandler (default impl)
     ├── TaskStore (required) — persists Task objects
     ├── ResultManager — processes events, updates TaskStore
     ├── AgentExecutor — user's agent logic
     └── ExecutionEventBus — event pub/sub
```

Key SDK types on the **server** side:

- `AgentExecutor` — interface with `execute(requestContext, eventBus)` and
  `cancelTask(taskId, eventBus)`.
- `TaskStore` — interface with `save(task)` and `load(taskId)`.
  `DefaultRequestHandler` requires it; `ResultManager` calls `save()` on every
  event.
- `A2ARequestHandler` — the transport-agnostic interface that all transport
  middleware accepts.

Key SDK types on the **client** side:

- `Client` / `ClientFactory` — create a client from a URL or agent card.
- `client.sendMessage(params)` / `client.sendMessageStream(params)` — send
  messages, get back `Message | Task` or an async stream of events.
- No persistence on the client side.

### A2A Request/Response Pattern

The initial implementation targets simple blocking `message/send`. A typical
exchange looks like this:

**Request** (JSON-RPC `message/send`):

```json
{
  "transport": "JSONRPC",
  "method": "message/send",
  "message": {
    "kind": "message",
    "messageId": "msg-123",
    "parts": [{ "kind": "text", "text": "what's the weather in london?" }],
    "role": "user"
  }
}
```

**Response** (A2A Task with completed status):

```json
{
  "id": "task-456",
  "kind": "task",
  "contextId": "ctx-789",
  "status": {
    "state": "completed",
    "timestamp": "2026-05-07T14:49:10.121Z"
  },
  "artifacts": [
    {
      "artifactId": "art-001",
      "parts": [
        {
          "kind": "text",
          "text": "The weather in london is sunny and 72 degrees."
        }
      ]
    }
  ],
  "history": [
    {
      "kind": "message",
      "messageId": "msg-123",
      "role": "user",
      "parts": [{ "kind": "text", "text": "what's the weather in london?" }],
      "contextId": "ctx-789",
      "taskId": "task-456"
    },
    {
      "kind": "message",
      "messageId": "msg-tool-call",
      "role": "agent",
      "parts": [
        {
          "kind": "data",
          "data": {
            "id": "call-1",
            "name": "get_weather",
            "args": { "location": "london" }
          },
          "metadata": { "genkit_type": "function_call" }
        }
      ]
    },
    {
      "kind": "message",
      "messageId": "msg-tool-resp",
      "role": "agent",
      "parts": [
        {
          "kind": "data",
          "data": {
            "id": "call-1",
            "name": "get_weather",
            "response": { "result": "sunny and 72 degrees" }
          },
          "metadata": { "genkit_type": "function_response" }
        }
      ]
    },
    {
      "kind": "message",
      "messageId": "msg-final",
      "role": "agent",
      "parts": [
        {
          "kind": "text",
          "text": "The weather in london is sunny and 72 degrees."
        }
      ]
    }
  ]
}
```

Key observations:

- The response is a **Task** object containing the full conversation history,
  artifacts, and a terminal status.
- Tool calls and responses appear in the history as data parts with type
  metadata (`genkit_type: "function_call"` / `"function_response"`).
- The final text response is the last message in history and also reflected in
  artifacts.
- `contextId` groups related messages into a conversation.
- `taskId` identifies this specific request/response exchange.

---

## Part 1: Consuming a Remote A2A Agent (`defineA2AAgent`)

### Goal

Make a remote A2A agent look identical to a local Genkit `Agent` so it can be
used interchangeably.

### Usage

```ts
import { defineA2AAgent } from '@genkit-ai/a2a';

const remoteAgent = ai.defineA2AAgent({
  name: 'remoteResearcher',
  agentUrl: 'https://remote-agent.example.com',
  // optional overrides:
  description: 'A remote research agent',
  store: mySessionStore, // enable server-managed state
});

// Use exactly like any other agent:
const session = remoteAgent.streamBidi({});
session.send({ messages: [{ role: 'user', content: [{ text: 'hello' }] }] });

// Or as a sub-agent via middleware:
const orchestrator = ai.defineAgent({
  name: 'orchestrator',
  use: [agents({ agents: ['remoteResearcher'] })],
});
```

### Implementation Strategy

Built on top of `defineCustomAgent`. The `AgentFn` handler:

1. Creates / reuses an A2A `Client` (via `ClientFactory`) for the remote URL.
2. Manages an A2A `contextId` per Genkit session invocation — stored in
   `session.custom.a2aContextId`.
3. For each turn (`sess.run`):
   - Converts Genkit `AgentInput` messages → A2A `Message` (via part mapping).
   - Calls `client.sendMessage({ message })` (blocking mode).
   - Parses the A2A response (Task or Message).
   - Extracts the agent's response messages from `task.history` and the final
     text from the last agent message or from `task.artifacts`.
   - Adds the response message to the session and streams artifact chunks.
4. Returns `AgentResult` with the final message and artifacts.

### Key Design Decisions

- **Blocking mode first**: We use `client.sendMessage()` (blocking) rather than
  `client.sendMessageStream()`. This keeps the initial implementation simple.
  Streaming can be layered on later by switching to `sendMessageStream()` and
  yielding `sendChunk()` calls as events arrive.

- **contextId management**: One A2A `contextId` per Genkit agent invocation.
  For multi-turn within a single invocation, all turns share the same contextId.
  For cross-invocation continuity (via snapshotId), the contextId is stored in
  `session.custom` and persisted.

- **Multi-turn**: Each turn within `sess.run()` sends a new message to the same
  A2A contextId, maintaining conversation continuity on the remote side.

- **No persistence conflict**: The A2A client has no persistence — it's the
  remote server's job. Our Genkit agent stores conversation state locally via its
  normal SessionStore / client-state mechanism.

- **Response extraction**: From the A2A Task response, we extract:
  - The final agent message (last `role: "agent"` entry in `history` with text
    parts).
  - Artifacts from `task.artifacts`.
  - We do NOT store intermediate tool-call/tool-response history in the Genkit
    session — those are internal to the remote agent. We only store the user
    message and the final agent response.

### Part Mapping Utilities

```ts
// Genkit Part → A2A Part
function mapGenkitPartToA2A(part: GenkitPart): A2APart {
  if (part.text !== undefined) return { kind: 'text', text: part.text };
  if (part.media) {
    // data: URI → file with bytes; remote URL → file with uri
    return { kind: 'file', file: { ... } };
  }
  if (part.toolRequest) {
    return {
      kind: 'data',
      data: { id: part.toolRequest.ref, name: part.toolRequest.name, args: part.toolRequest.input },
      metadata: { genkit_type: 'function_call' }
    };
  }
  if (part.toolResponse) {
    return {
      kind: 'data',
      data: { id: part.toolResponse.ref, name: part.toolResponse.name, response: part.toolResponse.output },
      metadata: { genkit_type: 'function_response' }
    };
  }
  return { kind: 'data', data: part };
}

// A2A Part → Genkit Part
function mapA2APartToGenkit(part: A2APart): GenkitPart {
  if (part.kind === 'text') return { text: part.text };
  if (part.kind === 'file') {
    // bytes → data: URI; uri → remote URL
    return { media: { url: ..., contentType: ... } };
  }
  if (part.kind === 'data') {
    // Check metadata.genkit_type for function_call / function_response
    if (part.metadata?.genkit_type === 'function_call') {
      return { toolRequest: { ref: part.data.id, name: part.data.name, input: part.data.args } };
    }
    if (part.metadata?.genkit_type === 'function_response') {
      return { toolResponse: { ref: part.data.id, name: part.data.name, output: part.data.response } };
    }
    return { data: part.data };
  }
}

// Artifact mapping
function mapGenkitArtifactToA2A(artifact: Artifact): A2AArtifact {
  return {
    artifactId: artifact.name,
    parts: artifact.parts.map(mapGenkitPartToA2A),
    ...artifact.metadata?.a2a, // forward A2A-specific overrides
  };
}

function mapA2AArtifactToGenkit(artifact: A2AArtifact): Artifact {
  return {
    name: artifact.artifactId,
    parts: artifact.parts.map(mapA2APartToGenkit),
    metadata: { a2a: { name: artifact.name, description: artifact.description } },
  };
}
```

---

## Part 2: Exposing a Genkit Agent as A2A (`GenkitA2ARequestHandler`)

### The Persistence Problem

The A2A SDK's `DefaultRequestHandler` **requires** a `TaskStore` (constructor
arg 2 is mandatory). Internally, `ResultManager` calls `taskStore.save()` on
virtually every event — status updates, artifact updates, messages. Meanwhile,
Genkit's `SessionRunner` does its own persistence via `SessionStore` snapshots.

If both are active, you get **dual-write**: the A2A SDK persists `Task` objects
(history, status, artifacts) to `TaskStore`, and Genkit persists
`SessionSnapshot` objects (messages, custom state, artifacts) to `SessionStore`.
These are fundamentally different state shapes tracking overlapping data.

### Approaches Considered

#### Option 1: DefaultRequestHandler + Bridging TaskStore

Provide a `TaskStore` implementation that bridges to Genkit's `SessionStore`.

**Rejected** because: still dual persistence, even if the bridging store is
thin. User must understand both stores.

#### Option 2: Implement `A2ARequestHandler` Directly ✅ (Selected)

Bypass `DefaultRequestHandler` entirely. Implement the `A2ARequestHandler`
interface directly, translating A2A protocol calls to agent flow's
`streamBidi()`.

**Rationale:**

- **Single source of truth**: Genkit's `SessionStore` IS the persistence layer.
  No dual writes.
- **Natural mapping**: A2A `taskId` ↔ Genkit `snapshotId`, A2A `contextId` ↔
  logical session, A2A `input-required` ↔ Genkit interrupt, A2A non-blocking ↔
  Genkit `detach`, A2A `cancelTask` ↔ `agent.abort()`.
- **Full SDK transport support**: `jsonRpcHandler`, `restHandler`,
  `agentCardHandler`, and even the gRPC server all accept `A2ARequestHandler`.
  We write zero transport/parsing code.

#### Option 3: Thin TaskStore Facade

**Rejected** because: hacky middle ground, `getTask` after disconnect wouldn't
work without extra logic.

### Implementation Plan (Blocking Mode First)

The initial implementation supports only `message/send` (blocking). The handler
runs the Genkit agent to completion and returns a Task with full history.

```ts
class GenkitA2ARequestHandler implements A2ARequestHandler {
  constructor(
    private agent: Agent,
    private agentCard: AgentCard
  ) {}

  async getAgentCard(): Promise<AgentCard> {
    return this.agentCard;
  }

  async getAuthenticatedExtendedAgentCard(context?) {
    return this.agentCard;
  }

  async sendMessage(params, context): Promise<Message | Task> {
    const incomingMessage = params.message;
    const contextId = incomingMessage.contextId || crypto.randomUUID();
    const taskId = incomingMessage.taskId || crypto.randomUUID();

    // 1. Map A2A message parts → Genkit parts
    const genkitParts = incomingMessage.parts.map(mapA2APartToGenkit);
    const isToolResponse = genkitParts.some((p) => 'toolResponse' in p);

    // 2. Determine init — resume from existing snapshot or fresh
    const init: AgentInit = {};
    if (incomingMessage.taskId) {
      init.snapshotId = incomingMessage.taskId;
    }
    init.newSnapshotId = taskId; // Use taskId as snapshotId for 1:1 mapping

    // 3. Run agent to completion (blocking)
    const result = await this.agent.run(
      {
        messages: [
          { role: isToolResponse ? 'tool' : 'user', content: genkitParts },
        ],
      },
      { init }
    );
    const output: AgentOutput = result.result;

    // 4. Build A2A Task response
    const history: A2AMessage[] = (output.state?.messages || []).map(
      (m, i) => ({
        kind: 'message',
        messageId: `msg-${taskId}-${i}`,
        role: m.role === 'user' || m.role === 'system' ? 'user' : 'agent',
        parts: m.content.map(mapGenkitPartToA2A),
        contextId,
        taskId,
      })
    );

    const artifacts = (output.artifacts || []).map(mapGenkitArtifactToA2A);

    return {
      kind: 'task',
      id: taskId,
      contextId,
      status: { state: 'completed', timestamp: new Date().toISOString() },
      history,
      artifacts,
    };
  }

  // Streaming — not supported initially, falls back to sendMessage behavior
  async *sendMessageStream(params, context) {
    const result = await this.sendMessage(params, context);
    yield result;
  }

  async getTask(params, context): Promise<Task> {
    // Map taskId → snapshotId, call agent.getSnapshotData()
    const snapshot = await this.agent.getSnapshotData(params.id);
    if (!snapshot) throw A2AError.taskNotFound(params.id);
    // Reconstruct A2A Task from snapshot
    return snapshotToTask(snapshot, params.id);
  }

  async cancelTask(params, context): Promise<Task> {
    const previousStatus = await this.agent.abort(params.id);
    if (previousStatus === undefined) throw A2AError.taskNotFound(params.id);
    const snapshot = await this.agent.getSnapshotData(params.id);
    return snapshotToTask(snapshot!, params.id);
  }

  // Push notifications — not supported initially
  async setTaskPushNotificationConfig() {
    throw A2AError.pushNotificationNotSupported();
  }
  async getTaskPushNotificationConfig() {
    throw A2AError.pushNotificationNotSupported();
  }
  async listTaskPushNotificationConfigs() {
    throw A2AError.pushNotificationNotSupported();
  }
  async deleteTaskPushNotificationConfig() {
    throw A2AError.pushNotificationNotSupported();
  }
  async *resubscribe() {
    throw A2AError.unsupportedOperation('resubscribe');
  }
}
```

### Wiring to Express

```ts
import {
  jsonRpcHandler,
  restHandler,
  agentCardHandler,
} from '@a2a-js/sdk/server/express';

const handler = new GenkitA2ARequestHandler(myAgent, agentCard);

// JSON-RPC transport (the primary A2A protocol)
app.use(
  '/a2a',
  jsonRpcHandler({
    requestHandler: handler,
    userBuilder: UserBuilder.noAuthentication,
  })
);

// REST transport (optional)
app.use(
  '/a2a/rest',
  restHandler({
    requestHandler: handler,
    userBuilder: UserBuilder.noAuthentication,
  })
);

// Agent card discovery
app.use(
  '/.well-known/agent-card.json',
  agentCardHandler({
    agentCardProvider: handler,
  })
);
```

### gRPC Support

Since `GenkitA2ARequestHandler` implements `A2ARequestHandler`, the SDK's gRPC
server module works without any additional code:

```ts
import { GrpcServer } from '@a2a-js/sdk/server/grpc';
const grpcServer = new GrpcServer(handler);
```

We don't need to implement any gRPC plumbing ourselves. The SDK handles all
transport concerns.

### Key Mapping Details

| A2A Concept         | Genkit Agent Flow Concept                     |
| ------------------- | --------------------------------------------- |
| `taskId`            | `snapshotId`                                  |
| `contextId`         | Session lineage (parentId chain)              |
| `Message.parts`     | `MessageData.content` (Part mapping)          |
| `Task.status.state` | Snapshot status (pending/done/failed/aborted) |
| `Task.history`      | `SessionState.messages`                       |
| `Task.artifacts`    | `SessionState.artifacts`                      |
| `input-required`    | Interrupt (toolRequest in final message)      |
| Non-blocking mode   | `detach: true` (future)                       |
| `cancelTask`        | `agent.abort(snapshotId)`                     |

### Open Questions

1. **contextId → session grouping**: Should we use `contextId` as a first-class
   concept in agent flow, or map it through custom state?
2. **Push notifications**: Support eventually or permanently unsupported? Could
   be implemented by watching snapshot state changes.
3. **Agent card generation**: Auto-generate from agent metadata, or always
   require explicit configuration?
4. **taskId ↔ snapshotId**: Direct 1:1 (using taskId as snapshotId via
   `init.newSnapshotId`), or maintain a lookup table?

---

## Implementation Priority

1. **`defineA2AAgent`** (consume) — first, simpler, no persistence conflict.
   Start with blocking `sendMessage`, add streaming later.
2. **Design doc** — this document
3. **`GenkitA2ARequestHandler`** (expose) — second, more complex. Start with
   blocking `sendMessage`, add `sendMessageStream` later.
4. **Convenience helpers** — `agentCardFromAgent()`, express wiring helpers
5. **Tests** — unit tests for mapping utilities and both adapters

---

## Future Enhancements

- **Streaming**: Switch from `sendMessage` to `sendMessageStream` on consume
  side; implement proper `sendMessageStream` yielding incremental events on
  expose side.
- **Detach / non-blocking**: Map to Genkit's `detach: true` for long-running
  agents. A2A clients can poll via `getTask`.
- **Push notifications**: Watch `SessionStore` snapshot changes and push
  notifications.
- **Agent card auto-generation**: Derive AgentCard from agent metadata (name,
  description, capabilities).
- **Multi-modal support**: Full file part mapping (images, audio, etc.).
