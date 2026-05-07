# Design Document: Agents (Session Flows) in JS/TS

**Status**: Implemented  
**Authors**: Antigravity  
**Target Package**: `@genkit-ai/ai` and `@genkit-ai/core`  

---

## 1. Objective

This document proposes a unified **Agent** primitive for Genkit JS/TS, drawing architectural parity with the reference Go implementation while addressing modern agentic needs (State, History, and Artifact Persistence) for Tooling and the Genkit Dev UI.

Agents act as a standard agent abstraction, allowing the framework to support multi-turn conversations and long-running agentic tasks with state persisted between independent runs.

---

## 2. Background & Relationship to Existing APIs

Genkit currently contains a beta `Session` class (`js/ai/src/session.ts`) and `SessionStore` interface.
**Decision**: We have decided to replace the existing beta `Session` and `SessionStore` APIs with the snapshot-based model described here to achieve full parity with the Go implementation and enable advanced tooling features.
- **Flow Integration**: Built on top of `defineBidiAction`, Agents natively support bidirectional streaming.
- **Durable Streaming**: Connects with Genkit's durable streaming protocol for robust reconnects over WebSockets.

---

## 3. Core Schemas & Wire Protocol

Agents enforce structured input/output payloads for predictable agent lifecycle management. These schemas match `genkit-tools/common/src/types/agent.ts` from the reference Go PR.

### Session State
```ts
import { z } from 'zod';
import { MessageSchema, PartSchema, ModelResponseChunkSchema } from './model-types.js';

export const ArtifactSchema = z.object({
  name: z.string().optional(),
  parts: z.array(PartSchema),
  metadata: z.record(z.any()).optional(),
});

export const SessionStateSchema = z.object({
  messages: z.array(MessageSchema).optional(),
  custom: z.any().optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});
```

### Wire Payloads
```ts
export const AgentInitSchema = z.object({
  snapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
});

export const AgentInputSchema = z.object({
  messages: z.array(MessageSchema).optional(),
  toolRestarts: z.array(PartSchema).optional(),
});

export const TurnEndSchema = z.object({
  snapshotId: z.string().optional(),
});

export const AgentStreamChunkSchema = z.object({
  modelChunk: ModelResponseChunkSchema.optional(),
  status: z.any().optional(),
  artifact: ArtifactSchema.optional(),
  turnEnd: TurnEndSchema.optional(),
});

export const AgentOutputSchema = z.object({
  snapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});
```

---

## 4. Persistence & The Snapshot System

To maintain state across environments, Genkit provides `SessionStore` abstractions for saving and loading point-in-time captures (`SessionSnapshot`).

### Interfaces (Strongly Typed)
```ts
export interface SnapshotContext<S = unknown> {
  state: SessionState<S>;
  prevState?: SessionState<S>;
  turnIndex: number;
  event: 'turnEnd' | 'invocationEnd';
}

export type SnapshotCallback<S = unknown> = (ctx: SnapshotContext<S>) => boolean;

export interface SessionSnapshot<S = unknown> {
  snapshotId: string;
  parentId?: string;
  createdAt: string;
  event: 'turnEnd' | 'invocationEnd';
  state: SessionState<S>;
}

export interface SessionStore<S = unknown> {
  getSnapshot(snapshotId: string): Promise<SessionSnapshot<S> | undefined>;
  saveSnapshot(snapshot: SessionSnapshot<S>): Promise<void>;
}
```

---

## 5. SDK APIs

### 5.1 `defineCustomAgent`
Allows programmatic declaration of agent logic with an injected `SessionRunner`.

```ts
export function defineCustomAgent<Stream = any, State = any>(
  registry: Registry,
  config: {
    name: string;
    description?: string;
    store?: SessionStore<State>;
    snapshotCallback?: SnapshotCallback<State>;
    toClient?: {
      messages?: (msgs: Message[]) => Message[];
      state?: (state: State) => Partial<State>;
    };
  },
  fn: AgentFn<Stream, State>
): Agent<Stream, State>;
```

### 5.2 `definePromptAgent`
Ergonomic shortcut for standard prompt-backed loop orchestration. Automatically manages history, tool restarts, and renders prompts. References a prompt defined separately via `definePrompt`.

```ts
export function definePromptAgent<State = any>(
  registry: Registry,
  config: {
    promptName: string;
    store?: SessionStore<State>;
  }
): Agent<State>;
```

### 5.3 `defineAgent`
The most ergonomic API — combines `definePrompt` and `definePromptAgent` into a single flat config. The config accepts all `PromptConfig` fields (name, model, system, tools, etc.) plus agent-specific fields (`store`, `snapshotCallback`).

```ts
export function defineAgent<State = unknown>(
  registry: Registry,
  config: AgentConfig<State>
): Agent<State>;

export interface AgentConfig<State = unknown> extends PromptConfig {
  store?: SessionStore<State>;
  snapshotCallback?: SnapshotCallback<State>;
}
```

---

## 6. Tooling & Dev UI Integration

- **Live State Playground**: Renders a continuous view of the accumulated `Artifacts` and `statePatch` streams.
- **Time-Travel Debugging**: Snapshots are tied directly to trace spans, enabling the developer to resume sessions from a past `snapshotId`.

---

## 7. Execution & Verification Plan

1. **Phase 1: Core Types & Schemas**
   - Declare Zod schemas in `js/ai/src/session-flow.ts`.
2. **Phase 2: Context Runner & Wrappers**
   - Construct the `SessionRunner` orchestrator.
   - Integrate snapshot event callbacks into execution loops.
3. **Phase 3: Prompt Engine Hooks**
   - Integrate prompt rendering into agent execution loops.
4. **Phase 4: Verification**
   - Test state retention between single-turn `.run()` bounds.
   - Simulate client disconnections over `.streamBidi()`.
