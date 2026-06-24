/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// IMPORTANT: Please keep schema/type definitions in sync with
//   genkit-tools/common/src/types/agent.ts
//
// This file is the single home for the agent/session wire schemas (and their
// types) shared between the JS runtime and the canonical tools definitions. It
// mirrors the order and shape of the tools file so the two stay easy to diff.
//
// Two intentional differences from the tools file:
//
//   1. The runtime variants of `SessionState`, `SessionSnapshot`, `AgentInit`,
//      and `AgentOutput` are generic interfaces (`<S>`) so callers can type the
//      custom session state, rather than the tools file's non-generic
//      `z.infer` aliases.
//   2. The structured error shape is defined here as `RuntimeErrorSchema`
//      (the JS runtime has no shared error-types module to import it from).
//

import { z } from '@genkit-ai/core';
import {
  MessageSchema,
  ModelResponseChunkSchema,
  PartSchema,
  type MessageData,
} from './model-types.js';
import { ToolRequestPartSchema, ToolResponsePartSchema } from './parts.js';

/**
 * Schema for the canonical Genkit error wire shape (`{status, message,
 * details}`). This is the form runtimes use when an error travels as data
 * inside another value (e.g. agent outputs and session snapshots).
 */
export const RuntimeErrorSchema = z.object({
  /** Canonical status name (e.g. `INTERNAL`, `FAILED_PRECONDITION`). */
  status: z.string().optional(),
  /** Human-readable error message. */
  message: z.string(),
  /** Optional structured details describing the failure. */
  details: z.any().optional(),
});
/** Structured error carried as data inside agent outputs and snapshots. */
export type RuntimeError = z.infer<typeof RuntimeErrorSchema>;

/**
 * Schema for tracking persistent artifacts generated during a session turn.
 */
export const ArtifactSchema = z.object({
  /** Name identifies the artifact (e.g., "generated_code.go", "diagram.png"). */
  name: z.string().optional(),
  /** Parts contains the artifact content (text, media, etc.). */
  parts: z.array(PartSchema),
  /** Metadata contains additional artifact-specific data. */
  metadata: z.record(z.any()).optional(),
});

/**
 * Artifact generated during a session turn.
 */
export type Artifact = z.infer<typeof ArtifactSchema>;

/**
 * Schema for a snapshot's lifecycle status.
 *
 * - `pending`: a detached invocation is still processing the queued inputs.
 *   The snapshot's state is empty until the background work finishes, at
 *   which point it is rewritten with the cumulative final state and a
 *   terminal status.
 * - `completed`: the snapshot captures a settled state.
 * - `aborted`: the snapshot's invocation was aborted via the `abort` companion
 *   action while detached.
 * - `failed`: the invocation terminated with an error. The snapshot's `error`
 *   field describes the failure and resume is rejected with that same error.
 * - `expired`: a `pending` snapshot whose detached background worker is
 *   presumed dead because its heartbeat went stale. Computed on read from a
 *   stale `heartbeatAt`; never persisted (the dead worker can no longer write
 *   a terminal status itself).
 */
export const SnapshotStatusSchema = z.enum([
  'pending',
  'completed',
  'aborted',
  'failed',
  'expired',
]);

/**
 * Lifecycle status of a session snapshot.
 */
export type SnapshotStatus = z.infer<typeof SnapshotStatusSchema>;

/**
 * Reason an agent turn (or whole invocation) finished.
 *
 * The first group mirrors the model-level `FinishReason` so a turn backed by a
 * single `generate` call can forward its reason verbatim. The remaining values
 * are agent-specific outcomes with no `generate`-level equivalent: `aborted`
 * (the turn/invocation was aborted), `detached` (the turn was moved to the
 * background), and `failed` (the turn ended in an error).
 */
export const AgentFinishReasonSchema = z.enum([
  // Mirror of generate's FinishReason:
  'stop',
  'length',
  'blocked',
  'interrupted',
  'other',
  'unknown',
  // Agent-specific additions:
  'aborted',
  'detached',
  'failed',
]);

/**
 * Reason an agent turn (or whole invocation) finished.
 */
export type AgentFinishReason = z.infer<typeof AgentFinishReasonSchema>;

/**
 * Schema for session execution state.
 */
export const SessionStateSchema = z.object({
  sessionId: z.string().optional(),
  messages: z.array(MessageSchema).optional(),
  custom: z.any().optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * State persisted for a session across turns.
 *
 * Runtime variant of {@link SessionStateSchema}, generic over the custom state
 * type `S` so callers can type `custom`.
 */
export interface SessionState<S = unknown> {
  sessionId?: string;
  messages?: MessageData[];
  custom?: S;
  artifacts?: Artifact[];
}

/**
 * Schema for agent input messages and commands.
 */
export const AgentInputSchema = z.object({
  /** User's input message for this turn. */
  message: MessageSchema.optional(),
  /** Options for resuming an interrupted generation. */
  resume: z
    .object({
      respond: z.array(ToolResponsePartSchema).optional(),
      restart: z.array(ToolRequestPartSchema).optional(),
    })
    .optional(),
  detach: z.boolean().optional(),
});

/**
 * Input received by an agent turn.
 */
export type AgentInput = z.infer<typeof AgentInputSchema>;

/**
 * Schema for initializing an agent turn.
 */
export const AgentInitSchema = z.object({
  snapshotId: z.string().optional(),
  sessionId: z.string().optional(),
  state: SessionStateSchema.optional(),
});

/**
 * Initialization options for an agent turn.
 *
 * For server-managed agents (with a `store`) provide a `snapshotId` (resume an
 * exact snapshot, required for branching/snapshotting clients) and/or a
 * `sessionId` (resume the session's latest snapshot - the simple case used by
 * `useChat`-style clients). When both are provided, `snapshotId` selects the
 * snapshot to resume and `sessionId` acts as an ownership guard: the snapshot
 * must belong to that session. For client-managed agents (no store) provide
 * the full `state`.
 *
 * Runtime variant of {@link AgentInitSchema}, generic over the custom state
 * type `S`.
 */
export interface AgentInit<S = unknown> {
  snapshotId?: string;
  sessionId?: string;
  state?: SessionState<S>;
}

/**
 * Schema for final results of an agent execution.
 */
export const AgentResultSchema = z.object({
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
  /** The reason the whole invocation finished (e.g. `stop`, `interrupted`). */
  finishReason: AgentFinishReasonSchema.optional(),
});

/**
 * Result returned upon completing an agent execution.
 */
export type AgentResult = z.infer<typeof AgentResultSchema>;

/**
 * Schema for output returned at turn completion.
 */
export const AgentOutputSchema = z.object({
  /**
   * ID of the session this invocation belongs to, assigned by the framework
   * when the invocation starts.
   */
  sessionId: z.string().optional(),
  snapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
  /** The reason the invocation finished (e.g. `stop`, `interrupted`). */
  finishReason: AgentFinishReasonSchema.optional(),
  /**
   * Present when `finishReason` is `failed`. Carries the original error
   * details (RuntimeError shape; the runtime resolves gracefully instead of
   * throwing). The accompanying `state`/`snapshotId` hold the last-good state -
   * the state the failed turn started with.
   */
  error: RuntimeErrorSchema.optional(),
});

/**
 * Output returned at turn completion.
 *
 * Runtime variant of {@link AgentOutputSchema}, generic over the custom state
 * type `S`.
 */
export interface AgentOutput<S = unknown> {
  sessionId?: string;
  artifacts?: Artifact[];
  message?: MessageData;
  snapshotId?: string;
  state?: SessionState<S>;
  finishReason?: AgentFinishReason;
  /**
   * Present when `finishReason` is `failed`. Carries the original error
   * details (RuntimeError shape); `state`/`snapshotId` hold the last-good state.
   */
  error?: RuntimeError;
}

/**
 * Schema identifying a turn termination event.
 */
export const TurnEndSchema = z.object({
  snapshotId: z.string().optional(),
  /** The reason this turn finished (e.g. `stop`, `interrupted`). */
  finishReason: AgentFinishReasonSchema.optional(),
});

/**
 * Identifies a turn termination event.
 */
export type TurnEnd = z.infer<typeof TurnEndSchema>;

/**
 * Schema for the operation kind of a JSON Patch operation (RFC 6902).
 */
export const JsonPatchOpSchema = z.enum([
  'add',
  'remove',
  'replace',
  'move',
  'copy',
  'test',
]);

/**
 * Operation kind of a JSON Patch (RFC 6902) operation.
 */
export type JsonPatchOp = z.infer<typeof JsonPatchOpSchema>;

/**
 * Schema for a single RFC 6902 (JSON Patch) operation.
 */
export const JsonPatchOperationSchema = z.object({
  op: JsonPatchOpSchema,
  /** A JSON Pointer (RFC 6901) to the target location, e.g. `"/agentStatus"`. */
  path: z.string(),
  /** Source pointer; required for `move` and `copy`. */
  from: z.string().optional(),
  /** New value; required for `add`, `replace`, and `test`. */
  value: z.any().optional(),
});

/**
 * Schema for an RFC 6902 JSON Patch: an ordered list of operations.
 */
export const JsonPatchSchema = z.array(JsonPatchOperationSchema);

/**
 * Schema for stream chunks emitted during agent execution.
 */
export const AgentStreamChunkSchema = z.object({
  modelChunk: ModelResponseChunkSchema.optional(),
  /**
   * An RFC 6902 JSON Patch describing a delta applied to the session's
   * `custom` state. The runtime auto-emits these whenever custom state is
   * mutated during a turn; clients apply them to keep their tracked custom
   * state live mid-stream.
   */
  customPatch: JsonPatchSchema.optional(),
  artifact: ArtifactSchema.optional(),
  turnEnd: TurnEndSchema.optional(),
});

/**
 * Streamed chunk emitted during agent execution.
 */
export type AgentStreamChunk = z.infer<typeof AgentStreamChunkSchema>;

/**
 * Zod schema mirroring {@link SessionSnapshot}. Used as the output schema for
 * the `getSnapshot` companion action so the snapshot shape is discoverable in
 * the registry (rather than an opaque `z.any()`).
 */
export const SessionSnapshotSchema = z.object({
  snapshotId: z.string(),
  sessionId: z.string().optional(),
  parentId: z.string().optional(),
  createdAt: z.string(),
  updatedAt: z.string().optional(),
  heartbeatAt: z.string().optional(),
  status: SnapshotStatusSchema.optional(),
  finishReason: AgentFinishReasonSchema.optional(),
  error: RuntimeErrorSchema.optional(),
  state: SessionStateSchema.optional(),
});

/**
 * Saved snapshot of a session's state at a given event point.
 *
 * Runtime variant of {@link SessionSnapshotSchema}, generic over the custom
 * state type `S`.
 */
export interface SessionSnapshot<S = unknown> {
  snapshotId: string;

  /**
   * ID of the session this snapshot belongs to. Assigned by the agent
   * framework when the conversation's first invocation starts and stamped on
   * every later snapshot in the chain, including across resumed invocations.
   * Stores preserve it across rewrites; rows written without one (data from
   * before session IDs existed) belong to no session.
   */
  sessionId?: string;

  /**
   * ID of the previous snapshot in this timeline. Informational lineage (for
   * debugging and UI history trees); plays no part in resolving a session's
   * latest snapshot.
   */
  parentId?: string;
  createdAt: string;
  /** When the snapshot was last written (RFC 3339). Equals `createdAt` until rewritten. */
  updatedAt?: string;

  /**
   * Heartbeat timestamp (RFC 3339) refreshed periodically while a detached
   * (background) turn is in flight. Used to detect a dead background worker:
   * if a `pending` snapshot's heartbeat goes stale (older than the configured
   * timeout), reads surface its status as `expired` (the dead process can no
   * longer persist a terminal status itself).
   */
  heartbeatAt?: string;
  status?: SnapshotStatus;

  /**
   * Semantic reason the turn/invocation finished (e.g. `interrupted`,
   * `stop`). Distinct from `status`, which tracks the persistence lifecycle.
   */
  finishReason?: AgentFinishReason;

  /**
   * Structured failure information (RuntimeError shape). `status` is the
   * canonical error category (e.g. `INTERNAL`, `FAILED_PRECONDITION`).
   */
  error?: RuntimeError;

  /**
   * Conversation state captured at this point. Empty on a pending snapshot
   * (the live state is not yet committed); populated on terminal snapshots
   * with the cumulative final state.
   */
  state?: SessionState<S>;
}

/**
 * Schema for the input of an agent's `getSnapshot` companion action. Provide
 * exactly one of `snapshotId` or `sessionId`.
 */
export const GetSnapshotRequestSchema = z.object({
  snapshotId: z.string().optional(),
  sessionId: z.string().optional(),
});

/**
 * Input identifying which snapshot to fetch.
 */
export type GetSnapshotRequest = z.infer<typeof GetSnapshotRequestSchema>;

/**
 * Schema for the input of the `abort` companion action.
 */
export const AgentAbortRequestSchema = z.object({
  snapshotId: z.string(),
});

/**
 * Input identifying which snapshot's invocation to abort.
 */
export type AgentAbortRequest = z.infer<typeof AgentAbortRequestSchema>;

/**
 * Schema for the output of the `abort` companion action.
 */
export const AgentAbortResponseSchema = z.object({
  snapshotId: z.string(),
  status: SnapshotStatusSchema.optional(),
});

/**
 * Result of an abort attempt.
 */
export type AgentAbortResponse = z.infer<typeof AgentAbortResponseSchema>;

/**
 * Schema for who owns session state for an agent.
 *
 * - `server`: a session store is configured and snapshots are persisted
 *   server-side.
 * - `client`: no store; state flows through the agent's invocation init and
 *   output payloads.
 */
export const AgentStateManagementSchema = z.enum(['server', 'client']);

/**
 * Who owns session state for an agent.
 */
export type AgentStateManagement = z.infer<typeof AgentStateManagementSchema>;

/**
 * Schema for the agent capability metadata placed under `metadata.agent` on an
 * agent's action descriptor. Lets the Dev UI and other reflective callers
 * render the right surface (e.g. hide the Abort button when the configured
 * store doesn't support it) without round-tripping through the reflection API.
 */
export const AgentMetadataSchema = z.object({
  /** Who owns session state for this agent. */
  stateManagement: AgentStateManagementSchema,
  /**
   * Whether the agent's invocations can be aborted. True only when the
   * configured store implements the abort lifecycle.
   */
  abortable: z.boolean(),
  /**
   * JSON schema for the agent's custom session state (the `custom` field of
   * `SessionState`), inferred from the agent's state type. Omitted when the
   * state type carries no schema to infer.
   */
  stateSchema: z.record(z.any()).optional(),
});

/**
 * Agent capability metadata placed under `metadata.agent`.
 */
export type AgentMetadata = z.infer<typeof AgentMetadataSchema>;
