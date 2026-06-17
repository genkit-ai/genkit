/**
 * Copyright 2025 Google LLC
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

/**
 * Zod schemas for the Agent wire protocol types.
 *
 * These schemas define the session state, agent input/output, streaming
 * chunks, and snapshot structures used by the Agent abstraction across
 * all Genkit language implementations.
 */

import { z } from 'zod';
import { RuntimeErrorSchema } from './error';
import { MessageSchema, ModelResponseChunkSchema } from './model';
import {
  PartSchema,
  ToolRequestPartSchema,
  ToolResponsePartSchema,
} from './parts';

// ---------------------------------------------------------------------------
// Snapshot events & finish reasons
// ---------------------------------------------------------------------------

/**
 * Reason an agent turn (or whole invocation) finished.
 *
 * Mirrors the generate `FinishReason` enum and adds two agent-specific
 * states: `detached` (the turn was moved to the background) and `failed`
 * (the turn ended in an error).
 */
export const AgentFinishReasonSchema = z.enum([
  // From generate's FinishReason:
  'stop',
  'length',
  'blocked',
  'aborted',
  'interrupted',
  'other',
  'unknown',
  // Agent additions:
  'detached',
  'failed',
]);
export type AgentFinishReason = z.infer<typeof AgentFinishReasonSchema>;

// ---------------------------------------------------------------------------
// Session state & artifacts
// ---------------------------------------------------------------------------

/**
 * Schema for an artifact generated during agent execution.
 */
export const ArtifactSchema = z.object({
  /** Name identifies the artifact (e.g., "generated_code.go", "diagram.png"). */
  name: z.string().optional(),
  /** Parts contains the artifact content (text, media, etc.). */
  parts: z.array(PartSchema),
  /** Metadata contains additional artifact-specific data. */
  metadata: z.record(z.any()).optional(),
});
export type Artifact = z.infer<typeof ArtifactSchema>;

/**
 * Schema for session state persisted across turns.
 */
export const SessionStateSchema = z.object({
  /**
   * ID of the session (conversation) this state belongs to.
   * Framework-owned: assigned when the conversation's first invocation
   * starts and re-stamped on outbound state, so client-managed callers
   * can round-trip the state object opaquely without tracking a separate
   * identifier. For server-managed agents the snapshot row's `sessionId`
   * is canonical and this field mirrors it.
   */
  sessionId: z.string().optional(),
  /** Conversation history (user/model exchanges). */
  messages: z.array(MessageSchema).optional(),
  /** User-defined state associated with this conversation. */
  custom: z.any().optional(),
  /** Named collections of parts produced during the conversation. */
  artifacts: z.array(ArtifactSchema).optional(),
});
export type SessionState = z.infer<typeof SessionStateSchema>;

// ---------------------------------------------------------------------------
// Agent init, input, output
// ---------------------------------------------------------------------------

/**
 * Schema for agent initialization options.
 */
export const AgentInitSchema = z.object({
  /**
   * Identifies the session (conversation) to resume. Only valid when the
   * agent is server-managed (a session store is configured); mutually
   * exclusive with state (a client-managed conversation carries its
   * identity inside `state.sessionId`). Alone it resumes the session
   * from its latest snapshot: the most recently updated one that is not
   * a failed/aborted dead end. A pending latest snapshot (a detached
   * invocation still running) rejects the resume rather than racing the
   * background work; if the session's history was forked by re-resuming
   * an earlier snapshot, the most recently updated branch wins, and
   * snapshotId can pick a branch explicitly. Combined with snapshotId,
   * it asserts which session the snapshot belongs to, and a mismatch is
   * rejected.
   */
  sessionId: z.string().optional(),
  /**
   * Loads state from a persisted snapshot (server-managed state only).
   * May be combined with sessionId to validate that the snapshot belongs
   * to that session. Mutually exclusive with state.
   */
  snapshotId: z.string().optional(),
  /**
   * Direct state for the invocation (client-managed state only). The
   * conversation's identity rides inside it (`state.sessionId`): the
   * framework mints one on the conversation's first invocation and
   * echoes it on the output state, so resending the state object keeps
   * the identity without tracking a separate field. Mutually exclusive
   * with sessionId and snapshotId.
   */
  state: SessionStateSchema.optional(),
});
export type AgentInit = z.infer<typeof AgentInitSchema>;

/**
 * Schema for agent input messages and commands.
 */
export const AgentInputSchema = z.object({
  /**
   * Detach signals that the client wishes to disconnect after this input is
   * accepted. The server writes a single pending snapshot (with empty
   * state), returns AgentOutput with that snapshot ID, and continues
   * processing any already-buffered inputs in a background context.
   * Streamed chunks emitted after detach are not forwarded over the wire;
   * only the final cumulative state is captured when the snapshot is
   * finalized (or the snapshot is aborted via `abortSnapshot`).
   */
  detach: z.boolean().optional(),
  /** User's input message for this turn. */
  message: MessageSchema.optional(),
  /** Options for resuming an interrupted generation. */
  resume: z
    .object({
      respond: z.array(ToolResponsePartSchema).optional(),
      restart: z.array(ToolRequestPartSchema).optional(),
    })
    .optional(),
});
export type AgentInput = z.infer<typeof AgentInputSchema>;

/**
 * Schema for turn termination event.
 */
export const TurnEndSchema = z.object({
  /**
   * ID of the snapshot persisted at the end of this turn. Empty if no
   * snapshot was written (no store configured, the callback declined,
   * nothing changed since the last snapshot, or snapshots were suspended
   * after detach).
   */
  snapshotId: z.string().optional(),
  /**
   * Why this turn finished (e.g. `stop`, `length`, `interrupted`). Lets a
   * caller react to a turn boundary (e.g. pause on `interrupted`) without
   * scanning the message content. Omitted when the turn reported no reason.
   *
   * `failed` reports a failed turn; unless the agent recovers and keeps
   * processing, the invocation then resolves with a failed output carrying
   * the error and the last-good state.
   */
  finishReason: AgentFinishReasonSchema.optional(),
});
export type TurnEnd = z.infer<typeof TurnEndSchema>;

/**
 * Schema for a single RFC 6902 (JSON Patch) operation.
 */
export const JsonPatchOperationSchema = z.object({
  op: z.enum(['add', 'remove', 'replace', 'move', 'copy', 'test']),
  /** A JSON Pointer (RFC 6901) to the target location, e.g. `"/agentStatus"`. */
  path: z.string(),
  /** Source pointer; required for `move` and `copy`. */
  from: z.string().optional(),
  /** New value; required for `add`, `replace`, and `test`. */
  value: z.any().optional(),
});
export type JsonPatchOperation = z.infer<typeof JsonPatchOperationSchema>;

/**
 * Schema for an RFC 6902 JSON Patch: an ordered list of operations.
 */
export const JsonPatchSchema = z.array(JsonPatchOperationSchema);
export type JsonPatch = z.infer<typeof JsonPatchSchema>;

/**
 * Schema for stream chunks emitted during agent execution.
 */
export const AgentStreamChunkSchema = z.object({
  /** Generation tokens from the model. */
  modelChunk: ModelResponseChunkSchema.optional(),
  /**
   * An RFC 6902 JSON Patch describing a delta applied to the session's
   * `custom` state. The runtime auto-emits these whenever custom state is
   * mutated during a turn; clients apply them to keep their tracked custom
   * state live mid-stream.
   */
  customPatch: JsonPatchSchema.optional(),
  /** A newly produced artifact. */
  artifact: ArtifactSchema.optional(),
  /**
   * Non-null when the agent has finished processing the current input.
   * Groups all turn-end signals; the client should stop iterating and may
   * send the next input.
   */
  turnEnd: TurnEndSchema.optional(),
});
export type AgentStreamChunk = z.infer<typeof AgentStreamChunkSchema>;

/**
 * Schema for the final results of an agent execution.
 */
export const AgentResultSchema = z.object({
  /** Last model response message from the conversation. */
  message: MessageSchema.optional(),
  /** Artifacts produced during the session. */
  artifacts: z.array(ArtifactSchema).optional(),
  /**
   * Why the invocation finished. Set by a custom agent to override the
   * default (the last turn's reason); omitted to accept the default.
   */
  finishReason: AgentFinishReasonSchema.optional(),
});
export type AgentResult = z.infer<typeof AgentResultSchema>;

/**
 * Schema for agent output returned at completion.
 */
export const AgentOutputSchema = z.object({
  /**
   * ID of the session this invocation belongs to, assigned by the
   * framework when the invocation starts. With server-managed state, a
   * fresh invocation mints a new ID, resumed invocations inherit the
   * chain's, and resuming a snapshot from before session IDs existed
   * mints a fresh one. With client-managed state it echoes the ID
   * carried inside the state object (`state.sessionId`), minting one on
   * the conversation's first invocation; only a session with persisted
   * snapshots can be resumed by this ID.
   */
  sessionId: z.string().optional(),
  /**
   * ID of the newest snapshot capturing this invocation: the
   * invocation-end snapshot, or the latest earlier snapshot when that
   * write was skipped. Empty when no store is configured or the
   * invocation persisted nothing. When `finishReason` is `detached` it
   * is the pending detach snapshot; when `failed`, the most recent
   * snapshot capturing the last-good state: everything through the last
   * successful turn (see the `recovery` snapshot event).
   */
  snapshotId: z.string().optional(),
  /**
   * Final conversation state (only when client-managed). When
   * `finishReason` is `failed`, this is the last-good state: everything
   * through the last successful turn, excluding the failed turn's
   * partial mutations.
   */
  state: SessionStateSchema.optional(),
  /** Last model response message from the conversation. */
  message: MessageSchema.optional(),
  /** Artifacts produced during the session. */
  artifacts: z.array(ArtifactSchema).optional(),
  /**
   * Why the invocation finished. `detached` when the client detached and
   * the work continues in the background; `failed` when the invocation
   * ended in failure (see `error`); otherwise the last turn's reason
   * (or the value a custom agent set on its result).
   */
  finishReason: AgentFinishReasonSchema.optional(),
  /**
   * Structured failure information when the invocation ended in failure
   * (`finishReason` is `failed`). `status` preserves the original error
   * category (e.g. `INVALID_ARGUMENT`, `FAILED_PRECONDITION`,
   * `INTERNAL`) so callers can still branch on it.
   */
  error: RuntimeErrorSchema.optional(),
});
export type AgentOutput = z.infer<typeof AgentOutputSchema>;

/**
 * Schema for the lookup input of the `getSnapshotData` action.
 *
 * Provide exactly one of `snapshotId` (an exact snapshot) or `sessionId`
 * (the session's latest leaf snapshot).
 */
export const GetSnapshotDataInputSchema = z.object({
  snapshotId: z.string().optional(),
  sessionId: z.string().optional(),
});
export type GetSnapshotDataInput = z.infer<typeof GetSnapshotDataInputSchema>;

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

export const SnapshotStatusSchema = z.enum([
  'pending',
  'completed',
  'aborted',
  'failed',
]);
export type SnapshotStatus = z.infer<typeof SnapshotStatusSchema>;

/**
 * Schema for a persisted session snapshot.
 */
export const SessionSnapshotSchema = z.object({
  /** Unique identifier for this snapshot (UUID). */
  snapshotId: z.string(),
  /**
   * ID of the session this snapshot belongs to. Assigned by the agent
   * framework when the conversation's first invocation starts and
   * stamped on every later snapshot in the chain, including across
   * resumed invocations. Stores preserve it across rewrites; rows
   * written without one (data from before session IDs existed) belong
   * to no session.
   */
  sessionId: z.string().optional(),
  /**
   * ID of the previous snapshot in this timeline. Informational lineage
   * (for debugging and UI history trees); plays no part in resolving a
   * session's latest snapshot.
   */
  parentId: z.string().optional(),
  /** When the snapshot was first written (RFC 3339). */
  createdAt: z.string(),
  /** When the snapshot was last written (RFC 3339). Equals `createdAt` until rewritten. */
  updatedAt: z.string().optional(),
  /** What triggered this snapshot. */
  /** Lifecycle state of this snapshot. Empty is treated as `succeeded`. */
  status: SnapshotStatusSchema.optional(),
  /**
   * Semantic reason the turn or invocation captured here ended (e.g.
   * `stop`, `interrupted`, `failed`, `aborted`). Complements `status` (the
   * persistence lifecycle) so a resumed or background task can report how it
   * ended without re-deriving it from the messages.
   */
  finishReason: AgentFinishReasonSchema.optional(),
  /** Structured failure information for a snapshot in `failed` status. */
  error: RuntimeErrorSchema.optional(),
  /**
   * Conversation state captured at this point. Empty on a pending snapshot
   * (the live state is not yet committed); populated on terminal snapshots
   * with the cumulative final state.
   */
  state: SessionStateSchema.optional(),
});
export type SessionSnapshot = z.infer<typeof SessionSnapshotSchema>;

/**
 * Zod schema for the input of an agent's `getSnapshot` companion action.
 * The action is registered under the agent's name (action type
 * `agent-snapshot`) when the agent has a session store configured. The
 * action returns the stored `SessionSnapshot`, with any configured state
 * transform applied to its state.
 */
export const GetSnapshotRequestSchema = z.object({
  sessionId: z.string().optional(),
  /** Identifies the snapshot to fetch. */
  snapshotId: z.string().optional(),
});
export type GetSnapshotRequest = z.infer<typeof GetSnapshotRequestSchema>;

/**
 * Zod schema for the input of the `abortSnapshot` companion action.
 */
export const AbortSnapshotRequestSchema = z.object({
  /** Identifies the snapshot whose invocation should be aborted. */
  snapshotId: z.string(),
});
export type AbortSnapshotRequest = z.infer<typeof AbortSnapshotRequestSchema>;

/**
 * Zod schema for the output of the `abortSnapshot` companion action.
 */
export const AbortSnapshotResponseSchema = z.object({
  /** Echoes the requested snapshot ID. */
  snapshotId: z.string(),
  /**
   * Snapshot's status after the abort attempt. For a pending snapshot
   * this is `aborted`. For an already-terminal snapshot this is the
   * existing terminal status (the abort is a no-op).
   */
  status: SnapshotStatusSchema.optional(),
});
export type AbortSnapshotResponse = z.infer<typeof AbortSnapshotResponseSchema>;

/**
 * Who owns session state for an agent.
 *
 * - `server`: a session store is configured and snapshots are persisted
 *   server-side.
 * - `client`: no store; state flows through the agent's invocation init
 *   and output payloads.
 */
export const AgentStateManagementSchema = z.enum(['server', 'client']);
export type AgentStateManagement = z.infer<typeof AgentStateManagementSchema>;

/**
 * Zod schema for the agent capability metadata placed under
 * `metadata.agent` on an agent's action descriptor. Lets the Dev UI
 * and other reflective callers render the right surface (e.g. hide
 * the Abort button when the configured store doesn't support it)
 * without round-tripping through the reflection API.
 */
export const AgentMetadataSchema = z.object({
  /** Who owns session state for this agent. */
  stateManagement: AgentStateManagementSchema,
  /**
   * Whether the agent's invocations can be aborted. True only when the
   * configured store implements the abort lifecycle.
   */
  abortable: z.boolean(),
});
export type AgentMetadata = z.infer<typeof AgentMetadataSchema>;
