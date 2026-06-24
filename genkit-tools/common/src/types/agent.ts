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

import { z } from 'zod';
import { RuntimeErrorSchema } from './error';
import { MessageSchema, ModelResponseChunkSchema } from './model';
import {
  PartSchema,
  ToolRequestPartSchema,
  ToolResponsePartSchema,
} from './parts';

/**
 * Zod schema for an artifact produced during a session.
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
 * Zod schema for a snapshot's lifecycle status.
 *
 * - `pending`: a detached invocation is still processing the queued inputs.
 *   The snapshot's state is empty until the background work finishes, at
 *   which point it is rewritten with the cumulative final state and a
 *   terminal status.
 * - `completed`: the snapshot captures a settled state.
 * - `aborted`: the snapshot's invocation was aborted via the
 *   `abort` companion action while detached.
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
export type SnapshotStatus = z.infer<typeof SnapshotStatusSchema>;

/**
 * Zod schema for the reason an agent turn or invocation finished.
 *
 * The first group mirrors the model-level `FinishReasonSchema` so a turn
 * backed by a single `generate` call can forward its reason verbatim:
 *
 * - `stop`: the model stopped naturally.
 * - `length`: generation hit the token limit.
 * - `blocked`: generation was blocked (e.g. safety).
 * - `interrupted`: the model paused on a tool request awaiting input
 *   (e.g. human approval); the turn can be resumed with a `resume` payload.
 * - `other` / `unknown`: anything else / unspecified.
 *
 * The remaining values are agent-specific outcomes with no `generate`-level
 * equivalent (they never arise from forwarding a model finish reason):
 *
 * - `aborted`: the turn or invocation was aborted (e.g. a detached
 *   invocation aborted via the `abort` companion action).
 * - `detached`: the invocation was moved to the background (the client
 *   detached). The returned output reports `detached`; the persisted
 *   snapshot is later finalized with how the background work actually ended.
 * - `failed`: the turn or invocation terminated with an error.
 */
export const AgentFinishReasonSchema = z.enum([
  'stop',
  'length',
  'blocked',
  'interrupted',
  'other',
  'unknown',
  'aborted',
  'detached',
  'failed',
]);
export type AgentFinishReason = z.infer<typeof AgentFinishReasonSchema>;

/**
 * Zod schema for session state.
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

/**
 * Zod schema for agent input (per-turn).
 */
export const AgentInputSchema = z.object({
  /**
   * Detach signals that the client wishes to disconnect after this input is
   * accepted. The server writes a single pending snapshot (with empty
   * state), returns AgentOutput with that snapshot ID, and continues
   * processing any already-buffered inputs in a background context.
   * Streamed chunks emitted after detach are not forwarded over the wire;
   * only the final cumulative state is captured when the snapshot is
   * finalized (or the snapshot is aborted via `abort`).
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
 * Zod schema for agent initialization.
 */
export const AgentInitSchema = z.object({
  /**
   * Identifies the session (conversation) to resume or start. Only valid
   * when the agent is server-managed (a session store is configured);
   * mutually exclusive with state (a client-managed conversation carries
   * its identity inside `state.sessionId`). Alone it resumes the session
   * from its latest snapshot: the most recently updated row, whatever its
   * status. If that row is a failed, aborted, or still-pending dead end
   * the resume is rejected (pass snapshotId to continue from a specific
   * earlier point); if the session's history was forked by re-resuming an
   * earlier snapshot, the most recently updated branch wins. If the
   * session has no snapshots yet, a brand-new conversation is started
   * under this caller-chosen ID, and every snapshot it persists carries
   * it. Combined with snapshotId, it asserts which session the snapshot
   * belongs to, and a mismatch is rejected.
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
 * Zod schema for agent result.
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
 * Zod schema for agent output.
 */
export const AgentOutputSchema = z.object({
  /**
   * ID of the session this invocation belongs to, assigned by the
   * framework when the invocation starts. With server-managed state, a
   * fresh invocation adopts the caller-supplied session ID (see
   * AgentInit.sessionId) or mints a new one, resumed invocations inherit
   * the chain's, and resuming a snapshot from before session IDs existed
   * mints a fresh one. With client-managed state it echoes the ID
   * carried inside the state object (`state.sessionId`), minting one on
   * the conversation's first invocation; only a session with persisted
   * snapshots can be resumed by this ID.
   */
  sessionId: z.string().optional(),
  /**
   * ID of the most recent turn-end snapshot for this invocation. Empty
   * when no store is configured or no turn committed. When `finishReason`
   * is `detached` it is the pending detach snapshot; when `failed`, it is
   * the last committed turn's snapshot (the resume point, holding state
   * through the last successful turn and excluding the failed turn's
   * partial mutations).
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
 * Zod schema for the turn-end signal emitted by an agent.
 *
 * A TurnEnd value is emitted exactly once per turn, regardless of whether a
 * snapshot was persisted. Grouping all turn-end signals here lets callers
 * detect turn boundaries with a single field check and leaves room for
 * additional turn-end metadata in the future.
 */
export const TurnEndSchema = z.object({
  /**
   * ID of the snapshot persisted at the end of this turn. Empty if no
   * snapshot was written (no store configured, the turn failed, or
   * snapshots were suspended after detach).
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
 * Zod schema for the operation kind of a JSON Patch operation (RFC 6902).
 */
export const JsonPatchOpSchema = z.enum([
  'add',
  'remove',
  'replace',
  'move',
  'copy',
  'test',
]);
export type JsonPatchOp = z.infer<typeof JsonPatchOpSchema>;

/**
 * Zod schema for a single RFC 6902 (JSON Patch) operation.
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
export type JsonPatchOperation = z.infer<typeof JsonPatchOperationSchema>;

/**
 * Zod schema for an RFC 6902 JSON Patch: an ordered list of operations
 * applied in sequence.
 */
export const JsonPatchSchema = z.array(JsonPatchOperationSchema);
export type JsonPatch = z.infer<typeof JsonPatchSchema>;

/**
 * Zod schema for agent stream chunk.
 */
export const AgentStreamChunkSchema = z.object({
  /** Generation tokens from the model. */
  modelChunk: ModelResponseChunkSchema.optional(),
  /**
   * An RFC 6902 JSON Patch describing a delta applied to the session's custom
   * state. Emitted automatically whenever the agent mutates custom state, so
   * the client can apply it to its tracked copy and keep custom live as the
   * turn streams. Pointers are rooted at the custom document (e.g.
   * `/agentStatus`), with no `/custom` prefix. The first patch of every turn is
   * a whole-document replace at the root pointer (`""`) that re-bases clients
   * which may not share the server's baseline; subsequent patches are
   * incremental diffs against the last sent value.
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
 * Zod schema for a persisted point-in-time capture of session state. It is
 * the canonical record written to and read from a session store; the wire
 * representation is shared across language runtimes and the Dev UI.
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
  /**
   * When the snapshot's state was last written (RFC 3339). Equals `createdAt`
   * until rewritten; a heartbeat refresh on a pending snapshot does not advance
   * it, so liveness stays distinct from state changes.
   */
  updatedAt: z.string().optional(),
  /**
   * Heartbeat timestamp (RFC 3339) refreshed periodically while a detached
   * (background) turn is in flight. Used to detect a dead background worker:
   * if a `pending` snapshot's heartbeat goes stale (older than the configured
   * timeout), reads surface its status as `expired` (the dead worker can no
   * longer persist a terminal status itself).
   */
  heartbeatAt: z.string().optional(),
  /** Lifecycle state of this snapshot. Empty is treated as `completed`. */
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
 *
 * At least one of `snapshotId` or `sessionId` must be set; they are not
 * mutually exclusive. `snapshotId` fetches a specific snapshot;
 * `sessionId` alone fetches the session's latest snapshot (via the
 * store's `GetLatestSnapshot`, whatever its status). When both are set
 * the fetched snapshot must belong to that session, or the request is
 * rejected.
 */
export const GetSnapshotRequestSchema = z.object({
  /**
   * Identifies the snapshot to fetch. Optional when `sessionId` is given;
   * when both are present the fetched snapshot must belong to that session.
   */
  snapshotId: z.string().optional(),
  /**
   * Identifies the session whose latest snapshot to fetch. Optional when
   * `snapshotId` is given. The latest snapshot is the session's most
   * recently updated row regardless of status (pending, failed, or
   * aborted included).
   */
  sessionId: z.string().optional(),
});
export type GetSnapshotRequest = z.infer<typeof GetSnapshotRequestSchema>;

/**
 * Zod schema for the input of the `abort` companion action.
 */
export const AgentAbortRequestSchema = z.object({
  /** Identifies the snapshot whose invocation should be aborted. */
  snapshotId: z.string(),
});
export type AgentAbortRequest = z.infer<typeof AgentAbortRequestSchema>;

/**
 * Zod schema for the output of the `abort` companion action.
 */
export const AgentAbortResponseSchema = z.object({
  /** Identifies the snapshot the abort attempt targeted. */
  snapshotId: z.string(),
  /**
   * Snapshot's status after the abort attempt. For a pending snapshot
   * this is `aborted`. For an already-terminal snapshot this is the
   * existing terminal status (the abort is a no-op).
   */
  status: SnapshotStatusSchema.optional(),
});
export type AgentAbortResponse = z.infer<typeof AgentAbortResponseSchema>;

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
  /**
   * JSON schema for the agent's custom session state (the `custom` field
   * of `SessionState`), inferred from the agent's state type. Lets the
   * Dev UI and other reflective callers render or validate state without
   * the agent describing it separately. Omitted when the state type
   * carries no schema to infer (e.g. an unstructured `any` state).
   */
  stateSchema: z.record(z.any()).optional(),
});
export type AgentMetadata = z.infer<typeof AgentMetadataSchema>;
