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

import { z } from 'zod';
import { MessageSchema, ModelResponseChunkSchema } from './model';
import { PartSchema } from './parts';

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
 * Zod schema for snapshot event.
 *
 * - `turnEnd`: snapshot was triggered at the end of a turn.
 * - `invocationEnd`: snapshot was triggered at the end of the invocation.
 * - `detach`: snapshot was created when the client detached the invocation
 *   and the flow continues in the background. Initially written with
 *   `pending` status (and empty state) and rewritten with a terminal
 *   status and the final cumulative state once the background work
 *   finishes.
 */
export const SnapshotEventSchema = z.enum([
  'turnEnd',
  'invocationEnd',
  'detach',
]);
export type SnapshotEvent = z.infer<typeof SnapshotEventSchema>;

/**
 * Zod schema for a snapshot's lifecycle status.
 *
 * - `pending`: a detached invocation is still processing the queued inputs.
 *   The snapshot's state is empty until the flow exits, at which point it
 *   is rewritten with the cumulative final state and a terminal status.
 * - `complete`: the snapshot captures a settled state.
 * - `canceled`: the snapshot's invocation was aborted via the
 *   `abortSnapshot` companion action while detached.
 * - `error`: the invocation terminated with an error. The snapshot's `error`
 *   field describes the failure and resume is rejected with that same error.
 */
export const SnapshotStatusSchema = z.enum([
  'pending',
  'complete',
  'canceled',
  'error',
]);
export type SnapshotStatus = z.infer<typeof SnapshotStatusSchema>;

/**
 * Zod schema for session state.
 */
export const SessionStateSchema = z.object({
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
   * finalized (or the snapshot is aborted via `abortSnapshot`).
   */
  detach: z.boolean().optional(),
  /** User's input messages for this turn. */
  messages: z.array(MessageSchema).optional(),
  /** Tool request parts to re-execute interrupted tools. */
  toolRestarts: z.array(PartSchema).optional(),
});
export type AgentInput = z.infer<typeof AgentInputSchema>;

/**
 * Zod schema for agent initialization.
 */
export const AgentInitSchema = z.object({
  /** Loads state from a persisted snapshot. Mutually exclusive with state. */
  snapshotId: z.string().optional(),
  /** Direct state for the invocation. Mutually exclusive with snapshotId. */
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
});
export type AgentResult = z.infer<typeof AgentResultSchema>;

/**
 * Zod schema for agent output.
 */
export const AgentOutputSchema = z.object({
  /** ID of the snapshot created at the end of this invocation. */
  snapshotId: z.string().optional(),
  /** Final conversation state (only when client-managed). */
  state: SessionStateSchema.optional(),
  /** Last model response message from the conversation. */
  message: MessageSchema.optional(),
  /** Artifacts produced during the session. */
  artifacts: z.array(ArtifactSchema).optional(),
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
   * snapshot was created (callback returned false, no store configured, or
   * snapshots were suspended after detach).
   */
  snapshotId: z.string().optional(),
});
export type TurnEnd = z.infer<typeof TurnEndSchema>;

/**
 * Zod schema for agent stream chunk.
 */
export const AgentStreamChunkSchema = z.object({
  /** Generation tokens from the model. */
  modelChunk: ModelResponseChunkSchema.optional(),
  /** User-defined structured status information. */
  status: z.any().optional(),
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
 * Zod schema for the metadata projection of a session snapshot. It exists
 * so callers can identify a snapshot and check its lifecycle status without
 * paying for a full state read.
 */
export const SnapshotMetadataSchema = z.object({
  /** Unique identifier for this snapshot (UUID). */
  snapshotId: z.string(),
  /** ID of the previous snapshot in this timeline. */
  parentId: z.string().optional(),
  /** When the snapshot was first written (RFC 3339). */
  createdAt: z.string(),
  /** When the snapshot was last written (RFC 3339). */
  updatedAt: z.string().optional(),
  /** What triggered this snapshot. */
  event: SnapshotEventSchema,
  /** Lifecycle state of this snapshot. Empty is treated as `complete`. */
  status: SnapshotStatusSchema.optional(),
  /** Failure message for a snapshot in `error` status. */
  error: z.string().optional(),
});
export type SnapshotMetadata = z.infer<typeof SnapshotMetadataSchema>;

/**
 * Zod schema for a persisted point-in-time capture of session state.
 */
export const SessionSnapshotSchema = SnapshotMetadataSchema.extend({
  /**
   * Conversation state. Empty on a pending snapshot (the live state is
   * not yet committed; the background invocation is still processing
   * queued inputs); populated on terminal snapshots with the cumulative
   * final state.
   */
  state: SessionStateSchema,
});
export type SessionSnapshot = z.infer<typeof SessionSnapshotSchema>;

/**
 * Zod schema for the input of an agent's `getSnapshot` companion action.
 * The action is registered at `{agentName}/getSnapshot` when the agent
 * is defined.
 */
export const GetSnapshotRequestSchema = z.object({
  /** Identifies the snapshot to fetch. */
  snapshotId: z.string(),
});
export type GetSnapshotRequest = z.infer<typeof GetSnapshotRequestSchema>;

/**
 * Zod schema for the output of the `getSnapshot` companion action. It is a
 * client-facing view of the stored snapshot: identifying metadata plus the
 * session state, with `WithStateTransform` applied if configured.
 */
export const GetSnapshotResponseSchema = z.object({
  /** Echoes the requested snapshot ID. */
  snapshotId: z.string(),
  /** When the snapshot record was first written (RFC 3339). */
  createdAt: z.string().optional(),
  /** When the snapshot record was last written (RFC 3339). */
  updatedAt: z.string().optional(),
  /** Lifecycle state of the snapshot. */
  status: SnapshotStatusSchema.optional(),
  /** Populated when status is `error`. */
  error: z.string().optional(),
  /**
   * Session state captured by the snapshot, after any configured transform.
   * Empty when status is `pending` or `error`.
   */
  state: SessionStateSchema.optional(),
});
export type GetSnapshotResponse = z.infer<typeof GetSnapshotResponseSchema>;

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
   * this is `canceled`. For an already-terminal snapshot this is the
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
export const AgentMetadataStateManagementSchema = z.enum(['server', 'client']);
export type AgentMetadataStateManagement = z.infer<
  typeof AgentMetadataStateManagementSchema
>;

/**
 * Zod schema for the agent capability metadata placed under
 * `metadata.agent` on an agent's action descriptor. Lets the Dev UI
 * and other reflective callers render the right surface (e.g. hide
 * the Abort button when the configured store doesn't support it)
 * without round-tripping through the reflection API.
 */
export const AgentMetadataSchema = z.object({
  /** Who owns session state for this agent. */
  stateManagement: AgentMetadataStateManagementSchema,
  /**
   * Whether the agent's invocations can be aborted. True only when the
   * configured store implements the abort lifecycle.
   */
  abortable: z.boolean(),
});
export type AgentMetadata = z.infer<typeof AgentMetadataSchema>;
