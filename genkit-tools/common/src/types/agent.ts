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
 *   `pending` status and rewritten with a terminal status once the
 *   background work finishes.
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
 *   The snapshot will be rewritten with a terminal status once the flow exits.
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
  /** Input used for session flows that require input variables. */
  inputVariables: z.any().optional(),
});
export type SessionState = z.infer<typeof SessionStateSchema>;

/**
 * Zod schema for session flow input (per-turn).
 */
export const SessionFlowInputSchema = z.object({
  /**
   * Detach signals that the client wishes to disconnect after this input is
   * accepted. The server writes a single pending snapshot capturing the
   * queued inputs (this one and any others already buffered), returns
   * SessionFlowOutput with that snapshot ID, and continues processing in a
   * background context. Streamed chunks emitted after detach are discarded;
   * only the final session state is captured when the snapshot is
   * finalized (or the snapshot is aborted via `abortSnapshot`).
   */
  detach: z.boolean().optional(),
  /** User's input messages for this turn. */
  messages: z.array(MessageSchema).optional(),
  /** Tool request parts to re-execute interrupted tools. */
  toolRestarts: z.array(PartSchema).optional(),
});
export type SessionFlowInput = z.infer<typeof SessionFlowInputSchema>;

/**
 * Zod schema for session flow initialization.
 */
export const SessionFlowInitSchema = z.object({
  /** Loads state from a persisted snapshot. Mutually exclusive with state. */
  snapshotId: z.string().optional(),
  /** Direct state for the invocation. Mutually exclusive with snapshotId. */
  state: SessionStateSchema.optional(),
});
export type SessionFlowInit = z.infer<typeof SessionFlowInitSchema>;

/**
 * Zod schema for session flow result.
 */
export const SessionFlowResultSchema = z.object({
  /** Last model response message from the conversation. */
  message: MessageSchema.optional(),
  /** Artifacts produced during the session. */
  artifacts: z.array(ArtifactSchema).optional(),
});
export type SessionFlowResult = z.infer<typeof SessionFlowResultSchema>;

/**
 * Zod schema for session flow output.
 */
export const SessionFlowOutputSchema = z.object({
  /** ID of the snapshot created at the end of this invocation. */
  snapshotId: z.string().optional(),
  /** Final conversation state (only when client-managed). */
  state: SessionStateSchema.optional(),
  /** Last model response message from the conversation. */
  message: MessageSchema.optional(),
  /** Artifacts produced during the session. */
  artifacts: z.array(ArtifactSchema).optional(),
});
export type SessionFlowOutput = z.infer<typeof SessionFlowOutputSchema>;

/**
 * Zod schema for the turn-end signal emitted by a session flow.
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
 * Zod schema for session flow stream chunk.
 */
export const SessionFlowStreamChunkSchema = z.object({
  /** Generation tokens from the model. */
  modelChunk: ModelResponseChunkSchema.optional(),
  /** User-defined structured status information. */
  status: z.any().optional(),
  /** A newly produced artifact. */
  artifact: ArtifactSchema.optional(),
  /**
   * Non-null when the session flow has finished processing the current
   * input. Groups all turn-end signals; the client should stop iterating and
   * may send the next input.
   */
  turnEnd: TurnEndSchema.optional(),
});
export type SessionFlowStreamChunk = z.infer<
  typeof SessionFlowStreamChunkSchema
>;

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
   * Inputs captured at detach time, in FIFO order. The first entry may be
   * the input that was in flight when detach landed (its turn was
   * suppressed because snapshots were suspended in the same atomic step
   * that captured it); the rest were queued behind it. Set only on
   * snapshots in `pending` status; cleared when the snapshot is finalized.
   */
  pendingInputs: z.array(SessionFlowInputSchema).optional(),
  /**
   * Conversation state. Empty on a pending snapshot (the queued inputs are
   * in `pendingInputs` and the live state is not yet committed); populated
   * on terminal snapshots.
   */
  state: SessionStateSchema,
});
export type SessionSnapshot = z.infer<typeof SessionSnapshotSchema>;

/**
 * Zod schema for the input of a session flow's `getSnapshot` companion
 * action. The action is registered at `{flowName}/getSnapshot` when the
 * flow is defined.
 */
export const GetSnapshotRequestSchema = z.object({
  /** Identifies the snapshot to fetch. */
  snapshotId: z.string(),
});
export type GetSnapshotRequest = z.infer<typeof GetSnapshotRequestSchema>;

/**
 * Zod schema for the output of the `getSnapshot` companion action. It is a
 * client-facing view of the stored snapshot: identifying metadata plus the
 * session state, with `WithSnapshotTransform` applied if configured.
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
  /** Queued inputs captured at detach time. Populated only when status is `pending`. */
  pendingInputs: z.array(SessionFlowInputSchema).optional(),
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
export type AbortSnapshotResponse = z.infer<
  typeof AbortSnapshotResponseSchema
>;
