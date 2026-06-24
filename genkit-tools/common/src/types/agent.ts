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
 * Events signifying a session snapshot persistence point.
 */
export const SnapshotEventSchema = z.enum(['turnEnd', 'invocationEnd']);
export type SnapshotEvent = z.infer<typeof SnapshotEventSchema>;

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
  name: z.string().optional(),
  parts: z.array(PartSchema),
  metadata: z.record(z.any()).optional(),
});
export type Artifact = z.infer<typeof ArtifactSchema>;

/**
 * Schema for session state persisted across turns.
 */
export const SessionStateSchema = z.object({
  sessionId: z.string().optional(),
  messages: z.array(MessageSchema).optional(),
  custom: z.any().optional(),
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
  snapshotId: z.string().optional(),
  sessionId: z.string().optional(),
  state: SessionStateSchema.optional(),
});
export type AgentInit = z.infer<typeof AgentInitSchema>;

/**
 * Schema for agent input messages and commands.
 */
export const AgentInputSchema = z.object({
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
export type AgentInput = z.infer<typeof AgentInputSchema>;

/**
 * Schema for turn termination event.
 */
export const TurnEndSchema = z.object({
  snapshotId: z.string().optional(),
  /** The reason this turn finished (e.g. `stop`, `interrupted`). */
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
export type AgentStreamChunk = z.infer<typeof AgentStreamChunkSchema>;

/**
 * Schema for the final results of an agent execution.
 */
export const AgentResultSchema = z.object({
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
  /** The reason the whole invocation finished (e.g. `stop`, `interrupted`). */
  finishReason: AgentFinishReasonSchema.optional(),
});
export type AgentResult = z.infer<typeof AgentResultSchema>;

export const RuntimeErrorSchema = z.object({
  status: z.string().optional(),
  message: z.string(),
  details: z.any().optional(),
});
export type RuntimeError = z.infer<typeof RuntimeErrorSchema>;

/**
 * Schema for agent output returned at completion.
 */
export const AgentOutputSchema = z.object({
  sessionId: z.string().optional(),
  snapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
  /** The reason the invocation finished (e.g. `stop`, `interrupted`). */
  finishReason: AgentFinishReasonSchema.optional(),
  /**
   * Present when `finishReason` is `failed`. Carries the original error
   * details (the runtime resolves gracefully instead of throwing). The
   * accompanying `state`/`snapshotId` hold the last-good state — the state
   * the failed turn started with.
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

export const SnapshotStatusSchema = z.enum([
  'pending',
  'completed',
  'aborted',
  'failed',
  'expired',
]);
export type SnapshotStatus = z.infer<typeof SnapshotStatusSchema>;

/**
 * Schema for a persisted session snapshot.
 */
export const SessionSnapshotSchema = z.object({
  snapshotId: z.string(),
  sessionId: z.string().optional(),
  parentId: z.string().optional(),
  createdAt: z.string(),
  updatedAt: z.string().optional(),
  heartbeatAt: z.string().optional(),
  state: SessionStateSchema,
  status: SnapshotStatusSchema.optional(),
  /**
   * Semantic reason the turn/invocation finished (e.g. `interrupted`,
   * `stop`). Distinct from `status`, which tracks the persistence lifecycle.
   */
  finishReason: AgentFinishReasonSchema.optional(),
  error: RuntimeErrorSchema.optional(),
});
export type SessionSnapshot = z.infer<typeof SessionSnapshotSchema>;
