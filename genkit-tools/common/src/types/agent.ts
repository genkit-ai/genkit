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
 */
export const SnapshotEventSchema = z.enum(['turnEnd', 'invocationEnd']);
export type SnapshotEvent = z.infer<typeof SnapshotEventSchema>;

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
  /** Input used for agents that require input variables. */
  inputVariables: z.any().optional(),
});
export type SessionState = z.infer<typeof SessionStateSchema>;

/**
 * Zod schema for agent input (per-turn).
 */
export const AgentInputSchema = z.object({
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
   * snapshot was created (callback returned false or no store configured).
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
