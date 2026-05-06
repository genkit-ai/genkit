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
import { PartSchema } from './parts';

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
  newSnapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
});
export type AgentInit = z.infer<typeof AgentInitSchema>;

/**
 * Schema for agent input messages and commands.
 */
export const AgentInputSchema = z.object({
  messages: z.array(MessageSchema).optional(),
  toolRestarts: z.array(PartSchema).optional(),
  detach: z.boolean().optional(),
});
export type AgentInput = z.infer<typeof AgentInputSchema>;

/**
 * Schema for turn termination event.
 */
export const TurnEndSchema = z.object({
  snapshotId: z.string().optional(),
});
export type TurnEnd = z.infer<typeof TurnEndSchema>;

/**
 * Schema for stream chunks emitted during agent execution.
 */
export const AgentStreamChunkSchema = z.object({
  modelChunk: ModelResponseChunkSchema.optional(),
  status: z.any().optional(),
  artifact: ArtifactSchema.optional(),
  turnEnd: TurnEndSchema.optional(),
});
export type AgentStreamChunk = z.infer<typeof AgentStreamChunkSchema>;

/**
 * Schema for agent output returned at completion.
 */
export const AgentOutputSchema = z.object({
  snapshotId: z.string().optional(),
  state: SessionStateSchema.optional(),
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});
export type AgentOutput = z.infer<typeof AgentOutputSchema>;

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/**
 * Schema for a persisted session snapshot.
 */
export const SessionSnapshotSchema = z.object({
  snapshotId: z.string(),
  parentId: z.string().optional(),
  createdAt: z.string(),
  event: z.enum(['turnEnd', 'invocationEnd']),
  state: SessionStateSchema,
  status: z.enum(['pending', 'done', 'failed', 'aborted']).optional(),
  error: z
    .object({
      status: z.string(),
      message: z.string(),
      details: z.any().optional(),
    })
    .optional(),
});
export type SessionSnapshot = z.infer<typeof SessionSnapshotSchema>;
