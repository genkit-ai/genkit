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
 * Zod schemas for the Agent conformance test spec format.
 *
 * These schemas define the structure of `tests/specs/agent.yaml` and are
 * shared across all language implementations. See
 * `docs/agents-conformance-testing.md` for the full spec format reference.
 *
 * General agent wire types (SessionState, AgentInput, etc.) are defined in
 * `./agent.ts`. This file only contains the conformance test spec schemas.
 */

import { z } from 'zod';
import {
  AgentFinishReasonSchema,
  AgentInitSchema,
  AgentInputSchema,
  AgentStreamChunkSchema,
  ArtifactSchema,
  SessionStateSchema,
} from './agent';

import {
  GenerateResponseChunkSchema,
  GenerateResponseSchema,
  MessageSchema,
} from './model';

// ---------------------------------------------------------------------------
// Conformance spec — output assertions
// ---------------------------------------------------------------------------

/**
 * Schema for output assertions on a `send` invocation.
 */
export const OutputAssertionsSchema = z.object({
  /** If present, output.message must deep-match this value. */
  message: MessageSchema.optional(),
  /** If true, asserts output.snapshotId is a non-empty string. */
  hasSnapshotId: z.boolean().optional(),
  /** If true, asserts output.state.sessionId is a non-empty string. */
  hasSessionId: z.boolean().optional(),
  /** Partial match on output.state. */
  stateContains: SessionStateSchema.partial().optional(),
  /** Partial match on output.artifacts. */
  artifactsContain: z.array(ArtifactSchema).optional(),
  /** If present, output.finishReason must equal this value exactly. */
  finishReason: AgentFinishReasonSchema.optional(),
  /**
   * If present, asserts output.error contains (at minimum) these fields
   * (`status` matched exactly, `message` as a substring). Set by the graceful
   * failure path when finishReason is `failed`.
   */
  errorContains: z.record(z.string(), z.any()).optional(),
});
export type OutputAssertions = z.infer<typeof OutputAssertionsSchema>;

// ---------------------------------------------------------------------------
// Conformance spec — snapshot assertions
// ---------------------------------------------------------------------------

/**
 * Schema for snapshot assertions on `getSnapshotData` and
 * `waitUntilCompleted` steps.
 */
export const SnapshotAssertionsSchema = z.object({
  /** Expected parentId. Supports {{name}} template references. */
  parentId: z.string().optional(),
  /** Expected status (e.g. "done", "pending", "failed", "aborted"). */
  status: z.string().optional(),
  /** If present, snapshot.finishReason must equal this value exactly. */
  finishReason: AgentFinishReasonSchema.optional(),
  /** If true, asserts snapshot.state.sessionId is a non-empty string. */
  hasSessionId: z.boolean().optional(),

  /** Partial match on snapshot.state. */
  stateContains: SessionStateSchema.partial().optional(),
  /** If present, asserts snapshot.error contains (at minimum) these fields. */
  errorContains: z.record(z.string(), z.any()).optional(),
});
export type SnapshotAssertions = z.infer<typeof SnapshotAssertionsSchema>;

// ---------------------------------------------------------------------------
// Conformance spec — invocation types
// ---------------------------------------------------------------------------

/**
 * Schema for a `send` invocation — sends inputs to the agent via bidi stream.
 */
export const SendInvocationSchema = z.object({
  type: z.literal('send'),
  /** Initialization payload (snapshotId, state, or empty). */
  init: AgentInitSchema.optional(),
  /** Ordered list of inputs to send. */
  inputs: z.array(AgentInputSchema).optional(),
  /** Pre-programmed model responses, one per generate call. */
  modelResponses: z.array(GenerateResponseSchema).optional(),
  /** Pre-programmed streaming chunks, indexed by model call. */
  streamChunks: z.array(z.array(GenerateResponseChunkSchema)).optional(),
  /** Strict ordered list of expected stream chunks. */
  expectChunks: z.array(AgentStreamChunkSchema).optional(),
  /** Expected fields on the AgentOutput. */
  expectOutput: OutputAssertionsSchema.optional(),
  /**
   * If present, the turn is expected to throw an error (rather than resolve
   * with a graceful `finishReason: 'failed'` output). Asserts on the thrown
   * error: `status` is matched exactly and `message` as a substring. Used for
   * API-misuse cases (e.g. sending `state` to a server-managed agent).
   */
  expectError: z
    .object({
      status: z.string().optional(),
      message: z.string().optional(),
    })
    .optional(),
  /** Capture output.snapshotId under this name for {{name}} references. */

  captureSnapshotId: z.string().optional(),
  /** Capture output.state under this name for {{name}} references. */
  captureState: z.string().optional(),
  /** Capture output.state.sessionId under this name for {{name}} references. */
  captureSessionId: z.string().optional(),
});
export type SendInvocation = z.infer<typeof SendInvocationSchema>;

/**
 * Schema for a `getSnapshotData` invocation — fetches a snapshot by
 * snapshotId (exact) or sessionId (latest leaf snapshot).
 *
 * Exactly one of `snapshotId` / `sessionId` must be provided.
 */
export const GetSnapshotDataInvocationSchema = z.object({
  type: z.literal('getSnapshotData'),
  /** Exact snapshot ID to fetch. Supports {{name}} references. */
  snapshotId: z.string().optional(),
  /**
   * Session ID (a UUID) whose latest (leaf) snapshot should be resolved.
   * Mutually exclusive with `snapshotId`.
   */
  sessionId: z.string().optional(),
  /** Assertions on the fetched snapshot. */
  expectSnapshot: SnapshotAssertionsSchema.optional(),
  /**
   * If present, the lookup is expected to throw an error whose message
   * contains this text (e.g. branching sessions reject sessionId lookups).
   */
  expectError: z.string().optional(),
});

export type GetSnapshotDataInvocation = z.infer<
  typeof GetSnapshotDataInvocationSchema
>;

/**
 * Schema for an `abort` invocation — aborts an agent by snapshot ID.
 */
export const AbortInvocationSchema = z.object({
  type: z.literal('abort'),
  /** Snapshot ID to abort. Supports {{name}} references. */
  snapshotId: z.string(),
  /** Expected previous status before abort. */
  expectPreviousStatus: z.string().optional(),
});
export type AbortInvocation = z.infer<typeof AbortInvocationSchema>;

/**
 * Schema for a `waitUntilCompleted` invocation — polls a snapshot until
 * it reaches a terminal status.
 */
export const WaitUntilCompletedInvocationSchema = z.object({
  type: z.literal('waitUntilCompleted'),
  /** Snapshot ID to poll. Supports {{name}} references. */
  snapshotId: z.string(),
  /** Max time to wait in milliseconds. Default: 5000. */
  timeoutMs: z.number().optional(),
  /** Assertions on the snapshot once it reaches terminal status. */
  expectSnapshot: SnapshotAssertionsSchema.optional(),
});
export type WaitUntilCompletedInvocation = z.infer<
  typeof WaitUntilCompletedInvocationSchema
>;

/**
 * Union schema for all invocation types.
 */
export const SpecStepSchema = z.discriminatedUnion('type', [
  SendInvocationSchema,
  GetSnapshotDataInvocationSchema,
  AbortInvocationSchema,
  WaitUntilCompletedInvocationSchema,
]);
export type SpecStep = z.infer<typeof SpecStepSchema>;

// ---------------------------------------------------------------------------
// Conformance spec — top-level
// ---------------------------------------------------------------------------

/**
 * Schema for a single conformance test case.
 */
export const SpecTestSchema = z.object({
  /** Human-readable test name. */
  name: z.string(),
  /** Optional description. */
  description: z.string().optional(),
  /** Name of the harness-provided agent to use. */
  agent: z.string(),
  /** Ordered sequence of steps to execute. */
  steps: z.array(SpecStepSchema),
});
export type SpecTest = z.infer<typeof SpecTestSchema>;

/**
 * Schema for the full conformance test suite (top-level of agent.yaml).
 */
export const SpecSuiteSchema = z.object({
  tests: z.array(SpecTestSchema),
});
export type SpecSuite = z.infer<typeof SpecSuiteSchema>;
