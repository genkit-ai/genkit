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
import { GenerationUsageSchema, ToolDefinitionSchema } from './model';
import { PartSchema } from './parts';

/**
 * Zod schema for a single tool, subagent, or interrupt action within a trajectory step.
 */
export const TrajectoryActionSchema = z.object({
  /**
   * Unique identifier for this action within the step. Set to `toolRequest.ref`
   * (the model's tool call ID) when present, otherwise `<modelSpanId>-action-<1-based index>`.
   */
  actionId: z.string(),
  /**
   * `spanId` of the paired tool execution span (or direct subagent span if invoked
   * without an enclosing tool wrapper). Omitted when no execution span was recorded.
   */
  spanId: z.string().optional(),
  name: z.string(),
  type: z.enum(['tool', 'subagent', 'interrupt']),
  status: z.enum(['success', 'error', 'interrupted', 'pending']),
  input: z.unknown().optional(),
  output: z.unknown().optional(),
  error: z.string().optional(),
  startTime: z.number().optional(),
  endTime: z.number().optional(),
  durationMs: z.number().optional(),
  /**
   * For `'subagent'` actions, the `trajectoryId` of the delegated child trajectory
   * in `TrajectoryBundle.trajectories` (set to the child subagent span's `spanId`).
   */
  subtrajectoryId: z.string().optional(),
  resume: z
    .object({
      strategy: z.enum(['respond', 'restart']),
      payload: z.unknown().optional(),
    })
    .optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type TrajectoryAction = z.infer<typeof TrajectoryActionSchema>;

/**
 * Zod schema for a single chronological step in an agent or model trajectory.
 */
export const TrajectoryStepSchema = z.object({
  /**
   * Unique identifier for this step, formatted as `<traceId>-step-<stepNumber>-<spanId>`.
   */
  stepId: z.string(),
  /** 1-based sequential index of this step within its `Trajectory.steps` array. */
  stepNumber: z.number(),
  /** `traceId` of the `TraceData` from which this step was extracted. */
  traceId: z.string(),
  /**
   * Source span's `spanId`: the model execution span for `'agent'` steps, or the
   * enclosing `runTurn-<N>` span (falling back to the model/scope span) for `'user'`
   * and `'system'` input steps.
   */
  spanId: z.string(),
  turnIndex: z.number().optional(),
  source: z.enum(['user', 'agent', 'system']),
  startTime: z.number(),
  endTime: z.number(),
  latencyMs: z.number(),
  model: z.string().optional(),
  content: z.array(PartSchema),
  actions: z.array(TrajectoryActionSchema),
  stopReason: z
    .enum([
      'completed',
      'tools',
      'interrupted',
      'errored',
      'cancelled',
      'max_tokens',
    ])
    .optional(),
  error: z
    .object({
      message: z.string(),
      stack: z.string().optional(),
    })
    .optional(),
  usage: GenerationUsageSchema.optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type TrajectoryStep = z.infer<typeof TrajectoryStepSchema>;

/**
 * Zod schema for a single agent or model trajectory (root or subagent).
 */
export const TrajectorySchema = z.object({
  /**
   * Unique identifier for this trajectory (and its key in `TrajectoryBundle.trajectories`).
   * Set to `trace.traceId` (or `'root'` if empty) for the root trajectory, and to the
   * subagent root span's `spanId` for nested subagent trajectories.
   */
  trajectoryId: z.string(),
  /**
   * `trajectoryId` of the enclosing parent trajectory when this trajectory represents
   * a nested subagent; omitted for the root trajectory.
   */
  parentTrajectoryId: z.string().optional(),
  agentName: z.string().optional(),
  systemInstructions: z.string().optional(),
  tools: z.array(ToolDefinitionSchema),
  steps: z.array(TrajectoryStepSchema),
  usage: GenerationUsageSchema.optional(),
  startTime: z.number(),
  endTime: z.number(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type Trajectory = z.infer<typeof TrajectorySchema>;

/**
 * Zod schema for a flat bundle of root and nested subagent trajectories.
 */
export const TrajectoryBundleSchema = z.object({
  /**
   * Key in `trajectories` for the top-level trajectory, set to `trace.traceId`
   * (or `'root'` if `traceId` is empty).
   */
  rootTrajectoryId: z.string(),
  /**
   * Session identifier extracted from the shallowest span attribute
   * `genkit:metadata:agent:sessionId` or `session.id`, if present.
   */
  sessionId: z.string().optional(),
  /**
   * Flat map of root and nested subagent trajectories keyed by `trajectoryId`.
   */
  trajectories: z.record(z.string(), TrajectorySchema),
});

export type TrajectoryBundle = z.infer<typeof TrajectoryBundleSchema>;
