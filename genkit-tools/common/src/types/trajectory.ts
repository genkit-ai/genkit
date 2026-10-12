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
  actionId: z.string(),
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
  subtrajectoryId: z.string().optional(),
  resume: z
    .object({
      strategy: z.enum(['respond', 'restart']),
      payload: z.unknown().optional(),
    })
    .optional(),
});

export type TrajectoryAction = z.infer<typeof TrajectoryActionSchema>;

/**
 * Zod schema for a single chronological step in an agent or model trajectory.
 */
export const TrajectoryStepSchema = z.object({
  stepId: z.string(),
  stepNumber: z.number(),
  traceId: z.string(),
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
});

export type TrajectoryStep = z.infer<typeof TrajectoryStepSchema>;
export type TrajectoryStepData = TrajectoryStep;

/**
 * Zod schema for a single agent or model trajectory (root or subagent).
 */
export const TrajectorySchema = z.object({
  trajectoryId: z.string(),
  parentTrajectoryId: z.string().optional(),
  agentName: z.string().optional(),
  systemInstructions: z.string().optional(),
  tools: z.array(ToolDefinitionSchema),
  steps: z.array(TrajectoryStepSchema),
  usage: GenerationUsageSchema.optional(),
  startTime: z.number(),
  endTime: z.number(),
});

export type Trajectory = z.infer<typeof TrajectorySchema>;

/**
 * Zod schema for a flat bundle of root and nested subagent trajectories.
 */
export const TrajectoryBundleSchema = z.object({
  rootTrajectoryId: z.string(),
  sessionId: z.string().optional(),
  trajectories: z.record(z.string(), TrajectorySchema),
});

export type TrajectoryBundle = z.infer<typeof TrajectoryBundleSchema>;
