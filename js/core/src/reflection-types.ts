/**
 * Copyright 2024 Google LLC
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
import { ActionMetadataSchema } from './action.js';

// NOTE: Keep this file in sync with genkit-tools/common/src/types/reflection.ts
// and genkit-tools/common/src/types/apis.ts.

/**
 * ReflectionRegisterSchema is the payload for the 'register' method.
 */
export const ReflectionRegisterParamsSchema = z.object({
  id: z.string(),
  pid: z.number(),
  name: z.string().optional(),
  genkitVersion: z.string().optional(),
  reflectionApiSpecVersion: z.number().optional(),
  envs: z.array(z.string()).optional(),
});

export type ReflectionRegisterParams = z.infer<
  typeof ReflectionRegisterParamsSchema
>;

/**
 * ReflectionStreamChunkSchema is the payload for the 'streamChunk' method.
 */
export const ReflectionStreamChunkParamsSchema = z.object({
  requestId: z.string(),
  chunk: z.any(),
});

export type ReflectionStreamChunkParams = z.infer<
  typeof ReflectionStreamChunkParamsSchema
>;

/**
 * ReflectionRunActionStateSchema is the payload for the 'runActionState' method.
 */
export const ReflectionRunActionStateParamsSchema = z.object({
  requestId: z.string(),
  state: z
    .object({
      traceId: z.string().optional(),
    })
    .optional(),
});

export type ReflectionRunActionStateParams = z.infer<
  typeof ReflectionRunActionStateParamsSchema
>;

/**
 * ReflectionConfigureSchema is the payload for the 'configure' method.
 */
export const ReflectionConfigureParamsSchema = z.object({
  telemetryServerUrl: z.string().optional(),
});

export type ReflectionConfigureParams = z.infer<
  typeof ReflectionConfigureParamsSchema
>;

/**
 * ReflectionListValuesSchema is the payload for the 'listValues' method.
 */
export const ReflectionListValuesParamsSchema = z.object({
  type: z.string(),
});

export type ReflectionListValuesParams = z.infer<
  typeof ReflectionListValuesParamsSchema
>;

/**
 * ReflectionRunActionSchema is the payload for the 'runAction' method.
 */
export const ReflectionRunActionParamsSchema = z.object({
  key: z.string(),
  input: z.any().optional(),
  init: z.any().optional(),
  context: z.any().optional(),
  telemetryLabels: z.record(z.string(), z.string()).optional(),
  stream: z.boolean().optional(),
  streamInput: z.boolean().optional(),
});

export type ReflectionRunActionParams = z.infer<
  typeof ReflectionRunActionParamsSchema
>;

/**
 * ReflectionCancelActionSchema is the payload for the 'cancelAction' method.
 */
export const ReflectionCancelActionParamsSchema = z.object({
  traceId: z.string(),
});

export type ReflectionCancelActionParams = z.infer<
  typeof ReflectionCancelActionParamsSchema
>;

/**
 * ReflectionSendInputStreamChunkSchema is the payload for the 'sendInputStreamChunk' method.
 */
export const ReflectionSendInputStreamChunkParamsSchema = z.object({
  requestId: z.string(),
  chunk: z.any(),
});

export type ReflectionSendInputStreamChunkParams = z.infer<
  typeof ReflectionSendInputStreamChunkParamsSchema
>;

/**
 * ReflectionEndInputStreamSchema is the payload for the 'endInputStream' method.
 */
export const ReflectionEndInputStreamParamsSchema = z.object({
  requestId: z.string(),
});

export type ReflectionEndInputStreamParams = z.infer<
  typeof ReflectionEndInputStreamParamsSchema
>;

/**
 * ReflectionListActionsResponseSchema is the result for the 'listActions' method.
 */
export const ReflectionListActionsResponseSchema = z.object({
  actions: z.record(z.string(), ActionMetadataSchema),
});

export type ReflectionListActionsResponse = z.infer<
  typeof ReflectionListActionsResponseSchema
>;

/**
 * ReflectionListValuesResponseSchema is the result for the 'listValues' method.
 */
export const ReflectionListValuesResponseSchema = z.object({
  values: z.record(z.any()),
});

export type ReflectionListValuesResponse = z.infer<
  typeof ReflectionListValuesResponseSchema
>;

/**
 * ReflectionCancelActionResponseSchema is the result for the 'cancelAction' method.
 */
export const ReflectionCancelActionResponseSchema = z.object({
  message: z.string(),
});

export type ReflectionCancelActionResponse = z.infer<
  typeof ReflectionCancelActionResponseSchema
>;
