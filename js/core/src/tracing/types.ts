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

// NOTE: Keep this file in sync with genkit-tools/common/src/types/trace.ts!
// Eventually tools will be source of truth for these types (by generating a
// JSON schema) but until then this file must be manually kept in sync

export const PathMetadataSchema = z.object({
  path: z.string(),
  status: z.string(),
  error: z.string().optional(),
  latency: z.number(),
});
export type PathMetadata = z.infer<typeof PathMetadataSchema>;

export const TraceMetadataSchema = z.object({
  featureName: z.string().optional(),
  paths: z.set(PathMetadataSchema).optional(),
  timestamp: z.number(),
});
export type TraceMetadata = z.infer<typeof TraceMetadataSchema>;

export const SpanMetadataSchema = z.object({
  name: z.string(),
  state: z.enum(['success', 'error']).optional(),
  input: z.any().optional(),
  output: z.any().optional(),
  init: z.any().optional(),
  isRoot: z.boolean().optional(),
  metadata: z.record(z.string(), z.string()).optional(),
  path: z.string().optional(),
  // Indicates a "leaf" span that is the source of a failure.
  isFailureSource: z.boolean().optional(),
});
export type SpanMetadata = z.infer<typeof SpanMetadataSchema>;

export const SpanStatusSchema = z.object({
  code: z.number(),
  message: z.string().optional(),
});

export const TimeEventSchema = z.object({
  time: z.number(),
  annotation: z.object({
    attributes: z.record(z.string(), z.any()),
    description: z.string(),
  }),
});

export const SpanContextSchema = z.object({
  traceId: z.string(),
  spanId: z.string(),
  isRemote: z.boolean().optional(),
  traceFlags: z.number(),
});

export const LinkSchema = z.object({
  context: SpanContextSchema.optional(),
  attributes: z.record(z.string(), z.any()).optional(),
  droppedAttributesCount: z.number().optional(),
});

export const InstrumentationLibrarySchema = z.object({
  name: z.string().readonly(),
  version: z.string().optional().readonly(),
  schemaUrl: z.string().optional().readonly(),
});

export const SpanDataSchema = z.object({
  spanId: z.string(),
  traceId: z.string(),
  parentSpanId: z.string().optional(),
  startTime: z.number(),
  endTime: z.number(),
  attributes: z.record(z.string(), z.any()),
  displayName: z.string(),
  links: z.array(LinkSchema).optional(),
  instrumentationLibrary: InstrumentationLibrarySchema,
  spanKind: z.string(),
  sameProcessAsParentSpan: z.object({ value: z.boolean() }).optional(),
  status: SpanStatusSchema.optional(),
  timeEvents: z
    .object({
      timeEvent: z.array(TimeEventSchema),
    })
    .optional(),
  truncated: z.boolean().optional(),
});
export type SpanData = z.infer<typeof SpanDataSchema>;

export const TraceDataSchema = z.object({
  traceId: z.string(),
  displayName: z.string().optional(),
  startTime: z
    .number()
    .optional()
    .describe('trace start time in milliseconds since the epoch'),
  endTime: z
    .number()
    .optional()
    .describe('end time in milliseconds since the epoch'),
  spans: z.record(z.string(), SpanDataSchema),
});

export type TraceData = z.infer<typeof TraceDataSchema>;
