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

import { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';
import * as z from 'zod';

extendZodWithOpenApi(z);

// NOTE: Keep this file in sync with js/core/src/tracing/types.ts!
// Eventually tools will be source of truth for these types (by generating a
// JSON schema) but until then this file must be manually kept in sync

/**
 * Zod schema for Path metadata.
 */
export const PathMetadataSchema = z.object({
  path: z.string(),
  status: z.string(),
  error: z.string().optional(),
  latency: z.number(),
});
export type PathMetadata = z.infer<typeof PathMetadataSchema>;

/**
 * Zod schema for Trace metadata.
 */
export const TraceMetadataSchema = z.object({
  featureName: z.string().optional(),
  paths: z.set(PathMetadataSchema).optional(),
  timestamp: z.number(),
});
export type TraceMetadata = z.infer<typeof TraceMetadataSchema>;

/**
 * Zod schema for span metadata.
 */
export const SpanMetadataSchema = z.object({
  name: z.string(),
  state: z.enum(['success', 'error']).optional(),
  input: z.any().optional(),
  output: z.any().optional(),
  isRoot: z.boolean().optional(),
  metadata: z.record(z.string(), z.string()).optional(),
  path: z.string().optional(),
});
export type SpanMetadata = z.infer<typeof SpanMetadataSchema>;

/**
 * Zod schema for span status.
 */
export const SpanStatusSchema = z.object({
  code: z.number(),
  message: z.string().optional(),
});

/**
 * Zod schema for time event.
 */
export const TimeEventSchema = z.object({
  time: z.number(),
  annotation: z.object({
    attributes: z.record(z.string(), z.unknown()),
    description: z.string(),
  }),
});

/**
 * Zod schema for span context.
 */
export const SpanContextSchema = z.object({
  traceId: z.string(),
  spanId: z.string(),
  isRemote: z.boolean().optional(),
  traceFlags: z.number(),
});

/**
 * Zod schema for Link.
 */
export const LinkSchema = z.object({
  context: SpanContextSchema.optional(),
  attributes: z.record(z.string(), z.unknown()).optional(),
  droppedAttributesCount: z.number().optional(),
});

/**
 * Zod schema for instrumentation library.
 */
export const InstrumentationLibrarySchema = z.object({
  name: z.string().readonly(),
  version: z.string().optional().readonly(),
  schemaUrl: z.string().optional().readonly(),
});

/**
 * Zod schema for span data.
 */
export const SpanDataSchema = z
  .object({
    spanId: z.string(),
    traceId: z.string(),
    parentSpanId: z.string().optional(),
    startTime: z.number(),
    endTime: z.number(),
    attributes: z.record(z.string(), z.unknown()),
    displayName: z.string(),
    links: z.array(LinkSchema).optional(),
    instrumentationLibrary: InstrumentationLibrarySchema,
    spanKind: z.string(),
    sameProcessAsParentSpan: z.object({ value: z.boolean() }).optional(),
    status: SpanStatusSchema.optional(),
    timeEvents: z
      .object({
        timeEvent: z.array(TimeEventSchema).optional(),
      })
      .optional(),
    truncated: z.boolean().optional(),
  })
  .openapi('SpanData');
export type SpanData = z.infer<typeof SpanDataSchema>;

/**
 * Zod schema for trace metadata.
 */
export const TraceDataSchema = z
  .object({
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
  })
  .openapi('TraceData');
export type TraceData = z.infer<typeof TraceDataSchema>;

export const SpantEventBaseSchema = z.object({
  traceId: z.string(),
  span: SpanDataSchema, // SpanData with endTime = 0
});

/**
 * Schema for span start event - sent when a span begins execution
 */
export const SpanStartEventSchema = SpantEventBaseSchema.extend({
  type: z.literal('span_start'),
});
export type SpanStartEvent = z.infer<typeof SpanStartEventSchema>;

/**
 * Schema for span end event - sent when a span completes
 */
export const SpanEndEventSchema = SpantEventBaseSchema.extend({
  type: z.literal('span_end'),
});
export type SpanEndEvent = z.infer<typeof SpanEndEventSchema>;

/**
 * Union type for all trace events
 */
export const TraceEventSchema = z.union([
  SpanStartEventSchema,
  SpanEndEventSchema,
]);
export type TraceEvent = z.infer<typeof TraceEventSchema>;

export const NestedSpanDataSchema = SpanDataSchema.extend({
  spans: z.lazy(() => z.array(SpanDataSchema)),
  placeholder: z.boolean().optional(),
});

export type NestedSpanData = z.infer<typeof SpanDataSchema> & {
  spans?: SpanData[];
  placeholder?: boolean;
};
