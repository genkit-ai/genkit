/**
 * @license
 *
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

/**
 * Tracing and telemetry: span/trace schemas, custom metadata helpers, and the
 * pluggable instrumentation API.
 *
 * Genkit records nothing outside the Developer UI unless an instrumentation is
 * configured, e.g. OpenTelemetry via `@genkit-ai/otel`:
 *
 * ```ts
 * import { GenAiInstrumentation } from '@genkit-ai/otel';
 * import { configureInstrumentation } from 'genkit/tracing';
 *
 * configureInstrumentation(new GenAiInstrumentation());
 * ```
 *
 * @module tracing
 */

export {
  DirectTelemetryInstrumentation,
  SPAN_TYPE_ATTR,
  SpanContextSchema,
  SpanDataSchema,
  SpanMetadataSchema,
  SpanStatusSchema,
  TimeEventSchema,
  TraceDataSchema,
  TraceMetadataSchema,
  configureInstrumentation,
  flushTracing,
  resetInstrumentation,
  runInNewSpan,
  setCustomMetadataAttribute,
  setCustomMetadataAttributes,
  setTelemetryServerUrl,
  toDisplayPath,
  type GenkitLogRecord,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationNext,
  type InstrumentationSpanInfo,
  type LogRecordingInstrumentation,
  type PathMetadata,
  type SpanData,
  type SpanMetadata,
  type TraceData,
  type TraceMetadata,
} from '@genkit-ai/core/tracing';
