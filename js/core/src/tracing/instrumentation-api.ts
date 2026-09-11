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

import type { Span as ApiSpan, Link } from '@opentelemetry/api';
import type { SpanMetadata } from './types.js';

/**
 * The genkit attribute namespace, e.g. `genkit:name`, `genkit:path`.
 * @hidden
 */
export const ATTR_PREFIX = 'genkit';
/** @hidden */
export const SPAN_TYPE_ATTR = ATTR_PREFIX + ':type';
/** @hidden */
export const TRACER_NAME = 'genkit-tracer';
/** @hidden */
export const TRACER_VERSION = 'v1';

/** ALS key under which the dispatcher stores the current span context. */
export const spanMetadataAlsKey = 'core.tracing.instrumentation.span';

/**
 * A backend-independent view of the span handed to the wrapped function.
 *
 * `traceId` / `spanId` are the composite-resolved ids (first non-empty across
 * the active instrumentation chain); both are '' when nothing is instrumented.
 *
 * @hidden
 */
export interface GenkitSpanContext {
  readonly traceId: string;
  readonly spanId: string;
  setMetadata(values: Record<string, unknown>): void;
}

/**
 * Everything an {@link Instrumentation} provider needs to open a span. The
 * dispatcher owns Genkit semantics (path, isRoot, custom metadata); providers
 * only encode this into their backend.
 *
 * @hidden
 */
export interface InstrumentationSpanInfo {
  metadata: SpanMetadata;
  labels?: Record<string, string>;
  links?: Link[];
}

/**
 * Continuation passed to a provider's `runInNewSpan`. Providers must call it
 * exactly once and return its result.
 *
 * @hidden
 */
export type InstrumentationNext<T> = (
  span: ApiSpan,
  ctx: GenkitSpanContext,
  isRoot: boolean
) => Promise<T>;

/**
 * A pluggable telemetry instrumentation provider. Middleware over span
 * creation: implementations open a backend span, call `next`, and finalize.
 *
 * @hidden
 */
export interface Instrumentation {
  runInNewSpan<T>(
    info: InstrumentationSpanInfo,
    next: InstrumentationNext<T>
  ): Promise<T>;
}

/**
 * Optional capability: a provider that needs cleanup on
 * {@link resetInstrumentation}.
 *
 * @hidden
 */
export interface DisposableInstrumentation {
  dispose(): void | Promise<void>;
}

/**
 * A correlation-free log record. Each provider attaches trace/span correlation
 * from its own context source (OTel active context vs Genkit ALS).
 *
 * @hidden
 */
export interface GenkitLogRecord {
  severity: 'debug' | 'info' | 'warn' | 'error';
  body: unknown;
  attributes?: Record<string, unknown>;
}

/**
 * Optional capability: a provider that also records logs (mirrors span
 * recording). `logging.ts` fans out to providers implementing this.
 *
 * @hidden
 */
export interface LogRecordingInstrumentation {
  recordLog(record: GenkitLogRecord): void;
}

/** @hidden */
export function hasRecordLog(
  p: Instrumentation
): p is Instrumentation & LogRecordingInstrumentation {
  return (
    typeof (p as Partial<LogRecordingInstrumentation>).recordLog === 'function'
  );
}

/** @hidden */
export function isDisposable(
  p: Instrumentation
): p is Instrumentation & DisposableInstrumentation {
  return (
    typeof (p as Partial<DisposableInstrumentation>).dispose === 'function'
  );
}

/** An OTel id made entirely of zeros is the no-op / unknown id. */
export function isValidId(id: string | undefined): id is string {
  return !!id && !/^0+$/.test(id);
}

/** First valid (non-empty, non-zero) id across the chain, or ''. */
export function firstNonEmpty(ids: (string | undefined)[]): string {
  for (const id of ids) {
    if (isValidId(id)) return id;
  }
  return '';
}

/**
 * Maps SpanMetadata to the flat `genkit:*` attribute set persisted on a span.
 * Shared by all providers so encodings stay identical.
 *
 * @hidden
 */
export function metadataToAttributes(
  metadata: SpanMetadata
): Record<string, string> {
  const out = {} as Record<string, string>;
  Object.keys(metadata).forEach((key) => {
    if (
      key === 'metadata' &&
      typeof metadata[key] === 'object' &&
      metadata.metadata
    ) {
      Object.entries(metadata.metadata).forEach(([metaKey, value]) => {
        out[ATTR_PREFIX + ':metadata:' + metaKey] = value;
      });
    } else if (
      key === 'input' ||
      key === 'init' ||
      typeof metadata[key] === 'object'
    ) {
      out[ATTR_PREFIX + ':' + key] = JSON.stringify(metadata[key]);
    } else {
      out[ATTR_PREFIX + ':' + key] = metadata[key];
    }
  });
  return out;
}

/** @hidden */
export function getErrorMessage(e: any): string {
  if (e instanceof Error) {
    return e.message;
  }
  return `${e}`;
}

const rootSpanDetectionKey = '__genkit_disableRootSpanDetection';

/** @hidden */
export function isDisableRootSpanDetection(): boolean {
  return global[rootSpanDetectionKey] === true;
}

/**
 * Disables Genkit's custom root span detection and leaves default Otel root span.
 *
 * This function attempts to control Genkit's internal OTel instrumentation behaviour,
 * since internal implementation details are subject to change at any time consider
 * this function "unstable" and subject to breaking changes as well.
 *
 * @hidden
 */
export function disableOTelRootSpanDetection() {
  global[rootSpanDetectionKey] = true;
}
