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

import type { SpanMetadata } from './types.js';

/**
 * The genkit attribute namespace, e.g. `genkit:name`, `genkit:path`.
 * @hidden
 */
export const ATTR_PREFIX = 'genkit';
/** @hidden */
export const SPAN_TYPE_ATTR = ATTR_PREFIX + ':type';

/** ALS key under which the dispatcher stores the current span context. */
export const spanMetadataAlsKey = 'core.tracing.instrumentation.span';

/**
 * A span attribute value. Same shape OpenTelemetry accepts, so OTel-backed
 * instrumentations can pass values straight through.
 *
 * @hidden
 */
export type SpanAttributeValue =
  | string
  | number
  | boolean
  | Array<null | undefined | string>
  | Array<null | undefined | number>
  | Array<null | undefined | boolean>;

/** @hidden */
export type SpanAttributes = Record<string, SpanAttributeValue | undefined>;

/**
 * A backend-independent view of the current span. Handed to `runInNewSpan`
 * callbacks and, per provider, to the dispatcher.
 *
 * `traceId` / `spanId` are the composite-resolved ids (first non-empty across
 * the active instrumentation chain). A provider may report '' when it has no
 * ids; `runInNewSpan` callbacks then see the all-zero W3C "invalid" ids.
 *
 * `setMetadata` records custom attributes on the span (fanned out to every
 * provider in the chain).
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
}

/**
 * Continuation passed to a provider's `runInNewSpan`. Providers must call it
 * exactly once, with their span's context, and return its result.
 *
 * @hidden
 */
export type InstrumentationNext<T> = (ctx: GenkitSpanContext) => Promise<T>;

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
 * A correlation-free log record. Each provider attaches trace/span correlation
 * from its own context source.
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

/** An id made entirely of zeros is the W3C/OTel no-op / unknown id. */
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
  (Object.keys(metadata) as (keyof SpanMetadata)[]).forEach((key) => {
    if (
      key === 'metadata' &&
      typeof metadata[key] === 'object' &&
      metadata.metadata
    ) {
      Object.entries(metadata.metadata).forEach(([metaKey, value]) => {
        out[ATTR_PREFIX + ':metadata:' + metaKey] = value as string;
      });
    } else if (
      key === 'input' ||
      key === 'init' ||
      typeof metadata[key] === 'object'
    ) {
      out[ATTR_PREFIX + ':' + key] = JSON.stringify(metadata[key]);
    } else {
      out[ATTR_PREFIX + ':' + key] = metadata[key] as string;
    }
  });
  return out;
}

/**
 * Normalizes custom attribute writes (`GenkitSpanContext.setMetadata`) into
 * valid attribute values. Primitives and homogeneous primitive arrays pass
 * through as-is; anything else is JSON-encoded. undefined/null are dropped.
 *
 * @hidden
 */
export function toSpanAttributes(
  values: Record<string, unknown>
): SpanAttributes {
  const attrs: SpanAttributes = {};
  for (const [k, v] of Object.entries(values)) {
    if (v === undefined || v === null) continue;
    attrs[k] = isAttributeValue(v) ? v : JSON.stringify(v);
  }
  return attrs;
}

function isPrimitiveAttribute(v: unknown): v is string | number | boolean {
  return (
    typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
  );
}

function isAttributeValue(v: unknown): v is SpanAttributeValue {
  if (isPrimitiveAttribute(v)) return true;
  if (!Array.isArray(v)) return false;
  const present = v.filter((x) => x !== null && x !== undefined);
  return (
    present.every(isPrimitiveAttribute) &&
    new Set(present.map((x) => typeof x)).size <= 1
  );
}

/** @hidden */
export function getErrorMessage(e: any): string {
  if (e instanceof Error) {
    return e.message;
  }
  return `${e}`;
}
