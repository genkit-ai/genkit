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

import {
  ROOT_CONTEXT,
  trace,
  TraceFlags,
  type Span as ApiSpan,
  type Link,
} from '@opentelemetry/api';
import { performance } from 'node:perf_hooks';
import { getAsyncContext } from '../async-context.js';
import type { HasRegistry, Registry } from '../registry.js';
import { ensureBasicTelemetryInstrumentation } from '../tracing.js';
import { DirectTelemetryInstrumentation } from './direct-instrumentation.js';
import { telemetryServerUrl } from './exporter.js';
import {
  ATTR_PREFIX,
  firstNonEmpty,
  metadataToAttributes,
  SPAN_TYPE_ATTR,
  spanMetadataAlsKey,
  TRACER_NAME,
  TRACER_VERSION,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationSpanInfo,
} from './instrumentation-api.js';
import { OTelInstrumentation } from './otel-instrumentation.js';
import type { PathMetadata, SpanMetadata, TraceMetadata } from './types.js';

export { DirectTelemetryInstrumentation } from './direct-instrumentation.js';
export {
  ATTR_PREFIX,
  disableOTelRootSpanDetection,
  SPAN_TYPE_ATTR,
  spanMetadataAlsKey,
  type DisposableInstrumentation,
  type GenkitLogRecord,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationSpanInfo,
  type LogRecordingInstrumentation,
} from './instrumentation-api.js';
export { OTelInstrumentation } from './otel-instrumentation.js';

type SpanContext = {
  metadata: SpanMetadata;
  labels?: Record<string, string>;
  spanId?: string;
} & TraceMetadata;

interface RunInNewSpanOpts {
  metadata: SpanMetadata;
  labels?: Record<string, string>;
  links?: Link[];
}

type RunInNewSpanFn<T> = (
  metadata: SpanMetadata,
  otSpan: ApiSpan
) => Promise<T>;

// ---------------------------------------------------------------------------
// Provider registry (global-keyed so it survives module duplication).
// ---------------------------------------------------------------------------

const configuredInstrumentationKey = '__GENKIT_CONFIGURED_INSTRUMENTATION';
const directInstrumentationKey = '__GENKIT_DIRECT_INSTRUMENTATION';
const defaultOTelInstrumentationKey = '__GENKIT_DEFAULT_OTEL_INSTRUMENTATION';

/**
 * Replaces the implicit default (`OTelInstrumentation`) with the provided
 * instrumentation. `DirectTelemetryInstrumentation` is still prepended in dev
 * (when a telemetry server is configured), so a typical dev chain becomes
 * `[Direct, x]`.
 *
 * @hidden
 */
export function configureInstrumentation(i: Instrumentation) {
  global[configuredInstrumentationKey] = i;
}

/**
 * Clears configured/auto-injected instrumentation. Tests only.
 *
 * @hidden
 */
export function resetInstrumentation() {
  delete global[configuredInstrumentationKey];
  delete global[directInstrumentationKey];
  delete global[defaultOTelInstrumentationKey];
}

function defaultOTelInstrumentation(): OTelInstrumentation {
  if (!global[defaultOTelInstrumentationKey]) {
    global[defaultOTelInstrumentationKey] = new OTelInstrumentation();
  }
  return global[defaultOTelInstrumentationKey];
}

/**
 * Lazily creates the dev instrumentation. Imported dynamically only when a
 * telemetry server is configured so the Direct sink stays out of prod paths.
 */
function directInstrumentation(): Instrumentation | undefined {
  const serverConfigured =
    !!telemetryServerUrl || !!process.env.GENKIT_TELEMETRY_SERVER;
  if (!serverConfigured) return undefined;
  if (!global[directInstrumentationKey]) {
    global[directInstrumentationKey] = new DirectTelemetryInstrumentation();
  }
  return global[directInstrumentationKey];
}

/**
 * Resolves the active instrumentation chain for this call.
 * - base: `[configured]` if `configureInstrumentation` was called, else the
 *   implicit `[OTelInstrumentation]` default.
 * - dev: `DirectTelemetryInstrumentation` is prepended when a telemetry server
 *   is configured, so ids from Direct win and the Dev UI is fed without OTel.
 *
 * @hidden
 */
export function activeInstrumentations(): Instrumentation[] {
  const base: Instrumentation =
    global[configuredInstrumentationKey] ?? defaultOTelInstrumentation();
  const direct = directInstrumentation();
  return direct ? [direct, base] : [base];
}

/** @hidden */
export function isInstrumentedBy(
  ctor: new (...args: any[]) => Instrumentation
): boolean {
  return activeInstrumentations().some((i) => i instanceof ctor);
}

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------

/**
 * Runs the provided function in a new span.
 * @deprecated
 * @hidden
 */
export async function runInNewSpan<T>(
  registry: Registry | HasRegistry,
  opts: RunInNewSpanOpts,
  fn: RunInNewSpanFn<T>
): Promise<T>;

/**
 * Runs the provided function in a new span.
 * @hidden
 */
export async function runInNewSpan<T>(
  opts: RunInNewSpanOpts,
  fn: RunInNewSpanFn<T>
): Promise<T>;

/**
 * Runs the provided function in a new span.
 * @hidden
 */
export async function runInNewSpan<T>(
  registryOrOprs: Registry | HasRegistry | RunInNewSpanOpts,
  optsOrFn: RunInNewSpanOpts | RunInNewSpanFn<T>,
  fnMaybe?: RunInNewSpanFn<T>
): Promise<T> {
  let opts: RunInNewSpanOpts;
  let fn: RunInNewSpanFn<T>;
  if (arguments.length === 3) {
    opts = optsOrFn as RunInNewSpanOpts;
    fn = fnMaybe as RunInNewSpanFn<T>;
  } else {
    opts = registryOrOprs as RunInNewSpanOpts;
    fn = optsOrFn as RunInNewSpanFn<T>;
  }
  await ensureBasicTelemetryInstrumentation();

  const parentStep =
    getAsyncContext().getStore<SpanContext>(spanMetadataAlsKey);
  if (!parentStep) opts.metadata.isRoot ||= true;

  // Genkit-semantic setup (backend independent). Providers only encode the
  // metadata/labels we build here.
  opts.metadata.path = buildPath(
    opts.metadata.name,
    parentStep?.metadata?.path || '',
    opts.labels
  );

  const isGenkitSpan = !!opts.labels?.[SPAN_TYPE_ATTR];
  const labels = { ...opts.labels };
  if (isGenkitSpan && parentStep) {
    const parentIsGenkit = !!parentStep.labels?.[SPAN_TYPE_ATTR];
    if (!parentIsGenkit && parentStep.spanId) {
      // Stitch orphaned genkit spans back to their nearest genkit ancestor.
      // Encoded as a label so every provider records it identically.
      labels[ATTR_PREFIX + ':lastKnownParentSpanId'] = parentStep.spanId;
    }
  }

  const spanContext = {
    ...parentStep,
    metadata: opts.metadata,
    labels,
  } as SpanContext;

  const info: InstrumentationSpanInfo = {
    metadata: opts.metadata,
    labels,
    links: opts.links,
  };

  // Innermost callback: every provider span/context is collected by now, so we
  // resolve the composite ids, seed the ALS store for children, run `fn`, and
  // set success/error state before the providers finalize their spans.
  const runBody = async (contexts: GenkitSpanContext[]): Promise<T> => {
    const compositeCtx = makeCompositeContext(contexts);
    if (isGenkitSpan) {
      spanContext.spanId = compositeCtx.spanId || undefined;
    }
    const compositeSpan = makeCompositeSpan(compositeCtx);

    return await getAsyncContext().run(spanMetadataAlsKey, spanContext, () =>
      runWithGenkitState(opts.metadata, spanContext, compositeSpan, fn)
    );
  };

  return await dispatch(activeInstrumentations(), info, runBody);
}

/**
 * Nests each provider's `runInNewSpan` (outermost first), collecting the
 * context each hands to `next`, and invokes `runBody` at the center.
 */
async function dispatch<T>(
  chain: Instrumentation[],
  info: InstrumentationSpanInfo,
  runBody: (contexts: GenkitSpanContext[]) => Promise<T>
): Promise<T> {
  const contexts: GenkitSpanContext[] = [];

  const at =
    (i: number) =>
    async (_span: ApiSpan, ctx: GenkitSpanContext): Promise<T> => {
      contexts[i] = ctx;
      if (i + 1 < chain.length) {
        return chain[i + 1].runInNewSpan(info, at(i + 1));
      }
      return runBody(contexts);
    };

  return chain[0].runInNewSpan(info, at(0));
}

/**
 * Genkit success/error bookkeeping that used to live inline in `runInNewSpan`.
 * Runs around `fn`; providers see the rethrown error and finalize afterwards.
 */
async function runWithGenkitState<T>(
  metadata: SpanMetadata,
  spanContext: SpanContext,
  compositeSpan: ApiSpan,
  fn: RunInNewSpanFn<T>
): Promise<T> {
  try {
    const output = await fn(metadata, compositeSpan);
    if (metadata.state !== 'error') {
      metadata.state = 'success';
    }
    recordPath(metadata, spanContext);
    return output;
  } catch (e) {
    recordPath(metadata, spanContext, e);
    metadata.state = 'error';
    // Mark the first failing span as the source of failure. Prevent parent
    // spans that catch re-thrown exceptions from also claiming to be the
    // source.
    if (typeof e === 'object' && e !== null) {
      if (!(e as any).ignoreFailedSpan) {
        metadata.isFailureSource = true;
      }
      (e as any).ignoreFailedSpan = true;
    }
    throw e;
  }
}

/**
 * The span handed to `fn`. `spanContext()` returns the composite-resolved ids;
 * `fn` callers only ever read those. Attribute writes fan out to providers via
 * the composite context.
 */
function makeCompositeSpan(ctx: GenkitSpanContext): ApiSpan {
  // Non-recording wrapper around the composite ids. Type-honest (real ApiSpan),
  // and does not boot the OTel SDK.
  return trace.wrapSpanContext({
    traceId: ctx.traceId || '0'.repeat(32),
    spanId: ctx.spanId || '0'.repeat(16),
    traceFlags: TraceFlags.SAMPLED,
  });
}

function makeCompositeContext(
  contexts: GenkitSpanContext[]
): GenkitSpanContext {
  // A third-party provider may call next() without a ctx, so guard each access.
  return {
    get traceId() {
      return firstNonEmpty(contexts.map((c) => c?.traceId));
    },
    get spanId() {
      return firstNonEmpty(contexts.map((c) => c?.spanId));
    },
    setMetadata(values: Record<string, unknown>) {
      for (const c of contexts) c?.setMetadata(values);
    },
  };
}

/**
 * Creates a new child span and attaches it to a previously created trace. This
 * is useful, for example, for adding deferred user engagement metadata.
 *
 * @hidden
 */
export async function appendSpan(
  traceId: string,
  parentSpanId: string,
  metadata: SpanMetadata,
  labels?: Record<string, string>
) {
  await ensureBasicTelemetryInstrumentation();

  const tracer = trace.getTracer(TRACER_NAME, TRACER_VERSION);

  const spanContext = trace.setSpanContext(ROOT_CONTEXT, {
    traceId: traceId,
    traceFlags: 1, // sampled
    spanId: parentSpanId,
  });

  // TODO(abrook): add explicit start time to align with parent
  const span = tracer.startSpan(metadata.name, {}, spanContext);
  span.setAttributes(metadataToAttributes(metadata));
  if (labels) {
    span.setAttributes(labels);
  }
  span.end();
}

/**
 * Sets provided attribute value in the current span.
 *
 * @hidden
 */
export function setCustomMetadataAttribute(key: string, value: string) {
  const currentStep = getCurrentSpan();
  if (!currentStep) {
    return;
  }
  if (!currentStep.metadata) {
    currentStep.metadata = {};
  }
  currentStep.metadata[key] = value;
}

/**
 * Sets provided attribute values in the current span.
 *
 * @hidden
 */
export function setCustomMetadataAttributes(values: Record<string, string>) {
  const currentStep = getCurrentSpan();
  if (!currentStep) {
    return;
  }
  if (!currentStep.metadata) {
    currentStep.metadata = {};
  }
  for (const [key, value] of Object.entries(values)) {
    currentStep.metadata[key] = value;
  }
}

/**
 * Converts a fully annotated path to a friendly display version for logs
 *
 * @hidden
 */
export function toDisplayPath(path: string): string {
  const pathPartRegex = /\{([^\,}]+),[^\}]+\}/g;
  return Array.from(path.matchAll(pathPartRegex), (m) => m[1]).join(' > ');
}

function getCurrentSpan(): SpanMetadata {
  const step = getAsyncContext().getStore<SpanContext>(spanMetadataAlsKey);
  if (!step) {
    throw new Error('running outside step context');
  }
  return step.metadata;
}

function buildPath(
  name: string,
  parentPath: string,
  labels?: Record<string, string>
) {
  const stepType =
    labels && labels['genkit:type']
      ? `,t:${labels['genkit:metadata:subtype'] === 'flow' ? 'flow' : labels['genkit:type']}`
      : '';
  return parentPath + `/{${name}${stepType}}`;
}

function recordPath(
  spanMeta: SpanMetadata,
  spanContext: SpanContext,
  err?: any
) {
  const path = spanMeta.path || '';
  const decoratedPath = decoratePathWithSubtype(spanMeta);
  // Only add the path if a child has not already been added. In the event that
  // an error is rethrown, we don't want to add each step in the unwind.
  const paths = Array.from(spanContext?.paths || new Set<PathMetadata>());
  const status = err ? 'failure' : 'success';
  if (!paths.some((p) => p.path.startsWith(path) && p.status === status)) {
    const now = performance.now();
    const start = spanContext?.timestamp || now;
    spanContext?.paths?.add({
      path: decoratedPath,
      error: err?.name,
      latency: now - start,
      status,
    });
  }
  spanMeta.path = decoratedPath;
}

function decoratePathWithSubtype(metadata: SpanMetadata): string {
  if (!metadata.path) {
    return '';
  }

  const pathComponents = metadata.path.split('}/{');

  if (pathComponents.length == 1) {
    return metadata.path;
  }

  const stepSubtype =
    metadata.metadata && metadata.metadata['subtype']
      ? `,s:${metadata.metadata['subtype']}`
      : '';
  const root = `${pathComponents.slice(0, -1).join('}/{')}}/`;
  const decoratedStep = `{${pathComponents.at(-1)?.slice(0, -1)}${stepSubtype}}`;
  return root + decoratedStep;
}
