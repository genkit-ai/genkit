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
  SpanOptions,
  SpanStatusCode,
  trace,
  type Span as ApiSpan,
  type Link,
} from '@opentelemetry/api';
import { performance } from 'node:perf_hooks';
import { getAsyncContext } from '../async-context.js';
import type { HasRegistry, Registry } from '../registry.js';
import { ensureBasicTelemetryInstrumentation } from '../tracing.js';
import type { PathMetadata, SpanMetadata, TraceMetadata } from './types.js';

export const spanMetadataAlsKey = 'core.tracing.instrumentation.span';

export const ATTR_PREFIX = 'genkit';
/** @hidden */
export const SPAN_TYPE_ATTR = ATTR_PREFIX + ':type';
const TRACER_NAME = 'genkit-tracer';
const TRACER_VERSION = 'v1';

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
  otSpan: ApiSpan,
  isRoot: boolean
) => Promise<T>;

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
  const tracer = trace.getTracer(TRACER_NAME, TRACER_VERSION);
  const parentStep =
    getAsyncContext().getStore<SpanContext>(spanMetadataAlsKey);
  const isInRoot = parentStep?.metadata?.isRoot === true;
  if (!parentStep) opts.metadata.isRoot ||= true;

  const spanOptions: SpanOptions = {
    links: opts.links,
    attributes: opts.labels,
  };
  if (!isDisableRootSpanDetection()) {
    spanOptions.root = opts.metadata.isRoot;
  }

  return await tracer.startActiveSpan(
    opts.metadata.name,
    spanOptions,
    async (otSpan) => {
      const spanContext = {
        ...parentStep,
        metadata: opts.metadata,
        labels: opts.labels,
      } as SpanContext;
      try {
        opts.metadata.path = buildPath(
          opts.metadata.name,
          parentStep?.metadata?.path || '',
          opts.labels
        );

        const isGenkitSpan = !!opts.labels?.[SPAN_TYPE_ATTR];
        if (isGenkitSpan && parentStep) {
          const parentIsGenkit = !!parentStep.labels?.[SPAN_TYPE_ATTR];
          if (!parentIsGenkit && parentStep.spanId) {
            otSpan.setAttribute(
              ATTR_PREFIX + ':lastKnownParentSpanId',
              parentStep.spanId
            );
          }
        }

        // Store spanId in context for Genkit spans so nested spans can reference it
        if (isGenkitSpan) {
          spanContext.spanId = otSpan.spanContext().spanId;
        }

        const output = await getAsyncContext().run(
          spanMetadataAlsKey,
          spanContext,
          () => fn(opts.metadata, otSpan, isInRoot)
        );
        if (opts.metadata.state !== 'error') {
          opts.metadata.state = 'success';
        }

        recordPath(opts.metadata, spanContext);
        return output;
      } catch (e) {
        recordPath(opts.metadata, spanContext, e);
        opts.metadata.state = 'error';
        otSpan.setStatus({
          code: SpanStatusCode.ERROR,
          message: getErrorMessage(e),
        });
        if (e instanceof Error) {
          otSpan.recordException(e);
        }

        // Mark the first failing span as the source of failure. Prevent parent
        // spans that catch re-thrown exceptions from also claiming to be the
        // source.
        if (typeof e === 'object') {
          if (!(e as any).ignoreFailedSpan) {
            opts.metadata.isFailureSource = true;
          }
          (e as any).ignoreFailedSpan = true;
        }

        throw e;
      } finally {
        otSpan.setAttributes(metadataToAttributes(opts.metadata));
        otSpan.end();
      }
    }
  );
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

function getErrorMessage(e: any): string {
  if (e instanceof Error) {
    return e.message;
  }
  return `${e}`;
}

function metadataToAttributes(metadata: SpanMetadata): Record<string, string> {
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
    } else if (key === 'input' || typeof metadata[key] === 'object') {
      out[ATTR_PREFIX + ':' + key] = JSON.stringify(metadata[key]);
    } else {
      out[ATTR_PREFIX + ':' + key] = metadata[key];
    }
  });
  return out;
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

const rootSpanDetectionKey = '__genkit_disableRootSpanDetection';

function isDisableRootSpanDetection(): boolean {
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
