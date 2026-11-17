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

// ai tests import this file, which pulls core sources into their compilation;
// those sources rely on core's global declarations.
/// <reference path="../typings.d.ts" />

import { randomBytes } from 'node:crypto';
import { getAsyncContext } from '../src/async-context.js';
import {
  metadataToAttributes,
  toSpanAttributes,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationNext,
  type InstrumentationSpanInfo,
  type SpanAttributes,
} from '../src/tracing/instrumentation-api.js';

export interface RecordedSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  displayName: string;
  attributes: Record<string, unknown>;
  status: 'ok' | 'error';
}

const parentAlsKey = 'core.tests.recordingInstrumentation.span';

/**
 * In-memory instrumentation for tests. Records spans when they finish (so
 * children come before parents), with the same attribute encoding as the
 * built-in providers: labels, then custom writes, then `genkit:*` metadata.
 *
 * Usage: `configureInstrumentation(recorder)`.
 */
export class TestInstrumentation implements Instrumentation {
  exportedSpans: RecordedSpan[] = [];

  async runInNewSpan<T>(
    info: InstrumentationSpanInfo,
    next: InstrumentationNext<T>
  ): Promise<T> {
    const parent = info.metadata.isRoot
      ? undefined
      : getAsyncContext().getStore<RecordedSpan>(parentAlsKey);
    const span: RecordedSpan = {
      traceId: parent?.traceId ?? randomBytes(16).toString('hex'),
      spanId: randomBytes(8).toString('hex'),
      parentSpanId: parent?.spanId,
      displayName: info.metadata.name,
      attributes: {},
      status: 'ok',
    };
    const custom: SpanAttributes = {};
    const ctx: GenkitSpanContext = {
      traceId: span.traceId,
      spanId: span.spanId,
      setMetadata: (values) => Object.assign(custom, toSpanAttributes(values)),
    };
    try {
      return await getAsyncContext().run(parentAlsKey, span, () => next(ctx));
    } catch (e) {
      span.status = 'error';
      throw e;
    } finally {
      // Drop undefined values (e.g. `genkit:input` for a no-input action), as
      // OTel and the Dev UI's JSON wire format both do.
      span.attributes = Object.fromEntries(
        Object.entries({
          ...info.labels,
          ...custom,
          ...metadataToAttributes(info.metadata),
        }).filter(([, v]) => v !== undefined)
      );
      this.exportedSpans.push(span);
    }
  }
}

export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
