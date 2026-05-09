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

import { SpanKind } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';

export class TestSpanExporter implements SpanExporter {
  exportedSpans: any[] = [];

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): void {
    this.exportedSpans.push(...spans.map((s) => this._exportInfo(s)));
    resultCallback({ code: 0 });
  }

  shutdown(): Promise<void> {
    return this.forceFlush();
  }

  private _exportInfo(span: ReadableSpan) {
    return {
      spanId: span.spanContext().spanId,
      traceId: span.spanContext().traceId,
      attributes: { ...span.attributes },
      displayName: span.name,
      links: span.links,
      spanKind: SpanKind[span.kind],
      parentSpanId: span.parentSpanId,
      sameProcessAsParentSpan: { value: !span.spanContext().isRemote },
      status: span.status,
      timeEvents: {
        timeEvent: span.events.map((e) => ({
          annotation: {
            attributes: e.attributes ?? {},
            description: e.name,
          },
        })),
      },
    };
  }
  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}

export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
