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

import {
  context,
  SpanOptions,
  SpanStatusCode,
  trace,
  type Span as ApiSpan,
} from '@opentelemetry/api';
import { logs, SeverityNumber } from '@opentelemetry/api-logs';
import {
  getErrorMessage,
  isDisableRootSpanDetection,
  metadataToAttributes,
  TRACER_NAME,
  TRACER_VERSION,
  type GenkitLogRecord,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationNext,
  type InstrumentationSpanInfo,
  type LogRecordingInstrumentation,
} from './instrumentation-api.js';

/**
 * The default, backwards-compatible instrumentation: exactly what
 * `runInNewSpan` did historically via `@opentelemetry/api`. Pure span creation;
 * spans only get exported if the user configured collection via
 * `enableTelemetry` (e.g. the GCP / Firebase plugins).
 *
 * When no collection is configured, `@opentelemetry/api`'s no-op tracer yields
 * all-zero ids; the composite treats those as empty.
 *
 * @hidden
 */
export class OTelInstrumentation
  implements Instrumentation, LogRecordingInstrumentation
{
  async runInNewSpan<T>(
    info: InstrumentationSpanInfo,
    next: InstrumentationNext<T>
  ): Promise<T> {
    const tracer = trace.getTracer(TRACER_NAME, TRACER_VERSION);
    const { metadata } = info;

    const spanOptions: SpanOptions = {
      links: info.links,
      attributes: info.labels,
    };
    if (!isDisableRootSpanDetection()) {
      spanOptions.root = metadata.isRoot;
    }

    return await tracer.startActiveSpan(
      metadata.name,
      spanOptions,
      async (otSpan: ApiSpan) => {
        const spanCtx: GenkitSpanContext = {
          get traceId() {
            return otSpan.spanContext().traceId;
          },
          get spanId() {
            return otSpan.spanContext().spanId;
          },
          setMetadata(values: Record<string, unknown>) {
            const attrs: Record<string, string> = {};
            for (const [k, v] of Object.entries(values)) {
              attrs[k] = typeof v === 'string' ? v : JSON.stringify(v);
            }
            otSpan.setAttributes(attrs);
          },
        };
        try {
          return await next(otSpan, spanCtx);
        } catch (e) {
          otSpan.setStatus({
            code: SpanStatusCode.ERROR,
            message: getErrorMessage(e),
          });
          if (e instanceof Error) {
            otSpan.recordException(e);
          }
          throw e;
        } finally {
          otSpan.setAttributes(metadataToAttributes(metadata));
          otSpan.end();
        }
      }
    );
  }

  /**
   * Logs via the OTel logs API, correlated through the active OTel context.
   * A no-op unless the user configured collection with a LoggerProvider. This
   * preserves historical behavior exactly.
   */
  recordLog(record: GenkitLogRecord): void {
    const otelLogger = logs.getLogger('genkit-logger');
    let activeContext;
    try {
      activeContext = context.active();
    } catch {
      // No-op if @opentelemetry/api is uninitialized right now.
    }
    otelLogger.emit({
      severityNumber: toSeverityNumber(record.severity),
      severityText: record.severity.toUpperCase(),
      body: record.body as any,
      attributes: (record.attributes as Record<string, any>) || {},
      ...(activeContext ? { context: activeContext } : {}),
    });
  }
}

function toSeverityNumber(level: string): SeverityNumber {
  switch (level) {
    case 'debug':
      return SeverityNumber.DEBUG;
    case 'info':
      return SeverityNumber.INFO;
    case 'warn':
      return SeverityNumber.WARN;
    case 'error':
      return SeverityNumber.ERROR;
    default:
      return SeverityNumber.UNSPECIFIED;
  }
}
