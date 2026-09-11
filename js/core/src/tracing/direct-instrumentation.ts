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
  SpanStatusCode,
  trace,
  TraceFlags,
  type Span as ApiSpan,
} from '@opentelemetry/api';
import { randomBytes } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { getAsyncContext } from '../async-context.js';
import { logger } from '../logging.js';
import { postToTelemetryServer } from './exporter.js';
import {
  getErrorMessage,
  metadataToAttributes,
  type GenkitLogRecord,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationNext,
  type InstrumentationSpanInfo,
  type LogRecordingInstrumentation,
} from './instrumentation-api.js';
import type { SpanData, TraceData } from './types.js';

const directAlsKey = 'core.tracing.direct.span';
const INSTRUMENTATION_LIBRARY = { name: 'genkit-tracer', version: 'v1' };

interface DirectParent {
  traceId: string;
  spanId: string;
}

/**
 * A self-contained instrumentation that feeds the Developer UI directly, with
 * no OpenTelemetry SDK. It mints its own ids, tracks parentage via Genkit's
 * async context, builds the same `SpanData`/`TraceData` shape the OTel-backed
 * `TraceServerExporter` produced, and POSTs to `${server}/api/traces`.
 *
 * Auto-prepended in dev when a telemetry server is configured.
 *
 * @hidden
 */
export class DirectTelemetryInstrumentation
  implements Instrumentation, LogRecordingInstrumentation
{
  async runInNewSpan<T>(
    info: InstrumentationSpanInfo,
    next: InstrumentationNext<T>
  ): Promise<T> {
    const parent = getAsyncContext().getStore<DirectParent>(directAlsKey);
    const traceId = parent?.traceId ?? genId(16);
    const spanId = genId(8);
    const startTime = Date.now();
    const startPerf = performance.now();

    // Ids surfaced to the composite and to log correlation.
    const spanCtx: GenkitSpanContext = {
      traceId,
      spanId,
      setMetadata() {
        // Direct encodes from metadata at span end; nothing to do live.
      },
    };
    // Non-recording OTel span carrying our ids, for providers/callbacks that
    // read `.spanContext()`. Never booted through the SDK.
    const otSpan: ApiSpan = trace.wrapSpanContext({
      traceId,
      spanId,
      traceFlags: TraceFlags.SAMPLED,
    });

    const exceptions: Array<Record<string, string>> = [];
    let status: { code: number; message?: string } = {
      code: SpanStatusCode.UNSET,
    };

    const exportSpan = (final: boolean) => {
      // In-progress spans report endTime 0. This matches the old OTel realtime
      // path (an open span's endTime is [0,0], and hrTimeToMilliseconds([0,0])
      // is 0). The telemetry server and Dev UI treat a falsy endTime as "still
      // running"; a real endTime here would render the span as already complete
      // and suppress the pending state.
      const endTime = final ? startTime + (performance.now() - startPerf) : 0;
      const spanData = buildSpanData({
        info,
        traceId,
        spanId,
        parentSpanId: parent?.spanId,
        startTime,
        endTime,
        status,
        exceptions,
      });
      const traceData: TraceData = {
        traceId,
        spans: { [spanId]: spanData },
      };
      if (!spanData.parentSpanId) {
        traceData.displayName = spanData.displayName;
        traceData.startTime = spanData.startTime;
        // Leave the trace endTime unset until the root span finishes.
        if (final) {
          traceData.endTime = spanData.endTime;
        }
      }
      // Fire-and-forget; telemetry must never block or fail the operation.
      postToTelemetryServer('/api/traces', traceData).catch((e) =>
        logger.debug(`Failed to save trace ${traceId}`, e)
      );
    };

    // Real-time updates: export the in-progress span on start too, so the Dev
    // UI can show running traces. Gated on the same flag the OTel-backed
    // RealtimeSpanProcessor used; the on-end export below is unconditional.
    if (process.env.GENKIT_ENABLE_REALTIME_TELEMETRY === 'true') {
      exportSpan(false);
    }

    try {
      const output = await getAsyncContext().run(
        directAlsKey,
        { traceId, spanId } as DirectParent,
        () => next(otSpan, spanCtx)
      );
      exportSpan(true);
      return output;
    } catch (e) {
      status = { code: SpanStatusCode.ERROR, message: getErrorMessage(e) };
      if (e instanceof Error) {
        exceptions.push({
          'exception.type': e.name || 'Error',
          'exception.message': e.message || String(e),
          ...(e.stack ? { 'exception.stacktrace': e.stack } : {}),
        });
      }
      exportSpan(true);
      throw e;
    }
  }

  /**
   * Correlates via Genkit async context (the span the Dev UI actually shows),
   * lowers to the OTLP-log shape, and POSTs to `${server}/api/otlp`.
   */
  recordLog(record: GenkitLogRecord): void {
    const parent = getAsyncContext().getStore<DirectParent>(directAlsKey);
    const attributes: any[] = [];
    for (const [k, v] of Object.entries(record.attributes ?? {})) {
      if (typeof v === 'string')
        attributes.push({ key: k, value: { stringValue: v } });
      else if (typeof v === 'number')
        attributes.push({ key: k, value: { intValue: v } });
      else if (typeof v === 'boolean')
        attributes.push({ key: k, value: { boolValue: v } });
    }
    let bodyValue: Record<string, unknown>;
    if (typeof record.body === 'string')
      bodyValue = { stringValue: record.body };
    else if (typeof record.body === 'number')
      bodyValue = { intValue: record.body };
    else if (typeof record.body === 'boolean')
      bodyValue = { boolValue: record.body };
    else {
      // A circular body would make JSON.stringify throw; keep the log record
      // rather than dropping it.
      try {
        bodyValue = { stringValue: JSON.stringify(record.body) };
      } catch {
        bodyValue = { stringValue: '[unserializable log body]' };
      }
    }

    const payload = {
      resourceLogs: [
        {
          resource: { attributes: [], droppedAttributesCount: 0 },
          scopeLogs: [
            {
              scope: {
                name: INSTRUMENTATION_LIBRARY.name,
                version: INSTRUMENTATION_LIBRARY.version,
              },
              logRecords: [
                {
                  timeUnixNano: (Date.now() * 1_000_000).toString(),
                  severityNumber: toSeverityNumber(record.severity),
                  severityText: record.severity.toUpperCase(),
                  body: bodyValue,
                  attributes,
                  ...(parent
                    ? { traceId: parent.traceId, spanId: parent.spanId }
                    : {}),
                },
              ],
            },
          ],
        },
      ],
    };
    postToTelemetryServer('/api/otlp', payload).catch(() => {
      // Best-effort; log shipping must never throw on the caller thread.
    });
  }
}

interface BuildSpanDataArgs {
  info: InstrumentationSpanInfo;
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  startTime: number;
  endTime: number;
  status: { code: number; message?: string };
  exceptions: Array<Record<string, string>>;
}

/**
 * Reproduces the `SpanData` shape `TraceServerExporter._exportInfo` produced,
 * so before/after Dev UI traces are structurally identical.
 */
function buildSpanData(args: BuildSpanDataArgs): SpanData {
  const { info, exceptions } = args;
  const attributes: Record<string, any> = {
    ...info.labels,
    ...metadataToAttributes(info.metadata),
  };
  const spanData: SpanData = {
    spanId: args.spanId,
    traceId: args.traceId,
    startTime: args.startTime,
    endTime: args.endTime,
    attributes,
    displayName: info.metadata.name,
    links: (info.links ?? []) as any,
    instrumentationLibrary: { ...INSTRUMENTATION_LIBRARY },
    spanKind: 'INTERNAL',
    sameProcessAsParentSpan: { value: true },
    status: args.status.message
      ? { code: args.status.code, message: args.status.message }
      : { code: args.status.code },
    timeEvents: {
      timeEvent: exceptions.map((attrs) => ({
        time: args.endTime,
        annotation: { attributes: attrs, description: 'exception' },
      })),
    },
  };
  if (args.parentSpanId) {
    spanData.parentSpanId = args.parentSpanId;
  }
  return spanData;
}

function genId(bytes: number): string {
  return randomBytes(bytes).toString('hex');
}

function toSeverityNumber(level: string): number {
  switch (level) {
    case 'debug':
      return 5;
    case 'info':
      return 9;
    case 'warn':
      return 13;
    case 'error':
      return 17;
    default:
      return 0;
  }
}
