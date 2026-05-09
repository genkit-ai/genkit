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

import { context, SpanKind, type HrTime } from '@opentelemetry/api';
import {
  ExportResultCode,
  hrTimeToMilliseconds,
  suppressTracing,
  type ExportResult,
} from '@opentelemetry/core';
import type {
  LogRecordExporter,
  ReadableLogRecord,
} from '@opentelemetry/sdk-logs';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { logger } from '../logging.js';
import { deleteUndefinedProps } from '../utils.js';
import type { SpanData, TraceData } from './types.js';

export let telemetryServerUrl: string | undefined;

/**
 * @hidden
 */
export function setTelemetryServerUrl(url: string) {
  telemetryServerUrl = url;
}

/**
 * Exports collected OpenTelemetetry spans to the telemetry server.
 */
export class TraceServerExporter implements SpanExporter {
  /**
   * Export spans.
   * @param spans
   * @param resultCallback
   */
  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): void {
    this._sendSpans(spans, resultCallback);
  }

  /**
   * Shutdown the exporter.
   */
  shutdown(): Promise<void> {
    this._sendSpans([]);
    return this.forceFlush();
  }

  /**
   * Converts span info into trace store format.
   * @param span
   */
  private _exportInfo(span: ReadableSpan): SpanData {
    const spanData: Partial<SpanData> = {
      spanId: span.spanContext().spanId,
      traceId: span.spanContext().traceId,
      startTime: transformTime(span.startTime),
      endTime: transformTime(span.endTime),
      attributes: { ...span.attributes },
      displayName: span.name,
      links: span.links,
      spanKind: SpanKind[span.kind],
      parentSpanId: span.parentSpanId,
      sameProcessAsParentSpan: { value: !span.spanContext().isRemote },
      status: span.status,
      timeEvents: {
        timeEvent: span.events.map((e) => ({
          time: transformTime(e.time),
          annotation: {
            attributes: e.attributes ?? {},
            description: e.name,
          },
        })),
      },
    };
    if (span.instrumentationLibrary !== undefined) {
      spanData.instrumentationLibrary = {
        name: span.instrumentationLibrary.name,
      };
      if (span.instrumentationLibrary.schemaUrl !== undefined) {
        spanData.instrumentationLibrary.schemaUrl =
          span.instrumentationLibrary.schemaUrl;
      }
      if (span.instrumentationLibrary.version !== undefined) {
        spanData.instrumentationLibrary.version =
          span.instrumentationLibrary.version;
      }
    }
    deleteUndefinedProps(spanData);
    return spanData as SpanData;
  }

  /**
   * Exports any pending spans in exporter
   */
  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  private async _sendSpans(
    spans: ReadableSpan[],
    done?: (result: ExportResult) => void
  ): Promise<void> {
    const traces = {} as Record<string, ReadableSpan[]>;
    for (const span of spans) {
      if (!traces[span.spanContext().traceId]) {
        traces[span.spanContext().traceId] = [];
      }
      traces[span.spanContext().traceId].push(span);
    }
    let error = false;
    for (const traceId of Object.keys(traces)) {
      try {
        await this.save(traceId, traces[traceId]);
      } catch (e) {
        error = true;
        logger.error(`Failed to save trace ${traceId}`, e);
      }
      if (done) {
        return done({
          code: error ? ExportResultCode.FAILED : ExportResultCode.SUCCESS,
        });
      }
    }
  }

  private async save(traceId, spans: ReadableSpan[]): Promise<void> {
    if (!telemetryServerUrl) {
      logger.debug(
        `Telemetry server is not configured, trace ${traceId} not saved!`
      );
      return;
    }
    // TODO: add interface for Firestore doc
    const data = {
      traceId,
      spans: {},
    } as TraceData;
    for (const span of spans) {
      const convertedSpan = this._exportInfo(span);
      data.spans[convertedSpan.spanId] = convertedSpan;
      if (!convertedSpan.parentSpanId) {
        data.displayName = convertedSpan.displayName;
        data.startTime = convertedSpan.startTime;
        data.endTime = convertedSpan.endTime;
      }
    }
    // Suppress tracing to prevent infinite loops when auto-instrumentation
    // (e.g., undici) is enabled. Without this, the fetch call would be traced,
    // creating new spans that trigger more exports, causing stack overflow.
    await context.with(suppressTracing(context.active()), () =>
      fetch(`${telemetryServerUrl}/api/traces`, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })
    );
  }
}

// Converts an HrTime to milliseconds.
function transformTime(time: HrTime) {
  return hrTimeToMilliseconds(time);
}

/**
 * Exports collected OpenTelemetetry logs to the telemetry server.
 */
export class LogServerExporter implements LogRecordExporter {
  export(
    logs: ReadableLogRecord[],
    resultCallback: (result: ExportResult) => void
  ): void {
    this._sendLogs(logs, resultCallback);
  }

  shutdown(): Promise<void> {
    return this.forceFlush();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  private async _sendLogs(
    logs: ReadableLogRecord[],
    done?: (result: ExportResult) => void
  ): Promise<void> {
    if (!telemetryServerUrl) {
      if (done) done({ code: ExportResultCode.SUCCESS });
      return;
    }

    try {
      const scopeLogsMap = new Map<string, any>();
      for (const log of logs) {
        const scopeName = log.instrumentationScope.name || 'unknown';
        if (!scopeLogsMap.has(scopeName)) {
          scopeLogsMap.set(scopeName, {
            scope: {
              name: scopeName,
              version: log.instrumentationScope.version || '',
            },
            logRecords: [],
          });
        }

        // TODO: Handle more complex types (arrays, maps, etc.).
        // See https://opentelemetry.io/docs/specs/otel/common/#anyvalue.
        const attributes: any[] = [];
        for (const [k, v] of Object.entries(log.attributes)) {
          if (typeof v === 'string')
            attributes.push({ key: k, value: { stringValue: v } });
          else if (typeof v === 'number')
            attributes.push({ key: k, value: { intValue: v } });
          else if (typeof v === 'boolean')
            attributes.push({ key: k, value: { boolValue: v } });
        }

        let bodyValue;
        if (typeof log.body === 'string') bodyValue = { stringValue: log.body };
        else if (typeof log.body === 'number')
          bodyValue = { intValue: log.body };
        else if (typeof log.body === 'boolean')
          bodyValue = { boolValue: log.body };
        else bodyValue = { stringValue: JSON.stringify(log.body) };

        scopeLogsMap.get(scopeName).logRecords.push({
          timeUnixNano: (
            hrTimeToMilliseconds(log.hrTime) * 1_000_000
          ).toString(),
          severityNumber: log.severityNumber,
          severityText: log.severityText,
          body: bodyValue,
          attributes,
          traceId: log.spanContext?.traceId,
          spanId: log.spanContext?.spanId,
        });
      }

      const payload = {
        resourceLogs: [
          {
            resource: { attributes: [], droppedAttributesCount: 0 },
            scopeLogs: Array.from(scopeLogsMap.values()),
          },
        ],
      };

      await context.with(suppressTracing(context.active()), () =>
        fetch(`${telemetryServerUrl}/api/otlp`, {
          method: 'POST',
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        })
      );
      if (done) done({ code: ExportResultCode.SUCCESS });
    } catch (e) {
      logger.error('Failed to export logs', e);
      if (done) done({ code: ExportResultCode.FAILED });
    }
  }
}
