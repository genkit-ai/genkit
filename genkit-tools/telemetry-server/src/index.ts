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

import type { LogStore } from '@genkit-ai/tools-common';
import {
  TraceDataSchema,
  TraceQueryFilterSchema,
  type SpanData,
} from '@genkit-ai/tools-common';
import { logger } from '@genkit-ai/tools-common/utils';
import cors from 'cors';
import express from 'express';
import type * as http from 'http';
import { BroadcastManager } from './broadcast-manager.js';
import type { TraceStore } from './types.js';
import { logDataFromOtlp, traceDataFromOtlp } from './utils/otlp.js';

export { LocalFileLogStore } from './file-log-store.js';
export { LocalFileTraceStore } from './file-trace-store.js';
export { TraceQuerySchema, type TraceQuery, type TraceStore } from './types';

let server: http.Server;
const broadcastManager = new BroadcastManager();

/** A live-trace event streamed to dev UI clients over SSE. */
export interface SpanBroadcastEvent {
  type: 'span_start' | 'span_end';
  traceId: string;
  span: SpanData;
}

/**
 * spanBroadcastEvents decides which span_start / span_end events to broadcast
 * for an incoming /api/traces save, given the trace's merged (post-save) spans.
 *
 * Live traces export each span twice — once as it starts (no endTime) and again
 * as it ends — and the senders do not guarantee start-before-end delivery. A
 * stale, still-in-progress "start" for a span the store already has as ended is
 * dropped, so it cannot re-open a finished span in the live view (which would
 * leave it spinning forever); this mirrors the merge in the file trace store
 * that already ignores such a start on disk. Events are returned in
 * chronological order, with span_start before span_end when times tie.
 */
export function spanBroadcastEvents(
  traceId: string,
  incomingSpans: SpanData[],
  mergedSpans: Record<string, SpanData> = {}
): SpanBroadcastEvent[] {
  const events: SpanBroadcastEvent[] = [];
  for (const span of incomingSpans) {
    const incomingEnded = span.endTime > 0;
    const mergedSpan = mergedSpans[span.spanId];
    const alreadyEnded = !!mergedSpan && mergedSpan.endTime > 0;
    if (!incomingEnded && alreadyEnded) {
      continue;
    }
    events.push({ type: 'span_start', traceId, span });
    if (incomingEnded) {
      events.push({ type: 'span_end', traceId, span });
    }
  }
  events.sort((a, b) => {
    const aTime = a.type === 'span_start' ? a.span.startTime : a.span.endTime;
    const bTime = b.type === 'span_start' ? b.span.startTime : b.span.endTime;
    if (aTime !== bTime) {
      return aTime - bTime;
    }
    return a.type === 'span_start' ? -1 : 1;
  });
  return events;
}

/**
 * Starts the telemetry server with the provided params
 */
export async function startTelemetryServer(params: {
  port: number;
  traceStore: TraceStore;
  logStore: LogStore;
  /**
   * Controls the maximum request body size. If this is a number,
   * then the value specifies the number of bytes; if it is a string,
   * the value is passed to the bytes library for parsing.
   *
   * Defaults to '5mb'.
   */
  maxRequestBodySize?: string | number;
  corsOrigin?: string | RegExp;
  /**
   * The network interface to bind to. Defaults to '127.0.0.1' so the
   * telemetry server is only reachable from the local machine. This is a
   * developer-facing tool with no authentication, so it should not be exposed
   * to other hosts by default.
   */
  host?: string;
}) {
  await params.traceStore.init();
  await params.logStore.init();

  const api = express();
  // Allow all origins and expose trace ID header
  api.use(
    cors({
      // By default, allow connections from localhost on any port.
      origin: params.corsOrigin || /^http:\/\/localhost:\d+$/,
      allowedHeaders: ['Content-Type'],
      exposedHeaders: ['X-Genkit-Trace-Id'],
    })
  );

  api.use(express.json({ limit: params.maxRequestBodySize ?? '100mb' }));

  api.get('/api/__health', async (_, response) => {
    response.status(200).send('OK');
  });

  api.get('/api/traces/:traceId', async (request, response, next) => {
    try {
      const { traceId } = request.params;
      response.json(await params.traceStore.load(traceId));
    } catch (e) {
      next(e);
    }
  });

  // SSE endpoint for live trace streaming
  api.get('/api/traces/:traceId/stream', async (request, response, next) => {
    try {
      const { traceId } = request.params;

      // Set SSE headers
      response.setHeader('Content-Type', 'text/event-stream');
      response.setHeader('Cache-Control', 'no-cache');
      response.setHeader('Connection', 'keep-alive');
      response.setHeader('Access-Control-Allow-Origin', '*');
      response.setHeader('Access-Control-Allow-Headers', 'Content-Type');

      // Send initial snapshot of current trace data
      const currentTrace = await params.traceStore.load(traceId);
      if (currentTrace) {
        const snapshot = JSON.stringify(currentTrace);
        response.write(`data: ${snapshot}\n\n`);
      }

      // Register this connection for broadcasts
      broadcastManager.subscribe(traceId, response);

      // Clean up on disconnect
      response.on('close', () => {
        broadcastManager.unsubscribe(traceId, response);
      });
    } catch (e) {
      next(e);
    }
  });

  api.post('/api/traces', async (request, response, next) => {
    try {
      const traceData = TraceDataSchema.parse(request.body);
      await params.traceStore.save(traceData.traceId, traceData);

      // Decide what to broadcast against the merged store so a stale in-progress
      // "start" cannot re-open an already-finished span in the live view (see
      // spanBroadcastEvents). Only an in-progress span can be a stale start, so
      // the extra read is skipped for the common all-ended save (e.g. the
      // non-realtime exporter, which sends each span only once, on end).
      const spans = Object.values(traceData.spans);
      const hasInProgress = spans.some((s) => !(s.endTime > 0));
      const merged = hasInProgress
        ? await params.traceStore.load(traceData.traceId)
        : undefined;
      const events = spanBroadcastEvents(
        traceData.traceId,
        spans,
        merged?.spans
      );
      for (const event of events) {
        broadcastManager.broadcast(traceData.traceId, event);
      }

      response.status(200).send('OK');
    } catch (e) {
      next(e);
    }
  });

  api.get('/api/traces', async (request, response, next) => {
    try {
      const { limit, continuationToken, filter } = request.query;
      response.json(
        await params.traceStore.list({
          limit: limit ? Number.parseInt(limit.toString()) : 10,
          continuationToken: continuationToken
            ? continuationToken.toString()
            : undefined,
          filter: filter
            ? TraceQueryFilterSchema.parse(JSON.parse(filter as string))
            : undefined,
        })
      );
    } catch (e) {
      next(e);
    }
  });

  api.get('/api/logs', async (request, response, next) => {
    try {
      const { limit, continuationToken } = request.query;
      response.json(
        await params.logStore.list({
          limit: limit ? Number.parseInt(limit.toString()) : 100,
          continuationToken: continuationToken
            ? continuationToken.toString()
            : undefined,
        })
      );
    } catch (e) {
      next(e);
    }
  });

  api.get('/api/traces/:traceId/logs', async (request, response, next) => {
    try {
      const { limit, continuationToken } = request.query;
      const { traceId } = request.params;
      response.json(
        await params.logStore.list({
          limit: limit ? Number.parseInt(limit.toString()) : 100,
          continuationToken: continuationToken
            ? continuationToken.toString()
            : undefined,
          traceId,
        })
      );
    } catch (e) {
      next(e);
    }
  });

  api.get(
    '/api/traces/:traceId/spans/:spanId/logs',
    async (request, response, next) => {
      try {
        const { limit, continuationToken } = request.query;
        const { traceId, spanId } = request.params;
        response.json(
          await params.logStore.list({
            limit: limit ? Number.parseInt(limit.toString()) : 100,
            continuationToken: continuationToken
              ? continuationToken.toString()
              : undefined,
            traceId,
            spanId,
          })
        );
      } catch (e) {
        next(e);
      }
    }
  );

  api.post(
    [
      '/api/otlp',
      '/api/otlp/v1/traces',
      '/api/otlp/v1/logs',
      '/api/otlp/v1/metrics',
    ],
    async (request, response) => {
      try {
        if (
          !request.body.resourceSpans?.length &&
          !request.body.resourceLogs?.length
        ) {
          // Acknowledge and ignore empty payloads.
          response.status(200).json({});
          return;
        }
        const traces = traceDataFromOtlp(request.body);
        for (const trace of traces) {
          const traceData = TraceDataSchema.parse(trace);
          await params.traceStore.save(traceData.traceId, traceData);

          // Convert each span to an event and broadcast individually
          for (const [_, span] of Object.entries(traceData.spans)) {
            const event: {
              type: 'span_start' | 'span_end';
              traceId: string;
              span: SpanData;
            } = {
              type: span.endTime > 0 ? 'span_end' : 'span_start',
              traceId: traceData.traceId,
              span,
            };
            broadcastManager.broadcast(traceData.traceId, event);
          }
        }

        // TODO: Add real time support and broadcast log events
        if (request.body.resourceLogs?.length) {
          const logs = logDataFromOtlp(request.body);
          if (logs.length > 0) {
            await params.logStore.save(logs);
          }
        }

        response.status(200).json({});
      } catch (err) {
        logger.error(`Error processing OTLP payload: ${err}`);
        response.status(500).json({
          code: 13, // INTERNAL
          message:
            'An internal error occurred while processing the OTLP payload.',
        });
      }
    }
  );

  api.use((err: any, req: any, res: any, next: any) => {
    logger.error(err.stack);
    const error = err as Error;
    const { message, stack } = error;
    const errorResponse = {
      code: 13, // StatusCodes.INTERNAL,
      message,
      details: {
        stack,
        traceId: err.traceId,
      },
    };
    res.status(500).json(errorResponse);
  });

  const host = params.host ?? '127.0.0.1';
  server = api.listen(params.port, host, () => {
    logger.info(`Telemetry API running on http://${host}:${params.port}`);
  });

  server.on('error', (error) => {
    logger.error(error);
  });

  process.on('SIGTERM', async () => await stopTelemetryApi());
}

/**
 * Stops Telemetry API and any running dependencies.
 */
export async function stopTelemetryApi() {
  await Promise.all([
    new Promise<void>((resolve) => {
      if (server) {
        server.close(() => {
          logger.debug('Telemetry API has succesfully shut down.');
          resolve();
        });
      } else {
        resolve();
      }
    }),
  ]);
}
