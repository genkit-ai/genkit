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
  TraceDataSchema,
  TraceQueryFilterSchema,
  type SpanData,
} from '@genkit-ai/tools-common';
import { logger } from '@genkit-ai/tools-common/utils';
import cors from 'cors';
import express from 'express';
import type * as http from 'http';
import { BroadcastManager } from './broadcast-manager.js';
import type { TraceStore } from './types';
import { traceDataFromOtlp } from './utils/otlp';

export { LocalFileTraceStore } from './file-trace-store.js';
export { TraceQuerySchema, type TraceQuery, type TraceStore } from './types';

let server: http.Server;
const broadcastManager = new BroadcastManager();

/**
 * Starts the telemetry server with the provided params
 */
export async function startTelemetryServer(params: {
  port: number;
  traceStore: TraceStore;
  /**
   * Controls the maximum request body size. If this is a number,
   * then the value specifies the number of bytes; if it is a string,
   * the value is passed to the bytes library for parsing.
   *
   * Defaults to '5mb'.
   */
  maxRequestBodySize?: string | number;
  corsOrigin?: string | RegExp;
}) {
  await params.traceStore.init();
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

      // Create events for all spans and sort them by start time to ensure
      // correct ordering.
      const allSpans = Object.values(traceData.spans);
      const events: {
        type: 'span_start' | 'span_end';
        traceId: string;
        span: SpanData;
      }[] = [];

      // Create span_start and span_end events
      for (const span of allSpans) {
        events.push({
          type: 'span_start',
          traceId: traceData.traceId,
          span,
        });
        if (span.endTime > 0) {
          events.push({
            type: 'span_end',
            traceId: traceData.traceId,
            span,
          });
        }
      }

      // Sort events chronologically. If times are equal, start comes before end.
      events.sort((a, b) => {
        const aTime =
          a.type === 'span_start' ? a.span.startTime : a.span.endTime;
        const bTime =
          b.type === 'span_start' ? b.span.startTime : b.span.endTime;
        if (aTime !== bTime) {
          return aTime - bTime;
        }
        return a.type === 'span_start' ? -1 : 1;
      });

      // Broadcast events in chronological order.
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

  api.post(
    '/api/otlp/:parentTraceId/:parentSpanId',
    async (request, response) => {
      try {
        const { parentTraceId, parentSpanId } = request.params;

        if (!request.body.resourceSpans?.length) {
          // Acknowledge and ignore empty payloads.
          response.status(200).json({});
          return;
        }
        const traces = traceDataFromOtlp(request.body);
        for (const traceData of traces) {
          traceData.traceId = parentTraceId;
          for (const span of Object.values(traceData.spans)) {
            span.attributes['genkit:otlp-traceId'] = span.traceId;
            span.traceId = parentTraceId;
            if (!span.parentSpanId) {
              span.parentSpanId = parentSpanId;
            }
          }
          await params.traceStore.save(parentTraceId, traceData);
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

  api.post('/api/otlp', async (request, response) => {
    try {
      if (!request.body.resourceSpans?.length) {
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
      response.status(200).json({});
    } catch (err) {
      logger.error(`Error processing OTLP payload: ${err}`);
      response.status(500).json({
        code: 13, // INTERNAL
        message:
          'An internal error occurred while processing the OTLP payload.',
      });
    }
  });

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

  server = api.listen(params.port, () => {
    logger.info(`Telemetry API running on http://localhost:${params.port}`);
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
