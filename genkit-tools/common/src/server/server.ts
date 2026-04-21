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

import * as trpcExpress from '@trpc/server/adapters/express';
import * as bodyParser from 'body-parser';
import * as clc from 'colorette';
import cors from 'cors';
import express, { type ErrorRequestHandler } from 'express';
import type { Server } from 'http';
import os from 'os';
import path from 'path';
import type { GenkitToolsError } from '../manager';
import type { BaseRuntimeManager } from '../manager/manager';
import { writeToolsInfoFile } from '../utils';
import { logger } from '../utils/logger';
import { toolsPackage } from '../utils/package';
import { downloadAndExtractUiAssets } from '../utils/ui-assets';
import { TOOLS_SERVER_ROUTER } from './router';

const MAX_PAYLOAD_SIZE = 30000000;
const UI_ASSETS_GCS_BUCKET = `https://storage.googleapis.com/genkit-assets`;
const UI_ASSETS_ZIP_FILE_NAME = `${toolsPackage.version}.zip`;
const UI_ASSETS_ZIP_GCS_PATH = `${UI_ASSETS_GCS_BUCKET}/${UI_ASSETS_ZIP_FILE_NAME}`;
const UI_ASSETS_ROOT = path.resolve(
  os.homedir(),
  '.genkit',
  'assets',
  toolsPackage.version
);
const UI_ASSETS_SERVE_PATH = path.resolve(UI_ASSETS_ROOT, 'ui', 'browser');
const API_BASE_PATH = '/api';

class PushableAsyncIterable<T> implements AsyncIterable<T> {
  private queue: T[] = [];
  private resolvers: ((value: IteratorResult<T>) => void)[] = [];
  private closed = false;

  [Symbol.asyncIterator]() {
    return {
      next: () => this.next(),
    };
  }

  next(): Promise<IteratorResult<T>> {
    if (this.queue.length > 0) {
      return Promise.resolve({ value: this.queue.shift()!, done: false });
    }
    if (this.closed) {
      return Promise.resolve({ value: undefined as any, done: true });
    }
    return new Promise((resolve) => this.resolvers.push(resolve));
  }

  push(value: T) {
    if (this.closed) return;
    if (this.resolvers.length > 0) {
      this.resolvers.shift()!({ value, done: false });
    } else {
      this.queue.push(value);
    }
  }

  close() {
    this.closed = true;
    while (this.resolvers.length > 0) {
      this.resolvers.shift()!({ value: undefined as any, done: true });
    }
  }
}

const activeInputStreams = new Map<string, PushableAsyncIterable<any>>();

/**
 * Starts up the Genkit Tools server which includes static files for the UI and the Tools API.
 */
export function startServer(manager: BaseRuntimeManager, port: number) {
  let server: Server;
  const app = express();

  // Allow all origins and expose trace ID header
  app.use(
    cors({
      origin: '*',
      allowedHeaders: ['Content-Type'],
      exposedHeaders: ['X-Genkit-Trace-Id'],
    })
  );

  // Download UI assets from public GCS bucket and serve locally
  downloadAndExtractUiAssets({
    fileUrl: UI_ASSETS_ZIP_GCS_PATH,
    extractPath: UI_ASSETS_ROOT,
    zipFileName: UI_ASSETS_ZIP_FILE_NAME,
  });
  app.use(express.static(UI_ASSETS_SERVE_PATH));

  // tRPC doesn't support simple streaming mutations (https://github.com/trpc/trpc/issues/4477).
  // Don't want a separate WebSocket server for subscriptions - https://trpc.io/docs/subscriptions.
  // TODO: migrate to streamingMutation when it becomes available in tRPC.
  app.post(
    '/api/runAction',
    bodyParser.json({ limit: MAX_PAYLOAD_SIZE }),
    async (req, res) => {
      // Set headers but don't flush yet - wait for trace ID (if realtime telemetry enabled)
      res.setHeader('Content-Type', 'application/json');
      res.statusCode = 200;

      // When realtime telemetry is disabled, flush headers immediately.
      // The trace ID will be available in the response body.
      if (manager.disableRealtimeTelemetry) {
        res.flushHeaders();
      }

      try {
        const onTraceIdCallback = !manager.disableRealtimeTelemetry
          ? (traceId: string) => {
              // Set trace ID header and flush - this fires before response body
              res.setHeader('X-Genkit-Trace-Id', traceId);
              // Force chunked encoding so we can flush headers early
              res.setHeader('Transfer-Encoding', 'chunked');
              res.flushHeaders();
            }
          : undefined;

        const result = await manager.runAction(
          req.body,
          undefined, // no streaming callback
          onTraceIdCallback
        );

        res.end(JSON.stringify(result));
      } catch (err) {
        const error = err as GenkitToolsError;

        // If headers not sent, we can send error status
        if (!res.headersSent) {
          res.writeHead(500, {
            'Content-Type': 'application/json',
          });
        }
        res.end(JSON.stringify({ error: error.data }));
      }
    }
  );

  app.post(
    '/api/streamAction',
    bodyParser.json({ limit: MAX_PAYLOAD_SIZE }),
    async (req, res) => {
      // Set streaming headers but don't flush yet - wait for trace ID (if realtime telemetry enabled)
      res.setHeader('Content-Type', 'text/plain');
      res.setHeader('Transfer-Encoding', 'chunked');
      res.statusCode = 200;

      // When realtime telemetry is disabled, flush headers immediately.
      // The trace ID will be available in the response body.
      if (manager.disableRealtimeTelemetry) {
        res.flushHeaders();
      }

      let inputStream: PushableAsyncIterable<any> | undefined;
      const bidi = req.query.bidi === 'true';
      if (bidi) {
        inputStream = new PushableAsyncIterable();
      }

      let capturedTraceId: string | undefined;

      try {
        const onTraceIdCallback = (traceId: string) => {
          capturedTraceId = traceId;
          // Set trace ID header and flush - this fires before first chunk
          if (!manager.disableRealtimeTelemetry) {
            res.setHeader('X-Genkit-Trace-Id', traceId);
            res.flushHeaders();
          }
          if (bidi && inputStream) {
            activeInputStreams.set(traceId, inputStream);
          }
        };

        const result = await manager.runAction(
          req.body,
          (chunk) => {
            res.write(JSON.stringify(chunk) + '\n');
          },
          onTraceIdCallback,
          inputStream
        );
        res.write(JSON.stringify(result));
      } catch (err) {
        res.write(JSON.stringify({ error: (err as GenkitToolsError).data }));
      } finally {
        if (capturedTraceId) {
          activeInputStreams.delete(capturedTraceId);
        }
      }
      res.end();
    }
  );

  app.post(
    '/api/sendBidiInput',
    bodyParser.json({ limit: MAX_PAYLOAD_SIZE }),
    (req, res) => {
      const { traceId, chunk } = req.body;
      const stream = activeInputStreams.get(traceId);
      if (stream) {
        stream.push(chunk);
        res.status(200).send('OK');
      } else {
        res.status(404).send('Stream not found');
      }
    }
  );

  app.post('/api/endBidiInput', bodyParser.json(), (req, res) => {
    const { traceId } = req.body;
    const stream = activeInputStreams.get(traceId);
    if (stream) {
      stream.close();
      // Don't delete here, wait for action to complete (finally block)
      // or delete if we want to ensure no more writes.
      // If we delete here, subsequent writes will fail, which is correct.
      // But finally block handles cleanup anyway.
      res.status(200).send('OK');
    } else {
      res.status(404).send('Stream not found');
    }
  });

  app.post(
    '/api/streamTrace',
    bodyParser.json({ limit: MAX_PAYLOAD_SIZE }),
    async (req, res) => {
      const { traceId } = req.body;

      if (!traceId) {
        res.status(400).json({ error: 'traceId is required' });
        return;
      }

      // Set streaming headers immediately
      res.setHeader('Content-Type', 'text/plain');
      res.setHeader('Transfer-Encoding', 'chunked');
      res.setHeader('X-Genkit-Trace-Id', traceId);
      res.statusCode = 200;
      res.flushHeaders();

      try {
        await manager.streamTrace({ traceId }, (chunk) => {
          // Forward each chunk from telemetry server as chunked JSON
          res.write(JSON.stringify(chunk) + '\n');
        });
        res.end();
      } catch (err) {
        const error = err as GenkitToolsError;
        if (!res.headersSent) {
          res.writeHead(500, {
            'Content-Type': 'application/json',
          });
        }
        res.write(
          JSON.stringify({ error: error.data || { message: error.message } })
        );
        res.end();
      }
    }
  );

  app.get('/api/__health', (_, res) => {
    res.status(200).send('');
  });

  app.post('/api/__quitquitquit', (_, res) => {
    logger.debug('Shutting down tools API');
    res.status(200).send('Server is shutting down');
    server.close(() => {
      process.exit(0);
    });
  });

  // Endpoints for CLI control
  app.use(
    API_BASE_PATH,
    trpcExpress.createExpressMiddleware({
      router: TOOLS_SERVER_ROUTER(manager),
      maxBodySize: MAX_PAYLOAD_SIZE,
    })
  );

  app.all('*', (_, res) => {
    res.status(200).sendFile('/', { root: UI_ASSETS_SERVE_PATH });
  });

  const errorHandler: ErrorRequestHandler = (
    error,
    request,
    response,
    // Poor API doesn't allow leaving off `next` without changing the entire signature...
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    next
  ) => {
    if (error instanceof Error) {
      logger.error(error.stack);
    }
    return response.status(500).send(error);
  };
  app.use(errorHandler);

  server = app.listen(port, async () => {
    const uiUrl = 'http://localhost:' + port;
    const projectRoot = manager.projectRoot;
    logger.info(`${clc.green(clc.bold('Project root:'))} ${projectRoot}`);
    logger.info(`${clc.green(clc.bold('Genkit Developer UI:'))} ${uiUrl}`);
    await writeToolsInfoFile(uiUrl, projectRoot);
  });

  return new Promise<void>((resolve) => {
    server.once('close', resolve);
  });
}
