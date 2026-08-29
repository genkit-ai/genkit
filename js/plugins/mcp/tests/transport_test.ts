/**
 * Copyright 2026 Google LLC
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

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import * as assert from 'assert';
import express from 'express';
import { genkit } from 'genkit';
import getPort from 'get-port';
import type { Server } from 'node:http';
import { describe, it } from 'node:test';
import { createMcpClient } from '../src/index.js';

describe('Streamable HTTP transport', () => {
  it('sends configured headers and overrides requestInit duplicates', async () => {
    const receivedHeaders: express.Request['headers'][] = [];
    const servers: McpServer[] = [];
    const app = express();
    app.use(express.json());
    app.post('/mcp', async (request, response) => {
      receivedHeaders.push(request.headers);
      const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
      });
      const server = new McpServer({
        name: 'test-server',
        version: '1.0.0',
      });
      servers.push(server);
      server.tool('test_tool', {}, async () => ({
        content: [{ type: 'text', text: 'ok' }],
      }));
      await server.connect(transport);
      await transport.handleRequest(request, response, request.body);
    });
    app.get('/mcp', (request, response) => {
      receivedHeaders.push(request.headers);
      response.status(405).end();
    });
    app.delete('/mcp', (request, response) => {
      receivedHeaders.push(request.headers);
      response.status(405).end();
    });

    const port = await getPort();
    const httpServer = await new Promise<Server>((resolve) => {
      const listener = app.listen(port, () => resolve(listener));
    });
    const ai = genkit({});
    const client = createMcpClient({
      name: 'authenticated-remote',
      mcpServer: {
        url: `http://localhost:${port}/mcp`,
        requestInit: {
          headers: {
            authorization: 'Bearer stale',
            'x-request-source': 'request-init',
          },
        },
        headers: {
          authorization: 'Bearer current',
          'x-api-key': 'test-key',
        },
      },
    });

    try {
      await client.ready();
      const tools = await client.getActiveTools(ai);

      assert.deepStrictEqual(
        tools.map((tool) => tool.__action.name),
        ['test-server/test_tool']
      );
      assert.ok(receivedHeaders.length >= 2);
      for (const headers of receivedHeaders) {
        assert.strictEqual(headers.authorization, 'Bearer current');
        assert.strictEqual(headers['x-api-key'], 'test-key');
        assert.strictEqual(headers['x-request-source'], 'request-init');
      }
    } finally {
      await client.disable();
      await Promise.all(servers.map((server) => server.close()));
      await new Promise<void>((resolve, reject) => {
        httpServer.close((error) => (error ? reject(error) : resolve()));
      });
    }
  });
});
