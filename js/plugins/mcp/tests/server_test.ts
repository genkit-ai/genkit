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

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import * as assert from 'assert';
import express from 'express';
import { GenkitBeta, genkit, z } from 'genkit/beta';
import { logger } from 'genkit/logging';
import getPort from 'get-port';
import * as http from 'http';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { createMcpServer } from '../src/index.js';
import { GenkitMcpServer } from '../src/server.js';
import { defineEchoModel } from './fakes.js';

logger.setLogLevel('debug');

describe('createMcpServer', async () => {
  let ai: GenkitBeta;
  let mcpServer: GenkitMcpServer;
  let mcpHttpServer: http.Server;
  let port: number;
  let client: Client;

  beforeEach(async () => {
    ai = genkit({});
    defineEchoModel(ai);

    ai.definePrompt({
      name: 'testPrompt',
      model: 'echoModel',
      prompt: 'prompt says: {{input}}',
    });

    ai.defineTool(
      {
        name: 'testTool',
        description: 'test tool',
        inputSchema: z.object({ foo: z.string() }),
      },
      async (input) => `yep ${JSON.stringify(input)}`
    );
    ai.defineTool(
      {
        name: 'weatherTool',
        description: 'weather tool',
        inputSchema: z.object({ city: z.string() }),
        outputSchema: z.object({ temp: z.number() }),
      },
      async () => ({ temp: 21 })
    );
    ai.defineTool(
      {
        name: 'screenshotTool',
        description: 'screenshot tool',
        multipart: true,
      },
      async () => ({
        output: 'ok',
        content: [
          { text: 'note' },
          {
            media: {
              url: 'data:image/png;base64,iVBORw0KGgo=',
              contentType: 'image/png',
            },
          },
          { media: { url: 'data:application/pdf;base64,JVBERi0=' } },
          {
            media: {
              url: 'https://example.com/a.png',
              contentType: 'image/png',
            },
          },
          { resource: { uri: 'my://resource' } },
        ],
      })
    );
    ai.defineResource(
      {
        name: 'testResouces',
        uri: 'my://resource',
      },
      async () => {
        return {
          content: [
            {
              text: 'my resource',
            },
          ],
        };
      }
    );

    ai.defineResource(
      {
        name: 'testTmpl',
        template: 'file://{path}',
      },
      async ({ uri }) => {
        return {
          content: [
            {
              text: `file contents for ${uri}`,
            },
          ],
        };
      }
    );

    mcpServer = createMcpServer(ai, { name: 'test-server', version: '0.0.1' });
    await mcpServer.setup();

    const app = express();
    let transport: SSEServerTransport | null = null;

    app.get('/sse', (req, res) => {
      transport = new SSEServerTransport('/messages', res);
      mcpServer.server!.connect(transport);
    });

    app.post('/messages', (req, res) => {
      if (transport) {
        transport.handlePostMessage(req, res);
      }
    });

    port = await getPort();
    let serverResolver;
    const serverPromise = new Promise((r) => {
      serverResolver = r;
    });
    mcpHttpServer = app.listen(port, () => {
      console.log(`MCP server listening on http://localhost:${port}`);
      serverResolver();
    });
    await serverPromise; // wait for server to start up

    client = new Client({ name: 'test', version: '0.0.1' });
    await client.connect(
      new SSEClientTransport(new URL(`http://localhost:${port}/sse`))
    );
  });

  afterEach(async () => {
    mcpHttpServer.close();
    await mcpServer.server?.close();
    await client?.close();
  });

  describe('tools', () => {
    it('should list tools', async () => {
      const r = await client.listTools();

      assert.deepStrictEqual(r.tools.map((t) => t.name).sort(), [
        'screenshotTool',
        'testTool',
        'weatherTool',
      ]);
      assert.deepStrictEqual(
        r.tools.find((t) => t.name === 'weatherTool')?.outputSchema,
        {
          $schema: 'http://json-schema.org/draft-07/schema#',
          properties: { temp: { type: 'number' } },
          required: ['temp'],
          type: 'object',
        }
      );
      assert.deepStrictEqual(
        r.tools.filter((t) => t.name === 'testTool'),
        [
          {
            description: 'test tool',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              properties: {
                foo: {
                  type: 'string',
                },
              },
              required: ['foo'],
              type: 'object',
            },
            name: 'testTool',
          },
        ]
      );
    });

    it('should return object output as structuredContent', async () => {
      const response = await client.callTool({
        name: 'weatherTool',
        arguments: { city: 'Paris' },
      });
      assert.deepStrictEqual(response, {
        content: [{ type: 'text', text: '{"temp":21}' }],
        structuredContent: { temp: 21 },
      });
    });

    it('should map multipart content to content blocks', async () => {
      const response = await client.callTool({
        name: 'screenshotTool',
        arguments: {},
      });
      assert.deepStrictEqual(response, {
        content: [
          { type: 'text', text: 'ok' },
          { type: 'text', text: 'note' },
          { type: 'image', mimeType: 'image/png', data: 'iVBORw0KGgo=' },
          {
            type: 'resource',
            resource: {
              uri: 'genkit://tool/screenshotTool/content/2',
              mimeType: 'application/pdf',
              blob: 'JVBERi0=',
            },
          },
          {
            type: 'resource_link',
            uri: 'https://example.com/a.png',
            name: 'https://example.com/a.png',
            mimeType: 'image/png',
          },
          {
            type: 'resource_link',
            uri: 'my://resource',
            name: 'my://resource',
          },
        ],
      });
    });

    it('should call the tool', async () => {
      const response = await client.callTool({
        name: 'testTool',
        arguments: {
          foo: 'bar',
        },
      });
      assert.deepStrictEqual(response, {
        content: [
          {
            text: 'yep {"foo":"bar"}',
            type: 'text',
          },
        ],
      });
    });
  });

  describe('prompts', () => {
    it('should list prompts', async () => {
      const r = await client.listPrompts();
      assert.deepStrictEqual(r.prompts, [{ name: 'testPrompt' }]);
    });

    it('should render prompt', async () => {
      const prompt = await client.getPrompt({
        name: 'testPrompt',
        arguments: {
          input: 'hello',
        },
      });

      assert.deepStrictEqual(prompt.messages, [
        {
          content: {
            text: 'prompt says: hello',
            type: 'text',
          },
          role: 'user',
        },
      ]);
    });
  });

  describe('resources', () => {
    it('should list resources', async () => {
      const r = await client.listResources();
      assert.deepStrictEqual(r.resources, [
        {
          name: 'testResouces',
          uri: 'my://resource',
        },
      ]);
    });
    it('should list templates', async () => {
      const r = await client.listResourceTemplates();
      assert.deepStrictEqual(r.resourceTemplates, [
        {
          name: 'testTmpl',
          uriTemplate: 'file://{path}',
        },
      ]);
    });

    it('should read resource', async () => {
      const resource = await client.readResource({
        uri: 'my://resource',
      });

      assert.deepStrictEqual(resource, {
        contents: [
          {
            text: 'my resource',
            uri: 'my://resource',
          },
        ],
      });
    });
  });
});
