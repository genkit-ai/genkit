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
import { validateSchema, type JSONSchema } from 'genkit/schema';
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

    ai.defineFlow(
      {
        name: 'testTool',
        inputSchema: z.object({ foo: z.string() }),
        outputSchema: z.string(),
      },
      async ({ foo }) => `flow ${foo}`
    );
    ai.defineFlow(
      { name: 'stringFlow', inputSchema: z.string(), outputSchema: z.string() },
      async (input) => input.toUpperCase()
    );
    ai.defineFlow('emptyFlow', async () => 'empty flow');
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

      assert.deepStrictEqual(r.tools, [
        {
          description: 'test tool',
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
            additionalProperties: true,
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
        {
          description: "Run the Genkit flow 'testTool'.",
          inputSchema: {
            $schema: 'http://json-schema.org/draft-07/schema#',
            additionalProperties: true,
            properties: {
              foo: {
                type: 'string',
              },
            },
            required: ['foo'],
            type: 'object',
          },
          name: 'flow_testTool',
        },
        {
          description: "Run the Genkit flow 'stringFlow'.",
          inputSchema: {
            type: 'object',
            properties: {
              input: {
                type: 'string',
                $schema: 'http://json-schema.org/draft-07/schema#',
              },
            },
            required: ['input'],
          },
          name: 'flow_stringFlow',
        },
        {
          description: "Run the Genkit flow 'emptyFlow'.",
          inputSchema: { type: 'object', properties: { input: {} } },
          name: 'flow_emptyFlow',
        },
      ]);
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

    it('should call a flow as a distinct MCP tool', async () => {
      const response = await client.callTool({
        name: 'flow_testTool',
        arguments: { foo: 'bar' },
      });
      assert.deepStrictEqual(response, {
        content: [{ text: 'flow bar', type: 'text' }],
      });
    });

    it('should wrap a scalar flow input in an MCP object', async () => {
      const response = await client.callTool({
        name: 'flow_stringFlow',
        arguments: { input: 'hello' },
      });
      assert.deepStrictEqual(response, {
        content: [{ text: 'HELLO', type: 'text' }],
      });
    });

    it('should call a flow without input', async () => {
      const response = await client.callTool({
        name: 'flow_emptyFlow',
        arguments: {},
      });
      assert.deepStrictEqual(response, {
        content: [{ text: 'empty flow', type: 'text' }],
      });
    });

    it('should reject an MCP tool name collision', async () => {
      ai.defineTool({ name: 'flow_testTool' }, async () => 'collision');
      const collidingServer = createMcpServer(ai, { name: 'colliding-server' });
      try {
        await assert.rejects(
          collidingServer.setup(),
          /Multiple actions would be exposed as MCP tool 'flow_testTool'/
        );
      } finally {
        await collidingServer.server?.close();
      }
    });

    async function isolatedServer(define: (ai: GenkitBeta) => void) {
      const isolatedAi = genkit({});
      define(isolatedAi);
      const server = createMcpServer(isolatedAi, { name: 'isolated-server' });
      await server.setup();
      return server;
    }

    async function toolSchema(server: GenkitMcpServer, name: string) {
      const { tools } = await server.listTools({ method: 'tools/list' });
      return tools.find((t) => t.name === name)?.inputSchema;
    }

    async function call(server: GenkitMcpServer, name: string, args?: object) {
      const result = await server.callTool({
        method: 'tools/call',
        params: { name, arguments: args },
      });
      return (result.content as Array<{ text: string }>)[0].text;
    }

    it('should forward input to a flow without an input schema', async () => {
      const server = await isolatedServer((ai) =>
        ai.defineFlow('echo', async (input) => input)
      );
      assert.deepStrictEqual(await toolSchema(server, 'flow_echo'), {
        type: 'object',
        properties: { input: {} },
      });
      assert.strictEqual(
        await call(server, 'flow_echo', { input: { greeting: 'hi' } }),
        '{"greeting":"hi"}'
      );
    });

    it('should call a z.void() flow without input', async () => {
      const server = await isolatedServer((ai) =>
        ai.defineFlow({ name: 'ping', inputSchema: z.void() }, async (input) =>
          input === undefined ? 'pong' : 'unexpected input'
        )
      );
      assert.deepStrictEqual(await toolSchema(server, 'flow_ping'), {
        type: 'object',
      });
      assert.strictEqual(await call(server, 'flow_ping', {}), 'pong');
    });

    it('should pass an empty object to an object flow when arguments are omitted', async () => {
      const server = await isolatedServer((ai) =>
        ai.defineFlow(
          {
            name: 'search',
            inputSchema: z.object({ limit: z.number().optional() }),
          },
          async ({ limit }) => `limit=${limit ?? 'none'}`
        )
      );
      assert.strictEqual(await call(server, 'flow_search'), 'limit=none');
    });

    it('should not require input for an optional scalar flow', async () => {
      const server = await isolatedServer((ai) =>
        ai.defineFlow(
          { name: 'greet', inputSchema: z.string().optional() },
          async (name) => `hi ${name}`
        )
      );
      const schema = await toolSchema(server, 'flow_greet');
      assert.strictEqual(schema?.required, undefined);
      assert.strictEqual(
        await call(server, 'flow_greet', { input: 'ann' }),
        'hi ann'
      );
    });

    it('should rebase $ref pointers in a wrapped flow input schema', async () => {
      const part = z.object({ label: z.string() });
      const server = await isolatedServer((ai) =>
        ai.defineFlow(
          { name: 'pair', inputSchema: z.tuple([part, part]) },
          async ([a, b]) => `${a.label}:${b.label}`
        )
      );
      const jsonSchema = (await toolSchema(server, 'flow_pair')) as JSONSchema;
      const valid = (input: unknown) =>
        validateSchema({ input }, { jsonSchema }).valid;
      assert.strictEqual(valid([{ label: 'a' }, { label: 'b' }]), true);
      assert.strictEqual(valid([{ label: 'a' }, { label: 1 }]), false);
      assert.strictEqual(
        await call(server, 'flow_pair', {
          input: [{ label: 'a' }, { label: 'b' }],
        }),
        'a:b'
      );
    });

    it('should skip flows with an initSchema', async () => {
      const server = await isolatedServer((ai) =>
        ai.defineFlow(
          { name: 'session', initSchema: z.object({ id: z.string() }) },
          async () => 'started'
        )
      );
      assert.strictEqual(await toolSchema(server, 'flow_session'), undefined);
      await assert.rejects(
        call(server, 'flow_session', {}),
        /could not be found/
      );
    });

    it('should call a namespaced tool by its MCP name or full action name', async () => {
      const server = await isolatedServer((ai) =>
        ai.defineTool(
          {
            name: 'ns/find',
            description: 'find',
            inputSchema: z.object({ q: z.string() }),
          },
          async ({ q }) => `found ${q}`
        )
      );
      assert.ok(await toolSchema(server, 'find'));
      assert.strictEqual(await call(server, 'find', { q: 'a' }), 'found a');
      assert.strictEqual(await call(server, 'ns/find', { q: 'b' }), 'found b');
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
