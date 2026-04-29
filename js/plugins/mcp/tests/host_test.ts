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

import * as assert from 'assert';
import { Genkit, genkit, ToolAction } from 'genkit';
import { logger } from 'genkit/logging';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { createMcpHost, GenkitMcpHost } from '../src/index.js';
import { defineEchoModel, FakeTransport } from './fakes.js';

logger.setLogLevel('debug');

describe('createMcpHost', () => {
  let ai: Genkit;

  beforeEach(async () => {
    ai = genkit({});
    defineEchoModel(ai);
  });

  describe('host', () => {
    let fakeTransport1: FakeTransport;
    let fakeTransport2: FakeTransport;
    let clientHost: GenkitMcpHost;

    beforeEach(() => {
      clientHost = createMcpHost({
        name: 'test-mcp-host',
      });

      fakeTransport1 = new FakeTransport();
      fakeTransport1.tools.push({
        name: 'testTool1',
        inputSchema: {
          type: 'object',
          properties: {
            foo: {
              type: 'string',
            },
          },
          required: ['foo'],
          additionalProperties: true,
          $schema: 'http://json-schema.org/draft-07/schema#',
        },
        description: 'test tool 1',
      });

      fakeTransport2 = new FakeTransport();
      fakeTransport2.tools.push({
        name: 'testTool2',
        inputSchema: {
          type: 'object',
          properties: {
            foo: {
              type: 'string',
            },
          },
          required: ['foo'],
          additionalProperties: true,
          $schema: 'http://json-schema.org/draft-07/schema#',
        },
        description: 'test tool 2',
      });
    });

    afterEach(async () => {
      await clientHost?.close();
    });

    it('should dynamically connect clients', async () => {
      // no server connected, no tools
      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        []
      );

      // connect fakeTransport1
      await clientHost.connect('test-mcp-host1', {
        transport: fakeTransport1,
      });

      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        ['test-mcp-host1/testTool1']
      );

      // connect fakeTransport2
      await clientHost.connect('test-mcp-host2', {
        transport: fakeTransport2,
      });

      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        ['test-mcp-host1/testTool1', 'test-mcp-host2/testTool2']
      );

      // disable
      await clientHost.disable('test-mcp-host1');

      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        ['test-mcp-host2/testTool2']
      );

      // reconnect
      await clientHost.enable('test-mcp-host1');

      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        ['test-mcp-host1/testTool1', 'test-mcp-host2/testTool2']
      );

      // disconnect
      await clientHost.disconnect('test-mcp-host1');

      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        ['test-mcp-host2/testTool2']
      );
    });

    it('updated roots', async () => {
      // no server connected, no tools
      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        []
      );

      // connect fakeTransport1
      await clientHost.connect('test-mcp-host1', {
        transport: fakeTransport1,
        roots: [
          {
            uri: `file:///foo`,
            name: 'foo',
          },
        ],
      });

      // MCP communicates roots async...
      await new Promise((r) => setTimeout(r, 10));

      assert.deepStrictEqual(fakeTransport1.roots, [
        {
          name: 'foo',
          uri: 'file:///foo',
        },
      ]);

      await clientHost.getClient('test-mcp-host1').updateRoots([
        {
          uri: `file:///bar`,
          name: 'bar',
        },
      ]);
      // MCP communicates roots async...
      await new Promise((r) => setTimeout(r, 10));

      assert.deepStrictEqual(fakeTransport1.roots, [
        {
          name: 'bar',
          uri: 'file:///bar',
        },
      ]);
    });
  });

  describe('tools', () => {
    let fakeTransport: FakeTransport;
    let clientHost: GenkitMcpHost;

    beforeEach(() => {
      fakeTransport = new FakeTransport();
      clientHost = createMcpHost({
        name: 'test-mcp-host',
        mcpServers: {
          'test-server': {
            transport: fakeTransport,
          },
        },
      });

      fakeTransport.tools.push({
        name: 'testTool',
        inputSchema: {
          type: 'object',
          properties: {
            foo: {
              type: 'string',
            },
          },
          required: ['foo'],
          additionalProperties: true,
          $schema: 'http://json-schema.org/draft-07/schema#',
        },
        description: 'test tool',
      });
    });

    afterEach(() => {
      clientHost?.close();
    });

    it('should list tools', async () => {
      assert.deepStrictEqual(
        (await clientHost.getActiveTools(ai)).map((t) => t.__action.name),
        ['test-server/testTool']
      );
    });

    it('should call the tool', async () => {
      fakeTransport.callToolResult = {
        content: [
          {
            type: 'text',
            text: 'yep {"foo":"bar"}',
          },
        ],
      };

      const tool: ToolAction = (await clientHost.getActiveTools(ai))[0];
      const response = await tool(
        {
          foo: 'bar',
        },
        { context: { mcp: { _meta: { soMeta: true } } } }
      );
      assert.deepStrictEqual(response, 'yep {"foo":"bar"}{"soMeta":true}');
    });

    it('should call the tool with _meta', async () => {
      fakeTransport.callToolResult = {
        content: [
          {
            type: 'text',
            text: 'yep {"foo":"bar"}',
          },
        ],
      };

      const tool: ToolAction = (await clientHost.getActiveTools(ai))[0];
      const response = await tool({
        foo: 'bar',
      });
      assert.deepStrictEqual(response, 'yep {"foo":"bar"}');
    });

    it('should call the multipart tool', async () => {
      const multipartHost = createMcpHost({
        name: 'test-multipart-host',
        multipart: true,
        mcpServers: {
          'test-server': {
            transport: fakeTransport,
          },
        },
      });
      await multipartHost.ready();

      fakeTransport.callToolResult = {
        content: [
          {
            type: 'text',
            text: 'yep {"foo":"bar"}',
          },
          {
            type: 'image',
            data: 'base64data',
            mimeType: 'image/png',
          },
          {
            type: 'resource',
            resource: {
              uri: 'file:///foo.txt',
              text: 'hello resource',
            },
          },
          {
            type: 'resource',
            resource: {
              uri: 'file:///blob.bin',
              blob: 'base64blob',
              mimeType: 'application/octet-stream',
            },
          },
        ],
        _meta: {
          someData: true,
        },
      };

      const tools = await multipartHost.getActiveTools(ai);
      const tool = tools[0];
      const response = await tool(
        { foo: 'bar' },
        { context: { mcp: { _meta: { soMeta: true } } } }
      );

      assert.deepStrictEqual(response, {
        content: [
          { text: 'yep {"foo":"bar"}' },
          {
            media: {
              url: 'data:image/png;base64,base64data',
              contentType: 'image/png',
            },
          },
          { text: 'Resource (file:///foo.txt):\nhello resource' },
          {
            media: {
              url: 'data:application/octet-stream;base64,base64blob',
              contentType: 'application/octet-stream',
            },
          },
          { text: '{"soMeta":true}' },
        ],
        metadata: {
          someData: true,
        },
      });
    });

    it('should call the multipart tool and handle errors', async () => {
      const multipartHost = createMcpHost({
        name: 'test-multipart-host',
        multipart: true,
        mcpServers: {
          'test-server': {
            transport: fakeTransport,
          },
        },
      });
      await multipartHost.ready();

      fakeTransport.callToolResult = {
        isError: true,
        content: [
          {
            type: 'text',
            text: 'Simulated tool failure',
          },
        ],
        _meta: {
          errorCode: 500,
        },
      };

      const tools = await multipartHost.getActiveTools(ai);
      const tool = tools[0];
      const response = await tool({ foo: 'bar' });

      assert.deepStrictEqual(response, {
        output: { error: 'Simulated tool failure' },
        metadata: {
          errorCode: 500,
        },
      });
    });

    it('should return raw tool response when rawToolResponses is true alongside multipart', async () => {
      const multipartHost = createMcpHost({
        name: 'test-multipart-host',
        multipart: true,
        rawToolResponses: true,
        mcpServers: {
          'test-server': {
            transport: fakeTransport,
          },
        },
      });
      await multipartHost.ready();

      fakeTransport.callToolResult = {
        content: [
          {
            type: 'text',
            text: 'yep {"foo":"bar"}',
          },
        ],
      };

      const tools = await multipartHost.getActiveTools(ai);
      const tool = tools[0];
      const response = await tool({ foo: 'bar' });

      // Genkit output schema for multipart expects `{ output: ... }` when returning raw responses
      assert.deepStrictEqual(response, {
        output: {
          content: [
            {
              type: 'text',
              text: 'yep {"foo":"bar"}',
            },
          ],
        },
      });
    });
  });

  describe('prompts', () => {
    let fakeTransport: FakeTransport;
    let clientHost: GenkitMcpHost;

    beforeEach(() => {
      fakeTransport = new FakeTransport();

      clientHost = createMcpHost({
        name: 'test-mcp-host',
        mcpServers: {
          'test-server': {
            transport: fakeTransport,
          },
        },
      });

      // Note: fakeTransport.prompts.push({ name: 'testPrompt' }); is moved to specific tests
      fakeTransport.getPromptResult = {
        messages: [
          {
            role: 'user',
            content: {
              type: 'text',
              text: 'prompt says: hello',
            },
          },
        ],
      };
    });

    afterEach(() => {
      clientHost?.close();
    });

    it('should list active prompts', async () => {
      // Initially no prompts
      assert.deepStrictEqual(await clientHost.getActivePrompts(ai), []);

      // Add a prompt to the first transport
      fakeTransport.prompts.push({
        name: 'testPrompt1',
        arguments: [
          {
            name: 'foo',
            description: 'foo arg',
            required: false,
          },
        ],
        description: 'descr',
        _meta: { foo: true },
      });
      let activePrompts = await clientHost.getActivePrompts(ai);
      assert.strictEqual(activePrompts.length, 1);
      assert.deepStrictEqual(await activePrompts[0].render(), {
        messages: [
          {
            role: 'user',
            content: [
              {
                text: 'prompt says: hello',
              },
            ],
          },
        ],
      });

      // Add a second transport with another prompt
      const fakeTransport2 = new FakeTransport();
      fakeTransport2.prompts.push({
        name: 'testPrompt2',
      });
      await clientHost.connect('test-server-2', {
        transport: fakeTransport2,
      });

      activePrompts = await clientHost.getActivePrompts(ai);
      assert.deepStrictEqual(activePrompts[0].ref.metadata, {
        arguments: [
          {
            description: 'foo arg',
            name: 'foo',
            required: false,
          },
        ],
        description: 'descr',
        mcp: { _meta: { foo: true } },
      });
      assert.deepStrictEqual(
        activePrompts.map((p) => p.ref.name),
        ['testPrompt1', 'testPrompt2']
      );

      // Disable the first server
      await clientHost.disable('test-server');
      activePrompts = await clientHost.getActivePrompts(ai);
      assert.deepStrictEqual(
        activePrompts.map((p) => p.ref.name),
        ['testPrompt2']
      );

      // Enable the first server again
      await clientHost.enable('test-server');
      activePrompts = await clientHost.getActivePrompts(ai);
      assert.deepStrictEqual(
        activePrompts.map((p) => p.ref.name),
        ['testPrompt1', 'testPrompt2']
      );
    });

    it('should execute prompt', async () => {
      fakeTransport.prompts.push({
        name: 'testPrompt',
      });
      const prompt = await clientHost.getPrompt(
        ai,
        'test-server',
        'testPrompt',
        { model: 'echoModel', config: { temperature: 11 } }
      );
      assert.ok(prompt);
      const { text } = await prompt({
        input: 'hello',
      });

      assert.strictEqual(
        text,
        'Echo: prompt says: hello; config: {"temperature":11}'
      );
    });

    it('should render prompt', async () => {
      fakeTransport.prompts.push({
        name: 'testPrompt',
      });
      const prompt = await clientHost.getPrompt(
        ai,
        'test-server',
        'testPrompt',
        { model: 'echoModel', config: { temperature: 11 } }
      );
      assert.ok(prompt);
      const request = await prompt.render({
        input: 'hello',
      });

      assert.deepStrictEqual(request.messages, [
        { role: 'user', content: [{ text: 'prompt says: hello' }] },
      ]);
    });

    it('should render prompt with _meta', async () => {
      fakeTransport.prompts.push({
        name: 'testPrompt',
      });
      const prompt = await clientHost.getPrompt(
        ai,
        'test-server',
        'testPrompt',
        { model: 'echoModel', config: { temperature: 11 } }
      );
      assert.ok(prompt);
      const request = await prompt.render(
        {
          input: 'hello',
        },
        { context: { mcp: { _meta: { soMeta: true } } } }
      );

      assert.deepStrictEqual(request.messages, [
        { role: 'user', content: [{ text: 'prompt says: hello' }] },
        { role: 'model', content: [{ text: '{"soMeta":true}' }] },
      ]);
    });

    it('should stream prompt', async () => {
      fakeTransport.prompts.push({
        name: 'testPrompt',
      });
      const prompt = await clientHost.getPrompt(
        ai,
        'test-server',
        'testPrompt',
        { model: 'echoModel', config: { temperature: 11 } }
      );
      assert.ok(prompt);
      const { stream, response } = prompt.stream({
        input: 'hello',
      });

      const chunks = [] as string[];
      for await (const chunk of stream) {
        chunks.push(chunk.text);
      }

      assert.deepStrictEqual(chunks, ['3', '2', '1']);
      assert.strictEqual(
        (await response).text,
        'Echo: prompt says: hello; config: {"temperature":11}'
      );
    });
  });

  describe('resources', () => {
    let fakeTransport: FakeTransport;
    let clientHost: GenkitMcpHost;

    beforeEach(() => {
      fakeTransport = new FakeTransport();

      clientHost = createMcpHost({
        name: 'test-mcp-host',
        mcpServers: {
          'test-server': {
            transport: fakeTransport,
          },
        },
      });
    });

    afterEach(() => {
      clientHost?.close();
    });

    it('should list active resources', async () => {
      // Initially no prompts
      assert.deepStrictEqual(await clientHost.getActiveResources(ai), []);

      // Add a prompt to the first transport
      fakeTransport.resources.push({
        name: 'testResource1',
        uri: 'test://resource/1',
        description: 'test resource 1',
        _meta: { foo: true },
      });
      fakeTransport.resourceTemplates.push({
        name: 'testResourceTmpl',
        uriTemplate: 'test://resource/{id}',
        description: 'test resource template',
        _meta: { foo: true },
      });
      let activeResources = await clientHost.getActiveResources(ai);
      assert.strictEqual(activeResources.length, 2);

      // Add a second transport with another prompt
      const fakeTransport2 = new FakeTransport();
      fakeTransport2.resources.push({
        name: 'testResource2',
        uri: 'test://resource/2',
        description: 'test resource 2',
        _meta: { foo: true },
      });
      await clientHost.connect('test-server-2', {
        transport: fakeTransport2,
      });

      activeResources = await clientHost.getActiveResources(ai);
      assert.deepStrictEqual(activeResources[0].__action.metadata, {
        type: 'resource',
        dynamic: true,
        resource: {
          template: undefined,
          uri: 'test://resource/1',
        },
        mcp: { _meta: { foo: true } },
      });
      assert.deepStrictEqual(
        activeResources.map((p) => p.__action.name),
        [
          'test-server/testResource1',
          'test-server/testResourceTmpl',
          'test-server-2/testResource2',
        ]
      );

      // Disable the first server
      await clientHost.disable('test-server');
      activeResources = await clientHost.getActiveResources(ai);
      assert.deepStrictEqual(
        activeResources.map((p) => p.__action.name),
        ['test-server-2/testResource2']
      );

      // Enable the first server again
      await clientHost.enable('test-server');
      activeResources = await clientHost.getActiveResources(ai);
      assert.deepStrictEqual(
        activeResources.map((p) => p.__action.name),
        [
          'test-server/testResource1',
          'test-server/testResourceTmpl',
          'test-server-2/testResource2',
        ]
      );
    });

    it('should render resource', async () => {
      fakeTransport.resources.push({
        name: 'testResource1',
        uri: 'test://resource/1',
        description: 'test resource 1',
        _meta: { foo: true },
      });
      fakeTransport.readResourceResult = {
        contents: [
          {
            uri: 'test://resource/1',
            text: 'text resource',
          },
          {
            uri: 'test://resource/1',
            blob: 'UmVzb3VyY2UgMjogVGhpcyBpcyBhIGJhc2U2NCBibG9i',
            mimeType: 'application/png',
          },
        ],
      };
      const prompt = (await clientHost.getActiveResources(ai))[0];
      assert.ok(prompt);

      const response = await prompt.attach(ai.registry)({
        uri: 'test://resource/1',
      });

      assert.deepStrictEqual(response, {
        content: [
          {
            text: 'text resource',
            metadata: {
              resource: {
                uri: 'test://resource/1',
              },
            },
          },
          {
            media: {
              contentType: 'application/png',
              url: 'data:application/png;base64,UmVzb3VyY2UgMjogVGhpcyBpcyBhIGJhc2U2NCBibG9i',
            },
            metadata: {
              resource: {
                uri: 'test://resource/1',
              },
            },
          },
        ],
      });
    });
  });
});
