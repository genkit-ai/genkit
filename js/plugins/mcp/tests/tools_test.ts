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

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import * as assert from 'assert';
import { genkit } from 'genkit';
import { describe, it } from 'node:test';
import { registerAllTools } from '../src/util/tools.js';
import { FakeTransport } from './fakes.js';

describe('registerAllTools', () => {
  it('should pass _meta from context to callTool request', async () => {
    const ai = genkit({});
    const transport = new FakeTransport();
    transport.tools.push({
      name: 'testTool',
      description: 'A tool for testing',
      inputSchema: {
        type: 'object',
        properties: {},
        $schema: 'http://json-schema.org/draft-07/schema#',
      },
    });

    transport.callToolResult = {
      content: [{ type: 'text', text: 'response' }],
    };

    const client = new Client({ name: 'test', version: '1.0' });
    await client.connect(transport);

    await registerAllTools(ai, client, {
      name: 'test-host',
      serverName: 'test-server',
    });

    const action = await ai.registry.lookupAction('/tool/test-server/testTool');
    assert.ok(action);

    const result = await action(
      {},
      { context: { mcp: { _meta: { progressToken: 'token-123' } } } }
    );

    // The fake transport appends the _meta object to the result content
    assert.deepStrictEqual(result, 'response{"progressToken":"token-123"}');
  });

  it('should pass _meta from context to callTool request for multipart tools', async () => {
    const ai = genkit({});
    const transport = new FakeTransport();
    transport.tools.push({
      name: 'testMultipartTool',
      description: 'A tool for testing multipart',
      inputSchema: {
        type: 'object',
        properties: {},
        $schema: 'http://json-schema.org/draft-07/schema#',
      },
    });

    transport.callToolResult = {
      content: [{ type: 'text', text: 'response' }],
    };

    const client = new Client({ name: 'test', version: '1.0' });
    await client.connect(transport);

    await registerAllTools(ai, client, {
      name: 'test-host',
      serverName: 'test-server',
      multipart: true,
    });

    const action = await ai.registry.lookupAction(
      '/tool.v2/test-server/testMultipartTool'
    );
    assert.ok(action);

    const result = await action(
      {},
      { context: { mcp: { _meta: { progressToken: 'token-multipart' } } } }
    );

    assert.deepStrictEqual(result, {
      output: 'response{"progressToken":"token-multipart"}',
      metadata: undefined,
    });
  });

  it('should return structuredContent as output', async () => {
    const ai = genkit({});
    const transport = new FakeTransport();
    transport.tools.push({
      name: 'testStructuredTool',
      description: 'A tool for testing structured content',
      inputSchema: {
        type: 'object',
        properties: {},
        $schema: 'http://json-schema.org/draft-07/schema#',
      },
    });

    transport.callToolResult = {
      content: [{ type: 'text', text: 'this text is ignored' }],
      structuredContent: { foo: 'bar', count: 42 },
    };

    const client = new Client({ name: 'test', version: '1.0' });
    await client.connect(transport);

    await registerAllTools(ai, client, {
      name: 'test-host',
      serverName: 'test-server',
      multipart: true,
    });

    const action = await ai.registry.lookupAction(
      '/tool.v2/test-server/testStructuredTool'
    );
    assert.ok(action);

    const result = await action({}, {});

    assert.deepStrictEqual(result, {
      output: { foo: 'bar', count: 42 },
      metadata: undefined,
    });
  });

  it('should map audio and resource_link to content parts', async () => {
    const ai = genkit({});
    const transport = new FakeTransport();
    transport.tools.push({
      name: 'testMediaTool',
      description: 'A tool for testing audio and resource links',
      inputSchema: {
        type: 'object',
        properties: {},
        $schema: 'http://json-schema.org/draft-07/schema#',
      },
    });

    transport.callToolResult = {
      content: [
        { type: 'audio', data: 'base64audio', mimeType: 'audio/wav' },
        {
          type: 'resource_link',
          uri: 'file:///my-file.txt',
          name: 'my-file.txt',
        },
      ],
    };

    const client = new Client({ name: 'test', version: '1.0' });
    await client.connect(transport);

    await registerAllTools(ai, client, {
      name: 'test-host',
      serverName: 'test-server',
      multipart: true,
    });

    const action = await ai.registry.lookupAction(
      '/tool.v2/test-server/testMediaTool'
    );
    assert.ok(action);

    const result = await action({}, {});

    assert.deepStrictEqual(result, {
      content: [
        {
          media: {
            url: 'data:audio/wav;base64,base64audio',
            contentType: 'audio/wav',
          },
        },
        {
          resource: {
            uri: 'file:///my-file.txt',
          },
        },
      ],
      metadata: undefined,
    });
  });
});
