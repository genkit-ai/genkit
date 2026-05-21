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
import { genkit } from 'genkit';
import { afterEach, beforeEach, describe, it, mock } from 'node:test';
import { goodmem, listEmbedders, listSpaces } from '../src/index.js';

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

function mockResponse(body: any, status = 200, ok = true): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Error',
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: new Headers(),
    redirected: false,
    type: 'basic' as ResponseType,
    url: '',
    clone: () => mockResponse(body, status, ok),
    body: null,
    bodyUsed: false,
    arrayBuffer: async () => new ArrayBuffer(0),
    blob: async () => new Blob(),
    formData: async () => new FormData(),
    bytes: async () => new Uint8Array(),
  } as Response;
}

function mockNdjsonResponse(lines: any[], status = 200): Response {
  const ndjson = lines.map((l) => JSON.stringify(l)).join('\n');
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'OK',
    json: async () => JSON.parse(ndjson),
    text: async () => ndjson,
    headers: new Headers(),
    redirected: false,
    type: 'basic' as ResponseType,
    url: '',
    clone: () => mockNdjsonResponse(lines, status),
    body: null,
    bodyUsed: false,
    arrayBuffer: async () => new ArrayBuffer(0),
    blob: async () => new Blob(),
    formData: async () => new FormData(),
    bytes: async () => new Uint8Array(),
  } as Response;
}

function mockErrorResponse(
  status: number,
  body: any = { message: 'Not Found' }
): Response {
  return mockResponse(body, status, false);
}

/**
 * Helper to call a registered Genkit tool by action key.
 */
async function callTool(ai: any, toolName: string, input: any): Promise<any> {
  const action = await ai.registry.lookupAction(`/tool/${toolName}`);
  if (!action) {
    throw new Error(`Tool not found: ${toolName}`);
  }
  return action(input);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GoodMem Plugin', () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    mock.restoreAll();
  });

  // ---- Plugin initialization ----

  describe('plugin initialization', () => {
    it('should throw error when baseUrl is missing', () => {
      assert.throws(
        () => goodmem({ baseUrl: '', apiKey: 'test-key' }),
        /GoodMem plugin requires a baseUrl/
      );
    });

    it('should throw error when apiKey is missing', () => {
      assert.throws(
        () => goodmem({ baseUrl: 'http://localhost:8080', apiKey: '' }),
        /GoodMem plugin requires an apiKey/
      );
    });

    it('should register all tools when initialized', async () => {
      globalThis.fetch = async () => mockResponse({});

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });

      // Wait for plugin initialization
      await new Promise((r) => setTimeout(r, 200));

      const toolNames = [
        'goodmem/list_embedders',
        'goodmem/list_spaces',
        'goodmem/get_space',
        'goodmem/create_space',
        'goodmem/update_space',
        'goodmem/delete_space',
        'goodmem/create_memory',
        'goodmem/list_memories',
        'goodmem/retrieve_memories',
        'goodmem/get_memory',
        'goodmem/delete_memory',
      ];

      for (const name of toolNames) {
        const action = await ai.registry.lookupAction(`/tool/${name}`);
        assert.ok(action, `Tool ${name} should be registered`);
      }
    });
  });

  // ---- Helper functions ----

  describe('listSpaces', () => {
    it('should return spaces array from response', async () => {
      const mockSpaces = [
        { spaceId: 'sp-1', name: 'test-space' },
        { spaceId: 'sp-2', name: 'another-space' },
      ];
      globalThis.fetch = async () => mockResponse({ spaces: mockSpaces });

      const result = await listSpaces({
        baseUrl: 'http://localhost:8080',
        apiKey: 'test-key',
      });

      assert.ok(Array.isArray(result));
      assert.strictEqual(result.length, 2);
      assert.strictEqual(result[0].spaceId, 'sp-1');
    });

    it('should handle array response directly', async () => {
      const mockSpaces = [{ spaceId: 'sp-1', name: 'test-space' }];
      globalThis.fetch = async () => mockResponse(mockSpaces);

      const result = await listSpaces({
        baseUrl: 'http://localhost:8080',
        apiKey: 'test-key',
      });

      assert.ok(Array.isArray(result));
      assert.strictEqual(result.length, 1);
    });

    it('should throw on API error', async () => {
      globalThis.fetch = async () =>
        mockErrorResponse(500, { message: 'Internal Server Error' });

      await assert.rejects(
        () =>
          listSpaces({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        /GoodMem API error \(500\)/
      );
    });
  });

  describe('listEmbedders', () => {
    it('should return embedders array from response', async () => {
      const mockEmbedders = [{ embedderId: 'emb-1', name: 'text-embedding' }];
      globalThis.fetch = async () => mockResponse({ embedders: mockEmbedders });

      const result = await listEmbedders({
        baseUrl: 'http://localhost:8080',
        apiKey: 'test-key',
      });

      assert.ok(Array.isArray(result));
      assert.strictEqual(result.length, 1);
      assert.strictEqual(result[0].embedderId, 'emb-1');
    });
  });

  // ---- createSpace tool ----

  describe('goodmem/create_space', () => {
    it('should create a new space', async () => {
      let callCount = 0;
      globalThis.fetch = async (url: any) => {
        callCount++;
        const urlStr = typeof url === 'string' ? url : url.toString();
        if (urlStr.endsWith('/v1/spaces') && callCount === 1) {
          // listSpaces call: no matching space
          return mockResponse({ spaces: [] });
        }
        // POST to create space
        return mockResponse({
          spaceId: 'new-space-id',
          name: 'test-space',
        });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/create_space', {
        name: 'test-space',
        embedderId: 'emb-1',
        chunkSize: 256,
        chunkOverlap: 25,
      });

      assert.strictEqual(result.success, true);
      assert.strictEqual(result.spaceId, 'new-space-id');
      assert.strictEqual(result.reused, false);
    });

    it('should reuse an existing space with the same name', async () => {
      globalThis.fetch = async () =>
        mockResponse({
          spaces: [{ spaceId: 'existing-id', name: 'test-space' }],
        });

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/create_space', {
        name: 'test-space',
        embedderId: 'emb-1',
      });

      assert.strictEqual(result.success, true);
      assert.strictEqual(result.spaceId, 'existing-id');
      assert.strictEqual(result.reused, true);
    });

    it('should handle API error when creating space', async () => {
      let callCount = 0;
      globalThis.fetch = async () => {
        callCount++;
        if (callCount === 1) {
          return mockResponse({ spaces: [] });
        }
        return mockErrorResponse(400, {
          message: 'Invalid embedder ID',
        });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/create_space', {
        name: 'test-space',
        embedderId: 'invalid-emb',
      });

      assert.strictEqual(result.success, false);
      assert.ok(result.error);
    });
  });

  // ---- createMemory tool ----

  describe('goodmem/create_memory', () => {
    it('should create a memory from text content', async () => {
      globalThis.fetch = async (url: any) => {
        const urlStr = typeof url === 'string' ? url : url.toString();
        if (urlStr.endsWith('/v1/memories')) {
          return mockResponse({
            memoryId: 'mem-123',
            spaceId: 'sp-1',
            processingStatus: 'PENDING',
          });
        }
        return mockResponse({});
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/create_memory', {
        spaceId: 'sp-1',
        textContent: 'The capital of France is Paris.',
        source: 'test',
        author: 'tester',
        tags: 'test,france',
      });

      assert.strictEqual(result.success, true);
      assert.strictEqual(result.memoryId, 'mem-123');
      assert.strictEqual(result.contentType, 'text/plain');
    });

    it('should fail gracefully when no content is provided', async () => {
      globalThis.fetch = async () => mockResponse({});

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/create_memory', {
        spaceId: 'sp-1',
      });

      assert.strictEqual(result.success, false);
      assert.ok(result.error);
      assert.match(result.error, /No content provided/);
    });

    it('should fail gracefully when file is not found', async () => {
      globalThis.fetch = async () => mockResponse({});

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/create_memory', {
        spaceId: 'sp-1',
        filePath: '/nonexistent/file.pdf',
      });

      assert.strictEqual(result.success, false);
      assert.ok(result.error);
      assert.match(result.error, /File not found/);
    });

    it('should handle API error when creating memory', async () => {
      globalThis.fetch = async () =>
        mockErrorResponse(500, { message: 'Internal error' });

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/create_memory', {
        spaceId: 'sp-1',
        textContent: 'some text',
      });

      assert.strictEqual(result.success, false);
      assert.ok(result.error);
    });
  });

  // ---- listMemories tool ----

  describe('goodmem/list_memories', () => {
    it('passes every set query parameter through the URL', async () => {
      let capturedUrl = '';
      globalThis.fetch = async (url: any) => {
        capturedUrl = typeof url === 'string' ? url : url.toString();
        return mockResponse({ memories: [] });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      await callTool(ai, 'goodmem/list_memories', {
        spaceId: 'sp-1',
        statusFilter: 'COMPLETED',
        includeContent: true,
        sortBy: 'created_at',
        sortOrder: 'DESCENDING',
      });

      assert.ok(
        capturedUrl.startsWith(
          'http://localhost:8080/v1/spaces/sp-1/memories?'
        ),
        `URL did not include a query string: ${capturedUrl}`
      );
      assert.match(capturedUrl, /statusFilter=COMPLETED/);
      assert.match(capturedUrl, /includeContent=true/);
      assert.match(capturedUrl, /sortBy=created_at/);
      assert.match(capturedUrl, /sortOrder=DESCENDING/);
    });

    it('sends no query string when no optional filters are set', async () => {
      let capturedUrl = '';
      globalThis.fetch = async (url: any) => {
        capturedUrl = typeof url === 'string' ? url : url.toString();
        return mockResponse({ memories: [] });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      await callTool(ai, 'goodmem/list_memories', { spaceId: 'sp-1' });

      assert.strictEqual(
        capturedUrl,
        'http://localhost:8080/v1/spaces/sp-1/memories'
      );
    });

    it('omits includeContent from the URL when set to false', async () => {
      let capturedUrl = '';
      globalThis.fetch = async (url: any) => {
        capturedUrl = typeof url === 'string' ? url : url.toString();
        return mockResponse({ memories: [] });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      await callTool(ai, 'goodmem/list_memories', {
        spaceId: 'sp-1',
        includeContent: false,
      });

      assert.strictEqual(
        capturedUrl,
        'http://localhost:8080/v1/spaces/sp-1/memories'
      );
    });

    it('keeps only the filters that are set', async () => {
      let capturedUrl = '';
      globalThis.fetch = async (url: any) => {
        capturedUrl = typeof url === 'string' ? url : url.toString();
        return mockResponse({ memories: [] });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      await callTool(ai, 'goodmem/list_memories', {
        spaceId: 'sp-1',
        statusFilter: 'PENDING',
      });

      assert.ok(capturedUrl.includes('statusFilter=PENDING'));
      assert.ok(!capturedUrl.includes('sortBy='));
      assert.ok(!capturedUrl.includes('sortOrder='));
      assert.ok(!capturedUrl.includes('includeContent='));
    });
  });

  // ---- retrieveMemories tool ----

  describe('goodmem/retrieve_memories', () => {
    it('should retrieve memories with NDJSON response', async () => {
      const ndjsonLines = [
        {
          resultSetBoundary: { resultSetId: 'rs-1' },
        },
        {
          retrievedItem: {
            chunk: {
              chunk: {
                chunkId: 'c-1',
                chunkText: 'Paris is the capital of France',
                memoryId: 'mem-1',
              },
              relevanceScore: 0.95,
              memoryIndex: 0,
            },
          },
        },
        {
          memoryDefinition: {
            memoryId: 'mem-1',
            spaceId: 'sp-1',
          },
        },
      ];

      globalThis.fetch = async () => mockNdjsonResponse(ndjsonLines);

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'What is the capital of France?',
        spaceIds: ['sp-1'],
        maxResults: 5,
        waitForIndexing: false,
      });

      assert.strictEqual(result.success, true);
      assert.strictEqual(result.totalResults, 1);
      assert.strictEqual(
        result.results[0].chunkText,
        'Paris is the capital of France'
      );
      assert.strictEqual(result.results[0].relevanceScore, 0.95);
      assert.strictEqual(result.memories.length, 1);
      assert.strictEqual(result.resultSetId, 'rs-1');
    });

    it('should fail with empty spaceIds', async () => {
      globalThis.fetch = async () => mockResponse({});

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'test',
        spaceIds: [],
        waitForIndexing: false,
      });

      assert.strictEqual(result.success, false);
      assert.match(result.error, /At least one space/);
    });

    it('should handle API error during retrieval', async () => {
      globalThis.fetch = async () =>
        mockErrorResponse(500, { message: 'Server error' });

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'test',
        spaceIds: ['sp-1'],
        waitForIndexing: false,
      });

      assert.strictEqual(result.success, false);
      assert.ok(result.error);
    });

    it('applies metadataFilter to every spaceKey when set', async () => {
      let capturedBody: any;
      globalThis.fetch = async (_url: any, init: any) => {
        capturedBody = JSON.parse(init.body as string);
        return mockNdjsonResponse([
          { resultSetBoundary: { resultSetId: 'rs-mf' } },
          {
            retrievedItem: {
              chunk: {
                chunk: {
                  chunkId: 'c-1',
                  chunkText: 'hello',
                  memoryId: 'mem-1',
                },
                relevanceScore: 0.5,
                memoryIndex: 0,
              },
            },
          },
        ]);
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const filter = "CAST(val('$.category') AS TEXT) = 'feat'";
      await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'q',
        spaceIds: ['sp-1', 'sp-2'],
        metadataFilter: filter,
        waitForIndexing: false,
      });

      assert.strictEqual(capturedBody.spaceKeys.length, 2);
      for (const key of capturedBody.spaceKeys) {
        assert.strictEqual(key.filter, filter);
      }
    });

    it('omits the filter key from spaceKeys when metadataFilter is not set', async () => {
      let capturedBody: any;
      globalThis.fetch = async (_url: any, init: any) => {
        capturedBody = JSON.parse(init.body as string);
        return mockNdjsonResponse([
          { resultSetBoundary: { resultSetId: 'rs-no-mf' } },
        ]);
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'q',
        spaceIds: ['sp-1', 'sp-2'],
        waitForIndexing: false,
      });

      assert.strictEqual(capturedBody.spaceKeys.length, 2);
      for (const key of capturedBody.spaceKeys) {
        assert.strictEqual(key.filter, undefined);
      }
    });

    it('treats an empty metadataFilter the same as unset', async () => {
      let capturedBody: any;
      globalThis.fetch = async (_url: any, init: any) => {
        capturedBody = JSON.parse(init.body as string);
        return mockNdjsonResponse([
          { resultSetBoundary: { resultSetId: 'rs-empty-mf' } },
        ]);
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'q',
        spaceIds: ['sp-1'],
        metadataFilter: '',
        waitForIndexing: false,
      });

      assert.strictEqual(capturedBody.spaceKeys[0].filter, undefined);
    });

    it('respects custom maxWaitSeconds and pollInterval while polling', async () => {
      let calls = 0;
      globalThis.fetch = async () => {
        calls++;
        return mockNdjsonResponse([
          { resultSetBoundary: { resultSetId: 'rs-empty' } },
        ]);
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const start = Date.now();
      const result = await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'q',
        spaceIds: ['sp-1'],
        waitForIndexing: true,
        maxWaitSeconds: 0.5,
        pollInterval: 0.1,
      });
      const elapsed = Date.now() - start;

      assert.ok(
        elapsed < 2000,
        `expected polling to stop quickly, took ${elapsed}ms`
      );
      assert.ok(
        calls >= 2,
        `expected polling to call fetch more than once, got ${calls}`
      );
      assert.strictEqual(result.success, true);
      assert.strictEqual(result.totalResults, 0);
      assert.match(result.message ?? '', /No results found/);
    });

    it('skips the polling loop when waitForIndexing is false', async () => {
      let calls = 0;
      globalThis.fetch = async () => {
        calls++;
        return mockNdjsonResponse([
          { resultSetBoundary: { resultSetId: 'rs-skip' } },
        ]);
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'q',
        spaceIds: ['sp-1'],
        waitForIndexing: false,
      });

      assert.strictEqual(calls, 1);
      assert.strictEqual(result.success, true);
      assert.strictEqual(result.totalResults, 0);
    });

    it('should handle abstractReply in NDJSON response', async () => {
      const ndjsonLines = [
        {
          resultSetBoundary: { resultSetId: 'rs-2' },
        },
        {
          retrievedItem: {
            chunk: {
              chunk: {
                chunkId: 'c-1',
                chunkText: 'Some text',
                memoryId: 'mem-1',
              },
              relevanceScore: 0.9,
              memoryIndex: 0,
            },
          },
        },
        {
          abstractReply: {
            text: 'Based on the context, the answer is...',
          },
        },
      ];

      globalThis.fetch = async () => mockNdjsonResponse(ndjsonLines);

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/retrieve_memories', {
        query: 'test query',
        spaceIds: ['sp-1'],
        waitForIndexing: false,
      });

      assert.strictEqual(result.success, true);
      assert.ok(result.abstractReply);
      assert.strictEqual(
        result.abstractReply.text,
        'Based on the context, the answer is...'
      );
    });
  });

  // ---- getMemory tool ----

  describe('goodmem/get_memory', () => {
    it('should get a memory by ID', async () => {
      let callCount = 0;
      globalThis.fetch = async (url: any) => {
        callCount++;
        const urlStr = typeof url === 'string' ? url : url.toString();
        if (urlStr.includes('/content')) {
          return mockResponse({
            contentType: 'text/plain',
            content: 'Hello world',
          });
        }
        return mockResponse({
          memoryId: 'mem-123',
          spaceId: 'sp-1',
          processingStatus: 'COMPLETED',
        });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/get_memory', {
        memoryId: 'mem-123',
        includeContent: true,
      });

      assert.strictEqual(result.success, true);
      assert.ok(result.memory);
      assert.strictEqual(result.memory.memoryId, 'mem-123');
      assert.ok(result.content);
    });

    it('should handle content fetch failure gracefully', async () => {
      let callCount = 0;
      globalThis.fetch = async (url: any) => {
        callCount++;
        const urlStr = typeof url === 'string' ? url : url.toString();
        if (urlStr.includes('/content')) {
          return mockErrorResponse(404, { message: 'Content not found' });
        }
        return mockResponse({
          memoryId: 'mem-123',
          spaceId: 'sp-1',
        });
      };

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/get_memory', {
        memoryId: 'mem-123',
        includeContent: true,
      });

      assert.strictEqual(result.success, true);
      assert.ok(result.memory);
      assert.ok(result.contentError);
    });

    it('should return error for invalid memory ID', async () => {
      globalThis.fetch = async () =>
        mockErrorResponse(404, { message: 'Memory not found' });

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/get_memory', {
        memoryId: '00000000-0000-0000-0000-000000000000',
      });

      assert.strictEqual(result.success, false);
      assert.ok(result.error);
    });
  });

  // ---- deleteMemory tool ----

  describe('goodmem/delete_memory', () => {
    it('should delete a memory successfully', async () => {
      globalThis.fetch = async () => mockResponse({});

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/delete_memory', {
        memoryId: 'mem-123',
      });

      assert.strictEqual(result.success, true);
      assert.strictEqual(result.memoryId, 'mem-123');
    });

    it('should return error for invalid memory ID', async () => {
      globalThis.fetch = async () =>
        mockErrorResponse(404, { message: 'Memory not found' });

      const ai = genkit({
        plugins: [
          goodmem({
            baseUrl: 'http://localhost:8080',
            apiKey: 'test-key',
          }),
        ],
      });
      await new Promise((r) => setTimeout(r, 200));

      const result = await callTool(ai, 'goodmem/delete_memory', {
        memoryId: '00000000-0000-0000-0000-000000000000',
      });

      assert.strictEqual(result.success, false);
      assert.ok(result.error);
    });
  });

  // ---- API headers ----

  describe('API headers', () => {
    it('should send correct headers with API key', async () => {
      let capturedHeaders: Record<string, string> = {};
      globalThis.fetch = async (_url: any, init: any) => {
        capturedHeaders = init?.headers || {};
        return mockResponse({ spaces: [] });
      };

      await listSpaces({
        baseUrl: 'http://localhost:8080',
        apiKey: 'gm_test_api_key',
      });

      assert.strictEqual(capturedHeaders['X-API-Key'], 'gm_test_api_key');
      assert.strictEqual(capturedHeaders['Content-Type'], 'application/json');
      assert.strictEqual(capturedHeaders['Accept'], 'application/json');
    });
  });

  // ---- URL normalization ----

  describe('URL normalization', () => {
    it('should strip trailing slash from baseUrl', async () => {
      let capturedUrl = '';
      globalThis.fetch = async (url: any) => {
        capturedUrl = typeof url === 'string' ? url : url.toString();
        return mockResponse({ spaces: [] });
      };

      await listSpaces({
        baseUrl: 'http://localhost:8080/',
        apiKey: 'test-key',
      });

      assert.strictEqual(capturedUrl, 'http://localhost:8080/v1/spaces');
    });
  });
});
