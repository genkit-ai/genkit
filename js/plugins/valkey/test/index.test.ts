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

import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import type { EmbedderArgument, Genkit } from 'genkit';
import { z } from 'genkit';
import { Document } from 'genkit/retriever';

// Mock @valkey/valkey-glide
const mockHset = jest.fn<any>().mockResolvedValue(1);
const mockBatchHset = jest.fn<any>().mockReturnValue(undefined);
const mockExec = jest.fn<any>().mockResolvedValue([]);
const mockClient = { hset: mockHset, exec: mockExec };

const mockGlideFtCreate = jest.fn<any>().mockResolvedValue('OK');
const mockGlideFtSearch = jest.fn<any>();

jest.mock('@valkey/valkey-glide', () => ({
  GlideClient: {
    createClient: jest.fn<any>().mockResolvedValue(mockClient),
  },
  GlideFt: {
    create: (...args: unknown[]) => mockGlideFtCreate(...args),
    search: (...args: unknown[]) => mockGlideFtSearch(...args),
  },
  Batch: jest.fn<any>().mockImplementation(() => {
    const instance = {
      hset: (...args: unknown[]) => {
        mockBatchHset(...args);
        return instance;
      },
    };
    return instance;
  }),
}));

// Import after mocks are set up
import {
  stableDocId,
  validateFilterExpression,
  valkeyIndexerRef,
  valkeyPlugin,
  valkeyRetrieverRef,
} from '../src';

// Mock Genkit instance
const mockEmbed = jest.fn<any>();
const mockEmbedderAction = jest.fn<any>();
const mockLookupAction = jest.fn<any>().mockResolvedValue(mockEmbedderAction);
const mockDefineRetriever = jest.fn<any>((config: any, handler: any) => ({
  config,
  retrieve: handler,
}));
const mockDefineIndexer = jest.fn<any>((config: any, handler: any) => ({
  config,
  index: handler,
}));

const mockGenkit: Genkit = {
  embed: mockEmbed,
  defineRetriever: mockDefineRetriever,
  defineIndexer: mockDefineIndexer,
  registry: {
    lookupAction: mockLookupAction,
  },
} as unknown as Genkit;

const mockEmbedder: EmbedderArgument<z.ZodTypeAny> = {
  name: 'mock-embedder',
  type: 'embedder',
  configSchema: z.object({}),
} as any;

describe('valkeyPlugin', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGlideFtCreate.mockResolvedValue('OK');
  });

  test('should create index on initialization', async () => {
    const plugin = valkeyPlugin([
      {
        indexName: 'test-index',
        embedder: mockEmbedder,
        dimension: 768,
        clientConfig: { addresses: [{ host: 'localhost', port: 6379 }] },
      },
    ]);

    // genkitPlugin returns a function that takes Genkit and returns {initializer}
    const pluginInstance = (plugin.plugin as any)(mockGenkit);
    await pluginInstance.initializer();

    expect(mockGlideFtCreate).toHaveBeenCalledWith(
      mockClient,
      'test-index',
      expect.arrayContaining([
        expect.objectContaining({
          type: 'VECTOR',
          name: 'embedding',
          attributes: expect.objectContaining({
            algorithm: 'HNSW',
            dimensions: 768,
            distanceMetric: 'COSINE',
            type: 'FLOAT32',
          }),
        }),
        expect.objectContaining({ type: 'TEXT', name: '_content' }),
        expect.objectContaining({ type: 'TEXT', name: '_metadata' }),
        expect.objectContaining({ type: 'TEXT', name: '_dataType' }),
      ]),
      expect.objectContaining({
        dataType: 'HASH',
        prefixes: ['test-index:'],
      })
    );
  });

  test('should handle duplicate index error gracefully', async () => {
    mockGlideFtCreate.mockRejectedValue(new Error('Index already exists'));

    const plugin = valkeyPlugin([
      {
        indexName: 'existing-index',
        embedder: mockEmbedder,
        dimension: 768,
        clientConfig: { addresses: [{ host: 'localhost', port: 6379 }] },
      },
    ]);

    const pluginInstance = (plugin.plugin as any)(mockGenkit);
    await expect(pluginInstance.initializer()).resolves.not.toThrow();
  });

  test('should propagate non-duplicate-index errors', async () => {
    mockGlideFtCreate.mockRejectedValue(new Error('Connection refused'));

    const plugin = valkeyPlugin([
      {
        indexName: 'fail-index',
        embedder: mockEmbedder,
        dimension: 768,
        clientConfig: { addresses: [{ host: 'localhost', port: 6379 }] },
      },
    ]);

    const pluginInstance = (plugin.plugin as any)(mockGenkit);
    await expect(pluginInstance.initializer()).rejects.toThrow(
      'Connection refused'
    );
  });

  test('should register both indexer and retriever', async () => {
    const plugin = valkeyPlugin([
      {
        indexName: 'test-index',
        embedder: mockEmbedder,
        dimension: 768,
        clientConfig: { addresses: [{ host: 'localhost', port: 6379 }] },
      },
    ]);

    const pluginInstance = (plugin.plugin as any)(mockGenkit);
    await pluginInstance.initializer();

    expect(mockDefineIndexer).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'valkey/test-index' }),
      expect.any(Function)
    );
    expect(mockDefineRetriever).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'valkey/test-index' }),
      expect.any(Function)
    );
  });

  test('should use custom prefix when provided', async () => {
    const plugin = valkeyPlugin([
      {
        indexName: 'test-index',
        embedder: mockEmbedder,
        dimension: 768,
        prefix: 'custom',
        clientConfig: { addresses: [{ host: 'localhost', port: 6379 }] },
      },
    ]);

    const pluginInstance = (plugin.plugin as any)(mockGenkit);
    await pluginInstance.initializer();

    expect(mockGlideFtCreate).toHaveBeenCalledWith(
      mockClient,
      'test-index',
      expect.any(Array),
      expect.objectContaining({
        prefixes: ['custom:'],
      })
    );
  });
});

describe('valkeyIndexer', () => {
  let indexerHandler: (docs: Document[]) => Promise<void>;

  beforeEach(async () => {
    jest.clearAllMocks();
    mockGlideFtCreate.mockResolvedValue('OK');

    const plugin = valkeyPlugin([
      {
        indexName: 'test-index',
        embedder: mockEmbedder,
        dimension: 3,
        clientConfig: { addresses: [{ host: 'localhost', port: 6379 }] },
      },
    ]);

    const pluginInstance = (plugin.plugin as any)(mockGenkit);
    await pluginInstance.initializer();

    indexerHandler = mockDefineIndexer.mock.calls[0][1] as any;
  });

  test('should embed documents and store as hashes', async () => {
    mockEmbedderAction.mockResolvedValue({
      embeddings: [{ embedding: [0.1, 0.2, 0.3] }],
    });

    const doc = new Document({
      content: [{ text: 'Hello world' }],
      metadata: { source: 'test' },
    });

    await indexerHandler([doc]);

    expect(mockEmbedderAction).toHaveBeenCalledWith(
      expect.objectContaining({
        input: expect.any(Array),
      })
    );
    expect(mockBatchHset).toHaveBeenCalledWith(
      expect.stringMatching(/^test-index:/),
      expect.objectContaining({
        embedding: expect.anything(),
        _content: expect.any(String),
        _metadata: expect.any(String),
        _dataType: expect.any(String),
      })
    );
    expect(mockExec).toHaveBeenCalledTimes(1);
  });

  test('should convert embedding to Float32 buffer', async () => {
    const embedding = [1.0, 2.0, 3.0];
    mockEmbedderAction.mockResolvedValue({
      embeddings: [{ embedding }],
    });

    const doc = new Document({ content: [{ text: 'test' }] });
    await indexerHandler([doc]);

    const hsetCall = mockBatchHset.mock.calls[0] as unknown[];
    const fields = hsetCall[1] as Record<string, any>;
    const embeddingValue = fields['embedding'];

    expect(embeddingValue).toBeDefined();
    expect(embeddingValue).toBeInstanceOf(Buffer);

    const expected = Buffer.from(new Float32Array(embedding).buffer);
    expect(embeddingValue).toEqual(expected);
  });

  test('should serialize metadata as JSON', async () => {
    mockEmbedderAction.mockResolvedValue({
      embeddings: [{ embedding: [0.1, 0.2, 0.3] }],
    });

    const metadata = { page: 1, source: 'docs', nested: { key: 'value' } };
    const doc = new Document({
      content: [{ text: 'test' }],
      metadata,
    });

    await indexerHandler([doc]);

    const hsetCall = mockBatchHset.mock.calls[0] as unknown[];
    const fields = hsetCall[1] as Record<string, any>;
    const metadataValue = fields['_metadata'];

    expect(metadataValue).toBeDefined();
    expect(JSON.parse(metadataValue)).toEqual(metadata);
  });

  test('should handle multiple documents', async () => {
    mockEmbedderAction.mockResolvedValue({
      embeddings: [
        { embedding: [0.1, 0.2, 0.3] },
        { embedding: [0.4, 0.5, 0.6] },
        { embedding: [0.7, 0.8, 0.9] },
      ],
    });

    const docs = [
      new Document({ content: [{ text: 'doc1' }] }),
      new Document({ content: [{ text: 'doc2' }] }),
      new Document({ content: [{ text: 'doc3' }] }),
    ];

    await indexerHandler(docs);

    // Batched: single call to embedder with all docs
    expect(mockEmbedderAction).toHaveBeenCalledTimes(1);
    expect(mockBatchHset).toHaveBeenCalledTimes(3);
    expect(mockExec).toHaveBeenCalledTimes(1);
  });
});

describe('valkeyRetriever', () => {
  let retrieverHandler: (content: any, options: any) => Promise<any>;

  beforeEach(async () => {
    jest.clearAllMocks();
    mockGlideFtCreate.mockResolvedValue('OK');

    const plugin = valkeyPlugin([
      {
        indexName: 'test-index',
        embedder: mockEmbedder,
        dimension: 3,
        clientConfig: { addresses: [{ host: 'localhost', port: 6379 }] },
      },
    ]);

    const pluginInstance = (plugin.plugin as any)(mockGenkit);
    await pluginInstance.initializer();

    retrieverHandler = mockDefineRetriever.mock.calls[0][1] as any;
  });

  test('should embed query and call GlideFt.search with KNN', async () => {
    mockEmbed.mockResolvedValue([{ embedding: [0.1, 0.2, 0.3] }]);
    mockGlideFtSearch.mockResolvedValue([0, []]);

    await retrieverHandler('What is Valkey?', { k: 5 });

    expect(mockEmbed).toHaveBeenCalledWith(
      expect.objectContaining({
        embedder: mockEmbedder,
        content: 'What is Valkey?',
      })
    );
    expect(mockGlideFtSearch).toHaveBeenCalledWith(
      mockClient,
      'test-index',
      '*=>[KNN $k @embedding $query_vec]',
      expect.objectContaining({
        params: expect.arrayContaining([
          expect.objectContaining({ key: 'k', value: '5' }),
          expect.objectContaining({ key: 'query_vec' }),
        ]),
        returnFields: expect.arrayContaining([
          expect.objectContaining({ fieldIdentifier: '_content' }),
          expect.objectContaining({ fieldIdentifier: '_metadata' }),
          expect.objectContaining({ fieldIdentifier: '_dataType' }),
        ]),
      })
    );
  });

  test('should parse search results into Document objects', async () => {
    mockEmbed.mockResolvedValue([{ embedding: [0.1, 0.2, 0.3] }]);
    mockGlideFtSearch.mockResolvedValue([
      2,
      [
        {
          key: 'test-index:abc123',
          value: [
            { key: '_content', value: 'Valkey is a key-value store.' },
            { key: '_metadata', value: '{"source":"docs"}' },
            { key: '_dataType', value: 'text' },
            { key: '__embedding_score', value: '0.95' },
          ],
        },
        {
          key: 'test-index:def456',
          value: [
            { key: '_content', value: 'ElastiCache supports Valkey.' },
            { key: '_metadata', value: '{}' },
            { key: '_dataType', value: '' },
            { key: '__embedding_score', value: '0.85' },
          ],
        },
      ],
    ]);

    const result = await retrieverHandler('What is Valkey?', { k: 2 });

    expect(result.documents).toHaveLength(2);
    expect(result.documents[0]).toBeInstanceOf(Document);
    expect(result.documents[1]).toBeInstanceOf(Document);
  });

  test('should handle Buffer values in search results', async () => {
    mockEmbed.mockResolvedValue([{ embedding: [0.1, 0.2, 0.3] }]);
    mockGlideFtSearch.mockResolvedValue([
      1,
      [
        {
          key: Buffer.from('test-index:abc123'),
          value: [
            { key: Buffer.from('_content'), value: Buffer.from('Hello') },
            { key: Buffer.from('_metadata'), value: Buffer.from('{}') },
            { key: Buffer.from('_dataType'), value: Buffer.from('text') },
          ],
        },
      ],
    ]);

    const result = await retrieverHandler('query', { k: 1 });

    expect(result.documents).toHaveLength(1);
    expect(result.documents[0]).toBeInstanceOf(Document);
  });

  test('should default k to 10 when not specified', async () => {
    mockEmbed.mockResolvedValue([{ embedding: [0.1, 0.2, 0.3] }]);
    mockGlideFtSearch.mockResolvedValue([0, []]);

    await retrieverHandler('query', {});

    expect(mockGlideFtSearch).toHaveBeenCalledWith(
      mockClient,
      'test-index',
      expect.any(String),
      expect.objectContaining({
        params: expect.arrayContaining([
          expect.objectContaining({ key: 'k', value: '10' }),
        ]),
      })
    );
  });

  test('should handle empty search results', async () => {
    mockEmbed.mockResolvedValue([{ embedding: [0.1, 0.2, 0.3] }]);
    mockGlideFtSearch.mockResolvedValue([0, []]);

    const result = await retrieverHandler('no results query', { k: 5 });

    expect(result.documents).toHaveLength(0);
  });

  test('should handle invalid metadata JSON gracefully', async () => {
    mockEmbed.mockResolvedValue([{ embedding: [0.1, 0.2, 0.3] }]);
    mockGlideFtSearch.mockResolvedValue([
      1,
      [
        {
          key: 'test-index:abc',
          value: [
            { key: '_content', value: 'content' },
            { key: '_metadata', value: 'not-valid-json' },
            { key: '_dataType', value: '' },
          ],
        },
      ],
    ]);

    const result = await retrieverHandler('query', { k: 1 });

    expect(result.documents).toHaveLength(1);
    // Should not throw, metadata should be undefined
    expect(result.documents[0]).toBeInstanceOf(Document);
  });
});

describe('valkeyRetrieverRef', () => {
  test('should create a retriever reference with correct name', () => {
    const ref = valkeyRetrieverRef({ indexName: 'my-docs' });
    expect(ref.name).toBe('valkey/my-docs');
  });

  test('should use custom display name', () => {
    const ref = valkeyRetrieverRef({
      indexName: 'my-docs',
      displayName: 'My Docs Retriever',
    });
    expect(ref.info?.label).toBe('My Docs Retriever');
  });

  test('should use default display name', () => {
    const ref = valkeyRetrieverRef({ indexName: 'my-docs' });
    expect(ref.info?.label).toBe('Valkey - my-docs');
  });
});

describe('valkeyIndexerRef', () => {
  test('should create an indexer reference with correct name', () => {
    const ref = valkeyIndexerRef({ indexName: 'my-docs' });
    expect(ref.name).toBe('valkey/my-docs');
  });

  test('should use custom display name', () => {
    const ref = valkeyIndexerRef({
      indexName: 'my-docs',
      displayName: 'My Docs Indexer',
    });
    expect(ref.info?.label).toBe('My Docs Indexer');
  });
});

describe('stableDocId', () => {
  test('same doc with nested metadata in different key orders produces same ID', () => {
    const id1 = stableDocId({
      data: 'hello',
      metadata: { a: 1, b: 2 },
      dataType: 'text',
    });
    const id2 = stableDocId({
      data: 'hello',
      metadata: { b: 2, a: 1 },
      dataType: 'text',
    });
    expect(id1).toBe(id2);
  });

  test('docs with different metadata produce different IDs', () => {
    const id1 = stableDocId({
      data: 'hello',
      metadata: { a: 1 },
      dataType: 'text',
    });
    const id2 = stableDocId({
      data: 'hello',
      metadata: { a: 2 },
      dataType: 'text',
    });
    expect(id1).not.toBe(id2);
  });

  test('metadata content is included in hash (not dropped)', () => {
    const idWithMeta = stableDocId({
      data: 'hello',
      metadata: { key: 'value' },
      dataType: 'text',
    });
    const idNoMeta = stableDocId({
      data: 'hello',
      metadata: {},
      dataType: 'text',
    });
    expect(idWithMeta).not.toBe(idNoMeta);
  });

  test('cross-language test vector matches Go and Python', () => {
    // Canonical JSON: {"data":"hello","dataType":"text","metadata":null}
    // MD5: 3c04c6b9f04e5e522404b4c567ad09b0
    const id = stableDocId({ data: 'hello', dataType: 'text' });
    expect(id).toBe('3c04c6b9f04e5e522404b4c567ad09b0');
  });
});

describe('validateFilterExpression', () => {
  test('valid filter expressions do not throw', () => {
    expect(() => validateFilterExpression('@price:[100 200]')).not.toThrow();
    expect(() => validateFilterExpression('@tag:{foo}')).not.toThrow();
    expect(() => validateFilterExpression('*')).not.toThrow();
    expect(() => validateFilterExpression('')).not.toThrow();
  });

  test.each([';', '|', '`', '$', '\\'])('blocks disallowed char %s', (char) => {
    expect(() =>
      validateFilterExpression(`@field:[0 10]${char}inject`)
    ).toThrow('disallowed characters');
  });

  test('blocks => KNN query injection sequence', () => {
    expect(() =>
      validateFilterExpression('@tag:{x})=>[KNN 99999 @embedding $query_vec]')
    ).toThrow('disallowed characters');
    expect(() => validateFilterExpression('foo=>bar')).toThrow(
      'disallowed characters'
    );
  });
});
