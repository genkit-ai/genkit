/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  jest,
  test,
} from '@jest/globals';

// Mocks for Genkit-specific components
import type { Genkit } from 'genkit';
import { z } from 'genkit';
import { EmbedderArgument } from 'genkit/embedder';
import { Document } from 'genkit/retriever';

// Import your actual source code
import { randomUUID } from 'node:crypto';
import { Column, PostgresEngine } from '../src/engine';
import {
  configurePostgresIndexer,
  configurePostgresRetriever,
} from '../src/index';
import { DistanceStrategy } from '../src/indexes';

// Load environment variables
import * as dotenv from 'dotenv';
dotenv.config();

// --- Constants for Test Setup ---
const TEST_TABLE = 'test_embeddings_integration';
const SCHEMA_NAME = 'test_schema_integration';
const ID_COLUMN = 'uuid';
const CONTENT_COLUMN = 'my_content';
const EMBEDDING_COLUMN = 'my_embedding';
// Assuming Column class exists and is structured like this
const METADATA_COLUMNS = [
  new Column('page', 'TEXT'),
  new Column('source', 'TEXT'),
];

// --- Mocks for Genkit Interactions ---
// Mock Genkit instance for `ai` parameter
const mockGenkit: Genkit = {
  embed: jest.fn(async ({ content }) => {
    let embeddingArray = new Array(1536).fill(0.05);
    if (content === 'first document content') {
      embeddingArray = new Array(1536).fill(0.1);
    } else if (content === 'another document') {
      embeddingArray = new Array(1536).fill(0.3);
    }
    return [{ embedding: embeddingArray, text: content }];
  }),

  embedMany: jest.fn(async ({ content }) => {
    return content.map((text) => ({
      embedding: new Array(1536).fill(0.1),
      text: Array.isArray(text) ? text.join(' ') : text,
    }));
  }),

  defineRetriever: jest.fn((config, handler) => ({
    config,
    retrieve: handler,
  })),

  defineIndexer: jest.fn(
    (
      config,
      handler: (
        input: { documents: Document[]; options?: any },
        options?: any
      ) => Promise<void>
    ) => ({
      config,
      index: (input: any, options?: any) => handler(input, options),
    })
  ),
} as unknown as Genkit;

// Mock EmbedderArgument as configured in Genkit plugin
const mockEmbedder: EmbedderArgument<z.ZodTypeAny> = {
  name: 'mock-embedder',
  type: 'embedder',
  configSchema: z.object({}),
} as any;

// Spy on console.log to capture output
const consoleSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

const REQUIRED_ENV_VARS = [
  'PROJECT_ID',
  'REGION',
  'INSTANCE_ID',
  'DATABASE_ID',
  'DB_USER',
  'DB_PASSWORD',
  'IP_ADDRESS',
  'DB_HOST',
];

let envVarsFound = true;
let missingEnvVars: string[] = [];

function validateEnvVars() {
  missingEnvVars = REQUIRED_ENV_VARS.filter((varName) => !process.env[varName]);
  if (missingEnvVars.length > 0) {
    envVarsFound = false;
    console.warn(
      `Skipping tests due to missing environment variables: ${missingEnvVars.join(', ')}`
    );
  }
}

// Validate environment variables once at the top level
validateEnvVars();

describe('configurePostgresRetriever Integration Tests', () => {
  // Conditionally skip the entire describe block
  if (!envVarsFound) {
    describe.skip('configurePostgresRetriever Integration Tests (skipped due to missing env vars)', () => {
      test('this test will be skipped', () => {
        // This block won't execute
      });
    });
    return; // Exit the describe block early
  }

  let engine: PostgresEngine;
  let retrieverInstance: any;

  beforeAll(async () => {
    // Initialize PostgresEngine
    try {
      engine = await PostgresEngine.fromEngineArgs({
        user: process.env.DB_USER!,
        password: process.env.DB_PASSWORD!,
        host: process.env.DB_HOST!,
        database: process.env.DATABASE_ID!,
        port: parseInt(process.env.DB_PORT || '5432'),
      });

      // Ensure database connection is established before proceeding
      await engine.pool.raw('SELECT 1;');

      // Create test schema and table
      await engine.pool.raw(`CREATE SCHEMA IF NOT EXISTS ${SCHEMA_NAME}`);
      await engine.pool.raw(
        `DROP TABLE IF EXISTS ${SCHEMA_NAME}.${TEST_TABLE}`
      );

      // Initialize the vectorstore table
      await engine.initVectorstoreTable(TEST_TABLE, 1536, {
        // Using 1536 as typical embedding size
        schemaName: SCHEMA_NAME,
        contentColumn: CONTENT_COLUMN,
        embeddingColumn: EMBEDDING_COLUMN,
        metadataColumns: METADATA_COLUMNS,
        metadataJsonColumn: 'metadata_json',
        idColumn: ID_COLUMN,
        overwriteExisting: true,
        storeMetadata: true,
      });

      // --- Populate Test Data ---
      // Insert some test documents into the table
      const doc1Id = randomUUID();
      const doc2Id = randomUUID();
      const doc3Id = randomUUID();
      await engine.pool.raw(`
       INSERT INTO ${SCHEMA_NAME}.${TEST_TABLE} (${ID_COLUMN}, ${CONTENT_COLUMN}, ${EMBEDDING_COLUMN}, page, source, metadata_json)
       VALUES
       ('${doc1Id}', 'This is the first test document content.', '[${new Array(1536).fill(0.1).join(',')}]', 'page1', 'sourceA', '{"key1": "value1"}'),
       ('${doc2Id}', 'Second document with different content.', '[${new Array(1536).fill(0.2).join(',')}]', 'page2', 'sourceB', '{"key2": "value2"}'),
       ('${doc3Id}', 'Another test document.', '[${new Array(1536).fill(0.3).join(',')}]', 'page1', 'sourceC', '{"key3": "value3"}');
     `);

      // Configure the Postgres Retriever using `configurePostgresRetriever`
      await configurePostgresRetriever(mockGenkit, {
        tableName: TEST_TABLE,
        engine: engine,
        schemaName: SCHEMA_NAME,
        contentColumn: CONTENT_COLUMN,
        embeddingColumn: EMBEDDING_COLUMN,
        idColumn: ID_COLUMN,
        metadataColumns: METADATA_COLUMNS.map((col) => col.name), // Pass metadata columns defined
        metadataJsonColumn: 'metadata_json', // Ensure this matches the table column name
        embedder: mockEmbedder,
        distanceStrategy: DistanceStrategy.COSINE_DISTANCE, // Explicitly set or use default
      });

      // Get the configured retriever instance that Genkit would use
      // This requires accessing the mock's internal state
      const defineRetrieverMock = mockGenkit.defineRetriever as jest.Mock;
      expect(defineRetrieverMock).toHaveBeenCalledWith(
        expect.objectContaining({
          name: `postgres/${TEST_TABLE}`,
        }),
        expect.any(Function)
      );
      retrieverInstance = defineRetrieverMock.mock.results[0].value;
    } catch (error) {
      console.error('Error during beforeAll setup:', error);
      if (error instanceof AggregateError) {
        console.error('Individual errors in AggregateError:');
        error.errors.forEach((e: any, index: number) => {
          console.error(`Error ${index + 1}:`, e);
          if (e.stack) {
            console.error('Stack:', e.stack);
          }
        });
      }
      // Re-throw the error so Jest still marks the test as failed
      throw error;
    }
  }, 60000); // Increased timeout for database operations

  afterEach(() => {
    jest.clearAllMocks(); // Clear mock call history after each test
  });

  afterAll(async () => {
    // Clean up the test table and schema after all tests are done
    if (engine?.pool) {
      await engine.pool.raw(
        `DROP TABLE IF EXISTS ${SCHEMA_NAME}.${TEST_TABLE}`
      );
      await engine.pool.raw(`DROP SCHEMA IF EXISTS ${SCHEMA_NAME} CASCADE`);
      await engine.closeConnection(); // Close the database connection pool
    }
    consoleSpy.mockRestore(); // Restore console.log
  }, 30000); // Increased timeout for cleanup

  test('should retrieve relevant documents based on a query', async () => {
    const queryText = 'first document content';
    const options = { k: 2 }; // Retrieve top 2 documents

    // Call the retrieve method on the configured retriever instance
    const result = await retrieverInstance.retrieve(queryText, options);

    expect(mockGenkit.embed).toHaveBeenCalledWith(
      expect.objectContaining({ content: queryText })
    );
    //expect(engine.pool.raw).toHaveBeenCalled();

    expect(result).toBeDefined();
    expect(result.documents).toHaveLength(2);

    // Verify the structure and content of the retrieved documents
    expect(result.documents[0]).toBeInstanceOf(Document);
    expect(result.documents[0].content).toBeDefined();
    expect(result.documents[0].metadata).toBeDefined();

    // Check content and metadata of the first retrieved document (assuming order from mock data)
    expect(result.documents[0].content).toContain('first test document');
    expect(result.documents[0].metadata).toEqual(
      expect.objectContaining({
        page: 'page1',
        source: 'sourceA',
        key1: 'value1', // From metadata_json
      })
    );
  });

  test('should retrieve documents with specific filter criteria', async () => {
    const queryText = 'another document';
    const options = { k: 1, filter: "page = 'page2'" };

    const result = await retrieverInstance.retrieve(queryText, options);

    expect(result).toBeDefined();
    expect(result.documents).toHaveLength(1); // Should only get the doc with page2

    expect(result.documents[0].content).toContain(
      'Second document with different content'
    );
    expect(result.documents[0].metadata).toEqual(
      expect.objectContaining({
        page: 'page2',
        source: 'sourceB',
        key2: 'value2',
      })
    );

    // Verify the SQL query contains the filter (implementation detail, but useful for integration)
    // const rawCalls = (engine.pool.raw as jest.Mock).mock.calls;
    // const selectQueryCall = rawCalls.find(call => typeof call[0] === 'string' && call[0].includes('SELECT') && call[0].includes('WHERE page = \'page2\''));
    // expect(selectQueryCall).toBeDefined();
  });

  test('should handle empty query text gracefully', async () => {
    const queryText = '';
    const options = { k: 1 };

    // Expect `ai.embed` to be called with empty string, and the retriever to return results (or empty results if no matching embeddings)
    const result = await retrieverInstance.retrieve(queryText, options);

    expect(mockGenkit.embed).toHaveBeenCalledWith(
      expect.objectContaining({ content: '' })
    );
    expect(result).toBeDefined();
    expect(result.documents).toHaveLength(1); // Still expects a document from the mock
  });

  // - Different `k` values
  // - Empty results
  // - Error scenarios from `engine.pool.raw` (e.g., network issues)
  // - Testing `ignoreMetadataColumns` and `metadataJsonColumn` interactions
  // - Testing `DistanceStrategy` variations
});

describe('configurePostgresIndexer Integration Tests', () => {
  // Conditionally skip the entire describe block
  if (!envVarsFound) {
    describe.skip('configurePostgresIndexer Integration Tests (skipped due to missing env vars)', () => {
      test('this test will be skipped', () => {
        // This block won't execute
      });
    });
    return; // Exit the describe block early
  }

  let engine: PostgresEngine;
  let indexerInstance: any;

  beforeAll(async () => {
    try {
      engine = await PostgresEngine.fromEngineArgs({
        user: process.env.DB_USER!,
        password: process.env.DB_PASSWORD!,
        host: process.env.DB_HOST!,
        database: process.env.DATABASE_ID!,
        port: parseInt(process.env.DB_PORT || '5432'),
      });

      await engine.pool.raw(`CREATE SCHEMA IF NOT EXISTS ${SCHEMA_NAME}`);
      await engine.pool.raw(
        `DROP TABLE IF EXISTS ${SCHEMA_NAME}.${TEST_TABLE}`
      );

      await engine.initVectorstoreTable(TEST_TABLE, 1536, {
        schemaName: SCHEMA_NAME,
        contentColumn: CONTENT_COLUMN,
        embeddingColumn: EMBEDDING_COLUMN,
        metadataColumns: METADATA_COLUMNS,
        metadataJsonColumn: 'metadata_json',
        idColumn: ID_COLUMN,
        overwriteExisting: true,
        storeMetadata: true,
      });

      configurePostgresIndexer(mockGenkit, {
        tableName: TEST_TABLE,
        engine: engine,
        schemaName: SCHEMA_NAME,
        contentColumn: CONTENT_COLUMN,
        embeddingColumn: EMBEDDING_COLUMN,
        idColumn: ID_COLUMN,
        metadataColumns: METADATA_COLUMNS.map((col) => col.name),
        metadataJsonColumn: 'metadata_json',
        embedder: mockEmbedder,
      });

      const defineIndexerMock = mockGenkit.defineIndexer as jest.Mock;
      indexerInstance = defineIndexerMock.mock.results[0].value;
    } catch (error) {
      console.error('Error during beforeAll setup:', error);
      throw error;
    }
  }, 60000);

  beforeEach(async () => {
    // Clear table before each test
    if (engine?.pool) {
      // Add a check here for engine and pool
      await engine.pool.raw(`TRUNCATE TABLE ${SCHEMA_NAME}.${TEST_TABLE}`);
    }
    jest.clearAllMocks();
  });

  afterAll(async () => {
    if (engine?.pool) {
      await engine.pool.raw(
        `DROP TABLE IF EXISTS ${SCHEMA_NAME}.${TEST_TABLE}`
      );
      await engine.pool.raw(`DROP SCHEMA IF EXISTS ${SCHEMA_NAME} CASCADE`);
      await engine.closeConnection();
    }
    consoleSpy.mockRestore();
  }, 30000);

  test('should index a single document correctly', async () => {
    const testDoc = new Document({
      content: [{ text: 'This is the first test document content.' }],
      metadata: { page: 'page1', source: 'sourceA', key1: 'value1' },
    });

    await indexerInstance.index({ documents: [testDoc], options: {} });

    const result = await engine.pool
      .withSchema(SCHEMA_NAME)
      .select([ID_COLUMN, CONTENT_COLUMN, EMBEDDING_COLUMN, 'metadata_json'])
      .from(TEST_TABLE);

    expect(result).toHaveLength(1);
    expect(result[0][CONTENT_COLUMN]).toBe(
      'This is the first test document content.'
    );
    expect(result[0]['metadata_json']).toEqual(
      expect.objectContaining({
        page: 'page1',
        source: 'sourceA',
        key1: 'value1',
      })
    );
    expect(result[0][EMBEDDING_COLUMN]).toBeDefined();
  });

  test('should index multiple documents in batch', async () => {
    const testDocs = [
      new Document({
        content: [{ text: 'Second document with different content.' }],
        metadata: { page: 'page2', source: 'sourceB', key2: 'value2' },
      }),
      new Document({
        content: [{ text: 'Another test document.' }],
        metadata: { page: 'page1', source: 'sourceC', key3: 'value3' },
      }),
    ];

    await indexerInstance.index({
      documents: testDocs,
      options: { batchSize: 2 },
    });

    const result = await engine.pool
      .withSchema(SCHEMA_NAME)
      .select([ID_COLUMN, CONTENT_COLUMN, EMBEDDING_COLUMN, 'metadata_json'])
      .from(TEST_TABLE)
      .orderBy(CONTENT_COLUMN);

    expect(result).toHaveLength(2);
    expect(result[0][CONTENT_COLUMN]).toBe('Another test document.');
    expect(result[1][CONTENT_COLUMN]).toBe(
      'Second document with different content.'
    );
  });

  test('should handle empty document array gracefully', async () => {
    await expect(
      indexerInstance.index({ documents: [], options: {} })
    ).resolves.not.toThrow();

    const result = await engine.pool
      .withSchema(SCHEMA_NAME)
      .select('*')
      .from(TEST_TABLE);

    expect(result).toHaveLength(0);
  });

  test('should use existing ID from metadata if provided', async () => {
    const testId = randomUUID();
    const testDoc = new Document({
      content: [{ text: 'Test with ID' }],
      metadata: { [ID_COLUMN]: testId, source: 'id-test' },
    });

    await indexerInstance.index({ documents: [testDoc], options: {} });

    const result = await engine.pool
      .withSchema(SCHEMA_NAME)
      .select(ID_COLUMN)
      .from(TEST_TABLE)
      .where(ID_COLUMN, testId)
      .first();

    expect(result).toBeDefined();
    expect(result[ID_COLUMN]).toBe(testId);
  });

  test('should throw error when engine is not provided', () => {
    expect(() => {
      configurePostgresIndexer(mockGenkit, {
        tableName: TEST_TABLE,
        // @ts-expect-error - Testing invalid input
        engine: undefined,
        contentColumn: CONTENT_COLUMN,
        embeddingColumn: EMBEDDING_COLUMN,
        idColumn: ID_COLUMN,
        embedder: mockEmbedder,
      });
    }).toThrow('Engine is required');
  });

  test('should throw error when both metadataColumns and ignoreMetadataColumns are provided', () => {
    expect(() => {
      configurePostgresIndexer(mockGenkit, {
        tableName: TEST_TABLE,
        engine,
        contentColumn: CONTENT_COLUMN,
        embeddingColumn: EMBEDDING_COLUMN,
        idColumn: ID_COLUMN,
        metadataColumns: ['col1'],
        ignoreMetadataColumns: ['col2'],
        embedder: mockEmbedder,
      });
    }).toThrow('Cannot use both metadataColumns and ignoreMetadataColumns');
  });

  test('should handle embedding generation errors', async () => {
    const errorMockGenkit = {
      ...mockGenkit,
      embedMany: jest.fn(() => {
        throw new Error('Embedding failed');
      }),
    } as unknown as Genkit;

    configurePostgresIndexer(errorMockGenkit, {
      tableName: TEST_TABLE,
      engine,
      schemaName: SCHEMA_NAME,
      contentColumn: CONTENT_COLUMN,
      embeddingColumn: EMBEDDING_COLUMN,
      idColumn: ID_COLUMN,
      embedder: mockEmbedder,
    });

    const defineIndexerMock = errorMockGenkit.defineIndexer as jest.Mock;
    const errorIndexerInstance = defineIndexerMock.mock.results[0].value as {
      index: (args: { documents: Document[]; options: {} }) => Promise<void>;
    };

    await expect(
      errorIndexerInstance.index({
        documents: [new Document({ content: [{ text: 'Fail me' }] })],
        options: {},
      })
    ).rejects.toThrow('Embedding failed');
  });
});
