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
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import type { PineconeConfiguration } from '@pinecone-database/pinecone';
import { genkit } from 'genkit';
import { Document } from 'genkit/retriever';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  configurePineconeIndexer,
  configurePineconeRetriever,
  pineconeIndexerRef,
  pineconeRetrieverRef,
} from '../src/index.js';

test('indexes and retrieves through the Pinecone SDK transport', async () => {
  const calls: Array<{ url: string; body: unknown }> = [];
  const indexName = 'sdk-compat-test';
  const indexHost = 'sdk-compat-test.svc.pinecone.io';
  const clientParams: PineconeConfiguration = {
    apiKey: 'test-api-key',
    controllerHostUrl: 'https://api.test.pinecone.io',
    fetchApi: async (input, init) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ url, body });
      const response = (data: unknown) =>
        new Response(JSON.stringify(data), {
          headers: { 'content-type': 'application/json' },
        });

      if (url === `https://api.test.pinecone.io/indexes/${indexName}`) {
        return response({
          name: indexName,
          host: indexHost,
          dimension: 3,
          metric: 'cosine',
          spec: { serverless: { cloud: 'aws', region: 'us-east-1' } },
          status: { ready: true, state: 'Ready' },
          vector_type: 'dense',
        });
      }
      if (url === `https://${indexHost}/vectors/upsert`) {
        return response({ upsertedCount: 1 });
      }
      if (url === `https://${indexHost}/query`) {
        return response({
          matches: [
            {
              id: 'record-1',
              score: 1,
              metadata: {
                _content: 'A test document',
                _contentType: 'text',
                docMetadata: JSON.stringify({ source: 'test' }),
              },
            },
          ],
          namespace: 'team',
          usage: { readUnits: 1 },
        });
      }
      throw new Error(`Unexpected Pinecone request: ${url}`);
    },
  };

  const ai = genkit({});
  const embedder = ai.defineEmbedder({ name: 'testEmbedder' }, async () => ({
    embeddings: [{ embedding: [0.1, 0.2, 0.3] }],
  }));
  configurePineconeIndexer(ai, { indexId: indexName, clientParams, embedder });
  configurePineconeRetriever(ai, {
    indexId: indexName,
    clientParams,
    embedder,
  });

  await ai.index({
    indexer: pineconeIndexerRef({ indexId: indexName }),
    documents: [Document.fromText('A test document', { source: 'test' })],
    options: { namespace: 'team' },
  });
  const documents = await ai.retrieve({
    retriever: pineconeRetrieverRef({ indexId: indexName }),
    query: 'test',
    options: { k: 1, namespace: 'team' },
  });

  assert.equal(documents[0].text, 'A test document');
  assert.deepEqual(documents[0].metadata, { source: 'test' });
  const upsert = calls.find((call) => call.url.endsWith('/vectors/upsert'));
  const query = calls.find((call) => call.url.endsWith('/query'));
  const upsertBody = upsert?.body as {
    vectors: Array<{
      id: string;
      values: number[];
      metadata: Record<string, string>;
    }>;
    namespace: string;
  };
  assert.equal(upsertBody.vectors.length, 1);
  assert.ok(upsertBody.vectors[0].id);
  assert.deepEqual(upsertBody.vectors[0].values, [0.1, 0.2, 0.3]);
  assert.equal(upsertBody.vectors[0].metadata._content, 'A test document');
  assert.equal(upsertBody.namespace, 'team');
  assert.equal((query?.body as { namespace: string }).namespace, 'team');
});
