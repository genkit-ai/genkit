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

import { GenkitError, z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { defineReranker, rerank, reranker } from '../../src/reranker';
import { Document } from '../../src/retriever';

initNodeFeatures();

describe('reranker', () => {
  describe('defineReranker()', () => {
    let registry: Registry;
    beforeEach(() => {
      registry = new Registry();
    });
    it('reranks documents based on custom logic', async () => {
      const customReranker = defineReranker(
        registry,
        {
          name: 'reranker',
          configSchema: z.object({
            k: z.number().optional(),
          }),
        },
        async (query, documents, options) => {
          // Custom reranking logic: score based on string length similarity to query
          const queryLength = query.text.length;
          const rerankedDocs = documents.map((doc) => {
            const score = Math.abs(queryLength - doc.text.length);
            return {
              ...doc,
              metadata: { ...doc.metadata, score },
            };
          });

          return {
            documents: rerankedDocs
              .sort((a, b) => a.metadata.score - b.metadata.score)
              .slice(0, options.k || 3),
          };
        }
      );
      // Sample documents for testing
      const documents = [
        Document.fromText('short'),
        Document.fromText('a bit longer'),
        Document.fromText('this is a very long document'),
      ];

      const query = Document.fromText('medium length');
      const rerankedDocuments = await rerank(registry, {
        reranker: customReranker,
        query,
        documents,
        options: { k: 2 },
      });
      // Validate the reranked results
      assert.equal(rerankedDocuments.length, 2);
      assert.ok(rerankedDocuments[0].text.includes('a bit longer'));
      assert.ok(rerankedDocuments[1].text.includes('short'));
    });

    it('reranks documents based on custom logic using dynamically defined reranker', async () => {
      const customReranker = reranker(
        {
          name: 'reranker',
          configSchema: z.object({
            k: z.number().optional(),
          }),
        },
        async (query, documents, options) => {
          // Custom reranking logic: score based on string length similarity to query
          const queryLength = query.text.length;
          const rerankedDocs = documents.map((doc) => {
            const score = Math.abs(queryLength - doc.text.length);
            return {
              ...doc,
              metadata: { ...doc.metadata, score },
            };
          });

          return {
            documents: rerankedDocs
              .sort((a, b) => a.metadata.score - b.metadata.score)
              .slice(0, options.k || 3),
          };
        }
      );
      // Sample documents for testing
      const documents = [
        Document.fromText('short'),
        Document.fromText('a bit longer'),
        Document.fromText('this is a very long document'),
      ];

      const query = Document.fromText('medium length');
      const rerankedDocuments = await rerank(registry, {
        reranker: customReranker,
        query,
        documents,
        options: { k: 2 },
      });
      // Validate the reranked results
      assert.equal(rerankedDocuments.length, 2);
      assert.ok(rerankedDocuments[0].text.includes('a bit longer'));
      assert.ok(rerankedDocuments[1].text.includes('short'));
    });

    it('handles missing options gracefully', async () => {
      const customReranker = defineReranker(
        registry,
        {
          name: 'reranker',
          configSchema: z.object({
            k: z.number().optional(),
          }),
        },
        async (query, documents, options) => {
          const rerankedDocs = documents.map((doc) => {
            const score = Math.random(); // Simplified scoring for testing
            return {
              ...doc,
              metadata: { ...doc.metadata, score },
            };
          });

          return {
            documents: rerankedDocs.sort(
              (a, b) => b.metadata.score - a.metadata.score
            ),
          };
        }
      );
      const documents = [Document.fromText('doc1'), Document.fromText('doc2')];

      const query = Document.fromText('test query');
      const rerankedDocuments = await rerank(registry, {
        reranker: customReranker,
        query,
        documents,
        options: { k: 2 },
      });
      assert.equal(rerankedDocuments.length, 2);
      assert.ok(typeof rerankedDocuments[0].metadata.score === 'number');
    });

    it('validates config schema and throws error on invalid input', async () => {
      const customReranker = defineReranker(
        registry,
        {
          name: 'reranker',
          configSchema: z.object({
            k: z.number().min(1),
          }),
        },
        async (query, documents, options) => {
          // Simplified scoring for testing
          const rerankedDocs = documents.map((doc) => ({
            ...doc,
            metadata: { score: Math.random() },
          }));
          return {
            documents: rerankedDocs.sort(
              (a, b) => b.metadata.score - a.metadata.score
            ),
          };
        }
      );
      const documents = [Document.fromText('doc1')];

      const query = Document.fromText('test query');

      try {
        await rerank(registry, {
          reranker: customReranker,
          query,
          documents,
          options: { k: 0 }, // Invalid input: k must be at least 1
        });
        assert.fail('Expected validation error');
      } catch (err) {
        assert.ok(err instanceof GenkitError);
        assert.equal(err.status, 'INVALID_ARGUMENT');
      }
    });

    it('preserves document metadata after reranking', async () => {
      const customReranker = defineReranker(
        registry,
        {
          name: 'reranker',
        },
        async (query, documents) => {
          const rerankedDocs = documents.map((doc, i) => ({
            ...doc,
            metadata: { ...doc.metadata, score: 2 - i },
          }));

          return {
            documents: rerankedDocs.sort(
              (a, b) => b.metadata.score - a.metadata.score
            ),
          };
        }
      );
      const documents = [
        new Document({ content: [], metadata: { originalField: 'test1' } }),
        new Document({ content: [], metadata: { originalField: 'test2' } }),
      ];

      const query = Document.fromText('test query');
      const rerankedDocuments = await rerank(registry, {
        reranker: customReranker,
        query,
        documents,
      });
      assert.equal(rerankedDocuments[0].metadata.originalField, 'test1');
      assert.equal(rerankedDocuments[1].metadata.originalField, 'test2');
    });

    it('handles errors thrown by the reranker', async () => {
      const customReranker = defineReranker(
        registry,
        {
          name: 'reranker',
        },
        async (query, documents) => {
          // Simulate an error in the reranker logic
          throw new GenkitError({
            message: 'Something went wrong during reranking',
            status: 'INTERNAL',
          });
        }
      );
      const documents = [Document.fromText('doc1'), Document.fromText('doc2')];
      const query = Document.fromText('test query');

      try {
        await rerank(registry, {
          reranker: customReranker,
          query,
          documents,
        });
        assert.fail('Expected an error to be thrown');
      } catch (err) {
        assert.ok(err instanceof GenkitError);
        assert.equal(err.status, 'INTERNAL');
        assert.equal(
          err.message,
          'INTERNAL: Something went wrong during reranking'
        );
      }
    });
  });
});
