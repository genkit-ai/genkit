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
import * as assert from 'assert';
import { describe, it } from 'node:test';
import { voyageai } from '../src/index.js';
import { embeddingsBody, setup } from './helpers.js';

const png =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

describe('voyageai plugin', () => {
  it('refuses to start without an API key', async () => {
    const saved = process.env.VOYAGE_API_KEY;
    delete process.env.VOYAGE_API_KEY;
    try {
      const { genkit } = await import('genkit');
      const ai = genkit({ plugins: [voyageai()] });
      await assert.rejects(
        ai.embed({ embedder: 'voyageai/voyage-4', content: 'x' }),
        /API key is missing/
      );
    } finally {
      if (saved !== undefined) process.env.VOYAGE_API_KEY = saved;
    }
  });

  it('has no default model: unknown names fail instead of falling back', async () => {
    assert.throws(
      () => voyageai.embedder('nope'),
      /Unknown Voyage AI embedding model 'nope'/
    );
    assert.throws(
      () => voyageai.reranker('nope'),
      /Unknown Voyage AI reranker model 'nope'/
    );
    assert.throws(
      () => voyageai.embedder(''),
      /Unknown Voyage AI embedding model/
    );
    const { ai, requests } = setup(() => ({ body: embeddingsBody(1) }));
    await assert.rejects(
      ai.embed({ embedder: 'voyageai/voyage-9', content: 'x' }),
      /voyage-9/
    );
    assert.equal(requests.length, 0);
  });

  it('surfaces the Voyage detail text of API errors', async () => {
    const { ai } = setup(() => ({
      status: 400,
      body: { detail: 'Provided API key is invalid.' },
    }));
    await assert.rejects(
      ai.embed({ embedder: voyageai.embedder('voyage-4'), content: 'x' }),
      (e: any) => {
        assert.equal(e.status, 'INVALID_ARGUMENT');
        assert.match(e.message, /Provided API key is invalid\./);
        return true;
      }
    );
  });

  describe('text embedders', () => {
    it('sends input_type and output_dimension only when requested', async () => {
      const { ai, requests } = setup((r) => ({
        body: embeddingsBody(r.body.input.length),
      }));
      await ai.embed({ embedder: voyageai.embedder('voyage-4'), content: 'a' });
      assert.equal(requests[0].url.endsWith('/embeddings'), true);
      assert.equal('input_type' in requests[0].body, false);
      assert.equal('output_dimension' in requests[0].body, false);
      await ai.embed({
        embedder: voyageai.embedder('voyage-4'),
        content: 'a',
        options: { inputType: 'query', outputDimension: 256 },
      });
      assert.equal(requests[1].body.input_type, 'query');
      assert.equal(requests[1].body.output_dimension, 256);
      assert.equal(requests[1].body.model, 'voyage-4');
    });

    it('accepts null options', async () => {
      const { ai, requests } = setup((r) => ({
        body: embeddingsBody(r.body.input.length),
      }));
      const action = await ai.registry.lookupAction(
        '/embedder/voyageai/voyage-4'
      );
      await action({ input: [{ content: [{ text: 'a' }] }], options: null });
      assert.equal(requests.length, 1);
    });

    it('rejects unsupported output dimensions before any request', async () => {
      const { ai, requests } = setup(() => ({ body: embeddingsBody(1) }));
      await assert.rejects(
        ai.embed({
          embedder: voyageai.embedder('voyage-4'),
          content: 'a',
          options: { outputDimension: 128 },
        }),
        /does not support outputDimension 128/
      );
      await assert.rejects(
        ai.embed({
          embedder: voyageai.embedder('voyage-law-2'),
          content: 'a',
          options: { outputDimension: 512 },
        }),
        /supported: 1024/
      );
      assert.equal(requests.length, 0);
    });

    it('rejects bad options and media', async () => {
      const { ai, requests } = setup((r) => ({
        body: embeddingsBody(r.body.input.length),
      }));
      const embedder = voyageai.embedder('voyage-4');
      await assert.rejects(
        ai.embed({
          embedder,
          content: 'a',
          options: { inputType: 'passage' } as any,
        }),
        /Invalid options/
      );
      await assert.rejects(
        ai.embed({
          embedder,
          content: 'a',
          options: { dimensions: 256 } as any,
        }),
        /Invalid options/
      );
      await assert.rejects(
        ai.embed({
          embedder,
          content: {
            content: [{ media: { url: png, contentType: 'image/png' } }],
          },
        }),
        /text only/
      );
      assert.equal(requests.length, 0);
    });

    it('fails when Voyage returns fewer vectors than inputs', async () => {
      const { ai } = setup(() => ({ body: embeddingsBody(1) }));
      await assert.rejects(
        ai.embedMany({
          embedder: voyageai.embedder('voyage-4'),
          content: ['a', 'b'],
        }),
        /returned 1 embeddings for 2 inputs/
      );
    });
  });

  describe('contextualized embedders', () => {
    const chunked = (r: { body: any }) => ({
      body: {
        object: 'list',
        data: r.body.inputs.map((_: unknown, i: number) => ({
          object: 'list',
          index: i,
          data: [
            {
              object: 'embedding',
              embedding: [1, 2],
              index: 0,
              text: `doc${i} first`,
            },
            {
              object: 'embedding',
              embedding: [3, 4],
              index: 1,
              text: `doc${i} second`,
            },
          ],
        })),
        usage: { total_tokens: 9 },
      },
    });

    it('requires an explicit input type', async () => {
      const { ai, requests } = setup(chunked);
      await assert.rejects(
        ai.embed({
          embedder: voyageai.embedder('voyage-context-4'),
          content: 'a',
        }),
        /Invalid options/
      );
      assert.equal(requests.length, 0);
    });

    it('always auto-chunks documents with 32,000-token chunks and returns chunk texts', async () => {
      const { ai, requests } = setup(chunked);
      const out = await ai.embedMany({
        embedder: voyageai.embedder('voyage-context-3'),
        content: ['first document', 'second document'],
        options: { inputType: 'document', outputDimension: 256 },
      });
      const body = requests[0].body;
      assert.equal(requests[0].url.endsWith('/contextualizedembeddings'), true);
      assert.equal(body.enable_auto_chunking, true);
      assert.equal(body.chunk_size, 32000);
      assert.equal(body.input_type, 'document');
      assert.equal(body.output_dimension, 256);
      assert.equal(body.inputs.length, 2);
      assert.deepEqual(
        out.map((e) => e.metadata),
        [
          { documentIndex: 0, chunkIndex: 0, text: 'doc0 first' },
          { documentIndex: 0, chunkIndex: 1, text: 'doc0 second' },
          { documentIndex: 1, chunkIndex: 0, text: 'doc1 first' },
          { documentIndex: 1, chunkIndex: 1, text: 'doc1 second' },
        ]
      );
    });

    it('does not auto-chunk queries and returns no chunk text', async () => {
      const { ai, requests } = setup((r) => ({
        body: {
          data: r.body.inputs.map((_: unknown, i: number) => ({
            index: i,
            data: [{ embedding: [1, 2], index: 0 }],
          })),
          usage: { total_tokens: 3 },
        },
      }));
      const out = await ai.embedMany({
        embedder: voyageai.embedder('voyage-context-4'),
        content: ['q1', 'q2'],
        options: { inputType: 'query' },
      });
      const body = requests[0].body;
      assert.equal('enable_auto_chunking' in body, false);
      assert.equal('chunk_size' in body, false);
      assert.equal(body.input_type, 'query');
      assert.deepEqual(
        out.map((e) => e.metadata),
        [
          { documentIndex: 0, chunkIndex: 0 },
          { documentIndex: 1, chunkIndex: 0 },
        ]
      );
    });

    it('rejects pre-chunked documents, media and bad dimensions', async () => {
      const { ai, requests } = setup(chunked);
      const embedder = voyageai.embedder('voyage-context-4');
      await assert.rejects(
        ai.embed({
          embedder,
          content: { content: [{ text: 'chunk 1' }, { text: 'chunk 2' }] },
          options: { inputType: 'document' },
        }),
        /pre-chunked documents are not supported/
      );
      await assert.rejects(
        ai.embed({
          embedder,
          content: {
            content: [{ media: { url: png, contentType: 'image/png' } }],
          },
          options: { inputType: 'document' },
        }),
        /text only/
      );
      await assert.rejects(
        ai.embed({
          embedder,
          content: 'a',
          options: { inputType: 'document', outputDimension: 100 },
        }),
        /does not support outputDimension 100/
      );
      assert.equal(requests.length, 0);
    });
  });

  describe('multimodal embedders', () => {
    const embeddings = (r: { body: any }) => ({
      body: {
        ...embeddingsBody(r.body.inputs.length),
        usage: { text_tokens: 1, image_pixels: 1, total_tokens: 2 },
      },
    });

    it('sends interleaved text and media in order', async () => {
      const { ai, requests } = setup(embeddings);
      await ai.embed({
        embedder: voyageai.embedder('voyage-multimodal-3.5'),
        content: {
          content: [
            { text: 'a banana' },
            {
              media: {
                url: 'https://example.com/b.jpg',
                contentType: 'image/jpeg',
              },
            },
            { media: { url: 'https://example.com/b.mp4' } },
          ],
        },
        options: { inputType: 'document', truncation: false },
      });
      const body = requests[0].body;
      assert.equal(requests[0].url.endsWith('/multimodalembeddings'), true);
      assert.deepEqual(body.inputs, [
        {
          content: [
            { type: 'text', text: 'a banana' },
            { type: 'image_url', image_url: 'https://example.com/b.jpg' },
            { type: 'video_url', video_url: 'https://example.com/b.mp4' },
          ],
        },
      ]);
      assert.equal(body.input_type, 'document');
      assert.equal(body.truncation, false);
    });

    it('sends data URLs as base64 items', async () => {
      const { ai, requests } = setup(embeddings);
      await ai.embed({
        embedder: voyageai.embedder('voyage-multimodal-3'),
        content: {
          content: [{ media: { url: png, contentType: 'image/png' } }],
        },
      });
      assert.deepEqual(requests[0].body.inputs[0].content, [
        { type: 'image_base64', image_base64: png },
      ]);
    });

    it('rejects mixed URL and base64 media in one request', async () => {
      const { ai, requests } = setup(embeddings);
      await assert.rejects(
        ai.embedMany({
          embedder: voyageai.embedder('voyage-multimodal-3.5'),
          content: [
            { content: [{ media: { url: png, contentType: 'image/png' } }] },
            {
              content: [
                {
                  media: {
                    url: 'https://example.com/b.png',
                    contentType: 'image/png',
                  },
                },
              ],
            },
          ],
        }),
        /only media URLs or only base64/
      );
      assert.equal(requests.length, 0);
    });

    it('rejects video for voyage-multimodal-3, unsupported types and outputDimension', async () => {
      const { ai, requests } = setup(embeddings);
      await assert.rejects(
        ai.embed({
          embedder: voyageai.embedder('voyage-multimodal-3'),
          content: {
            content: [
              {
                media: {
                  url: 'https://example.com/b.mp4',
                  contentType: 'video/mp4',
                },
              },
            ],
          },
        }),
        /does not accept/
      );
      await assert.rejects(
        ai.embed({
          embedder: voyageai.embedder('voyage-multimodal-3.5'),
          content: {
            content: [
              {
                media: {
                  url: 'https://example.com/b.bmp',
                  contentType: 'image/bmp',
                },
              },
            ],
          },
        }),
        /unsupported media type 'image\/bmp'/
      );
      await assert.rejects(
        ai.embed({
          embedder: voyageai.embedder('voyage-multimodal-3.5'),
          content: 'a',
          options: { outputDimension: 256 } as any,
        }),
        /outputDimension is not supported/
      );
      assert.equal(requests.length, 0);
      // An undefined outputDimension (e.g. from a pass-through config) is not a request for one.
      await ai.embed({
        embedder: voyageai.embedder('voyage-multimodal-3.5'),
        content: 'a',
        options: { outputDimension: undefined } as any,
      });
      assert.equal(requests.length, 1);
      requests.length = 0;
      await assert.rejects(
        ai.embed({
          embedder: voyageai.embedder('voyage-multimodal-3.5'),
          content: {
            content: [
              { media: { url: 'gs://bucket/b.png', contentType: 'image/png' } },
            ],
          },
        }),
        /neither an http\(s\) URL nor a base64 data URL/
      );
      assert.equal(requests.length, 0);
    });
  });

  describe('rerankers', () => {
    const docs = ['cats', 'dogs', 'birds'].map((text) => ({
      content: [{ text }],
    }));

    it('sends top_k and return_documents and orders by score', async () => {
      const { ai, requests } = setup(() => ({
        body: {
          object: 'list',
          data: [
            { index: 2, relevance_score: 0.9, document: 'birds' },
            { index: 0, relevance_score: 0.4, document: 'cats' },
          ],
          usage: { total_tokens: 7 },
        },
      }));
      const ranked = await ai.rerank({
        reranker: voyageai.reranker('rerank-3'),
        query: 'flying animals',
        documents: docs,
        options: { topK: 2 },
      });
      const body = requests[0].body;
      assert.equal(requests[0].url.endsWith('/rerank'), true);
      assert.equal(body.model, 'rerank-3');
      assert.equal(body.top_k, 2);
      assert.equal(body.return_documents, true);
      assert.deepEqual(body.documents, ['cats', 'dogs', 'birds']);
      assert.deepEqual(
        ranked.map((d) => [d.text, d.score()]),
        [
          ['birds', 0.9],
          ['cats', 0.4],
        ]
      );
    });

    it('fails closed on a missing score, bad index, wrong count or media', async () => {
      const bad = (data: unknown[]) =>
        setup(() => ({ body: { data, usage: { total_tokens: 1 } } }));
      const run = (ai: any) =>
        ai.rerank({
          reranker: voyageai.reranker('rerank-3-lite'),
          query: 'q',
          documents: docs,
        });
      await assert.rejects(
        run(bad([{ index: 0 }, { index: 1 }, { index: 2 }]).ai),
        /without a valid index or relevance score/
      );
      await assert.rejects(
        run(
          bad([
            { index: 7, relevance_score: 1 },
            { index: 1, relevance_score: 1 },
            { index: 2, relevance_score: 1 },
          ]).ai
        ),
        /without a valid index or relevance score/
      );
      await assert.rejects(
        run(bad([{ index: 0, relevance_score: 1 }]).ai),
        /returned 1 results, expected 3/
      );
      const { ai, requests } = setup(() => ({ body: { data: [] } }));
      await assert.rejects(
        ai.rerank({
          reranker: voyageai.reranker('rerank-3'),
          query: 'q',
          documents: [
            { content: [{ media: { url: png, contentType: 'image/png' } }] },
          ],
        }),
        /text only/
      );
      await assert.rejects(
        ai.rerank({
          reranker: voyageai.reranker('rerank-3'),
          query: 'q',
          documents: docs,
          options: { topK: 0 } as any,
        }),
        /topK/
      );
      assert.equal(requests.length, 0);
    });
  });
});
