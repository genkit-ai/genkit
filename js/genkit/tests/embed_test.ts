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

import { Document, embedderRef, type EmbedderAction } from '@genkit-ai/ai';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, type Genkit } from '../src/index.js';

describe('embed', () => {
  describe('default model', () => {
    let ai: Genkit;
    let embedder: EmbedderAction;

    beforeEach(() => {
      ai = genkit({});
      embedder = defineTestEmbedder(ai);
    });

    it('passes string content as docs', async () => {
      const response = (
        await ai.embed({
          embedder: 'echoEmbedder',
          content: 'hi',
        })
      )[0].embedding;
      assert.deepStrictEqual((embedder as any).lastRequest, [
        [Document.fromText('hi')],
        {
          version: undefined,
        },
      ]);
      assert.deepStrictEqual(response, [1, 2, 3, 4]);
    });

    it('passes docs content as docs', async () => {
      const response = await ai.embed({
        embedder: 'echoEmbedder',
        content: Document.fromText('hi'),
      });
      assert.deepStrictEqual((embedder as any).lastRequest, [
        [Document.fromText('hi')],
        {
          version: undefined,
        },
      ]);
      assert.deepStrictEqual(response, [{ embedding: [1, 2, 3, 4] }]);
    });
  });

  describe('config', () => {
    let ai: Genkit;
    let embedder: EmbedderAction;

    beforeEach(() => {
      ai = genkit({});
      embedder = defineTestEmbedder(ai);
    });

    it('takes config passed to generate', async () => {
      const response = await ai.embed({
        embedder: 'echoEmbedder',
        content: 'hi',
        options: {
          temperature: 11,
        },
      });
      assert.deepStrictEqual(response, [{ embedding: [1, 2, 3, 4] }]);
      assert.deepStrictEqual((embedder as any).lastRequest[1], {
        temperature: 11,
        version: undefined,
      });
    });

    it('merges config from the ref', async () => {
      const response = await ai.embed({
        embedder: embedderRef({
          name: 'echoEmbedder',
          config: {
            version: 'abc',
          },
        }),
        content: 'hi',
        options: {
          temperature: 11,
        },
      });
      assert.deepStrictEqual(response, [{ embedding: [1, 2, 3, 4] }]);
      assert.deepStrictEqual((embedder as any).lastRequest[1], {
        temperature: 11,
        version: 'abc',
      });
    });

    it('picks up the top-level version from the ref', async () => {
      const response = await ai.embed({
        embedder: embedderRef({
          name: 'echoEmbedder',
          version: 'abc',
        }),
        content: 'hi',
        options: {
          temperature: 11,
        },
      });
      assert.deepStrictEqual(response, [{ embedding: [1, 2, 3, 4] }]);
      assert.deepStrictEqual((embedder as any).lastRequest[1], {
        temperature: 11,
        version: 'abc',
      });
    });
  });
});

function defineTestEmbedder(ai: Genkit) {
  const embedder = ai.defineEmbedder(
    { name: 'echoEmbedder' },
    async (input, config) => {
      (embedder as any).lastRequest = [input, config];
      return {
        embeddings: [
          {
            embedding: [1, 2, 3, 4],
          },
        ],
      };
    }
  );
  return embedder;
}
