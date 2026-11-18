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

import { describe, expect, it, jest } from '@jest/globals';
import 'genkit';
import OpenAI from 'openai';
import { defineCompatOpenAIEmbedder } from '../src/embedder';

describe('defineCompatOpenAIEmbedder', () => {
  it('embeds when the request has no options', async () => {
    const create = jest.fn(async () => ({ data: [{ embedding: [1, 2] }] }));
    const client = { embeddings: { create } } as unknown as OpenAI;
    const embedder = defineCompatOpenAIEmbedder({ name: 'test', client });

    const response = await embedder({ input: [{ content: [{ text: 'hi' }] }] });

    expect(response).toEqual({ embeddings: [{ embedding: [1, 2] }] });
    expect(create).toHaveBeenCalledWith({
      model: 'test',
      input: ['hi'],
      encoding_format: undefined,
    });
  });
});
