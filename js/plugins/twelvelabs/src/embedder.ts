/**
 * Copyright 2025 Google LLC
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

import type { EmbedderAction, Genkit } from 'genkit';
import {
  DEFAULT_BASE_URL,
  type EmbedResponse,
  type EmbeddingModelDefinition,
} from './types.js';

interface DefineEmbedderParams {
  apiKey: string;
  baseUrl: string;
  embedder: EmbeddingModelDefinition;
}

/**
 * Registers a TwelveLabs Marengo embedder against the `/embed` endpoint.
 *
 * Marengo produces multimodal embeddings; here we expose text embedding, which
 * shares an embedding space with TwelveLabs' video embeddings and is the right
 * fit for Genkit's text-based `embed()` / retriever flows. The endpoint takes
 * `multipart/form-data` and returns the vector at
 * `text_embedding.segments[0].float`.
 */
export function defineTwelveLabsEmbedder(
  ai: Genkit,
  { apiKey, baseUrl, embedder }: DefineEmbedderParams
): EmbedderAction<any> {
  const modelName = embedder.modelName ?? embedder.name;
  return ai.defineEmbedder(
    {
      name: `twelvelabs/${embedder.name}`,
      info: {
        label: `TwelveLabs Embedding - ${embedder.name}`,
        dimensions: embedder.dimensions,
        supports: {
          input: ['text'],
        },
      },
    },
    async (input) => {
      // The /embed endpoint embeds one input per call, so map over documents.
      const embeddings = await Promise.all(
        input.map(async (doc) => {
          const form = new FormData();
          form.append('model_name', modelName);
          form.append('text', doc.text);

          const response = await fetch(`${baseUrl}/embed`, {
            method: 'POST',
            headers: { 'x-api-key': apiKey },
            body: form,
          });

          if (!response.ok) {
            const errMsg = await safeErrorMessage(response);
            throw new Error(
              `Error fetching embedding from TwelveLabs: ${response.statusText}. ${errMsg}`
            );
          }

          const payload = (await response.json()) as EmbedResponse;
          const embedding = payload.text_embedding?.segments?.[0]?.float;
          if (!embedding) {
            throw new Error(
              'TwelveLabs embedding response did not contain a text embedding.'
            );
          }
          return { embedding };
        })
      );
      return { embeddings };
    }
  );
}

async function safeErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { message?: string };
    return body.message ?? '';
  } catch {
    return '';
  }
}

export { DEFAULT_BASE_URL };
