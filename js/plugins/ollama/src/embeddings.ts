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
import type { Document, EmbedderAction } from 'genkit';
import { embedder } from 'genkit/plugin';
import type { EmbedRequest, EmbedResponse } from 'ollama';
import { DEFAULT_OLLAMA_SERVER_ADDRESS } from './constants.js';
import type { DefineOllamaEmbeddingParams, RequestHeaders } from './types.js';
import { OllamaEmbedderConfigSchema } from './types.js';

export async function toOllamaEmbedRequest(
  modelName: string,
  dimensions: number,
  documents: Document[],
  serverAddress: string,
  requestHeaders?: RequestHeaders
): Promise<{
  url: string;
  requestPayload: EmbedRequest;
  headers: Record<string, string>;
}> {
  const requestPayload: EmbedRequest = {
    model: modelName,
    input: documents.map((doc) => doc.text),
  };

  // Determine headers
  const extraHeaders = requestHeaders
    ? typeof requestHeaders === 'function'
      ? await requestHeaders({
          serverAddress,
          model: {
            name: modelName,
            dimensions,
          },
          embedRequest: requestPayload,
        })
      : requestHeaders
    : {};

  const headers = {
    'Content-Type': 'application/json',
    ...extraHeaders, // Add any dynamic headers
  };

  return {
    url: `${serverAddress}/api/embed`,
    requestPayload,
    headers,
  };
}

export function defineOllamaEmbedder({
  name,
  modelName,
  dimensions,
  options,
}: DefineOllamaEmbeddingParams): EmbedderAction<
  typeof OllamaEmbedderConfigSchema
> {
  return embedder(
    {
      name: `ollama/${name}`,
      configSchema: OllamaEmbedderConfigSchema,
      info: {
        label: 'Ollama Embedding - ' + name,
        dimensions,
        supports: {
          //  TODO: do any ollama models support other modalities?
          input: ['text'],
        },
      },
    },
    async ({ input, options: requestOptions }, config) => {
      const serverAddress =
        requestOptions?.serverAddress ||
        options.serverAddress ||
        DEFAULT_OLLAMA_SERVER_ADDRESS;
      const { url, requestPayload, headers } = await toOllamaEmbedRequest(
        modelName,
        dimensions,
        input,
        serverAddress,
        options.requestHeaders
      );

      const response: Response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestPayload),
      });

      if (!response.ok) {
        const errMsg = (await response.json()).error?.message || '';
        throw new Error(
          `Error fetching embedding from Ollama: ${response.statusText}. ${errMsg}`
        );
      }

      const payload = (await response.json()) as EmbedResponse;

      const embeddings: { embedding: number[] }[] = [];

      for (const embedding of payload.embeddings) {
        embeddings.push({ embedding });
      }
      return { embeddings };
    }
  );
}
