/**
 * Copyright 2024 The Fire Company
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

// import { defineEmbedder, embedderRef } from '@genkit-ai/ai/embedder';

import type { EmbedderAction, EmbedderReference } from 'genkit';
import type { EmbedRequest } from 'genkit/embedder';
import { embedder } from 'genkit/plugin';
import OpenAI, { APIError } from 'openai';
import type { EmbeddingCreateParams } from 'openai/resources/index.mjs';
import { PluginOptions } from './index.js';
import { toGenkitError, type ErrorStatusClassifier } from './model.js';
import { maybeCreateRequestScopedOpenAIClient, toModelName } from './utils.js';

/**
 * Hook to modify the embeddings request body. When set, it replaces the
 * default passthrough of embedder options, so it must copy any options the
 * provider needs.
 */
export type EmbedderRequestBuilder = (
  req: EmbedRequest,
  params: EmbeddingCreateParams
) => void;

/**
 * Method to define a new Genkit Embedder that is compatibale with the Open AI
 * Embeddings API. 
 *
 * @param params An object containing parameters for defining the OpenAI embedder.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the embedder.
 * @param params.client The OpenAI client instance.
 * @param params.embedderRef Optional reference to the embedder's configuration and
 * custom options.
 * @param params.requestBuilder Optional hook to modify the request body, see
 * {@link EmbedderRequestBuilder}.
 * @param params.errorStatus Optional error classifier. When set, API errors
 * are thrown as Genkit errors, with the status it returns or else one from the
 * HTTP status code. Without it, API errors are thrown as they are.

 * @returns the created {@link EmbedderAction}
 */
export function defineCompatOpenAIEmbedder(params: {
  name: string;
  client: OpenAI;
  pluginOptions?: PluginOptions;
  embedderRef?: EmbedderReference;
  requestBuilder?: EmbedderRequestBuilder;
  errorStatus?: ErrorStatusClassifier;
}): EmbedderAction {
  const {
    name,
    client: defaultClient,
    pluginOptions,
    embedderRef,
    requestBuilder,
    errorStatus,
  } = params;

  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    embedderRef?.name ?? `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return embedder(
    {
      name: actionName,
      configSchema: embedderRef?.configSchema,
      ...embedderRef?.info,
    },
    async (req) => {
      const { encodingFormat: encoding_format, ...restOfConfig } = req.options;
      const client = maybeCreateRequestScopedOpenAIClient(
        pluginOptions,
        req,
        defaultClient
      );
      let body: EmbeddingCreateParams = {
        model: modelName,
        input: req.input.map((d) => d.text),
        encoding_format,
      };
      if (requestBuilder) {
        requestBuilder(req, body);
      } else {
        body = { ...body, ...restOfConfig };
      }
      let embeddings: OpenAI.CreateEmbeddingResponse;
      try {
        embeddings = await client.embeddings.create(body);
      } catch (e) {
        if (errorStatus && e instanceof APIError) {
          throw toGenkitError(e, errorStatus);
        }
        throw e;
      }
      return {
        embeddings: embeddings.data.map((d) => ({ embedding: d.embedding })),
      };
    }
  );
}
