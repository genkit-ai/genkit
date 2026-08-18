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

import type {
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
  StreamingCallback,
} from 'genkit';
import { GenkitError } from 'genkit';
import { logger } from 'genkit/logging';
import type { ModelAction, ModelReference } from 'genkit/model';
import { model } from 'genkit/plugin';
import OpenAI, { APIError } from 'openai';
import { PluginOptions } from '../../index';
import { maybeCreateRequestScopedOpenAIClient, toModelName } from '../../utils';
import { toResponsesRequestBody } from './request';
import { fromResponsesResponse } from './response';
import { streamResponsesEvents } from './stream';
import { OpenAIResponsesConfigSchema } from './types';

/**
 * Map an HTTP status from an {@link APIError} to a Genkit `StatusName`.
 *
 * Mirrors the mapping in `openAIModelRunner` (Chat Completions). Duplicated
 * intentionally so we do not modify the existing `model.ts` (constraint:
 * other compat providers must remain bit-identical).
 */
function statusFromApiError(e: APIError): GenkitError['status'] {
  switch (e.status) {
    case 429:
      return 'RESOURCE_EXHAUSTED';
    case 401:
      return 'PERMISSION_DENIED';
    case 403:
      return 'UNAUTHENTICATED';
    case 400:
      return 'INVALID_ARGUMENT';
    case 500:
      return 'INTERNAL';
    case 503:
      return 'UNAVAILABLE';
    default:
      return 'UNKNOWN';
  }
}

/**
 * Creates the runner used by Genkit to drive a model via the OpenAI
 * Responses API (`/v1/responses`).
 *
 * Both non-streaming (`client.responses.create`) and streaming
 * (`client.responses.stream` + SSE event aggregation in
 * {@link streamResponsesEvents}) paths are supported. Streaming emits
 * Genkit chunks per token / per tool-call delta and uses the SDK's
 * `stream.finalResponse()` for the canonical final response shape.
 *
 * @param name Model id (e.g. `'gpt-5-mini'`) — bare, no namespace.
 * @param defaultClient OpenAI SDK client; per-request override via
 *   `config.apiKey` falls back to {@link maybeCreateRequestScopedOpenAIClient}.
 * @param pluginOptions Captured to forward into request-scoped clients.
 */
export function openAIResponsesModelRunner(
  name: string,
  defaultClient: OpenAI,
  pluginOptions?: Omit<PluginOptions, 'apiKey'>
) {
  return async (
    request: GenerateRequest<typeof OpenAIResponsesConfigSchema>,
    options?: {
      streamingRequested?: boolean;
      sendChunk?: StreamingCallback<GenerateResponseChunkData>;
      abortSignal?: AbortSignal;
    }
  ): Promise<GenerateResponseData> => {
    const client = maybeCreateRequestScopedOpenAIClient(
      pluginOptions,
      request as GenerateRequest,
      defaultClient
    );
    try {
      const body = toResponsesRequestBody(name, request);

      if (options?.streamingRequested && options.sendChunk) {
        // True streaming path — drive the SSE event stream and
        // emit Genkit chunks per token / per tool-call delta.
        // The aggregated final Response (canonical structure) is
        // pulled from `stream.finalResponse()` once events end.
        //
        // The SDK's `stream(...)` overload narrows `stream?: true`
        // (it adds it itself); we strip the `stream` field from
        // our non-streaming body to satisfy the type contract.
        const { stream: _stream, ...streamBody } = body as {
          stream?: unknown;
        };
        const stream = client.responses.stream(
          streamBody as Parameters<typeof client.responses.stream>[0],
          { signal: options.abortSignal }
        );
        await streamResponsesEvents(stream, options.sendChunk);
        const finalResponse = await stream.finalResponse();
        return fromResponsesResponse(finalResponse, request as GenerateRequest);
      }

      const response = await client.responses.create(body, {
        signal: options?.abortSignal,
      });
      return fromResponsesResponse(response, request as GenerateRequest);
    } catch (e) {
      if (e instanceof APIError) {
        throw new GenkitError({
          status: statusFromApiError(e),
          message: e.message,
        });
      }
      // Non-APIError (network failure, abort, type error, …). Surface a
      // structured GenkitError so flow-level error handling sees a
      // consistent shape across providers.
      const err = e as Error & { name?: string };
      logger.error(
        `[openai-responses] non-API error in model "${name}": ` +
          `${err.name ?? 'Error'}: ${err.message}`
      );
      const isAbort =
        err.name === 'AbortError' || options?.abortSignal?.aborted === true;
      throw new GenkitError({
        status: isAbort ? 'CANCELLED' : 'INTERNAL',
        message: `OpenAI Responses request failed: ${err.message}`,
      });
    }
  };
}

/**
 * Define a Genkit {@link ModelAction} backed by the Responses API.
 *
 * Always registers under the `'openai-responses'` namespace (set by
 * {@link openAIResponsesModelRef}) so it cannot collide with the existing
 * `'openai/...'` Chat Completions entries.
 *
 * The configSchema is locked to {@link OpenAIResponsesConfigSchema} —
 * {@link openAIResponsesModelRunner} reads Responses-specific fields
 * (`builtInTools`, `previousResponseId`, `reasoning`, …) from `request.config`,
 * so a custom schema would silently desync at runtime.
 */
export function defineCompatOpenAIResponsesModel(params: {
  name: string;
  client: OpenAI;
  modelRef?: ModelReference<typeof OpenAIResponsesConfigSchema>;
  pluginOptions?: PluginOptions;
}): ModelAction {
  const { name, client, pluginOptions, modelRef } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    modelRef?.name ??
    `${pluginOptions?.name ?? 'openai-responses'}/${modelName}`;

  // The runner's request type is parameterized by the Responses config
  // schema; `model()` accepts a generic runner so the cast is benign.
  const runner = openAIResponsesModelRunner(
    modelName,
    client,
    pluginOptions
  ) as unknown as Parameters<typeof model>[1];
  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema ?? OpenAIResponsesConfigSchema,
    },
    runner
  );
}
