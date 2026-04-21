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

import { GenkitError, StatusName } from 'genkit';
import { logger } from 'genkit/logging';
import {
  extractErrMsg,
  getGenkitClientHeader,
  processStream,
} from '../common/utils.js';
import {
  ClientOptions,
  CreateInteractionRequest,
  EmbedContentRequest,
  EmbedContentResponse,
  GeminiInteraction,
  GenerateContentRequest,
  GenerateContentResponse,
  GenerateContentStreamResult,
  ImagenPredictRequest,
  ImagenPredictResponse,
  ListModelsResponse,
  Model,
  VeoOperation,
  VeoPredictRequest,
} from './types.js';

/**
 * Creates an interaction using the Google AI API.
 *
 * @param apiKey The API key to authenticate the request.
 * @param createInteractionRequest The request object containing the interaction parameters.
 * @param clientOptions Optional options to customize the request
 * @returns A promise that resolves to the interaction response.
 */
export async function createInteraction(
  apiKey: string | undefined,
  createInteractionRequest: CreateInteractionRequest,
  clientOptions?: ClientOptions
): Promise<GeminiInteraction> {
  const url = getGoogleAIUrl({
    resourcePath: 'interactions',
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'POST',
    apiKey,
    clientOptions,
    body: JSON.stringify(createInteractionRequest),
  });

  const response = await makeRequest(url, fetchOptions);

  return await response.json();
}

/**
 * Gets an interaction using the Google AI API.
 *
 * @param apiKey The API key to authenticate the request.
 * @param interactionId The ID of the interaction to retrieve.
 * @param clientOptions Optional options to customize the request
 * @returns A promise that resolves to the interaction response.
 */
export async function getInteraction(
  apiKey: string | undefined,
  interactionId: string,
  clientOptions?: ClientOptions
): Promise<GeminiInteraction> {
  const url = getGoogleAIUrl({
    resourcePath: `interactions/${interactionId}`,
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'GET',
    apiKey,
    clientOptions,
  });

  const response = await makeRequest(url, fetchOptions);

  return await response.json();
}

/**
 * Cancels an interaction using the Google AI API.
 *
 * @param apiKey The API key to authenticate the request.
 * @param interactionId The ID of the interaction to cancel.
 * @param clientOptions Optional options to customize the request
 * @returns A promise that resolves to the interaction response.
 */
export async function cancelInteraction(
  apiKey: string | undefined,
  interactionId: string,
  clientOptions?: ClientOptions
): Promise<GeminiInteraction> {
  const url = getGoogleAIUrl({
    resourcePath: `interactions/${interactionId}/cancel`,
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'POST',
    apiKey,
    clientOptions,
  });

  try {
    await makeRequest(url, fetchOptions);
    // A successful cancellation will actually throw a
    // CANCELLED error. If we instead get a 200 OK here, then
    // it can mean a no-op cancellation (e.g. race condition between finish and cancelled)
    // We throw an error here so the exit logic is the same for both cases.
    throw new GenkitError({
      status: 'CANCELLED',
      message: 'successfully cancelled',
    });
  } catch (e: any) {
    if (e instanceof GenkitError && e.status === 'CANCELLED') {
      return {
        id: interactionId,
        status: 'cancelled',
      } as GeminiInteraction;
    }
    throw e;
  }
}

/**
 * Lists available models.
 *
 * https://ai.google.dev/api/models#method:-models.list
 *
 * @param apiKey The API key to authenticate the request.
 * @param clientOptions Optional options to customize the request
 * @returns A promise that resolves to an array of Model objects.
 */
export async function listModels(
  apiKey: string | undefined,
  clientOptions?: ClientOptions
): Promise<Model[]> {
  const url = getGoogleAIUrl({
    resourcePath: 'models',
    queryParams: 'pageSize=1000',
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'GET',
    apiKey,
    clientOptions,
  });
  const response = await makeRequest(url, fetchOptions);
  const modelResponse = JSON.parse(await response.text()) as ListModelsResponse;
  return modelResponse.models;
}

/**
 * Generates content using the Google AI API.
 *
 * @param {string} apiKey The API key for authentication.
 * @param {string} model The name of the model to use for content generation.
 * @param {GenerateContentRequest} generateContentRequest The request object containing the content generation parameters.
 * @param {ClientOptions} [clientOptions] Optional client options.
 * @returns {Promise<GenerateContentResponse>} A promise that resolves to the content generation response.
 * @throws {Error} If the API request fails or the response cannot be parsed.
 */
export async function generateContent(
  apiKey: string | undefined,
  model: string,
  generateContentRequest: GenerateContentRequest,
  clientOptions?: ClientOptions
): Promise<GenerateContentResponse> {
  const url = getGoogleAIUrl({
    resourcePath: `models/${model}`,
    resourceMethod: 'generateContent',
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'POST',
    apiKey,
    clientOptions,
    body: JSON.stringify(generateContentRequest),
  });
  const response = await makeRequest(url, fetchOptions);

  const responseJson = (await response.json()) as GenerateContentResponse;
  return responseJson;
}

/**
 * Generates a stream of content using the Google AI API.
 *
 * @param {string} apiKey The API key for authentication.
 * @param {string} model The name of the model to use for content generation.
 * @param {GenerateContentRequest} generateContentRequest The request object containing the content generation parameters.
 * @param {ClientOptions} [clientOptions] Optional client options.
 * @returns {Promise<GenerateContentStreamResult>} A promise that resolves to an object containing a both the stream and aggregated response.
 * @throws {Error} If the API request fails.
 */
export async function generateContentStream(
  apiKey: string | undefined,
  model: string,
  generateContentRequest: GenerateContentRequest,
  clientOptions?: ClientOptions
): Promise<GenerateContentStreamResult> {
  const url = getGoogleAIUrl({
    resourcePath: `models/${model}`,
    resourceMethod: 'streamGenerateContent',
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'POST',
    apiKey,
    clientOptions,
    body: JSON.stringify(generateContentRequest),
  });

  const response = await makeRequest(url, fetchOptions);
  return processStream(response);
}

/**
 * Embeds content using the Google AI API.
 *
 * @param {string} apiKey The API key for authentication.
 * @param {string} model The name of the model to use for content embedding.
 * @param {EmbedContentRequest} embedContentRequest The request object containing the content to embed.
 * @param {ClientOptions} [clientOptions] Optional client options.
 * @returns {Promise<EmbedContentResponse>} A promise that resolves to the embedding response.
 * @throws {Error} If the API request fails or the response cannot be parsed.
 */
export async function embedContent(
  apiKey: string | undefined,
  model: string,
  embedContentRequest: EmbedContentRequest,
  clientOptions?: ClientOptions
): Promise<EmbedContentResponse> {
  const url = getGoogleAIUrl({
    resourcePath: `models/${model}`,
    resourceMethod: 'embedContent',
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'POST',
    apiKey,
    clientOptions,
    body: JSON.stringify(embedContentRequest),
  });

  const response = await makeRequest(url, fetchOptions);
  return response.json();
}

export async function imagenPredict(
  apiKey: string | undefined,
  model: string,
  imagenPredictRequest: ImagenPredictRequest,
  clientOptions?: ClientOptions
): Promise<ImagenPredictResponse> {
  const url = getGoogleAIUrl({
    resourcePath: `models/${model}`,
    resourceMethod: 'predict',
    clientOptions,
  });

  const fetchOptions = getFetchOptions({
    method: 'POST',
    apiKey,
    clientOptions,
    body: JSON.stringify(imagenPredictRequest),
  });

  const response = await makeRequest(url, fetchOptions);
  return response.json() as Promise<ImagenPredictResponse>;
}

export async function veoPredict(
  apiKey: string | undefined,
  model: string,
  veoPredictRequest: VeoPredictRequest,
  clientOptions?: ClientOptions
): Promise<VeoOperation> {
  const url = getGoogleAIUrl({
    resourcePath: `models/${model}`,
    resourceMethod: 'predictLongRunning',
    clientOptions,
  });

  const fetchOptions = getFetchOptions({
    method: 'POST',
    apiKey,
    clientOptions,
    body: JSON.stringify(veoPredictRequest),
  });

  const response = await makeRequest(url, fetchOptions);
  return response.json() as Promise<VeoOperation>;
}

export async function veoCheckOperation(
  apiKey: string | undefined,
  operation: string,
  clientOptions?: ClientOptions
): Promise<VeoOperation> {
  const url = getGoogleAIUrl({
    resourcePath: operation,
    clientOptions,
  });
  const fetchOptions = getFetchOptions({
    method: 'GET',
    apiKey,
    clientOptions,
  });

  const response = await makeRequest(url, fetchOptions);
  return response.json() as Promise<VeoOperation>;
}

/**
 * Generates a Google AI URL.
 *
 * @param params - An object containing the parameters for the URL.
 * @param params.path - The path for the URL (the part after the version)
 * @param params.task - An optional task
 * @param params.queryParams - An optional string of '&' delimited query parameters.
 * @param params.clientOptions - An optional object containing client options.
 * @returns The generated Google AI URL.
 */
export function getGoogleAIUrl(params: {
  resourcePath: string;
  resourceMethod?: string;
  queryParams?: string;
  clientOptions?: ClientOptions;
}): string {
  // v1beta is the default because all the new experimental models
  // are found here but not in v1.
  const DEFAULT_API_VERSION = 'v1beta';
  const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com';

  const apiVersion = params.clientOptions?.apiVersion || DEFAULT_API_VERSION;
  const baseUrl = params.clientOptions?.baseUrl || DEFAULT_BASE_URL;

  let url = `${baseUrl}/${apiVersion}/${params.resourcePath}`;
  if (params.resourceMethod) {
    url += `:${params.resourceMethod}`;
  }
  if (params.queryParams) {
    url += `?${params.queryParams}`;
  }
  if (params.resourceMethod === 'streamGenerateContent') {
    url += `${params.queryParams ? '&' : '?'}alt=sse`;
  }
  return url;
}

function getFetchOptions(params: {
  method: 'POST' | 'GET';
  apiKey: string | undefined;
  body?: string;
  clientOptions?: ClientOptions;
}) {
  const fetchOptions: RequestInit = {
    method: params.method,
    headers: getHeaders(params.apiKey, params.clientOptions),
  };
  if (params.body) {
    fetchOptions.body = params.body;
  }
  const signal = getAbortSignal(params.clientOptions);
  if (signal) {
    fetchOptions.signal = signal;
  }
  return fetchOptions;
}

function getAbortSignal(
  clientOptions?: ClientOptions
): AbortSignal | undefined {
  const hasTimeout = (clientOptions?.timeout ?? -1) >= 0;
  if (clientOptions?.signal !== undefined || hasTimeout) {
    const controller = new AbortController();
    if (hasTimeout) {
      setTimeout(() => controller.abort(), clientOptions?.timeout);
    }
    if (clientOptions?.signal) {
      clientOptions.signal.addEventListener('abort', () => {
        controller.abort();
      });
    }
    return controller.signal;
  }
  return undefined;
}

/**
 * Constructs the headers for an API request.
 *
 * @param {string} apiKey The API key for authentication.
 * @param {ClientOptions} [clientOptions] Optional client options, containing custom headers.
 * @returns {HeadersInit} An object containing the headers to be included in the request.
 */
function getHeaders(
  apiKey?: string,
  clientOptions?: ClientOptions
): HeadersInit {
  let customHeaders = {};
  if (clientOptions?.customHeaders) {
    customHeaders = structuredClone(clientOptions.customHeaders);
    delete customHeaders['x-goog-api-key']; // Not allowed in user settings
    delete customHeaders['x-goog-api-client']; // Not allowed in user settings
  }
  const headers: HeadersInit = {
    ...customHeaders,
    'Content-Type': 'application/json',
    'x-goog-api-client': getGenkitClientHeader(),
  };

  if (apiKey) {
    headers['x-goog-api-key'] = apiKey;
  }

  return headers;
}

/**
 * Makes a request to the specified URL with the provided options.
 *
 * @param {string} url The URL to make the request to.
 * @param {RequestInit} fetchOptions The options to pass to the `fetch` API.
 * @returns {Promise<Response>} A promise that resolves to the Response
 * @throws {Error} If the request fails
 */
async function makeRequest(
  url: string,
  fetchOptions: RequestInit
): Promise<Response> {
  try {
    const response = await fetch(url, fetchOptions);
    if (!response.ok) {
      let errorText = await response.text();
      let errorMessage = errorText;
      let errorDetail: unknown;
      try {
        const json = JSON.parse(errorText);
        errorDetail = json;
        if (json.error && json.error.message) {
          errorMessage = json.error.message;
          if (Array.isArray(json.error.details)) {
            const detailsText = json.error.details
              .map((d: any) => {
                if (d.detail && typeof d.detail === 'string') {
                  const match = d.detail.match(/\[ORIGINAL ERROR\]\s*([^[]+)/);
                  return match ? match[1].trim() : d.detail;
                }
                return JSON.stringify(d);
              })
              .filter(Boolean)
              .join('\n');
            if (detailsText) {
              errorMessage += `\nDetails: ${detailsText}`;
            }
          }
        }
      } catch (e) {
        // Not JSON or expected format, use the raw text
      }
      let status: StatusName = 'UNKNOWN';
      switch (response.status) {
        case 429:
          status = 'RESOURCE_EXHAUSTED';
          break;
        case 400:
          status = 'INVALID_ARGUMENT';
          break;
        case 499:
          status = 'CANCELLED';
          break;
        case 500:
          status = 'INTERNAL';
          break;
        case 503:
          status = 'UNAVAILABLE';
          break;
      }
      throw new GenkitError({
        status,
        message: `Error fetching from ${url}: [${response.status} ${response.statusText}] ${errorMessage}`,
        detail: errorDetail,
      });
    }
    return response;
  } catch (e: unknown) {
    logger.error(e);
    if (e instanceof GenkitError) {
      throw e;
    }
    throw new Error(`Failed to fetch from ${url}: ${extractErrMsg(e)}`);
  }
}

export const TEST_ONLY = {
  getFetchOptions,
  getAbortSignal,
  getHeaders,
  makeRequest,
};
