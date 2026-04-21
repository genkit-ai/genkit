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
import { GoogleAuth } from 'google-auth-library';
import {
  CreateInteractionRequest,
  GeminiInteraction,
} from '../common/interaction-types.js';
import {
  extractErrMsg,
  getGenkitClientHeader,
  processStream,
} from '../common/utils.js';
import {
  ClientOptions,
  EmbedContentRequest,
  EmbedContentResponse,
  GenerateContentRequest,
  GenerateContentResponse,
  GenerateContentStreamResult,
  ImagenPredictRequest,
  ImagenPredictResponse,
  ListModelsResponse,
  LyriaPredictRequest,
  LyriaPredictResponse,
  Model,
  VeoOperation,
  VeoOperationRequest,
  VeoPredictRequest,
} from './types.js';
import { calculateApiKey, checkSupportedResourceMethod } from './utils.js';

export async function listModels(
  clientOptions: ClientOptions
): Promise<Model[]> {
  const url = getVertexAIUrl({
    includeProjectAndLocation: false,
    resourcePath: 'publishers/google/models',
    clientOptions,
  });
  const fetchOptions = await getFetchOptions({
    method: 'GET',
    clientOptions,
  });
  const response = await makeRequest(url, fetchOptions);
  const modelResponse = (await response.json()) as ListModelsResponse;
  return modelResponse.publisherModels;
}

export async function createInteraction(
  createInteractionRequest: CreateInteractionRequest,
  clientOptions: ClientOptions
): Promise<GeminiInteraction> {
  const url = getVertexAIUrl({
    includeProjectAndLocation: true,
    resourcePath: `interactions`,
    clientOptions,
  });

  const fetchOptions = await getFetchOptions({
    method: 'POST',
    clientOptions,
    body: JSON.stringify(createInteractionRequest),
  });

  const response = await makeRequest(url, fetchOptions);

  return await response.json();
}

export async function generateContent(
  model: string,
  generateContentRequest: GenerateContentRequest,
  clientOptions: ClientOptions
): Promise<GenerateContentResponse> {
  let url: string;
  if (model.includes('endpoints/')) {
    // Tuned model
    url = getVertexAIUrl({
      includeProjectAndLocation: !model.startsWith('projects/'),
      resourcePath: model,
      resourceMethod: 'generateContent',
      clientOptions,
    });
  } else {
    url = getVertexAIUrl({
      includeProjectAndLocation: true,
      resourcePath: `publishers/google/models/${model}`,
      resourceMethod: 'generateContent',
      clientOptions,
    });
  }
  const fetchOptions = await getFetchOptions({
    method: 'POST',
    clientOptions,
    body: JSON.stringify(generateContentRequest),
  });
  const response = await makeRequest(url, fetchOptions);

  const responseJson = (await response.json()) as GenerateContentResponse;
  return responseJson;
}

export async function generateContentStream(
  model: string,
  generateContentRequest: GenerateContentRequest,
  clientOptions: ClientOptions
): Promise<GenerateContentStreamResult> {
  let url: string;
  if (model.includes('endpoints/')) {
    // Tuned model
    url = getVertexAIUrl({
      includeProjectAndLocation: !model.startsWith('projects/'),
      resourcePath: model,
      resourceMethod: 'streamGenerateContent',
      clientOptions,
    });
  } else {
    url = getVertexAIUrl({
      includeProjectAndLocation: true,
      resourcePath: `publishers/google/models/${model}`,
      resourceMethod: 'streamGenerateContent',
      clientOptions,
    });
  }
  const fetchOptions = await getFetchOptions({
    method: 'POST',
    clientOptions,
    body: JSON.stringify(generateContentRequest),
  });
  const response = await makeRequest(url, fetchOptions);
  return processStream(response);
}

async function internalPredict(
  model: string,
  body: string,
  clientOptions: ClientOptions
): Promise<Response> {
  const url = getVertexAIUrl({
    includeProjectAndLocation: true,
    resourcePath: `publishers/google/models/${model}`,
    resourceMethod: 'predict',
    clientOptions,
  });

  const fetchOptions = await getFetchOptions({
    method: 'POST',
    clientOptions,
    body,
  });

  return await makeRequest(url, fetchOptions);
}

export async function embedContent(
  model: string,
  embedContentRequest: EmbedContentRequest,
  clientOptions: ClientOptions
): Promise<EmbedContentResponse> {
  const response = await internalPredict(
    model,
    JSON.stringify(embedContentRequest),
    clientOptions
  );
  return response.json() as Promise<EmbedContentResponse>;
}

export async function imagenPredict(
  model: string,
  imagenPredictRequest: ImagenPredictRequest,
  clientOptions: ClientOptions
): Promise<ImagenPredictResponse> {
  const response = await internalPredict(
    model,
    JSON.stringify(imagenPredictRequest),
    clientOptions
  );
  return response.json() as Promise<ImagenPredictResponse>;
}

export async function lyriaPredict(
  model: string,
  lyriaPredictRequest: LyriaPredictRequest,
  clientOptions: ClientOptions
): Promise<LyriaPredictResponse> {
  const response = await internalPredict(
    model,
    JSON.stringify(lyriaPredictRequest),
    clientOptions
  );
  return response.json() as Promise<LyriaPredictResponse>;
}

export async function veoPredict(
  model: string,
  veoPredictRequest: VeoPredictRequest,
  clientOptions: ClientOptions
): Promise<VeoOperation> {
  const url = getVertexAIUrl({
    includeProjectAndLocation: true,
    resourcePath: `publishers/google/models/${model}`,
    resourceMethod: 'predictLongRunning',
    clientOptions,
  });

  const fetchOptions = await getFetchOptions({
    method: 'POST',
    clientOptions,
    body: JSON.stringify(veoPredictRequest),
  });

  const response = await makeRequest(url, fetchOptions);
  const operation = await response.json();
  operation.clientOptions = clientOptions; // for the check
  return operation as Promise<VeoOperation>;
}

export async function veoCheckOperation(
  model: string,
  veoOperationRequest: VeoOperationRequest,
  clientOptions: ClientOptions
): Promise<VeoOperation> {
  const url = getVertexAIUrl({
    includeProjectAndLocation: true,
    resourcePath: `publishers/google/models/${model}`,
    resourceMethod: 'fetchPredictOperation',
    clientOptions,
  });
  const fetchOptions = await getFetchOptions({
    method: 'POST',
    clientOptions,
    body: JSON.stringify(veoOperationRequest),
  });

  const response = await makeRequest(url, fetchOptions);
  const operation = await response.json();
  operation.clientOptions = clientOptions; // for future checks
  return operation as Promise<VeoOperation>;
}

export function getVertexAIUrl(params: {
  includeProjectAndLocation: boolean; // False for listModels, true for most others
  resourcePath: string;
  resourceMethod?: string;
  queryParams?: string;
  clientOptions: ClientOptions;
}): string {
  checkSupportedResourceMethod(params);

  const DEFAULT_API_VERSION = 'v1beta1';
  const API_BASE_PATH = 'aiplatform.googleapis.com';

  let basePath: string;

  if (params.clientOptions.kind == 'regional') {
    basePath = `${params.clientOptions.location}-${API_BASE_PATH}`;
  } else {
    basePath = API_BASE_PATH;
  }

  let resourcePath = params.resourcePath;
  if (
    params.clientOptions.kind != 'express' &&
    params.includeProjectAndLocation
  ) {
    const parent = `projects/${params.clientOptions.projectId}/locations/${params.clientOptions.location}`;
    resourcePath = `${parent}/${params.resourcePath}`;
  }

  let url = `https://${basePath}/${DEFAULT_API_VERSION}/${resourcePath}`;
  if (params.resourceMethod) {
    url += `:${params.resourceMethod}`;
  }

  let joiner = '?';
  if (params.queryParams) {
    url += `${joiner}${params.queryParams}`;
    joiner = '&';
  }
  if (params.resourceMethod === 'streamGenerateContent') {
    url += `${joiner}alt=sse`;
    joiner = '&';
  }
  return url;
}

async function getFetchOptions(params: {
  method: 'POST' | 'GET';
  body?: string;
  clientOptions: ClientOptions;
}) {
  const fetchOptions: RequestInit = {
    method: params.method,
    headers: await getHeaders(params.clientOptions),
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

function getAbortSignal(clientOptions: ClientOptions): AbortSignal | undefined {
  const hasTimeout = (clientOptions.timeout ?? -1) >= 0;
  if (clientOptions.signal !== undefined || hasTimeout) {
    const controller = new AbortController();
    if (hasTimeout) {
      setTimeout(() => controller.abort(), clientOptions.timeout);
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

async function getHeaders(clientOptions: ClientOptions): Promise<HeadersInit> {
  if (clientOptions.kind == 'express') {
    const headers: HeadersInit = {
      'x-goog-api-key': calculateApiKey(clientOptions.apiKey, undefined),
      'Content-Type': 'application/json',
      'X-Goog-Api-Client': getGenkitClientHeader(),
      'User-Agent': getGenkitClientHeader(),
    };
    return headers;
  } else {
    const token = await getToken(clientOptions.authClient);
    const headers: HeadersInit = {
      Authorization: `Bearer ${token}`,
      'x-goog-user-project': clientOptions.projectId,
      'Content-Type': 'application/json',
      'X-Goog-Api-Client': getGenkitClientHeader(),
      'User-Agent': getGenkitClientHeader(),
    };
    if (clientOptions.apiKey) {
      headers['x-goog-api-key'] = clientOptions.apiKey;
    }
    return headers;
  }
}

async function getToken(authClient: GoogleAuth): Promise<string> {
  const CREDENTIAL_ERROR_MESSAGE =
    '\nUnable to authenticate your request\
        \nDepending on your run time environment, you can get authentication by\
        \n- if in local instance or cloud shell: `!gcloud auth login`\
        \n- if in Colab:\
        \n    -`from google.colab import auth`\
        \n    -`auth.authenticate_user()`\
        \n- if in service account or other: please follow guidance in https://cloud.google.com/docs/authentication';
  const token = await authClient.getAccessToken().catch((e) => {
    throw new Error(CREDENTIAL_ERROR_MESSAGE, e);
  });
  if (!token) {
    throw new Error(CREDENTIAL_ERROR_MESSAGE);
  }
  return token;
}

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
                  const match = d.detail.match(
                    /\[ORIGINAL ERROR\]\s*(?:generic::[^:]+:\s*)?(.*?)(?:\s+\[|\s+\d+\s+\{|$)/
                  );
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
