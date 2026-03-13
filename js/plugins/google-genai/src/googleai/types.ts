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

import {
  Content,
  FinishReason,
  GenerateContentCandidate,
  GenerateContentRequest,
  GenerateContentResponse,
  GenerateContentStreamResult,
  GenerationConfig,
  GoogleSearchRetrievalTool,
  HarmBlockThreshold,
  HarmCategory,
  ImagenInstance,
  ImagenParameters,
  ImagenPredictRequest,
  ImagenPredictResponse,
  ImagenPrediction,
  Part,
  SafetySetting,
  TaskType,
  TaskTypeSchema,
  Tool,
  ToolConfig,
  UrlContextTool,
} from '../common/types.js';

// This makes it easier to import all types from one place.
export {
  FinishReason,
  HarmBlockThreshold,
  HarmCategory,
  TaskTypeSchema,
  type Content,
  type GenerateContentCandidate,
  type GenerateContentRequest,
  type GenerateContentResponse,
  type GenerateContentStreamResult,
  type GenerationConfig,
  type GoogleSearchRetrievalTool,
  type ImagenInstance,
  type ImagenParameters,
  type ImagenPredictRequest,
  type ImagenPredictResponse,
  type ImagenPrediction,
  type Part,
  type SafetySetting,
  type Tool,
  type ToolConfig,
  type UrlContextTool,
};

export interface GoogleAIPluginOptions {
  /**
   * Provide the API key to use to authenticate with the Gemini API. By
   * default, an API key must be provided explicitly here or through the
   * `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables.
   *
   * If `false` is explicitly passed, the plugin will be configured to
   * expect an `apiKey` option to be provided to the model config at
   * call time.
   **/
  apiKey?: string | false;
  apiVersion?: string;
  baseUrl?: string;
  experimental_debugTraces?: boolean;
  /** Use `responseSchema` field instead of `responseJsonSchema`. */
  legacyResponseSchema?: boolean;
  /**
   * Additional headers to send along with the request.
   */
  customHeaders?: Record<string, string>;
}

/**
 * Options passed to the client
 * @public
 */
export interface ClientOptions {
  /**
   * An object that may be used to abort asynchronous requests. The request may
   * also be aborted due to the expiration of the timeout value, if provided.
   *
   * NOTE: AbortSignal is a client-only operation. Using it to cancel an
   * operation will not cancel the request in the service. You will still
   * be charged usage for any applicable operations.
   */
  signal?: AbortSignal;
  /**
   * Request timeout in milliseconds.
   */
  timeout?: number;
  /**
   * Api Key for Gemini API
   */
  apiKey?: string;
  /**
   * Version of API endpoint to call (e.g. "v1" or "v1beta"). If not specified,
   * defaults to 'v1beta'.
   */
  apiVersion?: string;
  /**
   * Additional attribution information to include in the x-goog-api-client header.
   * Used by wrapper SDKs.
   */
  apiClient?: string;
  /**
   * Base endpoint url. Defaults to "https://generativelanguage.googleapis.com"
   */
  baseUrl?: string;
  /**
   * Custom HTTP request headers.
   */
  customHeaders?: Headers | Record<string, string>;
}

/**
 * Params for calling embedContent
 * @public
 */
export interface EmbedContentRequest {
  content: Content;
  taskType?: TaskType;
  title?: string;
}

/**
 * Gemini model object
 * @public
 */
export interface Model {
  name: string;
  baseModelId: string;
  version: string;
  displayName: string;
  description: string;
  inputTokenLimit: number;
  outputTokenLimit: number;
  supportedGenerationMethods: string[];
  temperature: number;
  maxTemperature: number;
  topP: number;
  topK: number;
}

/**
 * Response from calling listModels
 * @public
 */
export interface ListModelsResponse {
  models: Model[];
  nextPageToken?: string;
}

/**
 * Response from calling embedContent
 * @public
 */
export interface EmbedContentResponse {
  embedding: ContentEmbedding;
}

/**
 * A single content embedding.
 * @public
 */
export interface ContentEmbedding {
  values: number[];
}

export declare interface VeoPredictRequest {
  instances: VeoInstance[];
  parameters: VeoParameters;
}

export declare interface VeoParameters {
  negativePrompt?: string;
  aspectRatio?: string;
  personGeneration?: string;
  durationSeconds?: number;
  enhancePrompt?: boolean;
  resolution?: string;
  seed?: number;
}

export declare interface VeoInstance {
  prompt: string;
  image?: VeoImage;
  video?: VeoVideo;
}

export declare interface VeoImage {
  bytesBase64Encoded: string;
  mimeType: string;
}

export declare interface VeoVideo {
  uri: string;
}

export declare interface VeoOperation {
  name: string;
  done?: boolean;
  error?: {
    message: string;
  };
  response?: {
    generateVideoResponse: {
      generatedSamples: { video: { uri: string } }[];
    };
  };
}
