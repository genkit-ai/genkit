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

import { z, type GenerateRequest } from 'genkit';
import type { EmbedRequest } from 'ollama';
// Define possible API types
export type ApiType = 'chat' | 'generate';

// Standard model definition
export interface ModelDefinition {
  name: string;
  type?: ApiType;
  supports?: {
    tools?: boolean;
  };
}

// Definition for embedding models
export interface EmbeddingModelDefinition {
  name: string;
  dimensions: number;
}

export const OllamaEmbeddingPredictionSchema = z.object({
  embedding: z.array(z.number()),
});

export type OllamaEmbeddingPrediction = z.infer<
  typeof OllamaEmbeddingPredictionSchema
>;

export interface DefineOllamaEmbeddingParams {
  name: string;
  modelName: string;
  dimensions: number;
  options: OllamaPluginParams;
}

/**
 * Parameters for the Ollama plugin configuration.
 */
export interface OllamaPluginParams {
  /**
   * Array of models to be defined.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [
   *     ollama({
   *       models: [{ name: 'gemma' }],
   *       serverAddress: 'http://127.0.0.1:11434', // default local address
   *     }),
   *   ],
   * });
   * ```
   */
  models?: ModelDefinition[];

  /**
   * Array of embedding models to be defined.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [
   *     ollama({
   *       serverAddress: 'http://localhost:11434',
   *       embedders: [{ name: 'nomic-embed-text', dimensions: 768 }],
   *     }),
   *   ],
   * });
   * ```
   */
  embedders?: EmbeddingModelDefinition[];

  /**
   * The address of the Ollama server. Default: http://localhost:11434
   */
  serverAddress?: string;

  /**
   * Optional request headers, which can be either static or dynamically generated.
   *
   * ```ts
   * const ai = genkit({
   *   plugins: [
   *     ollama({
   *       models: [...],
   *       serverAddress: 'https://my-deployment.server.address',
   *       requestHeaders: async (params) => {
   *         const headers = await fetchAuthHeaders(params.serverAddress);
   *         return { Authorization: headers['Authorization'] };
   *       },
   *     }),
   *   ],
   * });
   * ```
   */
  requestHeaders?: RequestHeaders;
}

// Function type for generating request headers dynamically
export type RequestHeaderFunction = (
  params: {
    serverAddress: string;
    model?: ModelDefinition | EmbeddingModelDefinition;
    modelRequest?: GenerateRequest;
    embedRequest?: EmbedRequest;
  },
  // @deprecated -- moved into params, here for backwards compatibility reasons.
  modelRequest?: GenerateRequest
) => Promise<Record<string, string> | void>;

// Union type for request headers, supporting both static and dynamic options
export type RequestHeaders = Record<string, string> | RequestHeaderFunction;

export type OllamaRole = 'assistant' | 'tool' | 'system' | 'user';

// Tool definition from Ollama Chat API
export interface OllamaTool {
  type: string;
  function: {
    name: string;
    description: string;
    // `parameters` should be valid JSON Schema
    parameters: Record<string, any>;
  };
}

// Tool Call from Ollama Chat API
export interface OllamaToolCall {
  function: {
    index?: number;
    name: string;
    arguments: Record<string, any>;
  };
}

// Message format as defined by Ollama API
export interface Message {
  role: string;
  content: string;
  thinking?: string;
  images?: string[];
  tool_calls?: any[];
  tool_name?: string;
}

// Ollama local model definition
export interface LocalModel {
  name: string;
  model: string;
  // ISO 8601 format date
  modified_at: string;
  size: number;
  digest: string;
  details?: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
}

// Ollama list local models response
export interface ListLocalModelsResponse {
  models: LocalModel[];
}
