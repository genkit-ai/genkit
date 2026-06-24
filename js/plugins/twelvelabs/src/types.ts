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

import { z } from 'genkit';

/** The default base URL for the TwelveLabs REST API. */
export const DEFAULT_BASE_URL = 'https://api.twelvelabs.io/v1.3';

/**
 * A Pegasus video-understanding model to expose as a Genkit model.
 *
 * `name` is the user-facing suffix (the action is registered as
 * `twelvelabs/<name>`). `modelName` is the identifier sent to the TwelveLabs
 * API; it defaults to `name` when omitted.
 */
export interface ModelDefinition {
  name: string;
  modelName?: string;
}

/**
 * A Marengo embedding model to expose as a Genkit embedder.
 *
 * `dimensions` is the size of the returned vector (Marengo returns 512-dim
 * float embeddings). `modelName` defaults to `name` when omitted.
 */
export interface EmbeddingModelDefinition {
  name: string;
  dimensions: number;
  modelName?: string;
}

/**
 * Configuration accepted by the TwelveLabs plugin.
 */
export interface TwelveLabsPluginParams {
  /**
   * TwelveLabs API key. Falls back to the `TWELVELABS_API_KEY` environment
   * variable when omitted. Grab a free key at https://twelvelabs.io.
   */
  apiKey?: string;

  /**
   * Pegasus video-understanding models to register, e.g.
   * `[{ name: 'pegasus1.5' }]`.
   */
  models?: ModelDefinition[];

  /**
   * Marengo embedding models to register, e.g.
   * `[{ name: 'marengo3.0', dimensions: 512 }]`.
   */
  embedders?: EmbeddingModelDefinition[];

  /**
   * Override the API base URL. Defaults to {@link DEFAULT_BASE_URL}.
   */
  baseUrl?: string;
}

/**
 * Per-request config for a Pegasus model. Mirrors the TwelveLabs `/analyze`
 * request options.
 */
export const TwelveLabsConfigSchema = z.object({
  temperature: z.number().min(0).max(1).optional(),
  maxTokens: z.number().int().positive().optional(),
  modelName: z
    .string()
    .describe('Override the TwelveLabs model_name sent to /analyze.')
    .optional(),
});

export type TwelveLabsConfig = z.infer<typeof TwelveLabsConfigSchema>;

// --- TwelveLabs REST API response shapes (the subset we consume) ---

/** Response from `POST /embed` with `model_name` + `text`. */
export interface EmbedResponse {
  model_name: string;
  text_embedding?: {
    segments: { float: number[] }[];
  };
}

/** Non-streaming response from `POST /analyze` (`stream: false`). */
export interface AnalyzeResponse {
  id: string;
  data: string;
  finish_reason?: string;
  usage?: { output_tokens?: number };
}
