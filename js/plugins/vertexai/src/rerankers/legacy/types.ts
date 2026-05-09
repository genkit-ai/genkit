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

import { z } from 'genkit';
import type { GoogleAuth } from 'google-auth-library';
import type { CommonPluginOptions } from '../../common/types.js';

// Define the schema for the options used in the Vertex AI reranker
export const VertexAIRerankerOptionsSchema = z.object({
  k: z.number().optional().describe('Number of top documents to rerank'), // Optional: Number of documents to rerank
  model: z.string().optional().describe('Model name for reranking'), // Optional: Model name, defaults to a pre-defined model
  location: z
    .string()
    .optional()
    .describe('Google Cloud location, e.g., "us-central1"'), // Optional: Location of the reranking model
});

// Type alias for the options schema
export type VertexAIRerankerOptions = z.infer<
  typeof VertexAIRerankerOptionsSchema
>;

export interface VertexRerankerConfig {
  // Optional: Name of the reranker
  name?: string;
  // Optional: Model name, defaults to a pre-defined model
  model?: string;
}

export interface VertexRerankOptions {
  authClient: GoogleAuth;
  location: string;
  projectId: string;
  rerankOptions: VertexRerankerConfig[];
}

export interface RerankerOptions {
  /** Configure reranker options */
  rerankers: (string | VertexRerankerConfig)[];
}

export interface PluginOptions extends CommonPluginOptions, RerankerOptions {}
