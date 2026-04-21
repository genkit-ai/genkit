/**
 * Copyright 2024 Bloom Labs Inc
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

import Anthropic from '@anthropic-ai/sdk';
import { genkitPluginV2, type GenkitPluginV2 } from 'genkit/plugin';

import type { Part } from 'genkit';
import { z, type ActionMetadata, type ModelReference } from 'genkit';
import { type ModelAction } from 'genkit/model';
import { type ActionType } from 'genkit/registry';
import { listActions } from './list.js';
import {
  KNOWN_CLAUDE_MODELS,
  claudeModel,
  claudeModelReference,
  type AnthropicConfigSchemaType,
  type ClaudeConfig,
  type ClaudeModelName,
  type KnownClaudeModels,
} from './models.js';
import {
  __testClient,
  type AnthropicDocumentOptions,
  type InternalPluginOptions,
  type PluginOptions,
} from './types.js';

// Re-export types and utilities for consumers
export type { AnthropicCacheControl, AnthropicCitation } from './types.js';
export { cacheControl } from './utils.js';

/**
 * Gets or creates an Anthropic client instance.
 * Supports test client injection for internal testing.
 */
function getAnthropicClient(options?: PluginOptions): Anthropic {
  // Check for test client injection first (internal use only)
  const internalOptions = options as InternalPluginOptions | undefined;
  if (internalOptions?.[__testClient]) {
    return internalOptions[__testClient];
  }

  // Production path: create real client
  const apiKey = options?.apiKey || process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error(
      'Please pass in the API key or set the ANTHROPIC_API_KEY environment variable'
    );
  }
  return new Anthropic({ apiKey });
}

/**
 * This module provides an interface to the Anthropic AI models through the Genkit plugin system.
 * It allows users to interact with various Claude models by providing an API key and optional configuration.
 *
 * The main export is the `anthropic` plugin, which can be configured with an API key either directly or through
 * environment variables. It initializes the Anthropic client and makes available the Claude models for use.
 *
 * Exports:
 * - anthropic: The main plugin function to interact with the Anthropic AI.
 *
 * Usage:
 * To use the Claude models, initialize the anthropic plugin inside `genkit()` and pass the configuration options. If no API key is provided in the options, the environment variable `ANTHROPIC_API_KEY` must be set.
 *
 * Example:
 * ```
 * import { anthropic } from '@genkit-ai/anthropic';
 * import { genkit } from 'genkit';
 *
 * const ai = genkit({
 *  plugins: [
 *    anthropic({ apiKey: 'your-api-key' })
 *    ... // other plugins
 *  ]
 * });
 *
 * // Access models via the plugin's model() method:
 * const model = anthropic.model('claude-sonnet-4');
 * ```
 */
function anthropicPlugin(options?: PluginOptions): GenkitPluginV2 {
  const client = getAnthropicClient(options);
  const defaultApiVersion = options?.apiVersion;

  let listActionsCache: ActionMetadata[] | null = null;

  return genkitPluginV2({
    name: 'anthropic',
    init: async () => {
      const actions: ModelAction[] = [];
      for (const name of Object.keys(KNOWN_CLAUDE_MODELS)) {
        const action = claudeModel({
          name,
          client,
          defaultApiVersion,
        });
        actions.push(action);
      }
      return actions;
    },
    resolve: (actionType: ActionType, name: string) => {
      if (actionType === 'model') {
        // Strip the 'anthropic/' namespace prefix if present
        const modelName = name.startsWith('anthropic/') ? name.slice(10) : name;
        return claudeModel({
          name: modelName,
          client,
          defaultApiVersion,
        });
      }
      return undefined;
    },
    list: async () => {
      if (listActionsCache) return listActionsCache;
      listActionsCache = await listActions(client);
      return listActionsCache;
    },
  });
}

export type AnthropicPlugin = {
  (pluginOptions?: PluginOptions): GenkitPluginV2;
  model(
    name: KnownClaudeModels | (ClaudeModelName & {}),
    config?: ClaudeConfig
  ): ModelReference<AnthropicConfigSchemaType>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
};

/**
 * Anthropic AI plugin for Genkit.
 * Includes Claude models (3, 3.5, and 4 series).
 */
export const anthropic = anthropicPlugin as AnthropicPlugin;
(anthropic as any).model = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  return claudeModelReference(name, config);
};

/**
 * Creates a custom part representing an Anthropic document with optional citations support.
 *
 * Use this to provide documents to Claude that can be cited in responses.
 * Citations must be enabled on all or none of the documents in a request.
 *
 * @example
 * ```ts
 * import { anthropic, anthropicDocument } from '@genkit-ai/anthropic';
 *
 * const { text } = await ai.generate({
 *   model: anthropic.model('claude-sonnet-4-5'),
 *   messages: [{
 *     role: 'user',
 *     content: [
 *       anthropicDocument({
 *         source: { type: 'text', data: 'The grass is green. The sky is blue.' },
 *         title: 'Nature Facts',
 *         citations: { enabled: true }
 *       }),
 *       { text: 'What color is the grass?' }
 *     ]
 *   }]
 * });
 * ```
 */
export function anthropicDocument(options: AnthropicDocumentOptions): Part {
  return {
    custom: {
      anthropicDocument: options,
    },
  };
}

export default anthropic;
