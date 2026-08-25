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
import { ModelInfo, ModelReference } from 'genkit/model';
import {
  ResponsesCommonConfigSchema,
  compatOaiResponsesModelRef,
} from '../responses.js';

/**
 * Models OpenAI serves exclusively over `/v1/responses`, so they register on
 * that transport because they have no other one.
 *
 * Hand-curated: no type in the pinned SDK enumerates this set, and the list is
 * chased by this plugin the same way `SUPPORTED_GPT_MODELS` is. Deep-research
 * and computer-use models belong on a background action rather than this
 * synchronous runner and are deliberately absent.
 */
export const RESPONSES_ONLY_MODELS = [
  'gpt-5-pro',
  'gpt-5.2-pro',
  'gpt-5.5-pro',
  'gpt-5-codex',
  'gpt-5.1-codex',
  'gpt-5.1-codex-mini',
  'gpt-5.1-codex-max',
  'gpt-5.2-codex',
  'gpt-5.3-codex',
  'codex-mini-latest',
  'o1-pro',
  'o3-pro',
] as const;

type ResponsesOnlyBaseName = (typeof RESPONSES_ONLY_MODELS)[number];

/**
 * A model name served only over the Responses API: one of
 * {@link RESPONSES_ONLY_MODELS} or any name extending one of them.
 */
export type ResponsesOnlyModelName =
  | ResponsesOnlyBaseName
  | (`${ResponsesOnlyBaseName}-${string}` & {});

/**
 * Checks whether a model name must be served over the Responses API.
 *
 * Any suffixed form of a curated base name matches, so `o3-pro-2025-06-10` and
 * `gpt-5-pro-preview` route like `o3-pro` and `gpt-5-pro` rather than falling
 * through to Chat Completions and failing with an opaque OpenAI 400. The bias
 * is deliberate: a name extending one of these bases is near-certainly served
 * on the same transport as the base, so unknown suffixes should fail toward the
 * transport that works.
 * @param name The bare model name, without the plugin namespace.
 */
export function isResponsesOnlyModelName(
  name?: string
): name is ResponsesOnlyModelName {
  if (!name) return false;
  return RESPONSES_ONLY_MODELS.some(
    (base) => name === base || name.startsWith(`${base}-`)
  );
}

/**
 * Responses-only models that reject `stream: true`. A streaming caller gets
 * the completed response delivered as a single chunk instead. Hand-curated
 * like {@link RESPONSES_ONLY_MODELS}, from the per-model feature tables.
 */
export const NON_STREAMING_RESPONSES_MODELS = ['o1-pro', 'o3-pro'] as const;

/**
 * Checks whether a Responses model name must not be streamed, with the same
 * suffix bias as {@link isResponsesOnlyModelName}.
 * @param name The bare model name, without the plugin namespace.
 */
export function isNonStreamingResponsesModelName(name?: string): boolean {
  if (!name) return false;
  return NON_STREAMING_RESPONSES_MODELS.some(
    (base) => name === base || name.startsWith(`${base}-`)
  );
}

/** OpenAI Responses API custom configuration schema. */
export const OpenAIResponsesConfigSchema = ResponsesCommonConfigSchema.extend({
  store: z.boolean().optional(),
  previousResponseId: z.string().optional(),
  reasoningEffort: z.enum(['minimal', 'low', 'medium', 'high']).optional(),
  reasoningSummary: z.enum(['auto', 'concise', 'detailed']).optional(),
  transport: z.enum(['responses']).optional(),
}).passthrough();

const RESPONSES_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    toolChoice: true,
    media: true,
    systemRole: true,
    output: ['text', 'json'],
    // See GENERIC_RESPONSES_MODEL_INFO in responses.ts on why this is declared.
    constrained: 'all',
  },
};

/** OpenAI ModelRef helper for models served over the Responses API. */
export function openAIResponsesModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: any;
}): ModelReference<typeof OpenAIResponsesConfigSchema> {
  return compatOaiResponsesModelRef({
    ...params,
    info: params.info ?? RESPONSES_MODEL_INFO,
    configSchema: OpenAIResponsesConfigSchema,
    namespace: 'openai',
  });
}

export const SUPPORTED_RESPONSES_MODELS = {
  'gpt-5-pro': openAIResponsesModelRef({ name: 'gpt-5-pro' }),
  'gpt-5.2-pro': openAIResponsesModelRef({ name: 'gpt-5.2-pro' }),
  'gpt-5.5-pro': openAIResponsesModelRef({ name: 'gpt-5.5-pro' }),
  'gpt-5-codex': openAIResponsesModelRef({ name: 'gpt-5-codex' }),
  'gpt-5.1-codex': openAIResponsesModelRef({ name: 'gpt-5.1-codex' }),
  'gpt-5.1-codex-mini': openAIResponsesModelRef({
    name: 'gpt-5.1-codex-mini',
  }),
  'gpt-5.1-codex-max': openAIResponsesModelRef({ name: 'gpt-5.1-codex-max' }),
  'gpt-5.2-codex': openAIResponsesModelRef({ name: 'gpt-5.2-codex' }),
  'gpt-5.3-codex': openAIResponsesModelRef({ name: 'gpt-5.3-codex' }),
  'codex-mini-latest': openAIResponsesModelRef({ name: 'codex-mini-latest' }),
  'o1-pro': openAIResponsesModelRef({ name: 'o1-pro' }),
  'o3-pro': openAIResponsesModelRef({ name: 'o3-pro' }),
} as const;
