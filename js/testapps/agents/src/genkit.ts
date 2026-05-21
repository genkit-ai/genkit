/**
 * Copyright 2026 Google LLC
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

import { anthropic } from '@genkit-ai/anthropic';
import { openAICompatible } from '@genkit-ai/compat-oai';
import { googleAI } from '@genkit-ai/google-genai';
import { vertexModelGarden } from '@genkit-ai/vertexai/modelgarden';
import type { ModelArgument } from 'genkit';
import { genkit } from 'genkit/beta';
import { GenkitPluginV2 } from 'genkit/plugin';

// ---------------------------------------------------------------------------
// Model provider configuration — controlled by GENKIT_MODEL_PROVIDER env var.
//
// Each provider entry specifies the plugins to load, a default (capable)
// model, and a lite (fast/cheap) model used for auxiliary tasks like safety
// checks and decomposition steps.
// ---------------------------------------------------------------------------

type ProviderConfig = {
  plugins: () => GenkitPluginV2[];
  defaultModel: ModelArgument;
  liteModel: ModelArgument;
};

const providers: Record<string, ProviderConfig> = {
  'google-ai': {
    plugins: () => [googleAI()],
    defaultModel: googleAI.model('gemini-flash-latest').withConfig({
      thinkingConfig: {
        thinkingLevel: 'HIGH',
        includeThoughts: true,
      },
    }),
    liteModel: 'googleai/gemini-flash-lite-latest',
  },
  anthropic: {
    plugins: () => [anthropic({ apiVersion: 'beta' })],
    defaultModel: anthropic.model('claude-opus-4-7'),
    liteModel: anthropic.model('claude-haiku-4-5'),
  },
  openai: {
    plugins: () => [openAICompatible({ name: 'openai' })],
    defaultModel: 'openai/gpt-5.5',
    liteModel: 'openai/gpt-5.5-mini',
  },
  'vertex-claude': {
    plugins: () => [vertexModelGarden({ location: 'us-east5' })],
    defaultModel: vertexModelGarden.model('claude-opus-4-6'),
    liteModel: vertexModelGarden.model('claude-haiku-4-5'),
  },
};

const providerName = process.env.GENKIT_MODEL_PROVIDER || 'google-ai';
const config = providers[providerName] ?? providers['google-ai'];

/** The default (capable) model for the active provider. */
export const defaultModel = config.defaultModel;

/** A fast/cheap model for auxiliary tasks (safety checks, decomposition, etc.). */
export const liteModel = config.liteModel;

export const ai = genkit({
  plugins: config.plugins(),
  model: defaultModel,
});
