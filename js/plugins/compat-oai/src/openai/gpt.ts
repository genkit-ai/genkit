/**
 * Copyright 2024 The Fire Company
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
import { ModelInfo, ModelReference } from 'genkit/model';
import {
  ChatCompletionCommonConfigSchema,
  compatOaiModelRef,
} from '../model.js';
import { lookup, toModelName } from '../utils.js';

/** OpenAI Custom configuration schema. */
export const OpenAIChatCompletionConfigSchema =
  ChatCompletionCommonConfigSchema.extend({
    store: z.boolean().optional(),
    strict: z.boolean().optional(),
  });

/** OpenAI ModelRef helper, with OpenAI specific config. */
export function openAIModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: any;
}): ModelReference<typeof OpenAIChatCompletionConfigSchema> {
  return compatOaiModelRef({
    ...params,
    info: params.info,
    configSchema: OpenAIChatCompletionConfigSchema,
    namespace: 'openai',
  });
}

/**
 * When a Chat Completions request may include function tools:
 * - `always`: no restriction.
 * - `effortNone`: only when the effective reasoning effort is `none`.
 * - `never`: not at all.
 */
type ToolsSupport = 'always' | 'effortNone' | 'never';

interface GptModelSpec {
  /** Capabilities other than tools and toolChoice, which follow from `tools`. */
  supports: Omit<NonNullable<ModelInfo['supports']>, 'tools' | 'toolChoice'>;
  tools: ToolsSupport;
  /** Announced shutdown date (YYYY-MM-DD); sets stage 'deprecated'. */
  shutdownDate?: string;
}

const CHAT_SUPPORTS = {
  multiturn: true,
  media: true,
  systemRole: true,
  output: ['text', 'json'],
  constrained: 'all',
} as const satisfies GptModelSpec['supports'];
const O_SERIES_SUPPORTS = {
  multiturn: true,
  media: true,
  systemRole: false,
  output: ['text', 'json'],
  constrained: 'all',
} as const satisfies GptModelSpec['supports'];
const SEARCH_SUPPORTS = {
  multiturn: true,
  media: true,
  systemRole: true,
} as const satisfies GptModelSpec['supports'];

/**
 * The OpenAI chat catalog: one entry per registered model id. Model info
 * (stage, supports.tools, supports.toolChoice) is derived from these entries.
 */
const GPT_MODEL_SPECS = {
  'gpt-4o': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2027-01-14',
  },
  'gpt-4o-mini': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2027-01-14',
  },
  'gpt-4o-mini-2024-07-18': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2027-01-14',
  },
  'gpt-4.1': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2027-01-14',
  },
  'gpt-4.1-mini': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2027-01-14',
  },
  o3: {
    supports: O_SERIES_SUPPORTS,
    tools: 'always',
    shutdownDate: '2026-12-11',
  },
  'gpt-5': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2026-12-11',
  },
  'gpt-5-mini': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2026-12-11',
  },
  'gpt-5-nano': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2026-12-11',
  },
  'gpt-5-search-api': { supports: SEARCH_SUPPORTS, tools: 'never' },
  'gpt-5.1': {
    supports: CHAT_SUPPORTS,
    tools: 'always',
    shutdownDate: '2027-04-01',
  },
  'gpt-5.2': { supports: CHAT_SUPPORTS, tools: 'always' },
  'gpt-5.4': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-5.4-mini': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-5.4-nano': {
    supports: CHAT_SUPPORTS,
    tools: 'effortNone',
    shutdownDate: '2027-04-01',
  },
  'gpt-5.5': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-5.6-luna': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-5.6-terra': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-5.6-sol': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-6-astra': { supports: CHAT_SUPPORTS, tools: 'never' },
  'gpt-6-sol': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-6-luna': { supports: CHAT_SUPPORTS, tools: 'effortNone' },
  'gpt-6.1-sol': { supports: CHAT_SUPPORTS, tools: 'never' },
  'gpt-6.1-terra': { supports: CHAT_SUPPORTS, tools: 'never' },
} as const satisfies Record<string, GptModelSpec>;

type GptModelId = keyof typeof GPT_MODEL_SPECS;

function gptModelInfo(spec: GptModelSpec): ModelInfo {
  const tools = spec.tools !== 'never';
  return {
    supports: { ...spec.supports, tools, toolChoice: tools },
    ...(spec.shutdownDate ? { stage: 'deprecated' as const } : {}),
  };
}

export const SUPPORTED_GPT_MODELS = Object.fromEntries(
  Object.entries(GPT_MODEL_SPECS).map(([id, spec]) => [
    id,
    openAIModelRef({ name: id, info: gptModelInfo(spec) }),
  ])
) as {
  [K in GptModelId]: ModelReference<typeof OpenAIChatCompletionConfigSchema>;
};

const SNAPSHOT_DATE_SUFFIX = /-\d{4}-\d{2}-\d{2}$/;

/** Shut-down snapshots of catalogued aliases, which must not get live info. */
const SHUT_DOWN_SNAPSHOTS: ReadonlySet<string> = new Set(['gpt-4o-2024-05-13']);

function gptModelSpec(name: string): GptModelSpec | undefined {
  const id = toModelName(name, 'openai');
  const spec = lookup(GPT_MODEL_SPECS, id);
  if (spec || SHUT_DOWN_SNAPSHOTS.has(id)) {
    return spec;
  }
  return lookup(GPT_MODEL_SPECS, id.replace(SNAPSHOT_DATE_SUFFIX, ''));
}

/**
 * Returns a chat model reference for an OpenAI model id. Catalogued ids and
 * their live dated snapshots carry catalog info (including stage); other ids
 * get generic info.
 */
export function gptModelRef(
  name: string
): ModelReference<typeof OpenAIChatCompletionConfigSchema> {
  const spec = gptModelSpec(name);
  return openAIModelRef({ name, info: spec && gptModelInfo(spec) });
}
