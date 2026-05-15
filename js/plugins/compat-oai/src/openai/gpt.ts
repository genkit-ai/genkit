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
import { ChatCompletionCommonConfigSchema, compatOaiModelRef } from '../model';

const MULTIMODAL_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: true,
    systemRole: true,
    output: ['text', 'json'],
    constrained: 'all',
  },
};

/** OpenAI Custom configuration schema. */
export const OpenAIChatCompletionConfigSchema =
  ChatCompletionCommonConfigSchema.extend({
    store: z.boolean().optional(),
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

const gpt45 = openAIModelRef({
  name: 'gpt-4.5',
  info: MULTIMODAL_MODEL_INFO,
});
const gpt45Preview = openAIModelRef({
  name: 'gpt-4.5-preview',
  info: MULTIMODAL_MODEL_INFO,
});

const gpt4o = openAIModelRef({
  name: 'gpt-4o',
  info: MULTIMODAL_MODEL_INFO,
});
const gpt4o20240513 = openAIModelRef({
  name: 'gpt-4o-2024-05-13',
  info: MULTIMODAL_MODEL_INFO,
});

const o1 = openAIModelRef({
  name: 'o1',
  info: {
    supports: {
      multiturn: true,
      tools: true,
      media: true,
      systemRole: false,
      output: ['text', 'json'],
    },
  },
});

const o3 = openAIModelRef({
  name: 'o3',
  info: {
    supports: {
      multiturn: true,
      tools: true,
      media: true,
      systemRole: false,
      output: ['text', 'json'],
    },
  },
});

const o3Mini = openAIModelRef({
  name: 'o3-mini',
  info: {
    supports: {
      multiturn: true,
      tools: true,
      media: false,
      systemRole: false,
      output: ['text', 'json'],
    },
  },
});

const o4Mini = openAIModelRef({
  name: 'o4-mini',
  info: {
    supports: {
      multiturn: true,
      tools: true,
      media: true,
      systemRole: false,
      output: ['text', 'json'],
    },
  },
});

const gpt4oMini = openAIModelRef({
  name: 'gpt-4o-mini',
  info: MULTIMODAL_MODEL_INFO,
});
const gpt4oMini20240718 = openAIModelRef({
  name: 'gpt-4o-mini-2024-07-18',
  info: MULTIMODAL_MODEL_INFO,
});

const gpt4Turbo = openAIModelRef({
  name: 'gpt-4-turbo',
  info: MULTIMODAL_MODEL_INFO,
});
const gpt4Turbo20240409 = openAIModelRef({
  name: 'gpt-4-turbo-2024-04-09',
  info: MULTIMODAL_MODEL_INFO,
});
const gpt4TurboPreview = openAIModelRef({
  name: 'gpt-4-turbo-preview',
  info: MULTIMODAL_MODEL_INFO,
});
const gpt40125Preview = openAIModelRef({
  name: 'gpt-4-0125-preview',
  info: MULTIMODAL_MODEL_INFO,
});
const gpt41106Preview = openAIModelRef({
  name: 'gpt-4-1106-preview',
  info: MULTIMODAL_MODEL_INFO,
});

const GPT_4_VISION_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: false,
    media: true,
    systemRole: true,
    output: ['text'],
  },
};
const gpt4Vision = openAIModelRef({
  name: 'gpt-4-vision',
  info: GPT_4_VISION_MODEL_INFO,
});
const gpt4VisionPreview = openAIModelRef({
  name: 'gpt-4-vision-preview',
  info: GPT_4_VISION_MODEL_INFO,
});
const gpt41106VisionPreview = openAIModelRef({
  name: 'gpt-4-1106-vision-preview',
  info: GPT_4_VISION_MODEL_INFO,
});

const GPT_4_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: false,
    systemRole: true,
    output: ['text'],
  },
};
const gpt4 = openAIModelRef({ name: 'gpt-4', info: GPT_4_MODEL_INFO });
const gpt40613 = openAIModelRef({
  name: 'gpt-4-0613',
  info: GPT_4_MODEL_INFO,
});
const gpt432k = openAIModelRef({
  name: 'gpt-4-32k',
  info: GPT_4_MODEL_INFO,
});
const gpt432k0613 = openAIModelRef({
  name: 'gpt-4-32k-0613',
  info: GPT_4_MODEL_INFO,
});

const GPT_35_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: false,
    systemRole: true,
    output: ['text', 'json'],
  },
};
const gpt35Turbo = openAIModelRef({
  name: 'gpt-3.5-turbo',
  info: GPT_35_MODEL_INFO,
});
const gpt35Turbo0125 = openAIModelRef({
  name: 'gpt-3.5-turbo-0125',
  info: GPT_35_MODEL_INFO,
});
const gpt35Turbo1106 = openAIModelRef({
  name: 'gpt-3.5-turbo-1106',
  info: GPT_35_MODEL_INFO,
});

const GPT_5_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: true,
    systemRole: true,
    output: ['text', 'json'],
  },
};
const gpt5 = openAIModelRef({
  name: 'gpt-5',
  info: GPT_5_MODEL_INFO,
});
const gpt5Mini = openAIModelRef({
  name: 'gpt-5-mini',
  info: GPT_5_MODEL_INFO,
});
const gpt5Nano = openAIModelRef({
  name: 'gpt-5-nano',
  info: GPT_5_MODEL_INFO,
});
const gpt5ChatLatest = openAIModelRef({
  name: 'gpt-5-chat-latest',
  info: {
    supports: {
      ...GPT_5_MODEL_INFO.supports,
      tools: false,
      output: ['text'],
    },
  },
});
const gpt5_1 = openAIModelRef({
  name: 'gpt-5.1',
  info: GPT_5_MODEL_INFO,
});

export const SUPPORTED_GPT_MODELS = {
  'gpt-4.5': gpt45,
  'gpt-4.5-preview': gpt45Preview,
  'gpt-4o': gpt4o,
  'gpt-4o-2024-05-13': gpt4o20240513,
  o1: o1,
  o3: o3,
  'o3-mini': o3Mini,
  'o4-mini': o4Mini,
  'gpt-4o-mini': gpt4oMini,
  'gpt-4o-mini-2024-07-18': gpt4oMini20240718,
  'gpt-4-turbo': gpt4Turbo,
  'gpt-4-turbo-2024-04-09': gpt4Turbo20240409,
  'gpt-4-turbo-preview': gpt4TurboPreview,
  'gpt-4-0125-preview': gpt40125Preview,
  'gpt-4-1106-preview': gpt41106Preview,
  'gpt-4-vision': gpt4Vision,
  'gpt-4-vision-preview': gpt4VisionPreview,
  'gpt-4-1106-vision-preview': gpt41106VisionPreview,
  'gpt-4': gpt4,
  'gpt-4-0613': gpt40613,
  'gpt-4-32k': gpt432k,
  'gpt-4-32k-0613': gpt432k0613,
  'gpt-3.5-turbo': gpt35Turbo,
  'gpt-3.5-turbo-0125': gpt35Turbo0125,
  'gpt-3.5-turbo-1106': gpt35Turbo1106,
  'gpt-5': gpt5,
  'gpt-5-mini': gpt5Mini,
  'gpt-5-nano': gpt5Nano,
  'gpt-5-chat-latest': gpt5ChatLatest,
  'gpt-5.1': gpt5_1,
} as const;
