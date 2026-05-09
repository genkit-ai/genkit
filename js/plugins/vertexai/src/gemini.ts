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

import {
  FunctionCallingMode,
  FunctionDeclarationSchemaType,
  UsageMetadata,
  type Content,
  type FunctionDeclaration,
  type Part as GeminiPart,
  type GenerateContentCandidate,
  type GenerateContentResponse,
  type GenerativeModelPreview,
  type GoogleSearchRetrieval,
  type GoogleSearchRetrievalTool,
  type HarmBlockThreshold,
  type HarmCategory,
  type SafetySetting,
  type Schema,
  type StartChatParams,
  type ToolConfig,
  type VertexAI,
} from '@google-cloud/vertexai';
import { ApiClient } from '@google-cloud/vertexai/build/src/resources/index.js';
import { GenkitError, z, type Genkit, type JSONSchema } from 'genkit';
import {
  GenerationCommonConfigDescriptions,
  GenerationCommonConfigSchema,
  getBasicUsageStats,
  modelRef,
  type CandidateData,
  type GenerateRequest,
  type MediaPart,
  type MessageData,
  type ModelAction,
  type ModelInfo,
  type ModelMiddleware,
  type ModelReference,
  type Part,
  type ToolDefinitionSchema,
} from 'genkit/model';
import {
  downloadRequestMedia,
  simulateSystemPrompt,
} from 'genkit/model/middleware';
import { runInNewSpan } from 'genkit/tracing';
import { GoogleAuth } from 'google-auth-library';
import { getGenkitClientHeader } from './common/index.js';
import type { PluginOptions } from './common/types.js';
import { handleCacheIfNeeded } from './context-caching/index.js';
import { extractCacheConfig } from './context-caching/utils.js';

// Extra type guard to keep the compiler happy and avoid a cast to any. The
// legacy Gemini SDK is no longer maintained, and doesn't have updated types.
// However, the REST API returns the data we want.
type ExtendedUsageMetadata = UsageMetadata & {
  thoughtsTokenCount?: number;
};

/**
 * @deprecated
 */
export const SafetySettingsSchema = z.object({
  category: z.enum([
    /** The harm category is unspecified. */
    'HARM_CATEGORY_UNSPECIFIED',
    /** The harm category is hate speech. */
    'HARM_CATEGORY_HATE_SPEECH',
    /** The harm category is dangerous content. */
    'HARM_CATEGORY_DANGEROUS_CONTENT',
    /** The harm category is harassment. */
    'HARM_CATEGORY_HARASSMENT',
    /** The harm category is sexually explicit content. */
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
  ]),
  threshold: z.enum([
    'BLOCK_LOW_AND_ABOVE',
    'BLOCK_MEDIUM_AND_ABOVE',
    'BLOCK_ONLY_HIGH',
    'BLOCK_NONE',
  ]),
});

const VertexRetrievalSchema = z.object({
  datastore: z
    .object({
      projectId: z.string().describe('Google Cloud Project ID.').optional(),
      location: z
        .string()
        .describe('Google Cloud region e.g. us-central1.')
        .optional(),
      dataStoreId: z
        .string()
        .describe(
          'The data store id, when project id and location are provided as ' +
            'separate options. Alternatively, the full path to the data ' +
            'store should be provided in the form: "projects/{project}/' +
            'locations/{location}/collections/default_collection/dataStores/{data_store}".'
        ),
    })
    .describe('Vertex AI Search data store details'),
  disableAttribution: z
    .boolean()
    .describe(
      'Disable using the search data in detecting grounding attribution. This ' +
        'does not affect how the result is given to the model for generation.'
    )
    .optional(),
});

const GoogleSearchRetrievalSchema = z.object({
  disableAttribution: z
    .boolean()
    .describe(
      'Disable using the search data in detecting grounding attribution. This ' +
        'does not affect how the result is given to the model for generation.'
    )
    .optional(),
});

/**
 * Zod schema of Gemini model options.
 * Please refer to: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig, for further information.
 * @deprecated
 */
export const GeminiConfigSchema = GenerationCommonConfigSchema.extend({
  temperature: z
    .number()
    .min(0.0)
    .max(2.0)
    .describe(
      GenerationCommonConfigDescriptions.temperature +
        ' The default value is 1.0.'
    )
    .optional(),
  topP: z
    .number()
    .min(0)
    .max(1.0)
    .describe(
      GenerationCommonConfigDescriptions.topP + ' The default value is 0.95.'
    )
    .optional(),
  location: z
    .string()
    .describe('Google Cloud region e.g. us-central1.')
    .optional(),

  /**
   * Safety filter settings. See: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters#configurable-filters
   *
   * E.g.
   *
   * ```js
   * config: {
   *   safetySettings: [
   *     {
   *       category: 'HARM_CATEGORY_HATE_SPEECH',
   *       threshold: 'BLOCK_LOW_AND_ABOVE',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
   *       threshold: 'BLOCK_MEDIUM_AND_ABOVE',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_HARASSMENT',
   *       threshold: 'BLOCK_ONLY_HIGH',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
   *       threshold: 'BLOCK_NONE',
   *     },
   *   ],
   * }
   * ```
   */
  safetySettings: z
    .array(SafetySettingsSchema)
    .describe(
      'Adjust how likely you are to see responses that could be harmful. ' +
        'Content is blocked based on the probability that it is harmful.'
    )
    .optional(),

  /**
   * Vertex retrieval options.
   *
   * E.g.
   *
   * ```js
   *   config: {
   *     vertexRetrieval: {
   *       datastore: {
   *         projectId: 'your-cloud-project',
   *         location: 'us-central1',
   *         collection: 'your-collection',
   *       },
   *       disableAttribution: true,
   *     }
   *   }
   * ```
   */
  vertexRetrieval: VertexRetrievalSchema.describe(
    'Retrieve from Vertex AI Search data store for grounding ' +
      'generative responses.'
  ).optional(),

  /**
   * Google Search retrieval options.
   *
   * ```js
   *   config: {
   *     googleSearchRetrieval: {
   *       disableAttribution: true,
   *     }
   *   }
   * ```
   */
  googleSearchRetrieval: GoogleSearchRetrievalSchema.describe(
    'Retrieve public web data for grounding, powered by Google Search.'
  ).optional(),

  /**
   * Function calling options.
   *
   * E.g. forced tool call:
   *
   * ```js
   *   config: {
   *     functionCallingConfig: {
   *       mode: 'ANY',
   *     }
   *   }
   * ```
   */
  functionCallingConfig: z
    .object({
      mode: z.enum(['MODE_UNSPECIFIED', 'AUTO', 'ANY', 'NONE']).optional(),
      allowedFunctionNames: z.array(z.string()).optional(),
    })
    .describe(
      'Controls how the model uses the provided tools (function declarations). ' +
        'With AUTO (Default) mode, the model decides whether to generate a ' +
        'natural language response or suggest a function call based on the ' +
        'prompt and context. With ANY, the model is constrained to always ' +
        'predict a function call and guarantee function schema adherence. ' +
        'With NONE, the model is prohibited from making function calls.'
    )
    .optional(),
  thinkingConfig: z
    .object({
      includeThoughts: z
        .boolean()
        .describe(
          'Indicates whether to include thoughts in the response.' +
            'If true, thoughts are returned only when available.'
        )
        .optional(),
      thinkingBudget: z
        .number()
        .min(0)
        .max(24576)
        .describe(
          'The thinking budget parameter gives the model guidance on the ' +
            'number of thinking tokens it can use when generating a response. ' +
            'A greater number of tokens is typically associated with more detailed ' +
            'thinking, which is needed for solving more complex tasks. ' +
            'Setting the thinking budget to 0 disables thinking.'
        )
        .optional(),
    })
    .optional(),
}).passthrough();

/**
 * Known model names, to allow code completion for convenience. Allows other model names.
 * @deprecated
 */
export type GeminiVersionString =
  | keyof typeof SUPPORTED_GEMINI_MODELS
  | (string & {});

/**
 * Returns a reference to a model that can be used in generate calls.
 *
 * ```js
 * await ai.generate({
 *   prompt: 'hi',
 *   model: gemini('gemini-1.5-flash')
 * });
 * ```
 * @deprecated
 */
export function gemini(
  version: GeminiVersionString,
  options: GeminiConfig = {}
): ModelReference<typeof GeminiConfigSchema> {
  const nearestModel = nearestGeminiModelRef(version);
  return modelRef({
    name: `vertexai/${version}`,
    config: options,
    configSchema: GeminiConfigSchema,
    info: {
      ...nearestModel.info,
      // If exact suffix match for a known model, use its label, otherwise create a new label
      label: nearestModel.name.endsWith(version)
        ? nearestModel.info?.label
        : `Vertex AI - ${version}`,
    },
  });
}

function nearestGeminiModelRef(
  version: GeminiVersionString,
  options: GeminiConfig = {}
): ModelReference<typeof GeminiConfigSchema> {
  const matchingKey = longestMatchingPrefix(
    version,
    Object.keys(SUPPORTED_GEMINI_MODELS)
  );
  if (matchingKey) {
    return SUPPORTED_GEMINI_MODELS[matchingKey].withConfig({
      ...options,
      version,
    });
  }
  return GENERIC_GEMINI_MODEL.withConfig({ ...options, version });
}

function longestMatchingPrefix(version: string, potentialMatches: string[]) {
  return potentialMatches
    .filter((p) => version.startsWith(p))
    .reduce(
      (longest, current) =>
        current.length > longest.length ? current : longest,
      ''
    );
}

/**
 * Gemini model configuration options.
 *
 * E.g.
 * ```js
 *   config: {
 *     temperature: 0.9,
 *     maxOutputTokens: 300,
 *     safetySettings: [
 *       {
 *         category: 'HARM_CATEGORY_HATE_SPEECH',
 *         threshold: 'BLOCK_LOW_AND_ABOVE',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
 *         threshold: 'BLOCK_MEDIUM_AND_ABOVE',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_HARASSMENT',
 *         threshold: 'BLOCK_ONLY_HIGH',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
 *         threshold: 'BLOCK_NONE',
 *       },
 *     ],
 *     functionCallingConfig: {
 *       mode: 'ANY',
 *     }
 *   }
 * ```
 * @deprecated
 */
export type GeminiConfig = z.infer<typeof GeminiConfigSchema>;

/**
 * @deprecated
 */
export const gemini10Pro = modelRef({
  name: 'vertexai/gemini-1.0-pro',
  info: {
    label: 'Vertex AI - Gemini Pro',
    versions: ['gemini-1.0-pro-001', 'gemini-1.0-pro-002'],
    supports: {
      multiturn: true,
      media: false,
      tools: true,
      systemRole: true,
      constrained: 'no-tools',
      toolChoice: true,
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini15Pro = modelRef({
  name: 'vertexai/gemini-1.5-pro',
  info: {
    label: 'Vertex AI - Gemini 1.5 Pro',
    versions: ['gemini-1.5-pro-001', 'gemini-1.5-pro-002'],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini15Flash = modelRef({
  name: 'vertexai/gemini-1.5-flash',
  info: {
    label: 'Vertex AI - Gemini 1.5 Flash',
    versions: ['gemini-1.5-flash-001', 'gemini-1.5-flash-002'],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini20Flash001 = modelRef({
  name: 'vertexai/gemini-2.0-flash-001',
  info: {
    label: 'Vertex AI - Gemini 2.0 Flash 001',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini20Flash = modelRef({
  name: 'vertexai/gemini-2.0-flash',
  info: {
    label: 'Vertex AI - Gemini 2.0 Flash',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini20FlashLite = modelRef({
  name: 'vertexai/gemini-2.0-flash-lite',
  info: {
    label: 'Vertex AI - Gemini 2.0 Flash Lite',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini20FlashLitePreview0205 = modelRef({
  name: 'vertexai/gemini-2.0-flash-lite-preview-02-05',
  info: {
    label: 'Vertex AI - Gemini 2.0 Flash Lite Preview 02-05',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini20ProExp0205 = modelRef({
  name: 'vertexai/gemini-2.0-pro-exp-02-05',
  info: {
    label: 'Vertex AI - Gemini 2.0 Flash Pro Experimental 02-05',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini25FlashPreview0417 = modelRef({
  name: 'vertexai/gemini-2.5-flash-preview-04-17',
  info: {
    label: 'Vertex AI - Gemini 2.5 Flash Preview 04-17',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini25ProExp0325 = modelRef({
  name: 'vertexai/gemini-2.5-pro-exp-03-25',
  info: {
    label: 'Vertex AI - Gemini 2.5 Pro Experimental 03-25',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated
 */
export const gemini25ProPreview0325 = modelRef({
  name: 'vertexai/gemini-2.5-pro-preview-03-25',
  info: {
    label: 'Vertex AI - Gemini 2.5 Pro Preview 03-25',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated After importing from the new plugin, use vertexAI.model('gemini-2.5-pro')
 */
export const gemini25Pro = modelRef({
  name: 'vertexai/gemini-2.5-pro',
  info: {
    label: 'Vertex AI - Gemini 2.5 Pro',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});
/**
 * @deprecated After importing from the new plugin, use vertexAI.model('gemini-2.5-flash')
 */
export const gemini25Flash = modelRef({
  name: 'vertexai/gemini-2.5-flash',
  info: {
    label: 'Vertex AI - Gemini 2.5 Flash',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/**
 * @deprecated After importing from the new plugin, use vertexAI.model('gemini-2.5-flash-lite')
 */
export const gemini25FlashLite = modelRef({
  name: 'vertexai/gemini-2.5-flash-lite',
  info: {
    label: 'Vertex AI - Gemini 2.5 Flash Lite',
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  configSchema: GeminiConfigSchema,
});

/** @deprecated */
export const GENERIC_GEMINI_MODEL = modelRef({
  name: 'vertexai/gemini',
  configSchema: GeminiConfigSchema,
  info: {
    label: 'Google Gemini',
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
    },
  },
});

const SUPPORTED_V1_MODELS = {
  'gemini-1.0-pro': gemini10Pro,
};

/** @deprecated */
export const SUPPORTED_V15_MODELS = {
  'gemini-1.5-pro': gemini15Pro,
  'gemini-1.5-flash': gemini15Flash,
  'gemini-2.0-flash': gemini20Flash,
  'gemini-2.0-flash-001': gemini20Flash001,
  'gemini-2.0-flash-lite': gemini20FlashLite,
  'gemini-2.0-flash-lite-preview-02-05': gemini20FlashLitePreview0205,
  'gemini-2.0-pro-exp-02-05': gemini20ProExp0205,
  'gemini-2.5-pro-exp-03-25': gemini25ProExp0325,
  'gemini-2.5-pro-preview-03-25': gemini25ProPreview0325,
  'gemini-2.5-flash-preview-04-17': gemini25FlashPreview0417,
  'gemini-2.5-flash': gemini25Flash,
  'gemini-2.5-pro': gemini25Pro,
  'gemini-2.5-flash-lite': gemini25FlashLite,
};

/** @deprecated */
export const SUPPORTED_GEMINI_MODELS = {
  ...SUPPORTED_V15_MODELS,
} as const;

function toGeminiRole(
  role: MessageData['role'],
  modelInfo?: ModelInfo
): string {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'model';
    case 'system':
      if (modelInfo && modelInfo.supports?.systemRole) {
        // We should have already pulled out the supported system messages,
        // anything remaining is unsupported; throw an error.
        throw new Error(
          'system role is only supported for a single message in the first position'
        );
      } else {
        throw new Error('system role is not supported');
      }
    case 'tool':
      return 'function';
    default:
      return 'user';
  }
}

/**
 * @hidden
 * @deprecated
 */
export const toGeminiTool = (
  tool: z.infer<typeof ToolDefinitionSchema>
): FunctionDeclaration => {
  const declaration: FunctionDeclaration = {
    name: tool.name.replace(/\//g, '__'), // Gemini throws on '/' in tool name
    description: tool.description,
    parameters: convertSchemaProperty(tool.inputSchema),
  };
  return declaration;
};

const toGeminiFileDataPart = (part: MediaPart): GeminiPart => {
  const media = part.media;
  if (media.url.startsWith('gs://') || media.url.startsWith('http')) {
    if (!media.contentType)
      throw new Error(
        'Must supply contentType when using media from http(s):// or gs:// URLs.'
      );
    return {
      fileData: {
        mimeType: media.contentType,
        fileUri: media.url,
      },
    };
  } else if (media.url.startsWith('data:')) {
    const dataUrl = media.url;
    const b64Data = dataUrl.substring(dataUrl.indexOf(',')! + 1);
    const contentType =
      media.contentType ||
      dataUrl.substring(dataUrl.indexOf(':')! + 1, dataUrl.indexOf(';'));
    return { inlineData: { mimeType: contentType, data: b64Data } };
  }

  throw Error(
    'Could not convert genkit part to gemini tool response part: missing file data'
  );
};

const toGeminiToolRequestPart = (part: Part): GeminiPart => {
  if (!part?.toolRequest?.input) {
    throw Error(
      'Could not convert genkit part to gemini tool response part: missing tool request data'
    );
  }
  return {
    functionCall: {
      name: part.toolRequest.name,
      args: part.toolRequest.input,
    },
  };
};

const toGeminiToolResponsePart = (part: Part): GeminiPart => {
  if (!part?.toolResponse?.output) {
    throw Error(
      'Could not convert genkit part to gemini tool response part: missing tool response data'
    );
  }
  return {
    functionResponse: {
      name: part.toolResponse.name,
      response: {
        name: part.toolResponse.name,
        content: part.toolResponse.output,
      },
    },
  };
};

/** @deprecated */
export function toGeminiSystemInstruction(message: MessageData): Content {
  return {
    role: 'user',
    parts: message.content.map(toGeminiPart),
  };
}

/** @deprecated */
export function toGeminiMessage(
  message: MessageData,
  modelInfo?: ModelInfo
): Content {
  let sortedParts = message.content;
  if (message.role === 'tool') {
    sortedParts = [...message.content].sort((a, b) => {
      const aRef = a.toolResponse?.ref;
      const bRef = b.toolResponse?.ref;
      if (!aRef && !bRef) return 0;
      if (!aRef) return 1;
      if (!bRef) return -1;
      return Number.parseInt(aRef, 10) - Number.parseInt(bRef, 10);
    });
  }
  return {
    role: toGeminiRole(message.role, modelInfo),
    parts: sortedParts.map(toGeminiPart),
  };
}

function fromGeminiFinishReason(
  reason: GenerateContentCandidate['finishReason']
): CandidateData['finishReason'] {
  if (!reason) return 'unknown';
  switch (reason) {
    case 'STOP':
      return 'stop';
    case 'MAX_TOKENS':
      return 'length';
    case 'SAFETY': // blocked for safety
    case 'RECITATION': // blocked for reciting training data
      return 'blocked';
    default:
      return 'unknown';
  }
}

function toGeminiPart(part: Part): GeminiPart {
  if (part.text) return { text: part.text };
  if (part.media) return toGeminiFileDataPart(part);
  if (part.toolRequest) return toGeminiToolRequestPart(part);
  if (part.toolResponse) return toGeminiToolResponsePart(part);
  if (typeof part.reasoning === 'string') return toGeminiThought(part);

  throw new Error(`Unsupported Gemini part type: ${JSON.stringify(part)}`);
}

function fromGeminiInlineDataPart(part: GeminiPart): MediaPart {
  // Check if the required properties exist
  if (
    !part.inlineData ||
    !part.inlineData.hasOwnProperty('mimeType') ||
    !part.inlineData.hasOwnProperty('data')
  ) {
    throw new Error('Invalid GeminiPart: missing required properties');
  }
  const { mimeType, data } = part.inlineData;
  // Combine data and mimeType into a data URL
  const dataUrl = `data:${mimeType};base64,${data}`;
  return {
    media: {
      url: dataUrl,
      contentType: mimeType,
    },
  };
}

function toGeminiThought(part: Part) {
  const outPart: any = { thought: true };
  if (part.metadata?.thoughtSignature)
    outPart.thoughtSignature = part.metadata.thoughtSignature;
  if (part.reasoning?.length) outPart.text = part.reasoning;
  return outPart;
}

function fromGeminiFileDataPart(part: GeminiPart): MediaPart {
  if (
    !part.fileData ||
    !part.fileData.hasOwnProperty('mimeType') ||
    !part.fileData.hasOwnProperty('url')
  ) {
    throw new Error(
      'Invalid Gemini File Data Part: missing required properties'
    );
  }

  return {
    media: {
      url: part.fileData?.fileUri,
      contentType: part.fileData?.mimeType,
    },
  };
}

function fromGeminiFunctionCallPart(part: GeminiPart, ref?: string): Part {
  if (!part.functionCall) {
    throw new Error(
      'Invalid Gemini Function Call Part: missing function call data'
    );
  }
  return {
    toolRequest: {
      name: part.functionCall.name,
      input: part.functionCall.args,
      ref,
    },
  };
}

function fromGeminiFunctionResponsePart(part: GeminiPart, ref?: string): Part {
  if (!part.functionResponse) {
    throw new Error(
      'Invalid Gemini Function Call Part: missing function call data'
    );
  }
  return {
    toolResponse: {
      name: part.functionResponse.name.replace(/__/g, '/'), // restore slashes
      output: part.functionResponse.response,
      ref,
    },
  };
}

// Converts vertex part to genkit part
function fromGeminiPart(
  part: GeminiPart,
  jsonMode: boolean,
  ref?: string
): Part {
  if ('thought' in part) return fromGeminiThought(part as any);
  if (typeof part.text === 'string') return { text: part.text };
  if (part.inlineData) return fromGeminiInlineDataPart(part);
  if (part.fileData) return fromGeminiFileDataPart(part);
  if (part.functionCall) return fromGeminiFunctionCallPart(part, ref);
  if (part.functionResponse) return fromGeminiFunctionResponsePart(part, ref);

  throw new Error(
    'Part type is unsupported/corrupted. Either data is missing or type cannot be inferred from type.'
  );
}

/** @deprecated */
export function fromGeminiCandidate(
  candidate: GenerateContentCandidate,
  jsonMode: boolean
): CandidateData {
  const parts = candidate.content.parts || [];

  const genkitCandidate: CandidateData = {
    index: candidate.index || 0,
    message: {
      role: 'model',
      content: parts.map((part, index) => {
        return fromGeminiPart(part, jsonMode, index.toString());
      }),
    },
    finishReason: fromGeminiFinishReason(candidate.finishReason),
    finishMessage: candidate.finishMessage,
    custom: {
      safetyRatings: candidate.safetyRatings,
      citationMetadata: candidate.citationMetadata,
    },
  };
  return genkitCandidate;
}

function fromGeminiThought(part: {
  thought: boolean;
  text?: string;
  thoughtSignature?: string;
}): Part {
  return {
    reasoning: part.text || '',
    metadata: { thoughtSignature: (part as any).thoughtSignature },
  };
}

// Translate JSON schema to Vertex AI's format. Specifically, the type field needs be mapped.
// Since JSON schemas can include nested arrays/objects, we have to recursively map the type field
// in all nested fields.
function convertSchemaProperty(property) {
  if (!property || !property.type) {
    return undefined;
  }
  const baseSchema = {} as Schema;
  if (property.description) {
    baseSchema.description = property.description;
  }
  if (property.enum) {
    baseSchema.enum = property.enum;
  }
  if (property.nullable) {
    baseSchema.nullable = property.nullable;
  }
  let propertyType;
  // nullable schema can ALSO be defined as, for example, type=['string','null']
  if (Array.isArray(property.type)) {
    const types = property.type as string[];
    if (types.includes('null')) {
      baseSchema.nullable = true;
    }
    // grab the type that's not `null`
    propertyType = types.find((t) => t !== 'null');
  } else {
    propertyType = property.type;
  }
  if (propertyType === 'object') {
    const nestedProperties = {};
    Object.keys(property.properties ?? {}).forEach((key) => {
      nestedProperties[key] = convertSchemaProperty(property.properties[key]);
    });
    return {
      ...baseSchema,
      type: FunctionDeclarationSchemaType.OBJECT,
      properties: nestedProperties,
      required: property.required,
    };
  } else if (propertyType === 'array') {
    return {
      ...baseSchema,
      type: FunctionDeclarationSchemaType.ARRAY,
      items: convertSchemaProperty(property.items),
    };
  } else {
    const schemaType = FunctionDeclarationSchemaType[
      propertyType.toUpperCase()
    ] as FunctionDeclarationSchemaType;
    if (!schemaType) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Unsupported property type ${propertyType.toUpperCase()}`,
      });
    }
    return {
      ...baseSchema,
      type: schemaType,
    };
  }
}

/** @deprecated */
export function cleanSchema(schema: JSONSchema): JSONSchema {
  const out = structuredClone(schema);
  for (const key in out) {
    if (key === '$schema' || key === 'additionalProperties') {
      delete out[key];
      continue;
    }
    if (typeof out[key] === 'object') {
      out[key] = cleanSchema(out[key]);
    }
    // Zod nullish() and picoschema optional fields will produce type `["string", "null"]`
    // which is not supported by the model API. Convert them to just `"string"`.
    if (key === 'type' && Array.isArray(out[key])) {
      // find the first that's not `null`.
      out[key] = out[key].find((t) => t !== 'null');
    }
  }
  return out;
}

/**
 * Define a Vertex AI Gemini model.
 * @deprecated
 */
export function defineGeminiKnownModel(
  ai: Genkit,
  name: string,
  vertexClientFactory: (
    request: GenerateRequest<typeof GeminiConfigSchema>
  ) => VertexAI,
  options: PluginOptions,
  debugTraces?: boolean
): ModelAction {
  const modelName = `vertexai/${name}`;

  const model: ModelReference<z.ZodTypeAny> = SUPPORTED_GEMINI_MODELS[name];
  if (!model) throw new Error(`Unsupported model: ${name}`);

  return defineGeminiModel({
    ai,
    modelName,
    version: name,
    modelInfo: model?.info,
    vertexClientFactory,
    options,
    debugTraces,
  });
}

/**
 * Define a Vertex AI Gemini model.
 * @deprecated
 */
export function defineGeminiModel({
  ai,
  modelName,
  version,
  modelInfo,
  vertexClientFactory,
  options,
  debugTraces,
}: {
  ai: Genkit;
  modelName: string;
  version: string;
  modelInfo: ModelInfo | undefined;
  vertexClientFactory: (
    request: GenerateRequest<typeof GeminiConfigSchema>
  ) => VertexAI;
  options: PluginOptions;
  debugTraces?: boolean;
}): ModelAction {
  const middlewares: ModelMiddleware[] = [];
  if (SUPPORTED_V1_MODELS[version]) {
    middlewares.push(simulateSystemPrompt());
  }
  if (modelInfo?.supports?.media) {
    // the gemini api doesn't support downloading media from http(s)
    middlewares.push(
      downloadRequestMedia({
        maxBytes: 1024 * 1024 * 20,
        filter: (part) => {
          try {
            const url = new URL(part.media.url);
            if (
              // Gemini can handle these URLs
              ['www.youtube.com', 'youtube.com', 'youtu.be'].includes(
                url.hostname
              )
            )
              return false;
          } catch {}
          return true;
        },
      })
    );
  }

  return ai.defineModel(
    {
      name: modelName,
      ...modelInfo,
      configSchema: GeminiConfigSchema,
      use: middlewares,
    },
    async (request, sendChunk) => {
      const vertex = vertexClientFactory(request);

      // Make a copy of messages to avoid side-effects
      const messages = [...request.messages];
      if (messages.length === 0) throw new Error('No messages provided.');

      // Handle system instructions separately
      let systemInstruction: Content | undefined = undefined;
      if (!SUPPORTED_V1_MODELS[version]) {
        const systemMessage = messages.find((m) => m.role === 'system');
        if (systemMessage) {
          messages.splice(messages.indexOf(systemMessage), 1);
          systemInstruction = toGeminiSystemInstruction(systemMessage);
        }
      }

      const requestConfig = request.config as z.infer<
        typeof GeminiConfigSchema
      >;
      const {
        functionCallingConfig,
        version: versionFromConfig,
        googleSearchRetrieval,
        tools: toolsFromConfig,
        vertexRetrieval,
        location, // location can be overridden via config, take it out.
        safetySettings,
        ...restOfConfig
      } = requestConfig;

      const tools = request.tools?.length
        ? [{ functionDeclarations: request.tools.map(toGeminiTool) }]
        : [];

      let toolConfig: ToolConfig | undefined;
      if (functionCallingConfig) {
        toolConfig = {
          functionCallingConfig: {
            allowedFunctionNames: functionCallingConfig.allowedFunctionNames,
            mode: toFunctionModeEnum(functionCallingConfig.mode),
          },
        };
      } else if (request.toolChoice) {
        toolConfig = {
          functionCallingConfig: {
            mode: toGeminiFunctionModeEnum(request.toolChoice),
          },
        };
      }

      // Cannot use tools and function calling at the same time
      const jsonMode =
        (request.output?.format === 'json' || !!request.output?.schema) &&
        tools.length === 0;

      const chatRequest: StartChatParams = {
        systemInstruction,
        tools,
        toolConfig,
        history: messages
          .slice(0, -1)
          .map((message) => toGeminiMessage(message, modelInfo)),
        generationConfig: {
          ...restOfConfig,
          candidateCount: request.candidates || undefined,
          responseMimeType: jsonMode ? 'application/json' : undefined,
        },
        safetySettings: toGeminiSafetySettings(safetySettings),
      };

      // Handle cache
      const modelVersion = (versionFromConfig || version) as string;
      const cacheConfigDetails = extractCacheConfig(request);

      const apiClient = new ApiClient(
        options.projectId!,
        options.location,
        'v1beta1',
        new GoogleAuth(options.googleAuth!)
      );

      const { chatRequest: updatedChatRequest, cache } =
        await handleCacheIfNeeded(
          apiClient,
          request,
          chatRequest,
          modelVersion,
          cacheConfigDetails
        );

      let genModel: GenerativeModelPreview;

      if (jsonMode && request.output?.constrained) {
        updatedChatRequest.generationConfig!.responseSchema = cleanSchema(
          request.output.schema
        );
      }

      if (toolsFromConfig) {
        if (!updatedChatRequest.tools) updatedChatRequest.tools = [];
        updatedChatRequest.tools.push(...(toolsFromConfig as any[]));
      }

      if (googleSearchRetrieval) {
        if (!updatedChatRequest.tools) updatedChatRequest.tools = [];
        // Gemini 1.5 models use googleSearchRetrieval, newer models use googleSearch.
        if (modelName.startsWith('vertexai/gemini-1.5')) {
          updatedChatRequest.tools.push({
            googleSearchRetrieval:
              googleSearchRetrieval as GoogleSearchRetrieval,
          } as GoogleSearchRetrievalTool);
        } else {
          updatedChatRequest.tools.push({
            googleSearch: googleSearchRetrieval as GoogleSearchRetrieval,
          } as GoogleSearchRetrievalTool);
        }
      }

      if (vertexRetrieval) {
        if (!updatedChatRequest.tools) updatedChatRequest.tools = [];
        const _projectId =
          vertexRetrieval.datastore.projectId || options.projectId;
        const _location =
          vertexRetrieval.datastore.location || options.location;
        const _dataStoreId = vertexRetrieval.datastore.dataStoreId;
        const datastore = `projects/${_projectId}/locations/${_location}/collections/default_collection/dataStores/${_dataStoreId}`;
        updatedChatRequest.tools.push({
          retrieval: {
            vertexAiSearch: {
              datastore,
            },
            disableAttribution: vertexRetrieval.disableAttribution,
          },
        });
      }

      const msg = toGeminiMessage(messages[messages.length - 1], modelInfo);

      if (cache) {
        genModel = vertex.preview.getGenerativeModelFromCachedContent(
          cache,
          {
            model: modelVersion,
          },
          {
            apiClient: getGenkitClientHeader(),
          }
        );
      } else {
        genModel = vertex.preview.getGenerativeModel(
          {
            model: modelVersion,
          },
          {
            apiClient: getGenkitClientHeader(),
          }
        );
      }

      const callGemini = async () => {
        let response: GenerateContentResponse;

        // Handle streaming and non-streaming responses
        if (sendChunk) {
          const result = await genModel
            .startChat(updatedChatRequest)
            .sendMessageStream(msg.parts);

          for await (const item of result.stream) {
            (item as GenerateContentResponse).candidates?.forEach(
              (candidate) => {
                const c = fromGeminiCandidate(candidate, jsonMode);
                sendChunk({
                  index: c.index,
                  content: c.message.content,
                });
              }
            );
          }

          response = await result.response;
        } else {
          const result = await genModel
            .startChat(updatedChatRequest)
            .sendMessage(msg.parts);

          response = result.response;
        }

        if (!response.candidates?.length) {
          throw new GenkitError({
            status: 'FAILED_PRECONDITION',
            message: 'No valid candidates returned.',
          });
        }

        const candidateData = response.candidates.map((c) =>
          fromGeminiCandidate(c, jsonMode)
        );

        const usageMetadata = response.usageMetadata as ExtendedUsageMetadata;

        return {
          candidates: candidateData,
          custom: response,
          usage: {
            ...getBasicUsageStats(request.messages, candidateData),
            inputTokens: usageMetadata?.promptTokenCount,
            outputTokens: usageMetadata?.candidatesTokenCount,
            totalTokens: usageMetadata?.totalTokenCount,
            thoughtsTokens: usageMetadata?.thoughtsTokenCount,
            cachedContentTokens: usageMetadata?.cachedContentTokenCount,
          },
        };
      };

      // If debugTraces is enable, we wrap the actual model call with a span, add raw
      // API params as for input.
      return debugTraces
        ? await runInNewSpan(
            ai.registry,
            {
              metadata: {
                name: sendChunk ? 'sendMessageStream' : 'sendMessage',
              },
            },
            async (metadata) => {
              metadata.input = {
                sdk: '@google-cloud/vertexai',
                cache: cache,
                model: genModel.getModelName(),
                chatOptions: updatedChatRequest,
                parts: msg.parts,
                options,
              };
              const response = await callGemini();
              metadata.output = response.custom;
              return response;
            }
          )
        : await callGemini();
    }
  );
}

/** Converts mode from the config, which follows Gemini naming convention. */
function toFunctionModeEnum(
  enumMode: string | undefined
): FunctionCallingMode | undefined {
  if (enumMode === undefined) {
    return undefined;
  }
  switch (enumMode) {
    case 'MODE_UNSPECIFIED': {
      return FunctionCallingMode.MODE_UNSPECIFIED;
    }
    case 'ANY': {
      return FunctionCallingMode.ANY;
    }
    case 'AUTO': {
      return FunctionCallingMode.AUTO;
    }
    case 'NONE': {
      return FunctionCallingMode.NONE;
    }
    default:
      throw new Error(`unsupported function calling mode: ${enumMode}`);
  }
}

function toGeminiSafetySettings(
  genkitSettings?: z.infer<typeof SafetySettingsSchema>[]
): SafetySetting[] | undefined {
  if (!genkitSettings) return undefined;
  return genkitSettings.map((s) => {
    return {
      category: s.category as HarmCategory,
      threshold: s.threshold as HarmBlockThreshold,
    };
  });
}

/** Converts mode from genkit tool choice. */
function toGeminiFunctionModeEnum(
  genkitMode: 'auto' | 'required' | 'none'
): FunctionCallingMode | undefined {
  if (genkitMode === undefined) {
    return undefined;
  }
  switch (genkitMode) {
    case 'required': {
      return FunctionCallingMode.ANY;
    }
    case 'auto': {
      return FunctionCallingMode.AUTO;
    }
    case 'none': {
      return FunctionCallingMode.NONE;
    }
    default:
      throw new Error(`unsupported function calling mode: ${genkitMode}`);
  }
}
