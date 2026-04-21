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

import type {
  ContentBlock as AnthropicContent,
  ImageBlockParam,
  Message,
  MessageCreateParamsBase,
  MessageParam,
  TextBlock,
  TextBlockParam,
  TextDelta,
  Tool,
  ToolResultBlockParam,
  ToolUseBlock,
  ToolUseBlockParam,
} from '@anthropic-ai/sdk/resources/messages';
import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';
import {
  ActionMetadata,
  z,
  type GenerateRequest,
  type Part as GenkitPart,
  type MessageData,
  type ModelReference,
  type ModelResponseData,
  type Part,
} from 'genkit';
import {
  GenerationCommonConfigSchema,
  ModelInfo,
  getBasicUsageStats,
  modelRef,
  type ModelAction,
} from 'genkit/model';
import { model as pluginModel } from 'genkit/plugin';
import { getGenkitClientHeader } from '../../common/index.js';
import { PluginOptions } from './types.js';
import { checkModelName } from './utils.js';

export const AnthropicConfigSchema = GenerationCommonConfigSchema.extend({
  location: z.string().optional(),
}).passthrough();
export type AnthropicConfigSchemaType = typeof AnthropicConfigSchema;
export type AnthropicConfig = z.infer<AnthropicConfigSchemaType>;

// All the config schema types
type ConfigSchemaType = AnthropicConfigSchemaType;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = AnthropicConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertex-model-garden/${name}`,
    configSchema,
    info: info ?? {
      supports: {
        multiturn: true,
        media: true,
        tools: true,
        systemRole: true,
        output: ['text'],
      },
    },
  });
}

export const GENERIC_MODEL = commonRef('anthropic');

export const KNOWN_MODELS = {
  'claude-sonnet-4-6': commonRef('claude-sonnet-4-6'),
  'claude-opus-4-6': commonRef('claude-opus-4-6'),
  'claude-haiku-4-5@20251001': commonRef('claude-haiku-4-5@20251001'),
  'claude-sonnet-4-5@20250929': commonRef('claude-sonnet-4-5@20250929'),
  'claude-sonnet-4@20250514': commonRef('claude-sonnet-4@20250514'),
  'claude-opus-4-5@20251101': commonRef('claude-opus-4-5@20251101'),
  'claude-opus-4-1@20250805': commonRef('claude-opus-4-1@20250805'),
  'claude-opus-4@20250514': commonRef('claude-opus-4@20250514'),
};
export type KnownModels = keyof typeof KNOWN_MODELS;
export type AnthropicModelName = `claude-${string}`;
export function isAnthropicModelName(
  value?: string
): value is AnthropicModelName {
  return !!value?.startsWith('claude-');
}

export function model(
  version: string,
  options: AnthropicConfig = {}
): ModelReference<AnthropicConfigSchemaType> {
  const name = checkModelName(version);

  return modelRef({
    name: `vertex-model-garden/${name}`,
    config: options,
    configSchema: AnthropicConfigSchema,
    info: {
      ...GENERIC_MODEL.info,
    },
  });
}

export interface ClientOptions {
  location: string; // e.g. 'us-central1' or 'global'
  projectId: string;
}

export function listActions(clientOptions: ClientOptions): ActionMetadata[] {
  // TODO: figure out where to get the list of models.
  return [];
}

export function listKnownModels(
  clientOptions: ClientOptions,
  pluginOptions?: PluginOptions
) {
  return Object.keys(KNOWN_MODELS).map((name) =>
    defineModel(name, clientOptions, pluginOptions)
  );
}

export function defineModel(
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: PluginOptions
): ModelAction {
  const clients: Record<string, AnthropicVertex> = {};
  const clientFactory = (region: string): AnthropicVertex => {
    if (!clients[region]) {
      clients[region] = new AnthropicVertex({
        region: region,
        projectId: clientOptions.projectId,
        defaultHeaders: {
          'X-Goog-Api-Client': getGenkitClientHeader(),
        },
      });
    }
    return clients[region];
  };
  const ref = model(name);

  return pluginModel(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
    },
    async (request, { streamingRequested, sendChunk }) => {
      const client = clientFactory(
        request.config?.location || clientOptions.location
      );
      const modelVersion = checkModelName(ref.name);
      const anthropicRequest = toAnthropicRequest(modelVersion, request);
      if (!streamingRequested) {
        // Non-streaming
        const response = await client.messages.create({
          ...anthropicRequest,
          stream: false,
        });
        return fromAnthropicResponse(request, response);
      } else {
        // Streaming
        const stream = await client.messages.stream(anthropicRequest);
        for await (const event of stream) {
          if (event.type === 'content_block_delta') {
            sendChunk({
              index: 0,
              content: [
                {
                  text: (event.delta as TextDelta).text,
                },
              ],
            });
          }
        }
        return fromAnthropicResponse(request, await stream.finalMessage());
      }
    }
  );
}

export function toAnthropicRequest(
  model: string,
  input: GenerateRequest<typeof AnthropicConfigSchema>
): MessageCreateParamsBase {
  let system: string | undefined = undefined;
  const messages: MessageParam[] = [];
  for (const msg of input.messages) {
    if (msg.role === 'system') {
      system = msg.content
        .map((c) => {
          if (!c.text) {
            throw new Error(
              'Only text context is supported for system messages.'
            );
          }
          return c.text;
        })
        .join();
    }
    // If the last message is a tool response, we need to add a user message.
    // https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-result-content-blocks
    else if (msg.content[msg.content.length - 1].toolResponse) {
      messages.push({
        role: 'user',
        content: toAnthropicContent(msg.content),
      });
    } else {
      messages.push({
        role: toAnthropicRole(msg.role),
        content: toAnthropicContent(msg.content),
      });
    }
  }
  const request = {
    model,
    messages,
    // https://docs.anthropic.com/claude/docs/models-overview#model-comparison
    max_tokens: input.config?.maxOutputTokens ?? 4096,
  } as MessageCreateParamsBase;
  if (system) {
    request['system'] = system;
  }
  if (input.tools) {
    request.tools = input.tools?.map((tool) => {
      return {
        name: tool.name,
        description: tool.description,
        input_schema: tool.inputSchema,
      };
    }) as Array<Tool>;
  }
  if (input.config?.stopSequences) {
    request.stop_sequences = input.config?.stopSequences;
  }
  if (input.config?.temperature) {
    request.temperature = input.config?.temperature;
  }
  if (input.config?.topK) {
    request.top_k = input.config?.topK;
  }
  if (input.config?.topP) {
    request.top_p = input.config?.topP;
  }
  return request;
}

function toAnthropicContent(
  content: GenkitPart[]
): Array<
  TextBlockParam | ImageBlockParam | ToolUseBlockParam | ToolResultBlockParam
> {
  return content.map((p) => {
    if (p.text) {
      return {
        type: 'text',
        text: p.text,
      };
    }
    if (p.media) {
      let b64Data = p.media.url;
      if (b64Data.startsWith('data:')) {
        b64Data = b64Data.substring(b64Data.indexOf(',')! + 1);
      }

      return {
        type: 'image',
        source: {
          type: 'base64',
          data: b64Data,
          media_type: p.media.contentType as
            | 'image/jpeg'
            | 'image/png'
            | 'image/gif'
            | 'image/webp',
        },
      };
    }
    if (p.toolRequest) {
      return toAnthropicToolRequest(p.toolRequest);
    }
    if (p.toolResponse) {
      return toAnthropicToolResponse(p);
    }
    throw new Error(`Unsupported content type: ${JSON.stringify(p)}`);
  });
}

function toAnthropicRole(role): 'user' | 'assistant' {
  if (role === 'model') {
    return 'assistant';
  }
  if (role === 'user') {
    return 'user';
  }
  if (role === 'tool') {
    return 'assistant';
  }
  throw new Error(`Unsupported role type ${role}`);
}

function fromAnthropicTextPart(part: TextBlock): Part {
  return {
    text: part.text,
  };
}

function fromAnthropicToolCallPart(part: ToolUseBlock): Part {
  return {
    toolRequest: {
      name: part.name,
      input: part.input,
      ref: part.id,
    },
  };
}

// Converts an Anthropic part to a Genkit part.
function fromAnthropicPart(part: AnthropicContent): Part {
  if (part.type === 'text') return fromAnthropicTextPart(part);
  if (part.type === 'tool_use') return fromAnthropicToolCallPart(part);
  throw new Error(
    'Part type is unsupported/corrupted. Either data is missing or type cannot be inferred from type.'
  );
}

// Converts an Anthropic response to a Genkit response.
export function fromAnthropicResponse(
  input: GenerateRequest<typeof AnthropicConfigSchema>,
  response: Message
): ModelResponseData {
  const parts = response.content as AnthropicContent[];
  const message: MessageData = {
    role: 'model',
    content: parts.map(fromAnthropicPart),
  };
  return {
    message,
    finishReason: toGenkitFinishReason(
      response.stop_reason as
        | 'end_turn'
        | 'max_tokens'
        | 'stop_sequence'
        | 'tool_use'
        | null
    ),
    custom: {
      id: response.id,
      model: response.model,
      type: response.type,
    },
    usage: {
      ...getBasicUsageStats(input.messages, message),
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    },
  };
}

function toGenkitFinishReason(
  reason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | null
): ModelResponseData['finishReason'] {
  switch (reason) {
    case 'end_turn':
      return 'stop';
    case 'max_tokens':
      return 'length';
    case 'stop_sequence':
      return 'stop';
    case 'tool_use':
      return 'stop';
    case null:
      return 'unknown';
    default:
      return 'other';
  }
}

function toAnthropicToolRequest(tool: Record<string, any>): ToolUseBlock {
  if (!tool.name) {
    throw new Error('Tool name is required');
  }
  // Validate the tool name, Anthropic only supports letters, numbers, and underscores.
  // https://docs.anthropic.com/en/docs/build-with-claude/tool-use#specifying-tools
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(tool.name)) {
    throw new Error(
      `Tool name ${tool.name} contains invalid characters.
      Only letters, numbers, and underscores are allowed,
      and the name must be between 1 and 64 characters long.`
    );
  }
  const declaration: ToolUseBlock = {
    type: 'tool_use',
    id: tool.ref,
    name: tool.name,
    input: tool.input,
  };
  return declaration;
}

function toAnthropicToolResponse(part: Part): ToolResultBlockParam {
  if (!part.toolResponse?.ref) {
    throw new Error('Tool response reference is required');
  }

  if (!part.toolResponse.output) {
    throw new Error('Tool response output is required');
  }

  return {
    type: 'tool_result',
    tool_use_id: part.toolResponse.ref,
    content: JSON.stringify(part.toolResponse.output),
  };
}
