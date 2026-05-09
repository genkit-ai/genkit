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

import { MistralGoogleCloud } from '@mistralai/mistralai-gcp';
import {
  ChatCompletionChoiceFinishReason,
  ToolTypes,
  type AssistantMessage,
  type ChatCompletionRequest,
  type ChatCompletionResponse,
  type CompletionChunk,
  type FunctionCall,
  type Tool as MistralTool,
  type SystemMessage,
  type ToolCall,
  type ToolMessage,
  type UserMessage,
} from '@mistralai/mistralai-gcp/models/components';
import {
  GenerationCommonConfigSchema,
  z,
  type GenerateRequest,
  type Genkit,
  type MessageData,
  type ModelReference,
  type ModelResponseData,
  type Part,
  type Role,
  type ToolRequestPart,
} from 'genkit';
import {
  GenerationCommonConfigDescriptions,
  modelRef,
  type ModelAction,
} from 'genkit/model';
import { getGenkitClientHeader } from '../../common/index.js';

/**
 * See https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
 * @deprecated
 */
export const MistralConfigSchema = GenerationCommonConfigSchema.extend({
  // TODO: Update this with all the parameters in
  // https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post.
  location: z.string().optional(),
  topP: z
    .number()
    .describe(
      GenerationCommonConfigDescriptions.topP + ' The default value is 1.'
    )
    .optional(),
});

/**
 * @deprecated
 */
export const mistralLarge = modelRef({
  name: 'vertexai/mistral-large',
  info: {
    label: 'Vertex AI Model Garden - Mistral Large',
    versions: ['mistral-large-2411', 'mistral-large-2407'],
    supports: {
      multiturn: true,
      media: false,
      tools: true,
      systemRole: true,
      output: ['text'],
    },
  },
  configSchema: MistralConfigSchema,
});

/**
 * @deprecated
 */
export const mistralNemo = modelRef({
  name: 'vertexai/mistral-nemo',
  info: {
    label: 'Vertex AI Model Garden - Mistral Nemo',
    versions: ['mistral-nemo-2407'],
    supports: {
      multiturn: true,
      media: false,
      tools: false,
      systemRole: true,
      output: ['text'],
    },
  },
  configSchema: MistralConfigSchema,
});

/**
 * @deprecated
 */
export const codestral = modelRef({
  name: 'vertexai/codestral',
  info: {
    label: 'Vertex AI Model Garden - Codestral',
    versions: ['codestral-2405'],
    supports: {
      multiturn: true,
      media: false,
      tools: false,
      systemRole: true,
      output: ['text'],
    },
  },
  configSchema: MistralConfigSchema,
});

/**
 * @deprecated
 */
export const SUPPORTED_MISTRAL_MODELS: Record<
  string,
  ModelReference<typeof MistralConfigSchema>
> = {
  'mistral-large': mistralLarge,
  'mistral-nemo': mistralNemo,
  codestral: codestral,
};

// TODO: Do they export a type for this?
type MistralRole = 'assistant' | 'user' | 'tool' | 'system';

function toMistralRole(role: Role): MistralRole {
  switch (role) {
    case 'model':
      return 'assistant';
    case 'user':
      return 'user';
    case 'tool':
      return 'tool';
    case 'system':
      return 'system';
    default:
      throw new Error(`Unknwon role ${role}`);
  }
}
function toMistralToolRequest(toolRequest: Record<string, any>): FunctionCall {
  if (!toolRequest.name) {
    throw new Error('Tool name is required');
  }

  return {
    name: toolRequest.name,
    // Mistral expects arguments as either a string or object
    arguments:
      typeof toolRequest.input === 'string'
        ? toolRequest.input
        : JSON.stringify(toolRequest.input),
  };
}

/**
 * @deprecated
 */
export function toMistralRequest(
  model: string,
  input: GenerateRequest<typeof MistralConfigSchema>
): ChatCompletionRequest {
  const messages = input.messages.map((msg) => {
    // Handle regular text messages
    if (msg.content.every((part) => part.text)) {
      const content = msg.content.map((part) => part.text || '').join('');
      return {
        role: toMistralRole(msg.role),
        content,
      };
    }

    // Handle assistant's tool/function calls
    const toolRequest = msg.content.find((part) => part.toolRequest);
    if (toolRequest?.toolRequest) {
      const functionCall = toMistralToolRequest(toolRequest.toolRequest);
      return {
        role: 'assistant' as const,
        content: null,
        toolCalls: [
          {
            id: toolRequest.toolRequest.ref,
            type: ToolTypes.Function,
            function: {
              name: functionCall.name,
              arguments: functionCall.arguments,
            },
          },
        ],
      };
    }

    // Handle tool responses
    const toolResponse = msg.content.find((part) => part.toolResponse);
    if (toolResponse?.toolResponse) {
      return {
        role: 'tool' as const,
        name: toolResponse.toolResponse.name,
        content: JSON.stringify(toolResponse.toolResponse.output),
        toolCallId: toolResponse.toolResponse.ref, // This must match the id from tool_calls
      };
    }

    return {
      role: toMistralRole(msg.role),
      content: msg.content.map((part) => part.text || '').join(''),
    };
  });

  validateToolSequence(messages); // This line exists but might not be running?

  const request: ChatCompletionRequest = {
    model,
    messages,
    maxTokens: input.config?.maxOutputTokens ?? 1024,
    temperature: input.config?.temperature ?? 0.7,
    ...(input.config?.topP && { topP: input.config.topP }),
    ...(input.config?.stopSequences && { stop: input.config.stopSequences }),
    ...(input.tools && {
      tools: input.tools.map((tool) => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema || {},
        },
      })) as MistralTool[],
    }),
  };

  return request;
}
// Helper to convert Mistral AssistantMessage content into Genkit parts
function fromMistralTextPart(content: string): Part {
  return {
    text: content,
  };
}

// Helper to convert Mistral ToolCall into Genkit parts
function fromMistralToolCall(toolCall: ToolCall): ToolRequestPart {
  if (!toolCall.function) {
    throw new Error('Tool call must include a function definition');
  }

  return {
    toolRequest: {
      ref: toolCall.id,
      name: toolCall.function.name,
      input:
        typeof toolCall.function.arguments === 'string'
          ? JSON.parse(toolCall.function.arguments)
          : toolCall.function.arguments,
    },
  };
}

// Converts Mistral AssistantMessage content into Genkit parts
function fromMistralMessage(message: AssistantMessage): Part[] {
  const parts: Part[] = [];

  // Handle textual content
  if (typeof message.content === 'string') {
    parts.push(fromMistralTextPart(message.content));
  } else if (Array.isArray(message.content)) {
    // If content is an array of ContentChunk, handle each chunk
    message.content.forEach((chunk) => {
      if (chunk.type === 'text') {
        parts.push(fromMistralTextPart(chunk.text));
      }
      // Add support for other ContentChunk types here if needed
    });
  }

  // Handle tool calls if present
  if (message.toolCalls) {
    message.toolCalls.forEach((toolCall) => {
      parts.push(fromMistralToolCall(toolCall));
    });
  }

  return parts;
}

/**
 * Maps Mistral finish reasons to Genkit finish reasons
 * @deprecated
 */
export function fromMistralFinishReason(
  reason: ChatCompletionChoiceFinishReason | undefined
): 'length' | 'unknown' | 'stop' | 'blocked' | 'other' {
  switch (reason) {
    case ChatCompletionChoiceFinishReason.Stop:
      return 'stop';
    case ChatCompletionChoiceFinishReason.Length:
    case ChatCompletionChoiceFinishReason.ModelLength:
      return 'length';
    case ChatCompletionChoiceFinishReason.Error:
      return 'other'; // Map generic errors to "other"
    case ChatCompletionChoiceFinishReason.ToolCalls:
      return 'stop'; // Assuming tool calls signify a "stop" in processing
    default:
      return 'other'; // For undefined or unmapped reasons
  }
}

/**
 * Converts a Mistral response to a Genkit response
 * @deprecated
 */
export function fromMistralResponse(
  _input: GenerateRequest<typeof MistralConfigSchema>,
  response: ChatCompletionResponse
): ModelResponseData {
  const firstChoice = response.choices?.[0];
  // Convert content from Mistral response to Genkit parts
  const contentParts: Part[] = firstChoice?.message
    ? fromMistralMessage(firstChoice.message)
    : [];

  const message: MessageData = {
    role: 'model',
    content: contentParts,
  };

  return {
    message,
    finishReason: fromMistralFinishReason(firstChoice?.finishReason),
    usage: {
      inputTokens: response.usage.promptTokens,
      outputTokens: response.usage.completionTokens,
    },
    custom: {
      id: response.id,
      model: response.model,
      created: response.created,
    },
    raw: response, // Include the raw response for debugging or additional context
  };
}

/** @deprecated */
export function mistralModel(
  ai: Genkit,
  modelName: string,
  projectId: string,
  region: string
): ModelAction {
  const getClient = createClientFactory(projectId);

  const model = SUPPORTED_MISTRAL_MODELS[modelName];
  if (!model) {
    throw new Error(`Unsupported Mistral model name ${modelName}`);
  }

  return ai.defineModel(
    {
      name: model.name,
      label: model.info?.label,
      configSchema: MistralConfigSchema,
      supports: model.info?.supports,
      versions: model.info?.versions,
    },
    async (input, sendChunk) => {
      const client = getClient(input.config?.location || region);

      const versionedModel =
        input.config?.version ?? model.info?.versions?.[0] ?? model.name;

      if (!sendChunk) {
        const mistralRequest = toMistralRequest(versionedModel, input);

        const response = await client.chat.complete(mistralRequest, {
          fetchOptions: {
            headers: {
              'X-Goog-Api-Client': getGenkitClientHeader(),
            },
          },
        });

        return fromMistralResponse(input, response);
      } else {
        const mistralRequest = toMistralRequest(versionedModel, input);
        const stream = await client.chat.stream(mistralRequest, {
          fetchOptions: {
            headers: {
              'X-Goog-Api-Client': getGenkitClientHeader(),
            },
          },
        });

        for await (const event of stream) {
          const parts = fromMistralCompletionChunk(event.data);
          if (parts.length > 0) {
            sendChunk({
              content: parts,
            });
          }
        }

        // Get the complete response after streaming
        const completeResponse = await client.chat.complete(mistralRequest, {
          fetchOptions: {
            headers: {
              'X-Goog-Api-Client': getGenkitClientHeader(),
            },
          },
        });

        return fromMistralResponse(input, completeResponse);
      }
    }
  );
}

function createClientFactory(projectId: string) {
  const clients: Record<string, MistralGoogleCloud> = {};

  return (region: string): MistralGoogleCloud => {
    if (!region) {
      throw new Error('Region is required to create Mistral client');
    }

    try {
      if (!clients[region]) {
        clients[region] = new MistralGoogleCloud({
          region,
          projectId,
        });
      }
      return clients[region];
    } catch (error) {
      throw new Error(
        `Failed to create/retrieve Mistral client for region ${region}: ${error}`
      );
    }
  };
}

type MistralMessage =
  | AssistantMessage
  | ToolMessage
  | SystemMessage
  | UserMessage;

// Helper function to validate tool calls and responses match
function validateToolSequence(messages: MistralMessage[]) {
  const toolCalls = (
    messages.filter((m) => {
      return m.role === 'assistant' && m.toolCalls;
    }) as AssistantMessage[]
  ).reduce((acc: ToolCall[], m) => {
    if (m.toolCalls) {
      return [...acc, ...m.toolCalls];
    }
    return acc;
  }, []);

  const toolResponses = messages.filter(
    (m) => m.role === 'tool'
  ) as ToolMessage[];

  if (toolCalls.length !== toolResponses.length) {
    throw new Error(
      `Mismatch between tool calls (${toolCalls.length}) and responses (${toolResponses.length})`
    );
  }

  toolResponses.forEach((response) => {
    const matchingCall = toolCalls.find(
      (call) => call.id === response.toolCallId
    );
    if (!matchingCall) {
      throw new Error(
        `Tool response with ID ${response.toolCallId} has no matching call`
      );
    }
  });
}

/** @deprecated */
export function fromMistralCompletionChunk(chunk: CompletionChunk): Part[] {
  if (!chunk.choices?.[0]?.delta) return [];

  const delta = chunk.choices[0].delta;
  const parts: Part[] = [];

  if (typeof delta.content === 'string') {
    parts.push({ text: delta.content });
  }

  if (delta.toolCalls) {
    delta.toolCalls.forEach((toolCall) => {
      if (!toolCall.function) return;

      parts.push({
        toolRequest: {
          ref: toolCall.id,
          name: toolCall.function.name,
          input:
            typeof toolCall.function.arguments === 'string'
              ? JSON.parse(toolCall.function.arguments)
              : toolCall.function.arguments,
        },
      });
    });
  }

  return parts;
}
