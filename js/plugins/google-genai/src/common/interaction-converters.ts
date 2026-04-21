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

import { GenerateResponseData, MessageData, Operation, Part } from 'genkit';
import { ToolDefinition } from 'genkit/model';
import {
  AudioContent,
  Content,
  DocumentContent,
  FunctionCallContent,
  FunctionResultContent,
  GeminiInteraction,
  ImageContent,
  InteractionFunctionTool,
  InteractionTool,
  TextContent,
  ThoughtContent,
  Turn,
  VideoContent,
} from './interaction-types.js';
import { cleanSchema } from './utils.js';

/**
 * Ensures that all tool requests and responses in a list of messages have unique reference IDs.
 *
 * This function performs two passes:
 * 1. Assigns generated IDs to tool requests that lack a `ref`.
 * 2. Assigns matching IDs to tool responses that lack a `ref`, assuming they correspond
 *    sequentially to the requests. If a response has no matching request, it gets an orphaned ID.
 *
 * @param messages - The list of messages to process.
 * @returns A deep copy of the messages with tool IDs ensured.
 */
export function ensureToolIds(messages: MessageData[]): MessageData[] {
  const generatedIds: string[] = [];
  let nextIdCounter = 0;

  // Deep copy to avoid mutating original request messages
  const newMessages = structuredClone(messages) as MessageData[];

  // First pass: find ToolRequests without ref
  for (const message of newMessages) {
    for (const part of message.content) {
      if (part.toolRequest && !part.toolRequest.ref) {
        const newId = `genkit-auto-id-${nextIdCounter++}`;
        part.toolRequest.ref = newId;
        generatedIds.push(newId);
      }
    }
  }

  // Second pass: find ToolResponses without ref and assign from queue
  // Note: This assumes responses are in the same order as requests.
  for (const message of newMessages) {
    for (const part of message.content) {
      if (part.toolResponse && !part.toolResponse.ref) {
        const id = generatedIds.shift();
        if (id) {
          part.toolResponse.ref = id;
        } else {
          // No matching request found (or queue empty).
          // Generate unique one to avoid empty string rejection.
          part.toolResponse.ref = `genkit-orphan-id-${nextIdCounter++}`;
        }
      }
    }
  }

  return newMessages;
}

/**
 * Converts a Genkit ToolDefinition to an InteractionTool format.
 *
 * Maps the name, description, and input schema (cleaned) to the interaction tool structure.
 *
 * @param tool - The Genkit tool definition.
 * @returns The converted InteractionTool.
 */
export function toInteractionTool(tool: ToolDefinition): InteractionTool {
  const func: InteractionFunctionTool = {
    type: 'function',
    name: tool.name,
    description: tool.description,
  };
  if (tool.inputSchema) {
    func.parameters = cleanSchema(tool.inputSchema);
  }
  return func;
}

/**
 * Converts a Genkit Part to an Interaction Content object.
 *
 * Handles various part types including text, media, tool requests (mapped to function calls),
 * and tool responses (mapped to function results).
 *
 * @param part - The Genkit message part.
 * @returns The corresponding Interaction Content object.
 * @throws Error if the part type is unsupported.
 */
export function toInteractionContent(part: Part): Content {
  if (part.text !== undefined) {
    return { type: 'text', text: part.text };
  }
  if (part.media) {
    return toInteractionMedia(part);
  }
  if (part.toolRequest) {
    return {
      type: 'function_call',
      name: part.toolRequest.name,
      arguments: part.toolRequest.input as Record<string, any>,
      id: part.toolRequest.ref || '',
    };
  }
  if (part.toolResponse) {
    let output = part.toolResponse.output;
    if (
      typeof output !== 'object' &&
      typeof output !== 'string' &&
      output !== undefined
    ) {
      output = { result: output };
    }
    return {
      type: 'function_result',
      name: part.toolResponse.name,
      result: output as Record<string, any> | string,
      call_id: part.toolResponse.ref || '',
    };
  }
  throw new Error('Unsupported part type for Interaction input');
}

function toInteractionMedia(part: Part): Content {
  if (!part.media) throw new Error('Media part missing media');
  const { url, contentType } = part.media;
  if (!contentType) throw new Error('Media part missing contentType');

  let data: string | undefined;
  let uri: string | undefined;

  if (url.startsWith('data:')) {
    data = url.substring(url.indexOf(',') + 1);
  } else {
    uri = url;
  }

  const out: Partial<Content> = { mime_type: contentType };
  if (data) out.data = data;
  if (uri) out.uri = uri;

  if (contentType.startsWith('image/')) {
    out.type = 'image';
    return out as ImageContent;
  }
  if (contentType.startsWith('audio/')) {
    out.type = 'audio';
    return out as AudioContent;
  }
  if (contentType.startsWith('video/')) {
    out.type = 'video';
    return out as VideoContent;
  }
  if (contentType === 'application/pdf') {
    out.type = 'document';
    return out as DocumentContent;
  }

  throw new Error(`Unsupported media type: ${contentType}`);
}

/**
 * Maps a Genkit message role to the corresponding Interaction API role.
 *
 * - 'user' -> 'user'
 * - 'model' -> 'model'
 * - 'tool' -> 'user' (Tool outputs are treated as user turns in this context)
 *
 * @param role - The Genkit message role.
 * @returns The mapped Interaction role string.
 * @throws Error if the role is 'system', as system instructions are handled separately.
 */
export function toInteractionRole(role: MessageData['role']): string {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'model';
    case 'tool':
      return 'user';
    case 'system':
      throw new Error(
        `System role should be handled as system_instruction, not part of turns.`
      );
    default:
      return 'user';
  }
}

/**
 * Converts a Genkit MessageData object into an Interaction Turn.
 *
 * Maps the role using `toInteractionRole` and converts all content parts using `toInteractionContent`.
 *
 * @param message - The Genkit message data.
 * @returns The converted Interaction Turn.
 */
export function toInteractionTurn(message: MessageData): Turn {
  return {
    role: toInteractionRole(message.role),
    content: message.content.map(toInteractionContent),
  };
}

/**
 * Converts an Interaction Content object back into a Genkit Part.
 *
 * Supports text, image, thought, function calls, and function results.
 *
 * @param content - The Interaction Content object.
 * @returns The corresponding Genkit Part.
 * @throws Error if the content type is unsupported.
 */
export function fromInteractionContent(content: Content): Part {
  switch (content.type) {
    case 'text':
      return fromTextContent(content);
    case 'image':
      return fromImageContent(content);
    case 'audio':
    case 'document':
      return fromMediaContent(content);
    case 'video':
      return fromVideoContent(content);
    case 'thought':
      return fromThoughtContent(content);
    case 'function_call':
      return fromFunctionCallContent(content);
    case 'function_result':
      return fromFunctionResultContent(content);
    default:
      // We need the 'any' because currently this is an exhaustive list
      throw new Error(`Unsupported content type: ${(content as any).type}`);
  }
}

function fromMediaContent(
  content: ImageContent | AudioContent | VideoContent | DocumentContent
): Part {
  let url = content.uri;
  if (content.data && content.mime_type) {
    url = `data:${content.mime_type};base64,${content.data}`;
  }
  return {
    media: {
      url: url || '',
      contentType: content.mime_type,
    },
  };
}

function fromTextContent(content: TextContent): Part {
  return {
    text: content.text || '',
    metadata: {
      annotations: content.annotations,
    },
  };
}

function fromImageContent(content: ImageContent): Part {
  const part = fromMediaContent(content);
  part.metadata = { resolution: content.resolution };
  return part;
}

function fromVideoContent(content: VideoContent): Part {
  const part = fromMediaContent(content);
  part.metadata = { resolution: content.resolution };
  return part;
}

function fromThoughtContent(content: ThoughtContent): Part {
  let reasoning = '';
  if (content.summary) {
    reasoning = content.summary
      .map((c) => {
        if (c.type === 'text') return c.text;
        return '[Image]';
      })
      .join('\n');
  }

  return {
    reasoning,
    metadata: {
      thoughtSignature: content.signature,
    },
    custom: {
      thought: content,
    },
  };
}

function fromFunctionCallContent(content: FunctionCallContent): Part {
  return {
    toolRequest: {
      name: content.name,
      input: content.arguments,
      ref: content.id,
    },
  };
}

function fromFunctionResultContent(content: FunctionResultContent): Part {
  return {
    toolResponse: {
      name: content.name,
      output: content.result,
      ref: content.call_id,
    },
  };
}

export function fromInteractionSync(
  interaction: GeminiInteraction
): GenerateResponseData {
  if (interaction.status === 'failed') {
    throw new Error('Interaction failed');
  }

  const response: GenerateResponseData = {
    finishReason: 'stop',
    message: {
      role: 'model',
      content: [],
    },
    custom: interaction,
  };

  if (interaction.status === 'cancelled') {
    response.finishReason = 'aborted';
    response.finishMessage = 'Operation cancelled';
    response.message!.content = [{ text: 'Operation cancelled.' }];
    return response;
  }

  const outputs = interaction.outputs;
  if (outputs?.length) {
    response.message!.content = outputs.map(fromInteractionContent);

    if (interaction.usage) {
      response.usage = {
        inputTokens: interaction.usage.total_input_tokens,
        outputTokens: interaction.usage.total_output_tokens,
        totalTokens: interaction.usage.total_tokens,
        cachedContentTokens: interaction.usage.total_cached_tokens,
        thoughtsTokens: interaction.usage.total_thought_tokens,
      };
      if (interaction.usage.input_tokens_by_modality) {
        for (const modalityToken of interaction.usage
          .input_tokens_by_modality) {
          switch (modalityToken.modality) {
            case 'text':
              response.usage.inputCharacters = modalityToken.tokens;
              break;
            case 'image':
              response.usage.inputImages = modalityToken.tokens;
              break;
            case 'audio':
              response.usage.inputAudioFiles = modalityToken.tokens;
              break;
          }
        }
      }
      if (interaction.usage.output_tokens_by_modality) {
        for (const modalityToken of interaction.usage
          .output_tokens_by_modality) {
          switch (modalityToken.modality) {
            case 'text':
              response.usage.outputCharacters = modalityToken.tokens;
              break;
            case 'image':
              response.usage.outputImages = modalityToken.tokens;
              break;
            case 'audio':
              response.usage.outputAudioFiles = modalityToken.tokens;
              break;
          }
        }
      }
    }
  }
  return response;
}

export function fromInteraction<T extends Object>(
  interaction: GeminiInteraction,
  clientOptions?: T
): Operation<GenerateResponseData> {
  const op = { id: interaction.id } as Operation<GenerateResponseData>;
  if (clientOptions) {
    op.metadata = { clientOptions };
  }
  if (interaction.status === 'in_progress') {
    op.done = false;
  } else if (interaction.status === 'cancelled') {
    op.done = true;
    op.output = {
      finishReason: 'aborted',
      finishMessage: 'Operation cancelled',
      message: {
        role: 'model',
        content: [{ text: 'Operation cancelled.' }],
      },
    };
  } else if (interaction.status === 'completed') {
    op.done = true;
    const outputs = interaction.outputs;
    if (outputs?.length) {
      const content = outputs.map(fromInteractionContent);
      op.output = {
        finishReason: 'stop',
        message: {
          role: 'model',
          content,
        },
        custom: interaction,
      };
      if (interaction.usage) {
        op.output.usage = {
          inputTokens: interaction.usage.total_input_tokens,
          outputTokens: interaction.usage.total_output_tokens,
          totalTokens: interaction.usage.total_tokens,
          cachedContentTokens: interaction.usage.total_cached_tokens,
          thoughtsTokens: interaction.usage.total_thought_tokens,
        };
        if (interaction.usage.input_tokens_by_modality) {
          for (const modalityToken of interaction.usage
            .input_tokens_by_modality) {
            switch (modalityToken.modality) {
              case 'text':
                op.output.usage.inputCharacters = modalityToken.tokens;
                break;
              case 'image':
                op.output.usage.inputImages = modalityToken.tokens;
                break;
              case 'audio':
                op.output.usage.inputAudioFiles = modalityToken.tokens;
                break;
            }
          }
        }
        if (interaction.usage.output_tokens_by_modality) {
          for (const modalityToken of interaction.usage
            .output_tokens_by_modality) {
            switch (modalityToken.modality) {
              case 'text':
                op.output.usage.outputCharacters = modalityToken.tokens;
                break;
              case 'image':
                op.output.usage.outputImages = modalityToken.tokens;
                break;
              case 'audio':
                op.output.usage.outputAudioFiles = modalityToken.tokens;
                break;
            }
          }
        }
      }
    }
  }
  return op;
}
