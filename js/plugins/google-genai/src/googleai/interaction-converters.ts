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

import { GenerateResponseData, MessageData, Operation, Part, z } from 'genkit';
import { ToolDefinition } from 'genkit/model';
import {
  AudioContent,
  CodeExecutionCallStep,
  CodeExecutionResultStep,
  Content,
  DocumentContent,
  FunctionCallContent,
  FunctionResultContent,
  GeminiInteraction,
  GoogleSearchCallStep,
  GoogleSearchResultStep,
  ImageContent,
  InteractionFunctionTool,
  InteractionTool,
  Step,
  TextContent,
  ThoughtContent,
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
export function toInteractionContent(part: Part): Content | undefined {
  if (part.text !== undefined) {
    return { type: 'text', text: part.text };
  }
  if (part.media) {
    return toInteractionMedia(part);
  }
  console.warn(
    `Unsupported part type for Interaction input: ${JSON.stringify(part)}`
  );
  return undefined;
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

const GoogleSearchArgsSchema = z.object({ queries: z.array(z.string()) });
const RecordUnknownSchema = z.record(z.unknown());
const RecordUnknownOrStringSchema = z.union([RecordUnknownSchema, z.string()]);
const OptionalStringSchema = z.string().optional();

const GoogleSearchCallSchema = z.object({
  id: z.string(),
  arguments: GoogleSearchArgsSchema,
});

const GoogleSearchResultSchema = z.object({
  callId: z.string(),
  result: RecordUnknownSchema,
});

const ExecutableCodeSchema = z.object({
  code: z.string(),
  language: z.string().default('PYTHON'),
});

const CodeExecutionResultSchema = z.object({
  output: z.string(),
  outcome: z.string().optional(),
});

/**
 * Converts an array of Genkit MessageData objects into an array of Interaction Steps.
 */
export function toInteractionSteps(messages: MessageData[]): Step[] {
  const steps: Step[] = [];

  for (const message of messages) {
    const normalContent: Content[] = [];

    for (const part of message.content) {
      if (part.toolRequest) {
        steps.push({
          type: 'function_call',
          name: part.toolRequest.name,
          arguments: RecordUnknownSchema.optional().parse(
            part.toolRequest.input
          ),
          id: part.toolRequest.ref || '',
        });
      } else if (part.toolResponse) {
        let output = part.toolResponse.output;
        if (
          typeof output !== 'object' &&
          typeof output !== 'string' &&
          output !== undefined
        ) {
          output = { result: output };
        }
        steps.push({
          type: 'function_result',
          name: part.toolResponse.name,
          result: RecordUnknownOrStringSchema.optional().parse(output),
          call_id: part.toolResponse.ref || '',
        });
      } else if (part.custom?.googleSearchCall) {
        const gsCall = GoogleSearchCallSchema.parse(
          part.custom.googleSearchCall
        );
        steps.push({
          type: 'google_search_call',
          id: gsCall.id,
          arguments: gsCall.arguments,
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.custom?.googleSearchResult) {
        const gsResult = GoogleSearchResultSchema.parse(
          part.custom.googleSearchResult
        );
        steps.push({
          type: 'google_search_result',
          call_id: gsResult.callId,
          result: gsResult.result,
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.custom?.executableCode) {
        const execCode = ExecutableCodeSchema.parse(part.custom.executableCode);
        steps.push({
          type: 'code_execution_call',
          id: z.string().parse(part.metadata?.callId),
          arguments: {
            code: execCode.code,
            language: execCode.language,
          },
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.custom?.codeExecutionResult) {
        const execResult = CodeExecutionResultSchema.parse(
          part.custom.codeExecutionResult
        );
        steps.push({
          type: 'code_execution_result',
          call_id: z.string().parse(part.metadata?.callId),
          result: execResult.output,
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else if (part.reasoning) {
        steps.push({
          type: 'thought',
          summary: [{ type: 'text', text: part.reasoning }],
          signature: OptionalStringSchema.parse(
            part.metadata?.thoughtSignature
          ),
        });
      } else {
        const content = toInteractionContent(part);
        if (content) {
          normalContent.push(content);
        }
      }
    }

    if (normalContent.length > 0) {
      if (message.role === 'model') {
        steps.push({
          type: 'model_output',
          content: normalContent,
        });
      } else {
        steps.push({
          type: 'user_input',
          content: normalContent,
        });
      }
    }
  }

  return steps;
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
      return {
        custom: {
          unknownContent: content,
        },
      };
  }
}

function maybeAddGeminiThoughtSignature(step: Step, part: Part): Part {
  let updatedPart = part;

  if ('signature' in step && step.signature) {
    updatedPart.metadata = {
      ...(part.metadata ?? {}),
      thoughtSignature: step.signature,
    };
  }
  return updatedPart;
}

export function fromGoogleSearchCall(step: GoogleSearchCallStep): Part {
  const part: Part = {
    custom: {
      googleSearchCall: {
        id: step.id,
        arguments: step.arguments,
      },
    },
  };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromGoogleSearchResult(step: GoogleSearchResultStep): Part {
  const part: Part = {
    custom: {
      googleSearchResult: {
        callId: step.call_id,
        result: step.result,
      },
    },
  };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromCodeExecutionCall(step: CodeExecutionCallStep): Part {
  const part: Part = {
    custom: {
      executableCode: {
        code: step.arguments.code,
        language: step.arguments.language || 'PYTHON',
      },
    },
  };
  part.metadata = { callId: step.id };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromCodeExecutionResult(step: CodeExecutionResultStep): Part {
  const part: Part = {
    custom: {
      codeExecutionResult: {
        output:
          typeof step.result === 'string'
            ? step.result
            : JSON.stringify(step.result),
        outcome: 'OUTCOME_OK',
      },
    },
  };
  part.metadata = { callId: step.call_id };
  return maybeAddGeminiThoughtSignature(step, part);
}

export function fromServerFunctionCall(step: FunctionCallContent): Part {
  return {
    custom: {
      serverFunctionCall: {
        id: step.id,
        name: step.name,
        arguments: step.arguments,
      },
    },
  };
}

export function fromServerFunctionResult(step: FunctionResultContent): Part {
  return {
    custom: {
      serverFunctionResult: {
        callId: step.call_id,
        name: step.name,
        result: step.result,
        isError: step.is_error,
      },
    },
  };
}

export function fromInteractionStep(step: Step): Part[] {
  switch (step.type) {
    case 'model_output':
      return step.content.map(fromInteractionContent);
    case 'user_input':
      return [];
    case 'google_search_call':
      return [fromGoogleSearchCall(step)];
    case 'google_search_result':
      return [fromGoogleSearchResult(step)];
    case 'code_execution_call':
      return [fromCodeExecutionCall(step)];
    case 'code_execution_result':
      return [fromCodeExecutionResult(step)];
    case 'thought':
      return [fromThoughtContent(step)];
    case 'function_call':
      return [fromServerFunctionCall(step)];
    case 'function_result':
      return [fromServerFunctionResult(step)];
  }

  return [{ custom: { unknownStep: step } }];
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
  if (content.resolution !== undefined) {
    part.metadata = { resolution: content.resolution };
  }
  return part;
}

function fromVideoContent(content: VideoContent): Part {
  const part = fromMediaContent(content);
  if (content.resolution !== undefined) {
    part.metadata = { resolution: content.resolution };
  }
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
      ...(interaction.id || interaction.environment_id
        ? {
            metadata: {
              ...(interaction.id ? { interactionId: interaction.id } : {}),
              ...(interaction.environment_id
                ? { environmentId: interaction.environment_id }
                : {}),
            },
          }
        : {}),
    },
    custom: interaction,
    raw: interaction,
  };

  if (interaction.status === 'cancelled') {
    response.finishReason = 'aborted';
    response.finishMessage = 'Operation cancelled';
    response.message!.content = [{ text: 'Operation cancelled.' }];
    return response;
  }

  const steps = interaction.steps;
  if (steps?.length) {
    response.message!.content = steps
      .flatMap(fromInteractionStep)
      .filter((p) => p && Object.keys(p).length > 0);

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
        ...(interaction.id || interaction.environment_id
          ? {
              metadata: {
                ...(interaction.id ? { interactionId: interaction.id } : {}),
                ...(interaction.environment_id
                  ? { environmentId: interaction.environment_id }
                  : {}),
              },
            }
          : {}),
      },
    };
  } else if (interaction.status === 'completed') {
    op.done = true;
    const steps = interaction.steps;
    if (steps?.length) {
      const content = steps
        .flatMap(fromInteractionStep)
        .filter((p) => p && Object.keys(p).length > 0);
      op.output = {
        finishReason: 'stop',
        message: {
          role: 'model',
          content,
          ...(interaction.id || interaction.environment_id
            ? {
                metadata: {
                  ...(interaction.id ? { interactionId: interaction.id } : {}),
                  ...(interaction.environment_id
                    ? { environmentId: interaction.environment_id }
                    : {}),
                },
              }
            : {}),
        },
        custom: interaction,
        raw: interaction,
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
