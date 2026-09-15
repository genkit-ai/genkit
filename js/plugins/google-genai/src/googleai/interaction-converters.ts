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
import { extractMimeType } from '../common/utils.js';
import {
  AudioContent,
  CodeExecutionCallStep,
  CodeExecutionResultStep,
  Content,
  DocumentContent,
  FunctionCallContent,
  FunctionCallStep,
  FunctionResultContent,
  FunctionResultStep,
  GeminiInteraction,
  GoogleSearchCallStep,
  GoogleSearchResultStep,
  ImageContent,
  InteractionDynamicTool,
  InteractionFileSearchTool,
  InteractionFunctionTool,
  InteractionGoogleSearchTool,
  InteractionTool,
  ModelGenerationConfig,
  ResponseModality,
  Step,
  StepDeltaData,
  TextContent,
  ThoughtContent,
  VideoContent,
} from './interaction-types.js';
import {
  camelToSnakeCase,
  cleanSchema,
  convertObjectKeysToSnakeCase,
  isObject,
  toSnakeCaseObj,
} from './utils.js';

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

export function toInteractionConfigTool(toolRaw: unknown): InteractionTool {
  if (!isObject(toolRaw)) {
    throw new Error(
      `Invalid tool configuration: Expected an object, got ${typeof toolRaw}`
    );
  }
  const tool = toolRaw;

  if ('googleSearch' in tool || 'google_search' in tool) {
    return toInteractionGoogleSearch(tool.googleSearch || tool.google_search);
  }
  if ('codeExecution' in tool || 'code_execution' in tool) {
    const config = tool.codeExecution || tool.code_execution;
    if (config === true || config === undefined) {
      return { type: 'code_execution' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for codeExecution tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'code_execution',
      ...toSnakeCaseObj(config),
    };
  }
  if ('fileSearch' in tool || 'file_search' in tool) {
    const config = tool.fileSearch || tool.file_search;
    if (config === true || config === undefined) {
      return { type: 'file_search' } as InteractionFileSearchTool;
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for fileSearch tool: Expected object, got ${typeof config}`
      );
    }
    const result: InteractionFileSearchTool = { type: 'file_search' };

    const fileSearchStoreNames =
      config.fileSearchStoreNames || config.file_search_store_names;
    const restFileSearch = { ...config };
    delete restFileSearch.fileSearchStoreNames;
    delete restFileSearch.file_search_store_names;

    if (fileSearchStoreNames !== undefined) {
      if (
        !Array.isArray(fileSearchStoreNames) ||
        !fileSearchStoreNames.every((n) => typeof n === 'string')
      ) {
        throw new Error('fileSearchStoreNames must be an array of strings.');
      }
      result.file_search_store_names = fileSearchStoreNames;
    }
    return {
      ...result,
      ...toSnakeCaseObj(restFileSearch),
    };
  }
  if ('urlContext' in tool) {
    const config = tool.urlContext;
    if (config === true || config === undefined) {
      return { type: 'url_context' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for urlContext tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'url_context',
      ...toSnakeCaseObj(config),
    };
  }
  if ('googleMaps' in tool) {
    const config = tool.googleMaps;
    if (config === true || config === undefined) {
      return { type: 'google_maps' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for googleMaps tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'google_maps',
      ...toSnakeCaseObj(config),
    };
  }
  if ('computerUse' in tool) {
    const config = tool.computerUse;
    if (config === true || config === undefined) {
      return { type: 'computer_use' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for computerUse tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'computer_use',
      ...toSnakeCaseObj(config),
    };
  }
  if ('retrieval' in tool) {
    const config = tool.retrieval;
    if (config === true || config === undefined) {
      return { type: 'retrieval' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for retrieval tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'retrieval',
      ...toSnakeCaseObj(config),
    };
  }
  if ('mcpServer' in tool) {
    const config = tool.mcpServer;
    if (config === true || config === undefined) {
      return { type: 'mcp_server' };
    }
    if (!isObject(config)) {
      throw new Error(
        `Invalid configuration for mcpServer tool: Expected object or true, got ${typeof config}`
      );
    }
    return {
      type: 'mcp_server',
      ...toSnakeCaseObj(config),
    };
  }

  // Pass through any other properties/custom tools, ensuring snake_case format
  return toSnakeCaseObj(tool) as InteractionDynamicTool;
}

export function toInteractionGoogleSearch(
  gs: boolean | unknown
): InteractionGoogleSearchTool {
  const result: InteractionGoogleSearchTool = { type: 'google_search' };

  if (gs === true || gs === undefined) {
    return result;
  }

  if (!isObject(gs)) {
    throw new Error(
      `Invalid configuration for googleSearch tool: Expected object or true, got ${typeof gs}`
    );
  }

  const searchTypesObj = gs.searchTypes || gs.search_types;
  if (searchTypesObj !== undefined) {
    const searchTypes = new Set<string>();

    if (Array.isArray(searchTypesObj)) {
      for (const type of searchTypesObj) {
        if (typeof type === 'string') {
          if (type === 'webSearch' || type === 'web_search') {
            searchTypes.add('web_search');
          } else if (type === 'imageSearch' || type === 'image_search') {
            searchTypes.add('image_search');
          } else if (
            type === 'enterpriseWebSearch' ||
            type === 'enterprise_web_search'
          ) {
            searchTypes.add('enterprise_web_search');
          } else {
            // Passthrough for any unknown string elements
            searchTypes.add(camelToSnakeCase(type));
          }
        } else {
          throw new Error(
            `Invalid search type: Expected string, got ${typeof type}`
          );
        }
      }
    } else if (isObject(searchTypesObj)) {
      for (const [key, value] of Object.entries(searchTypesObj)) {
        if (value) {
          if (key === 'webSearch' || key === 'web_search') {
            searchTypes.add('web_search');
          } else if (key === 'imageSearch' || key === 'image_search') {
            searchTypes.add('image_search');
          } else if (
            key === 'enterpriseWebSearch' ||
            key === 'enterprise_web_search'
          ) {
            searchTypes.add('enterprise_web_search');
          } else {
            // Passthrough for any unknown properties
            searchTypes.add(camelToSnakeCase(key));
          }
        }
      }
    } else {
      throw new Error(
        `Invalid searchTypes configuration: Expected array or object, got ${typeof searchTypesObj}`
      );
    }

    if (searchTypes.size > 0) {
      result.search_types = Array.from(
        searchTypes
      ) as InteractionGoogleSearchTool['search_types'];
    }
  }

  // Handle any other properties on gs (passthrough)
  const restConfig = { ...gs };
  delete restConfig.searchTypes;
  delete restConfig.search_types;

  Object.assign(result, toSnakeCaseObj(restConfig));

  return result;
}

export function toInteractionResponseModalities(
  modalities: string[]
): ResponseModality[] {
  return modalities.map((m) => m.toLowerCase());
}

export function toInteractionGenerationConfig(
  config: Record<string, unknown>
): ModelGenerationConfig {
  const result = convertObjectKeysToSnakeCase(config) as Record<
    string,
    unknown
  >;

  if (isObject(result.thinking_config)) {
    const tc = result.thinking_config;
    let hasOtherProps = false;
    const newTc: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(tc)) {
      if (key === 'thinking_level') {
        if (typeof value === 'string') {
          result.thinking_level = value.toLowerCase();
        } else {
          result.thinking_level = value;
        }
      } else if (key === 'include_thoughts') {
        if (typeof value === 'boolean') {
          result.thinking_summaries = value ? 'auto' : 'none';
        } else {
          result.thinking_summaries = value;
        }
      } else {
        hasOtherProps = true;
        newTc[key] = value;
      }
    }

    if (hasOtherProps) {
      result.thinking_config = newTc;
    } else {
      delete result.thinking_config;
    }
  }

  return result;
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
  const { url } = part.media;
  const contentType = part.media.contentType || extractMimeType(url);
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

const MediaResolutionSchema = z.enum(['low', 'medium', 'high', 'ultra_high']);

const TextAnnotationSchema = z.object({
  type: z.string().optional(),
  start_index: z.number().optional(),
  end_index: z.number().optional(),
  url: z.string().optional(),
  title: z.string().optional(),
  source: z.string().optional(),
});

const TextContentSchema = z.object({
  type: z.literal('text'),
  text: z.string().optional(),
  annotations: z.array(TextAnnotationSchema).optional(),
});

const ImageContentSchema = z.object({
  type: z.literal('image'),
  data: z.string().optional(),
  uri: z.string().optional(),
  mime_type: z.string().optional(),
  resolution: MediaResolutionSchema.optional(),
});

const FunctionResultArraySchema = z.array(
  z.union([ImageContentSchema, TextContentSchema])
);

const RecordUnknownOrStringOrArraySchema = z.union([
  RecordUnknownSchema,
  z.string(),
  FunctionResultArraySchema,
]);

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
        let result: unknown = part.toolResponse.output;

        if (part.toolResponse.content && part.toolResponse.content.length > 0) {
          const contentParts: Content[] = [];
          if (result !== undefined) {
            const outputText =
              typeof result === 'string' ? result : JSON.stringify(result);
            contentParts.push({ type: 'text', text: outputText });
          }
          for (const p of part.toolResponse.content) {
            const mapped = toInteractionContent(p);
            if (mapped) {
              contentParts.push(mapped);
            }
          }
          result = contentParts;
        } else if (
          typeof result !== 'object' &&
          typeof result !== 'string' &&
          result !== undefined
        ) {
          result = { result: result };
        }

        steps.push({
          type: 'function_result',
          name: part.toolResponse.name,
          result: RecordUnknownOrStringOrArraySchema.parse(result ?? {}),
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
export function fromInteractionDelta(delta: StepDeltaData): Part[] {
  switch (delta.type) {
    case 'text':
      return [{ text: delta.text }];
    case 'image':
    case 'audio':
    case 'document':
    case 'video': {
      let url = delta.uri;
      if (delta.data && delta.mime_type) {
        url = `data:${delta.mime_type};base64,${delta.data}`;
      }
      const part: Part = {
        media: {
          url: url || '',
          contentType: delta.mime_type,
        },
      };
      if (
        (delta.type === 'image' || delta.type === 'video') &&
        delta.resolution !== undefined
      ) {
        part.metadata = { resolution: delta.resolution };
      }
      return [part];
    }
    case 'thought_summary':
      return delta.content ? [fromInteractionContent(delta.content)] : [];
    case 'thought_signature':
      return [
        {
          metadata: { thoughtSignature: delta.signature },
          custom: { thoughtSignatureDelta: delta.signature },
        },
      ];
    case 'function_call':
      return [
        {
          toolRequest: {
            name: delta.name,
            ref: delta.id,
            input: delta.arguments || {},
            partial: true,
          },
        },
      ];
    case 'arguments_delta':
      return [];
    case 'code_execution_call': {
      const part: Part = {
        custom: {
          executableCode: {
            code: delta.arguments.code || '',
            language: delta.arguments.language || 'PYTHON',
          },
        },
      };
      if (delta.signature) {
        part.metadata = { thoughtSignature: delta.signature };
      }
      return [part];
    }
    case 'code_execution_result': {
      const part: Part = {
        custom: {
          codeExecutionResult: {
            output: delta.result,
            outcome: delta.is_error ? 'OUTCOME_FAILED' : 'OUTCOME_OK',
          },
        },
      };
      if (delta.signature) {
        part.metadata = { thoughtSignature: delta.signature };
      }
      return [part];
    }
    case 'google_search_call': {
      const part: Part = {
        custom: {
          googleSearchCall: {
            id: '',
            arguments: delta.arguments,
          },
        },
      };
      if (delta.signature) {
        part.metadata = { thoughtSignature: delta.signature };
      }
      return [part];
    }
    case 'google_search_result': {
      const part: Part = {
        custom: {
          googleSearchResult: {
            callId: '',
            result: delta.result || [],
          },
        },
      };
      if (delta.signature) {
        part.metadata = { thoughtSignature: delta.signature };
      }
      return [part];
    }
    case 'function_result':
      return [
        {
          custom: {
            serverFunctionResult: {
              callId: delta.call_id,
              name: delta.name || '',
              result: delta.result,
              isError: delta.is_error,
            },
          },
        },
      ];
    default:
      return [];
  }
}

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

export function fromPendingFunctionCall(
  step: FunctionCallContent | FunctionCallStep
): Part {
  return {
    toolRequest: {
      name: step.name,
      ref: step.id,
      input: step.arguments,
    },
  };
}

export function fromServerFunctionCall(
  step: FunctionCallContent | FunctionCallStep
): Part {
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

export function fromServerFunctionResult(
  step: FunctionResultContent | FunctionResultStep
): Part {
  return {
    custom: {
      serverFunctionResult: {
        callId: step.call_id,
        name: step.name || '',
        result: step.result,
        isError: step.is_error,
      },
    },
  };
}

export function fromInteractionStep(step: Step, isPending?: boolean): Part[] {
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
      return isPending
        ? [fromPendingFunctionCall(step)]
        : [fromServerFunctionCall(step)];
    case 'function_result':
      return [fromServerFunctionResult(step)];
  }

  return [{ custom: { unknownStep: step } }];
}

function getPendingFunctionCallIds(
  steps: Step[],
  status?: string
): Set<string> {
  const pendingIds = new Set<string>();
  if (status !== 'requires_action') {
    return pendingIds;
  }

  const resultSet = new Set<string>();
  for (const step of steps) {
    if (step.type === 'function_result' && step.call_id) {
      resultSet.add(step.call_id);
    }
  }

  for (const step of steps) {
    if (step.type === 'function_call' && step.id) {
      if (!resultSet.has(step.id)) {
        pendingIds.add(step.id);
      }
    }
  }

  return pendingIds;
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
  if (Array.isArray(content.result)) {
    return {
      toolResponse: {
        name: content.name,
        content: content.result.map((c) => fromInteractionContent(c)),
        ref: content.call_id,
      },
    };
  }
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
    const pendingIds = getPendingFunctionCallIds(steps, interaction.status);

    response.message!.content = steps
      .flatMap((step) => {
        const isPending =
          step.type === 'function_call' && step.id
            ? pendingIds.has(step.id)
            : false;
        return fromInteractionStep(step, isPending);
      })
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

export function fromInteraction(
  interaction: GeminiInteraction
): Operation<GenerateResponseData> {
  const op = { id: interaction.id } as Operation<GenerateResponseData>;
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
        .flatMap((step) => fromInteractionStep(step, false))
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
