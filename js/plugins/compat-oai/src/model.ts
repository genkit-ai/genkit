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

import type {
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
  MessageData,
  ModelReference,
  Part,
  Role,
  StreamingCallback,
  ToolRequestPart,
} from 'genkit';
import {
  GenerationCommonConfigSchema,
  GenkitError,
  Message,
  StatusName,
  modelRef,
  z,
  type ErrorResponseMetadata,
} from 'genkit';
import { parsePartialJson } from 'genkit/extract';
import type { ModelAction, ModelInfo, ToolDefinition } from 'genkit/model';
import { model } from 'genkit/plugin';
import OpenAI, { APIError } from 'openai';
import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionContentPart,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionRole,
  ChatCompletionStreamOptions,
  ChatCompletionTool,
} from 'openai/resources/index.mjs';
import { PluginOptions } from './index.js';
import { maybeCreateRequestScopedOpenAIClient, toModelName } from './utils.js';

/**
 * Parses a `Retry-After` header value into milliseconds.
 * Supports delay-seconds and HTTP-date formats (RFC 7231 §7.1.3).
 */
function parseRetryAfterMs(value: string): number | undefined {
  if (!value || !value.trim()) return undefined;
  const seconds = Number(value);
  if (!isNaN(seconds) && seconds >= 0) return seconds * 1000;
  const date = new Date(value);
  if (!isNaN(date.getTime())) return Math.max(0, date.getTime() - Date.now());
  return undefined;
}

const VisualDetailLevelSchema = z.enum(['auto', 'low', 'high']).optional();

type VisualDetailLevel = z.infer<typeof VisualDetailLevelSchema>;

export type ModelRequestBuilder = (
  req: GenerateRequest,
  params: ChatCompletionCreateParams
) => void;

export const ChatCompletionCommonConfigSchema =
  GenerationCommonConfigSchema.extend({
    temperature: z.number().min(0).max(2).optional(),
    frequencyPenalty: z.number().min(-2).max(2).optional(),
    logProbs: z.boolean().optional(),
    presencePenalty: z.number().min(-2).max(2).optional(),
    topLogProbs: z.number().int().min(0).max(20).optional(),
  });

export function toOpenAIRole(role: Role): ChatCompletionRole {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'assistant';
    case 'system':
      return 'system';
    case 'tool':
      return 'tool';
    default:
      throw new Error(`role ${role} doesn't map to an OpenAI role.`);
  }
}

/**
 * Converts a Genkit ToolDefinition to an OpenAI ChatCompletionTool object.
 * @param tool The Genkit ToolDefinition to convert.
 * @returns The converted OpenAI ChatCompletionTool object.
 */
export function toOpenAITool(tool: ToolDefinition): ChatCompletionTool {
  return {
    type: 'function',
    function: {
      name: tool.name,
      parameters: tool.inputSchema !== null ? tool.inputSchema : undefined,
    },
  };
}

/**
 * Checks if a content type is an image type.
 * @param contentType The content type to check.
 * @returns True if the content type is an image type.
 */
function isImageContentType(contentType?: string): boolean {
  if (!contentType) return false;
  return contentType.startsWith('image/');
}

/**
 * Extracts the base64 data and content type from a data URL.
 * @param url The data URL to parse.
 * @returns The base64 data and content type, or null if invalid.
 */
function extractDataFromBase64Url(url: string): {
  data: string;
  contentType: string;
} | null {
  const match = url.match(/^data:([^;]+);base64,(.+)$/);
  return (
    match && {
      contentType: match[1],
      data: match[2],
    }
  );
}

/**
 * Map of content types to file extensions.
 */
const FILE_EXTENSIONS: Record<string, string> = {
  'application/pdf': 'pdf',
  'application/msword': 'doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
    'docx',
  'text/plain': 'txt',
  'text/csv': 'csv',
};

/**
 * Generates a filename from a content type.
 * @param contentType The content type.
 * @returns A filename with appropriate extension.
 */
function generateFilenameFromContentType(contentType: string): string {
  const ext = FILE_EXTENSIONS[contentType] || '';
  return ext ? `file.${ext}` : 'file';
}

/**
 * Converts a Genkit Part to the corresponding OpenAI ChatCompletionContentPart.
 * @param part The Genkit Part to convert.
 * @param visualDetailLevel The visual detail level to use for media parts.
 * @returns The corresponding OpenAI ChatCompletionContentPart.
 * @throws Error if the part contains unsupported fields for the current message role.
 */
export function toOpenAITextAndMedia(
  part: Part,
  visualDetailLevel: VisualDetailLevel
): ChatCompletionContentPart {
  if (part.text) {
    return {
      type: 'text',
      text: part.text,
    };
  } else if (part.media) {
    // Determine the content type from the media part or data URL
    let contentType = part.media.contentType;
    if (!contentType && part.media.url.startsWith('data:')) {
      const extracted = extractDataFromBase64Url(part.media.url);
      if (extracted) {
        contentType = extracted.contentType;
      }
    }

    // Check if this is an image type
    // If no contentType is provided, preserve legacy behavior by treating the media
    // as an image URL (e.g. signed URLs or remote images without metadata)
    if (!contentType || isImageContentType(contentType)) {
      return {
        type: 'image_url',
        image_url: {
          url: part.media.url,
          detail: visualDetailLevel,
        },
      };
    }

    // For non-image types (like PDF), use the file type
    // OpenAI expects the full data URL (with data: prefix) in file_data
    if (part.media.url.startsWith('data:')) {
      const extracted = extractDataFromBase64Url(part.media.url);
      if (!extracted) {
        throw Error(
          `Invalid data URL format for media: ${part.media.url.substring(0, 50)}...`
        );
      }
      return {
        type: 'file',
        file: {
          filename: generateFilenameFromContentType(extracted.contentType),
          file_data: part.media.url, // Full data URL with prefix
        },
      } as ChatCompletionContentPart;
    }

    // If it's a remote URL with non-image content type, this is not supported
    // for chat completions according to OpenAI docs
    throw Error(
      `File URLs are not supported for chat completions. Only base64-encoded files and image URLs are supported. Content type: ${contentType}`
    );
  }
  throw Error(
    `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(part)}.`
  );
}

/**
 * Converts a Genkit MessageData array to an OpenAI ChatCompletionMessageParam array.
 * @param messages The Genkit MessageData array to convert.
 * @param visualDetailLevel The visual detail level to use for media parts.
 * @returns The converted OpenAI ChatCompletionMessageParam array.
 */
export function toOpenAIMessages(
  messages: MessageData[],
  visualDetailLevel: VisualDetailLevel = 'auto'
): ChatCompletionMessageParam[] {
  const apiMessages: ChatCompletionMessageParam[] = [];
  for (const message of messages) {
    const msg = new Message(message);
    const role = toOpenAIRole(message.role);
    switch (role) {
      case 'user':
        const content = msg.content.map((part) =>
          toOpenAITextAndMedia(part, visualDetailLevel)
        );
        // Check if we have only text content
        const onlyTextContent = content.some((item) => item.type !== 'text');

        // If all items are strings, just add them as text
        if (!onlyTextContent) {
          content.forEach((item) => {
            if (item.type === 'text') {
              apiMessages.push({
                role: role,
                content: item.text,
              });
            }
          });
        } else {
          apiMessages.push({
            role: role,
            content: content,
          });
        }
        break;
      case 'system':
        apiMessages.push({
          role: role,
          content: msg.text,
        });
        break;
      case 'assistant': {
        const toolCalls: ChatCompletionMessageToolCall[] = msg.content
          .filter(
            (
              part
            ): part is Part & {
              toolRequest: NonNullable<Part['toolRequest']>;
            } => Boolean(part.toolRequest)
          )
          .map((part) => ({
            id: part.toolRequest.ref ?? '',
            type: 'function',
            function: {
              name: part.toolRequest.name,
              arguments: JSON.stringify(part.toolRequest.input),
            },
          }));
        const text = msg.text;
        // Providers with thinking modes (e.g. DeepSeek) expect reasoning from
        // all previous assistant turns to be echoed back when the request
        // carries tools, so it is sent on every assistant message, not only
        // tool-calling ones. Most other providers ignore the field; providers
        // that reject unknown fields (e.g. Mistral) must remove it in their
        // request builder.
        const reasoning = msg.reasoning;
        // `reasoning_content` is a provider extension, not part of the
        // OpenAI SDK types.
        const assistantMessage: ChatCompletionAssistantMessageParam & {
          reasoning_content?: string;
        } =
          toolCalls.length > 0
            ? {
                role: role,
                ...(text ? { content: text } : {}),
                tool_calls: toolCalls,
              }
            : { role: role, content: text };
        if (reasoning) {
          assistantMessage.reasoning_content = reasoning;
        }
        apiMessages.push(assistantMessage);
        break;
      }
      case 'tool': {
        const toolResponseParts = msg.toolResponseParts();
        toolResponseParts.map((part) => {
          apiMessages.push({
            role: role,
            tool_call_id: part.toolResponse.ref ?? '',
            content:
              typeof part.toolResponse.output === 'string'
                ? part.toolResponse.output
                : JSON.stringify(part.toolResponse.output),
          });
        });
        break;
      }
    }
  }
  return apiMessages;
}

/**
 * A content chunk in an assistant message or stream delta, in Mistral's
 * `ContentChunk` shape. OpenAI always sends `content` as a string; Mistral can
 * send an array of typed chunks, with reasoning in `thinking` chunks.
 */
interface ContentChunk {
  type?: string;
  text?: string;
  thinking?: { type?: string; text?: string }[];
  /** Signature of a `thinking` chunk, for replaying it on later turns. */
  signature?: string | null;
}

/**
 * Converts an array of Mistral `ContentChunk`s to Genkit parts. `text` chunks
 * become text parts and `thinking` chunks become reasoning parts, with the
 * chunk's `signature` (if any) in `metadata.signature`. Consecutive chunks of
 * the same kind are merged. Other chunk types (`image_url`, `reference`,
 * `document_url` and so on) are dropped.
 */
function fromContentChunks(chunks: ContentChunk[]): Part[] {
  const parts: Part[] = [];
  for (const chunk of chunks) {
    const last = parts[parts.length - 1];
    if (chunk.type === 'text') {
      if (!chunk.text) continue;
      if (last?.text !== undefined) {
        last.text += chunk.text;
      } else {
        parts.push({ text: chunk.text });
      }
    } else if (chunk.type === 'thinking') {
      const text = (chunk.thinking ?? [])
        .map((t) => (t.type === 'text' ? (t.text ?? '') : ''))
        .join('');
      if (!text && !chunk.signature) continue;
      let part: Part;
      if (last?.reasoning !== undefined) {
        part = last;
        part.reasoning += text;
      } else {
        part = { reasoning: text };
        parts.push(part);
      }
      if (chunk.signature) {
        part.metadata = { ...part.metadata, signature: chunk.signature };
      }
    }
  }
  return parts;
}

const finishReasonMap: Partial<
  Record<
    ChatCompletion.Choice['finish_reason'] | 'model_length',
    GenerateResponseData['finishReason']
  >
> = {
  length: 'length',
  // Mistral: generation hit the model's context length.
  model_length: 'length',
  stop: 'stop',
  tool_calls: 'stop',
  content_filter: 'blocked',
};

/**
 * Converts an OpenAI tool call to a Genkit ToolRequestPart.
 * @param toolCall The OpenAI tool call to convert.
 * @returns The converted Genkit ToolRequestPart.
 */
export function fromOpenAIToolCall(
  toolCall:
    | ChatCompletionMessageToolCall
    | ChatCompletionChunk.Choice.Delta.ToolCall,
  choice: ChatCompletion.Choice | ChatCompletionChunk.Choice
): ToolRequestPart {
  if (!toolCall.function) {
    throw Error(
      `Unexpected openAI chunk choice. tool_calls was provided but one or more tool_calls is missing.`
    );
  }
  const f = toolCall.function;

  // Only parse arguments when it is a JSON object and the finish reason is tool_calls to avoid parsing errors
  if (choice.finish_reason === 'tool_calls') {
    return {
      toolRequest: {
        name: f.name!,
        ref: toolCall.id,
        input: f.arguments ? JSON.parse(f.arguments) : f.arguments,
      },
    };
  } else {
    return {
      toolRequest: {
        name: f.name!,
        ref: toolCall.id,
        input: '',
      },
    };
  }
}

/**
 * Converts an OpenAI message event to a Genkit GenerateResponseData object.
 * `message.content` is normally a string. An array is read as Mistral's
 * `ContentChunk` shape: text and thinking chunks are kept, other chunk types
 * (`image_url`, `reference` and so on) are dropped.
 * @param choice The OpenAI message event to convert.
 * @param jsonMode Whether the event is a JSON response.
 * @returns The converted Genkit GenerateResponseData object.
 */
export function fromOpenAIChoice(
  choice: ChatCompletion.Choice,
  jsonMode = false
): GenerateResponseData {
  const toolRequestParts = choice.message.tool_calls?.map((toolCall) =>
    fromOpenAIToolCall(toolCall, choice)
  );

  // Build content array based on what's present in the message
  const content: Part[] = [];
  const hasToolRequests = !!toolRequestParts && toolRequestParts.length > 0;

  // Handle reasoning_content if present
  if (
    'reasoning_content' in choice.message &&
    choice.message.reasoning_content
  ) {
    content.push({ reasoning: choice.message.reasoning_content as string });
  }

  // Handle regular content if present. Text that accompanies tool calls is a
  // preamble, not JSON output, so it is never parsed. An array is Mistral's
  // `ContentChunk` shape; in JSON mode its text chunks are joined and parsed.
  const messageContent: unknown = choice.message.content;
  if (Array.isArray(messageContent)) {
    const chunkParts = fromContentChunks(messageContent);
    if (jsonMode && !hasToolRequests) {
      content.push(...chunkParts.filter((p) => p.reasoning !== undefined));
      const text = chunkParts.map((p) => p.text ?? '').join('');
      if (text) {
        content.push({ data: JSON.parse(text) });
      }
    } else {
      content.push(...chunkParts);
    }
  } else if (choice.message.content) {
    content.push(
      jsonMode && !hasToolRequests
        ? { data: JSON.parse(choice.message.content!) }
        : { text: choice.message.content! }
    );
  }

  if (hasToolRequests) {
    content.push(...(toolRequestParts as ToolRequestPart[]));
  }

  return {
    finishReason: finishReasonMap[choice.finish_reason] || 'other',
    message: {
      role: 'model',
      content,
    },
  };
}

/**
 * Accumulates the streamed fragments of a single tool call. OpenAI streams tool
 * calls incrementally: the first delta carries `name`/`id` (with empty or
 * partial `arguments`) and subsequent deltas carry only fragments of the
 * `arguments` JSON string, keyed by the tool call `index` within the message.
 */
export interface ToolCallAccumulator {
  ref?: string;
  name?: string;
  /** Concatenated `arguments` JSON string fragments received so far. */
  args: string;
}

/**
 * Converts a streamed tool-call delta into a partial Genkit ToolRequestPart,
 * accumulating `arguments` fragments across chunks so that the emitted
 * `toolRequest` carries the best-effort parsed input instead of `input: ''`.
 * @param toolCall The streamed tool-call delta.
 * @param accumulators Per-message tool-call accumulators, keyed by tool call
 *   index. Mutated in place as fragments arrive.
 * @returns The partial ToolRequestPart, or `undefined` if there is no
 *   meaningful data to emit yet.
 */
function fromOpenAIToolCallChunk(
  toolCall: ChatCompletionChunk.Choice.Delta.ToolCall,
  accumulators: ToolCallAccumulator[]
): ToolRequestPart | undefined {
  const index = toolCall.index ?? 0;
  const acc = (accumulators[index] ??= { args: '' });

  // The first delta for a tool call carries its `id` and function `name`;
  // subsequent deltas omit them, so carry the accumulated values forward.
  if (toolCall.id) {
    acc.ref = toolCall.id;
  }
  if (toolCall.function?.name) {
    acc.name = toolCall.function.name;
  }
  if (toolCall.function?.arguments) {
    acc.args += toolCall.function.arguments;
  }

  // Without a tool name there is nothing meaningful to emit yet; the name
  // arrives with the first delta of a tool call, so wait for it.
  if (!acc.name) {
    return undefined;
  }

  // Best-effort parse of the (possibly incomplete) accumulated arguments.
  let input: unknown = {};
  if (acc.args) {
    try {
      input = parsePartialJson(acc.args);
    } catch {
      input = {};
    }
  }

  return {
    toolRequest: {
      name: acc.name,
      ref: acc.ref,
      input,
      partial: true,
    },
  };
}

/**
 * Converts an OpenAI message stream event to a Genkit GenerateResponseData
 * object. `delta.content` is normally a string. An array is read as Mistral's
 * `ContentChunk` shape: text and thinking chunks are kept, other chunk types
 * (`image_url`, `reference` and so on) are dropped.
 * @param choice The OpenAI message stream event to convert.
 * @param _jsonMode Unused; kept for signature compatibility. Streamed content
 *   is always emitted as text, see the comment in the body.
 * @param toolCallAccumulators Optional per-message tool-call accumulators (keyed
 *   by tool call index) used to assemble streamed `arguments` fragments into
 *   partial tool requests. When omitted, tool-call deltas are converted
 *   statelessly (legacy behavior).
 * @returns The converted Genkit GenerateResponseData object.
 */
export function fromOpenAIChunkChoice(
  choice: ChatCompletionChunk.Choice,
  _jsonMode = false,
  toolCallAccumulators?: ToolCallAccumulator[]
): GenerateResponseData {
  const toolRequestParts = toolCallAccumulators
    ? choice.delta.tool_calls
        ?.map((toolCall) =>
          fromOpenAIToolCallChunk(toolCall, toolCallAccumulators)
        )
        .filter((p): p is ToolRequestPart => !!p)
    : choice.delta.tool_calls?.map((toolCall) =>
        fromOpenAIToolCall(toolCall, choice)
      );

  // Build content array based on what's present in the delta
  const content: Part[] = [];
  const hasToolRequests = !!toolRequestParts && toolRequestParts.length > 0;

  // Handle reasoning_content if present
  if ('reasoning_content' in choice.delta && choice.delta.reasoning_content) {
    content.push({ reasoning: choice.delta.reasoning_content as string });
  }

  // Deltas are always emitted as text, even in JSON mode. A delta is a fragment
  // of the response; one that happens to parse on its own (OpenAI streams
  // `{}` and `[]` as standalone tokens) is still part of a larger document.
  // Emitting it as `data` would drop it from `accumulatedText`, which the
  // JSON format parser uses to build partial output. The final message is
  // parsed from the complete response. An array is Mistral's `ContentChunk`
  // shape, see fromContentChunks.
  const deltaContent: unknown = choice.delta.content;
  if (Array.isArray(deltaContent)) {
    content.push(...fromContentChunks(deltaContent));
  } else if (choice.delta.content) {
    content.push({ text: choice.delta.content });
  }

  if (hasToolRequests) {
    content.push(...(toolRequestParts as ToolRequestPart[]));
  }

  return {
    finishReason: choice.finish_reason
      ? finishReasonMap[choice.finish_reason] || 'other'
      : 'unknown',
    message: {
      role: 'model',
      content,
    },
  };
}

/**
 * Converts an OpenAI request to an OpenAI API request body.
 * @param modelName The name of the OpenAI model to use.
 * @param request The Genkit GenerateRequest to convert.
 * @returns The converted OpenAI API request body.
 * @throws An error if the specified model is not supported or if an unsupported output format is requested.
 */
export function toOpenAIRequestBody(
  modelName: string,
  request: GenerateRequest,
  requestBuilder?: ModelRequestBuilder
) {
  const messages = toOpenAIMessages(
    request.messages,
    request.config?.visualDetailLevel
  );
  const {
    temperature,
    maxOutputTokens, // unused
    topK, // unused
    topP: top_p,
    frequencyPenalty: frequency_penalty,
    logProbs: logprobs,
    presencePenalty: presence_penalty,
    topLogProbs: top_logprobs,
    stopSequences: stop,
    version: modelVersion,
    tools: toolsFromConfig,
    apiKey,
    strict,
    ...restOfConfig
  } = request.config ?? {};

  const tools: ChatCompletionTool[] = request.tools?.map(toOpenAITool) ?? [];
  if (toolsFromConfig) {
    tools.push(...(toolsFromConfig as any[]));
  }
  let body = {
    model: modelVersion ?? modelName,
    messages,
    tools: tools.length > 0 ? tools : undefined,
    temperature,
    top_p,
    stop,
    frequency_penalty,
    presence_penalty,
    top_logprobs,
    logprobs,
  } as ChatCompletionCreateParamsNonStreaming;
  if (requestBuilder) {
    // If override provided, apply the override to the OpenAI request.
    // User must control passthrough config too.
    requestBuilder(request, body);
  } else {
    body = { ...body, ...restOfConfig }; // passthrough for other config
  }
  const response_format = request.output?.format;
  if (
    strict === true &&
    (response_format !== 'json' || !request.output?.schema)
  ) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'OpenAI strict output requires output.format="json" and an output schema.',
    });
  }
  if (response_format === 'json') {
    if (request.output?.schema) {
      body.response_format = {
        type: 'json_schema',
        json_schema: {
          name: 'output',
          schema: request.output!.schema,
          ...(strict !== undefined ? { strict } : {}),
        },
      };
    } else {
      body.response_format = {
        type: 'json_object',
      };
    }
  } else if (response_format === 'text') {
    body.response_format = {
      type: 'text',
    };
  }
  const bodyRecord = body as unknown as Record<string, unknown>;
  for (const key in bodyRecord) {
    if (
      !bodyRecord[key] ||
      (Array.isArray(bodyRecord[key]) && !(bodyRecord[key] as unknown[]).length)
    )
      delete bodyRecord[key];
  }
  return body;
}

/** Options that change how {@link openAIModelRunner} calls the API. */
export interface ModelRunnerOptions {
  /**
   * `stream_options` sent with streaming requests. Defaults to
   * `{ include_usage: true }`. Set to `false` for providers that reject the
   * field.
   */
  streamOptions?: ChatCompletionStreamOptions | false;
  /**
   * Set for providers that stream `content` as an array of chunks (Mistral's
   * `ContentChunk` shape). The runner then keeps the streamed chunks and puts
   * them on the final completion, because the OpenAI SDK builds the final
   * message by concatenating `content` as strings. Off by default.
   */
  contentChunks?: boolean;
}

/**
 * Creates the runner used by Genkit to interact with an OpenAI compatible
 * model.
 * @param name The name of the GPT model.
 * @param client The OpenAI client instance.
 * @param requestBuilder Optional hook to modify the request body.
 * @param pluginOptions Optional plugin options, used for request scoped clients.
 * @param runnerOptions Optional options, see {@link ModelRunnerOptions}.
 * @returns The runner that Genkit will call when the model is invoked.
 */
export function openAIModelRunner(
  name: string,
  defaultClient: OpenAI,
  requestBuilder?: ModelRequestBuilder,
  pluginOptions?: Omit<PluginOptions, 'apiKey'>,
  runnerOptions?: ModelRunnerOptions
) {
  const streamOptions = runnerOptions?.streamOptions ?? {
    include_usage: true,
  };
  return async (
    request: GenerateRequest,
    options?: {
      streamingRequested?: boolean;
      sendChunk?: StreamingCallback<GenerateResponseChunkData>;
      abortSignal?: AbortSignal;
    }
  ): Promise<GenerateResponseData> => {
    const client = maybeCreateRequestScopedOpenAIClient(
      pluginOptions,
      request,
      defaultClient
    );
    try {
      let response: ChatCompletion;
      const body = toOpenAIRequestBody(name, request, requestBuilder);
      if (options?.streamingRequested) {
        const stream = client.beta.chat.completions.stream(
          {
            ...body,
            stream: true,
            ...(streamOptions ? { stream_options: streamOptions } : {}),
          },
          { signal: options?.abortSignal }
        );
        // Tracks streamed tool-call argument fragments per choice index so we
        // can accumulate them and emit `partial` tool requests with parsed
        // input instead of a flurry of empty `input: ''` chunks. OpenAI streams
        // tool calls incrementally: the first delta carries `name`/`id` and
        // subsequent deltas carry only fragments of the `arguments` JSON string.
        const toolCallAccumulators = new Map<number, ToolCallAccumulator[]>();
        // Only for providers that opt in with `contentChunks` (Mistral): they
        // can stream `content` as an array of chunks, which the SDK's final
        // completion mangles by concatenating content as strings. Keep the
        // chunks per choice and restore them below.
        const contentChunks = runnerOptions?.contentChunks
          ? new Map<number, ContentChunk[]>()
          : undefined;
        let streamedContentChunks = false;
        for await (const chunk of stream) {
          chunk.choices?.forEach((chunk) => {
            if (contentChunks) {
              const deltaContent: unknown = chunk.delta?.content;
              let chunks = contentChunks.get(chunk.index);
              if (!chunks) {
                chunks = [];
                contentChunks.set(chunk.index, chunks);
              }
              if (Array.isArray(deltaContent)) {
                streamedContentChunks = true;
                chunks.push(...deltaContent);
              } else if (typeof deltaContent === 'string' && deltaContent) {
                chunks.push({ type: 'text', text: deltaContent });
              }
            }
            let accumulators = toolCallAccumulators.get(chunk.index);
            if (!accumulators) {
              accumulators = [];
              toolCallAccumulators.set(chunk.index, accumulators);
            }
            const c = fromOpenAIChunkChoice(chunk, undefined, accumulators);
            options?.sendChunk!({
              index: chunk.index,
              content: c.message?.content ?? [],
            });
          });
        }
        response = await stream.finalChatCompletion();
        if (contentChunks && streamedContentChunks) {
          for (const choice of response.choices) {
            const chunks = contentChunks.get(choice.index);
            if (chunks) {
              (choice.message as { content: unknown }).content = chunks;
            }
          }
        }
      } else {
        response = await client.chat.completions.create(body, {
          signal: options?.abortSignal,
        });
      }
      const standardResponse: GenerateResponseData = {
        usage: {
          inputTokens: response.usage?.prompt_tokens,
          outputTokens: response.usage?.completion_tokens,
          totalTokens: response.usage?.total_tokens,
        },
        raw: response,
      };
      if (response.choices.length === 0) {
        return standardResponse;
      } else {
        const choice = response.choices[0];
        return {
          ...fromOpenAIChoice(choice, request.output?.format === 'json'),
          ...standardResponse,
        };
      }
    } catch (e) {
      if (e instanceof APIError) {
        let status: StatusName = 'UNKNOWN';
        switch (e.status) {
          case 429:
            status = 'RESOURCE_EXHAUSTED';
            break;
          case 401:
            status = 'UNAUTHENTICATED';
            break;
          case 403:
            status = 'PERMISSION_DENIED';
            break;
          case 400:
          // Request validation errors, e.g. from Mistral's closed schema.
          case 422:
            status = 'INVALID_ARGUMENT';
            break;
          case 500:
            status = 'INTERNAL';
            break;
          case 503:
            status = 'UNAVAILABLE';
            break;
        }
        const retryAfterHeader =
          e.headers?.get?.('retry-after') ??
          (e.headers as any)?.['retry-after'];
        const retryAfterMs = retryAfterHeader
          ? parseRetryAfterMs(retryAfterHeader)
          : undefined;
        const responseMetadata: ErrorResponseMetadata | undefined =
          retryAfterMs !== undefined ? { retryAfterMs } : undefined;
        throw new GenkitError({
          status,
          message: e.message,
          responseMetadata,
        });
      }
      throw e;
    }
  };
}

/**
 * Method to define a new Genkit Model that is compatible with Open AI
 * Chat Completions API.
 *
 * These models are to be used to chat with a large language model.
 *
 * @param params An object containing parameters for defining the OpenAI
 * Chat model.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.
 * @param params.runnerOptions Optional options for the model runner, see
 * {@link ModelRunnerOptions}.

 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAIModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  client: OpenAI;
  modelRef?: ModelReference<CustomOptions>;
  requestBuilder?: ModelRequestBuilder;
  pluginOptions?: PluginOptions;
  runnerOptions?: ModelRunnerOptions;
}): ModelAction {
  const {
    name,
    client,
    pluginOptions,
    modelRef,
    requestBuilder,
    runnerOptions,
  } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    modelRef?.name ?? `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    openAIModelRunner(
      modelName,
      client,
      requestBuilder,
      pluginOptions,
      runnerOptions
    )
  );
}

const GENERIC_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    media: true,
    tools: true,
    toolChoice: true,
    systemRole: true,
  },
};

/** ModelRef helper, with reasonable defaults for OpenAI-compatible providers */
export function compatOaiModelRef<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  info?: ModelInfo;
  configSchema?: CustomOptions;
  config?: any;
  namespace?: string;
}): ModelReference<CustomOptions> {
  const {
    name,
    info = GENERIC_MODEL_INFO,
    configSchema,
    config = undefined,
    namespace,
  } = params;
  return modelRef({
    name,
    configSchema: configSchema || (ChatCompletionCommonConfigSchema as any),
    info: info,
    config,
    namespace,
  });
}
