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
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
  MessageData,
  ModelReference,
  Part,
  StreamingCallback,
} from 'genkit';
import {
  GenerationCommonConfigSchema,
  GenkitError,
  Message,
  modelRef,
  z,
} from 'genkit';
import { parsePartialJson } from 'genkit/extract';
import type { ModelAction, ModelInfo, ToolDefinition } from 'genkit/model';
import { model } from 'genkit/plugin';
import type OpenAI from 'openai';
import type {
  FunctionTool,
  Response as OpenAIResponse,
  ResponseCreateParamsNonStreaming,
  ResponseIncludable,
  ResponseInput,
  ResponseInputContent,
  ResponseInputItem,
  ResponseReasoningItem,
  ResponseStreamEvent,
  Tool,
} from 'openai/resources/responses/responses.mjs';
import { PluginOptions } from './index.js';
import {
  extractDataFromBase64Url,
  generateFilenameFromContentType,
  isImageContentType,
  maybeCreateRequestScopedOpenAIClient,
  rethrowOpenAIError,
  toModelName,
} from './utils.js';

type VisualDetailLevel = 'auto' | 'low' | 'high';

/**
 * Generation config shared by models served over the OpenAI Responses API.
 *
 * `topK` and `stopSequences` are omitted because the Responses API has no
 * equivalent for either, so advertising them would offer fields that are
 * silently dropped. The schema is otherwise narrower than the Responses request
 * surface; anything not declared here still reaches the wire through the
 * passthrough in {@link toOpenAIResponsesRequestBody}.
 */
export const ResponsesCommonConfigSchema = GenerationCommonConfigSchema.omit({
  topK: true,
  stopSequences: true,
}).extend({
  temperature: z.number().min(0).max(2).optional(),
});

/**
 * Converts a Genkit Part into a Responses API input content item.
 * @param part The Genkit Part to convert.
 * @param visualDetailLevel The visual detail level to use for image parts.
 * @returns The corresponding Responses API input content item.
 * @throws GenkitError if the part cannot be represented as Responses API input.
 */
export function toOpenAIResponsesContent(
  part: Part,
  visualDetailLevel: VisualDetailLevel = 'auto'
): ResponseInputContent {
  if (part.text) {
    return { type: 'input_text', text: part.text };
  }
  if (part.media) {
    let contentType = part.media.contentType;
    if (!contentType && part.media.url.startsWith('data:')) {
      contentType = extractDataFromBase64Url(part.media.url)?.contentType;
    }

    // Media without a content type is treated as an image, preserving the
    // behaviour of the Chat Completions converter for signed/remote URLs.
    if (!contentType || isImageContentType(contentType)) {
      return {
        type: 'input_image',
        detail: visualDetailLevel,
        image_url: part.media.url,
      };
    }

    if (part.media.url.startsWith('data:')) {
      const extracted = extractDataFromBase64Url(part.media.url);
      if (!extracted) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message: `Invalid data URL format for media: ${part.media.url.substring(0, 50)}...`,
        });
      }
      return {
        type: 'input_file',
        filename: generateFilenameFromContentType(extracted.contentType),
        file_data: part.media.url,
      };
    }

    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `File URLs are not supported. Only base64-encoded files and image URLs are supported. Content type: ${contentType}`,
    });
  }
  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(part)}.`,
  });
}

/**
 * Converts a Genkit ToolDefinition into a Responses API function tool.
 *
 * The Responses `FunctionTool` shape is flat and requires `strict`, so the Chat
 * Completions converter cannot be reused. `strict` stays null: genkit tool
 * schemas are not authored for OpenAI strict mode, which rejects any schema
 * missing `additionalProperties: false`.
 * @param tool The Genkit ToolDefinition to convert.
 * @returns The corresponding Responses API function tool.
 */
export function toOpenAIResponsesTool(tool: ToolDefinition): FunctionTool {
  return {
    type: 'function',
    name: tool.name,
    description: tool.description,
    parameters: (tool.inputSchema ?? null) as Record<string, unknown> | null,
    strict: null,
  };
}

/**
 * Serializes a prior model turn into Responses API input items, preserving the
 * order of text, tool-call and reasoning parts.
 *
 * Structured output comes back as a `data` part carrying no text, so replaying
 * only the turn's text would send an empty assistant message and lose the
 * model's own previous answer. Reasoning parts replay only through the
 * encrypted round-trip: the item id and encrypted payload ride `part.metadata`,
 * and a summary-only reasoning part carries nothing the API can resume from.
 * @param parts The content of the model message.
 * @returns The input items, empty when the turn carries nothing this transport
 * can replay.
 * @throws GenkitError if a part cannot be represented in assistant history.
 */
function toModelTurnItems(parts: Part[]): ResponseInputItem[] {
  const items: ResponseInputItem[] = [];
  let text = '';
  const flushText = () => {
    if (text) {
      items.push({ role: 'assistant', content: text });
      text = '';
    }
  };
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (part.text !== undefined) {
      text += part.text;
    } else if (part.data !== undefined) {
      text += JSON.stringify(part.data);
    } else if (part.toolRequest !== undefined) {
      flushText();
      const itemId = part.metadata?.itemId as string | undefined;
      items.push({
        type: 'function_call',
        call_id: part.toolRequest.ref ?? '',
        name: part.toolRequest.name,
        arguments: JSON.stringify(part.toolRequest.input ?? {}),
        ...(itemId ? { id: itemId } : {}),
      });
    } else if (part.reasoning !== undefined) {
      const itemId = part.metadata?.itemId as string | undefined;
      if (!itemId) continue;
      flushText();
      // One Responses reasoning item was fanned out into one part per summary;
      // fold consecutive parts with the same item id back into a single item.
      const summary: ResponseReasoningItem['summary'] = [];
      let encrypted: string | undefined;
      let j = i;
      for (
        ;
        j < parts.length &&
        parts[j].reasoning !== undefined &&
        parts[j].metadata?.itemId === itemId;
        j++
      ) {
        if (parts[j].reasoning) {
          summary.push({ type: 'summary_text', text: parts[j].reasoning! });
        }
        encrypted ??= parts[j].metadata?.encryptedContent as string | undefined;
      }
      i = j - 1;
      items.push({
        type: 'reasoning',
        id: itemId,
        summary,
        ...(encrypted ? { encrypted_content: encrypted } : {}),
      });
    } else {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(part)}.`,
      });
    }
  }
  flushText();
  return items;
}

/**
 * Converts Genkit messages into Responses API `input` items.
 *
 * System messages are hoisted into `instructions`, which is where the Responses
 * API takes them; the remaining messages keep their order in `input`.
 * @param messages The Genkit messages to convert.
 * @param visualDetailLevel The visual detail level to use for image parts.
 * @returns The `input` items and, when any system message was present, the
 * joined `instructions` string.
 */
export function toOpenAIResponsesInput(
  messages: MessageData[],
  visualDetailLevel: VisualDetailLevel = 'auto'
): { input: ResponseInput; instructions?: string } {
  const input: ResponseInput = [];
  const instructions: string[] = [];
  for (const message of messages) {
    const msg = new Message(message);
    switch (message.role) {
      case 'system':
        if (msg.text) instructions.push(msg.text);
        break;
      case 'user':
        input.push({
          role: 'user',
          content: msg.content.map((part) =>
            toOpenAIResponsesContent(part, visualDetailLevel)
          ),
        });
        break;
      case 'model':
        input.push(...toModelTurnItems(msg.content));
        break;
      case 'tool':
        for (const part of msg.content) {
          if (part.toolResponse === undefined) {
            throw new GenkitError({
              status: 'INVALID_ARGUMENT',
              message: `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(part)}.`,
            });
          }
          input.push({
            type: 'function_call_output',
            call_id: part.toolResponse.ref ?? '',
            // `output` is required on the wire; a void tool's undefined would
            // vanish from the serialized JSON entirely.
            output:
              typeof part.toolResponse.output === 'string'
                ? part.toolResponse.output
                : JSON.stringify(part.toolResponse.output ?? null),
          });
        }
        break;
      default:
        throw new GenkitError({
          status: 'UNIMPLEMENTED',
          message: `role ${message.role} is not supported by the OpenAI Responses API transport.`,
        });
    }
  }
  return {
    input,
    instructions: instructions.length ? instructions.join('\n\n') : undefined,
  };
}

/**
 * Checks whether a model belongs to a reasoning family, i.e. can return
 * reasoning items with encrypted content.
 *
 * Hand-curated by family like the transport lists in openai/responses.ts:
 * requesting `reasoning.encrypted_content` from a non-reasoning model is a
 * 400, not a no-op. Chat-tuned variants (`gpt-5-chat-latest`) are the
 * non-reasoning exceptions inside a reasoning family.
 * @param name The bare model name, without the plugin namespace.
 */
export function isReasoningModelName(name?: string): boolean {
  if (!name) return false;
  if (name.includes('chat')) return false;
  return /^o\d|^gpt-5|^codex/.test(name);
}

/** Drops keys whose value is `undefined` so they never show up in traces. */
function stripUndefined<T extends object>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, v]) => v !== undefined)
  ) as T;
}

/**
 * Converts a Genkit request into an OpenAI Responses API request body.
 * @param modelName The name of the model to use.
 * @param request The Genkit GenerateRequest to convert.
 * @returns The Responses API request body.
 */
export function toOpenAIResponsesRequestBody(
  modelName: string,
  request: GenerateRequest
): ResponseCreateParamsNonStreaming {
  const { input, instructions } = toOpenAIResponsesInput(
    request.messages,
    request.config?.visualDetailLevel
  );
  const {
    temperature,
    maxOutputTokens: max_output_tokens,
    topP: top_p,
    topK, // the Responses API has no equivalent
    stopSequences, // the Responses API has no equivalent
    visualDetailLevel, // consumed while building the input items above
    version: modelVersion,
    store,
    previousResponseId,
    reasoningEffort,
    reasoningSummary,
    instructions: instructionsFromConfig,
    tools: toolsFromConfig,
    include: includeFromConfig,
    // Selects the OpenAI transport; it is a plugin concept, not a wire field.
    transport,
    stream,
    background,
    apiKey,
    ...restOfConfig
  } = request.config ?? {};

  if (transport !== undefined && transport !== 'responses') {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Unsupported transport '${transport}'; this model is served over the Responses API.`,
    });
  }

  // Either key through the passthrough would change the response envelope out
  // from under this runner: a Stream or a queued response read as a completed
  // one comes back as an empty message with a clean finish.
  if (stream !== undefined) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `'stream' is not a config option; use Genkit's streaming API (generateStream) instead.`,
    });
  }
  if (background !== undefined) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Background responses are not supported.`,
    });
  }

  // Joined rather than passed through: a passthrough `instructions` landing
  // after the spread would silently discard every hoisted system message.
  const mergedInstructions =
    [instructions, instructionsFromConfig as string | undefined]
      .filter(Boolean)
      .join('\n\n') || undefined;

  const tools: Tool[] = request.tools?.map(toOpenAIResponsesTool) ?? [];
  if (toolsFromConfig) {
    tools.push(...(toolsFromConfig as Tool[]));
  }

  const storeValue = store ?? false;
  // A bare-string `include` from the passthrough config would otherwise be
  // spread into single characters below.
  const callerInclude =
    typeof includeFromConfig === 'string'
      ? [includeFromConfig as ResponseIncludable]
      : (includeFromConfig as ResponseIncludable[] | undefined);
  // Under the stateless default the encrypted reasoning payload is the only
  // context a reasoning model can resume from, so it is requested for every
  // model that can return one; asking a non-reasoning model for it is a 400,
  // not a no-op, so the gate matters for dual-transport models.
  const include: ResponseIncludable[] | undefined =
    !storeValue && isReasoningModelName((modelVersion as string) ?? modelName)
      ? [
          ...new Set<ResponseIncludable>([
            ...(callerInclude ?? []),
            'reasoning.encrypted_content',
          ]),
        ]
      : callerInclude;

  const body: ResponseCreateParamsNonStreaming = {
    model: modelVersion ?? modelName,
    input,
    instructions: mergedInstructions,
    max_output_tokens,
    temperature,
    top_p,
    tools: tools.length ? tools : undefined,
    tool_choice: request.toolChoice,
    include,
    previous_response_id: previousResponseId,
    // The Responses API retains requests and responses server-side by default;
    // pinning it off matches the Chat Completions retention posture.
    store: storeValue,
    ...restOfConfig,
  };

  // Composed onto any raw `reasoning` object from the passthrough so the
  // declared fields win without discarding the rest of it.
  if (reasoningEffort !== undefined || reasoningSummary !== undefined) {
    body.reasoning = {
      ...body.reasoning,
      ...(reasoningEffort !== undefined ? { effort: reasoningEffort } : {}),
      ...(reasoningSummary !== undefined ? { summary: reasoningSummary } : {}),
    };
  }

  // Composed onto any raw `text` object from the passthrough (e.g. verbosity)
  // rather than replacing it wholesale.
  const format = request.output?.format;
  if (format === 'json') {
    body.text = {
      ...body.text,
      format: request.output?.schema
        ? {
            type: 'json_schema',
            name: 'output',
            // Unlike Chat Completions, the Responses API validates the schema
            // under strict mode unless told otherwise, and genkit schemas are
            // not authored for it (`additionalProperties: false` everywhere).
            strict: false,
            schema: request.output.schema,
          }
        : { type: 'json_object' },
    };
  } else if (format === 'text') {
    body.text = { ...body.text, format: { type: 'text' } };
  }

  return stripUndefined(body);
}

/**
 * Maps the terminal state of a Response onto a Genkit finish reason, keeping
 * whatever the API said about it as the finish message.
 */
function toFinishInfo(response: OpenAIResponse): {
  finishReason: GenerateResponseData['finishReason'];
  finishMessage?: string;
} {
  const reason = response.incomplete_details?.reason;
  if (reason === 'max_output_tokens') return { finishReason: 'length' };
  if (reason === 'content_filter') return { finishReason: 'blocked' };
  switch (response.status) {
    case undefined:
    case 'completed':
      return { finishReason: 'stop' };
    case 'incomplete':
      return {
        finishReason: 'other',
        finishMessage: `Response incomplete: ${reason ?? 'no reason given'}.`,
      };
    case 'cancelled':
      return { finishReason: 'other', finishMessage: 'Response cancelled.' };
    default:
      // 'failed' is unreachable: fromOpenAIResponse throws before mapping it.
      return { finishReason: 'unknown' };
  }
}

/**
 * Best-effort conversion of json-mode output text into a data part. A response
 * truncated by `max_output_tokens` or a content filter can end with partial
 * JSON, which must surface through the finish reason rather than as a
 * SyntaxError.
 */
function toJsonData(text: string): Part {
  try {
    return { data: JSON.parse(text) };
  } catch {
    try {
      return { data: parsePartialJson(text) };
    } catch {
      return { text };
    }
  }
}

/**
 * Best-effort parse of a function call's `arguments` string. A response
 * truncated by `max_output_tokens` can end with partial JSON, which must
 * surface as a `length` finish reason rather than a SyntaxError.
 */
function parseFunctionCallArguments(args: string): unknown {
  if (!args) return {};
  try {
    return JSON.parse(args);
  } catch {
    try {
      return parsePartialJson(args);
    } catch {
      return args;
    }
  }
}

/**
 * Converts an OpenAI Response into Genkit response data.
 * @param response The Response to convert.
 * @param jsonMode Whether the response text is expected to be JSON.
 * @returns The converted Genkit GenerateResponseData object.
 * @throws GenkitError if the response failed or asks for a custom tool call.
 */
export function fromOpenAIResponse(
  response: OpenAIResponse,
  jsonMode = false
): GenerateResponseData {
  // Returning a failed response as an empty completion would hide the failure
  // entirely: the message object is present, so genkit's validation passes.
  if (response.status === 'failed') {
    throw new GenkitError({
      status: 'INTERNAL',
      message: response.error
        ? `OpenAI Responses API request failed (${response.error.code}): ${response.error.message}`
        : 'OpenAI Responses API request failed without an error payload.',
    });
  }

  const content: Part[] = [];
  let refused = false;
  for (const item of response.output ?? []) {
    if (item.type === 'reasoning') {
      // The item id and encrypted payload ride part metadata so the item can
      // be reassembled and replayed on the next stateless turn. The payload
      // rides only the item's first part; the id marks the rest as one item.
      const meta: Record<string, unknown> = { itemId: item.id };
      if (item.encrypted_content) {
        meta.encryptedContent = item.encrypted_content;
      }
      const summaries = (item.summary ?? []).filter((s) => s.text);
      if (summaries.length === 0) {
        // A reasoning model asked for no summary still returns the item; the
        // empty part exists purely to carry the payload through history.
        if (item.encrypted_content) {
          content.push({ reasoning: '', metadata: meta });
        }
      } else {
        summaries.forEach((summary, index) => {
          content.push({
            reasoning: summary.text,
            metadata: index === 0 ? meta : { itemId: item.id },
          });
        });
      }
    } else if (item.type === 'message') {
      for (const contentItem of item.content ?? []) {
        if (contentItem.type === 'output_text') {
          const part: Part = jsonMode
            ? toJsonData(contentItem.text)
            : { text: contentItem.text };
          if (contentItem.annotations?.length) {
            part.metadata = { annotations: contentItem.annotations };
          }
          content.push(part);
        } else if (contentItem.type === 'refusal') {
          refused = true;
          content.push({ text: contentItem.refusal });
        }
      }
    } else if (item.type === 'function_call') {
      content.push({
        toolRequest: {
          name: item.name,
          ref: item.call_id,
          input: parseFunctionCallArguments(item.arguments),
        },
        ...(item.id ? { metadata: { itemId: item.id } } : {}),
      });
    } else if ((item.type as string) === 'custom_tool_call') {
      // The pinned SDK's output-item union predates custom tools, but the live
      // API can still return one.
      throw new GenkitError({
        status: 'UNIMPLEMENTED',
        message: `Custom tool calls are not supported on the OpenAI Responses API transport; the model returned a 'custom_tool_call' item.`,
      });
    }
    // Records of tools OpenAI ran itself (`web_search_call`, `mcp_call`, ...);
    // the caller-visible result arrives in the message item alongside them.
  }

  const finish = refused
    ? { finishReason: 'blocked' as const }
    : toFinishInfo(response);

  return {
    ...finish,
    message: {
      role: 'model',
      content,
    },
    usage: {
      inputTokens: response.usage?.input_tokens,
      outputTokens: response.usage?.output_tokens,
      totalTokens: response.usage?.total_tokens,
      thoughtsTokens: response.usage?.output_tokens_details?.reasoning_tokens,
      cachedContentTokens: response.usage?.input_tokens_details?.cached_tokens,
    },
    raw: response,
  };
}

/**
 * Accumulates the streamed argument fragments of one function call. The
 * `response.output_item.added` event carries the call's `name` and `call_id`;
 * subsequent `response.function_call_arguments.delta` events carry only
 * fragments of the `arguments` JSON string, keyed by `output_index`.
 */
export interface ResponsesToolCallAccumulator {
  name: string;
  ref: string;
  /** Concatenated `arguments` JSON string fragments received so far. */
  args: string;
}

/**
 * Converts one Responses API stream event into a Genkit response chunk.
 *
 * Annotation events are deliberately ignored: annotations attach to the final
 * text part's metadata in {@link fromOpenAIResponse}, which every streamed
 * request still runs through.
 * @param event The stream event to convert.
 * @param toolCalls Per-request function-call accumulators, keyed by
 * `output_index`. Mutated in place as fragments arrive.
 * @returns The chunk to emit, or `undefined` for events with no chunk-visible
 * payload.
 *
 * Chunks carry no `index`: that field is the position of the message in the
 * conversation, not the response's `output_index`, and core fills it with the
 * tool-loop message index when omitted.
 */
export function fromOpenAIResponsesStreamEvent(
  event: ResponseStreamEvent,
  toolCalls: Map<number, ResponsesToolCallAccumulator>
): GenerateResponseChunkData | undefined {
  switch (event.type) {
    case 'response.output_text.delta':
      return { content: [{ text: event.delta }] };
    case 'response.refusal.delta':
      return { content: [{ text: event.delta }] };
    case 'response.reasoning_summary_text.delta':
      return { content: [{ reasoning: event.delta }] };
    case 'response.output_item.added': {
      if (event.item.type !== 'function_call') return undefined;
      const acc: ResponsesToolCallAccumulator = {
        name: event.item.name,
        ref: event.item.call_id,
        args: event.item.arguments ?? '',
      };
      toolCalls.set(event.output_index, acc);
      return {
        content: [
          {
            toolRequest: {
              name: acc.name,
              ref: acc.ref,
              input: {},
              partial: true,
            },
          },
        ],
      };
    }
    case 'response.function_call_arguments.delta': {
      const acc = toolCalls.get(event.output_index);
      if (!acc) return undefined;
      acc.args += event.delta;
      let input: unknown = {};
      try {
        input = acc.args ? parsePartialJson(acc.args) : {};
      } catch {
        input = {};
      }
      return {
        content: [
          {
            toolRequest: { name: acc.name, ref: acc.ref, input, partial: true },
          },
        ],
      };
    }
    default:
      return undefined;
  }
}

/**
 * Creates the runner used by Genkit to interact with a model over the OpenAI
 * Responses API.
 * @param name The name of the model.
 * @param defaultClient The OpenAI client instance.
 * @param pluginOptions Options of the plugin that owns the model.
 * @returns The runner that Genkit will call when the model is invoked.
 */
export function openAIResponsesModelRunner(
  name: string,
  defaultClient: OpenAI,
  pluginOptions?: Omit<PluginOptions, 'apiKey'>,
  modelOptions?: { streaming?: boolean }
) {
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
    // Some Responses-only models (o1-pro, o3-pro) reject `stream: true`, and
    // genkit's default paths always request streaming, so they must fall back
    // to a non-streaming request answered as a single chunk.
    const canStream = modelOptions?.streaming !== false;
    try {
      const body = toOpenAIResponsesRequestBody(name, request);
      let response: OpenAIResponse;
      if (options?.streamingRequested && canStream) {
        const stream = client.responses.stream(
          { ...body, stream: true },
          { signal: options?.abortSignal }
        );
        const toolCalls = new Map<number, ResponsesToolCallAccumulator>();
        // The SDK's finalResponse() snapshot only folds in `response.completed`,
        // so a stream ending in `response.incomplete` or `response.failed`
        // would come back as a stale in-progress response with the truncation
        // or failure erased. Capture the terminal event's response instead.
        let terminalResponse: OpenAIResponse | undefined;
        for await (const event of stream) {
          if (
            event.type === 'response.completed' ||
            event.type === 'response.incomplete' ||
            event.type === 'response.failed'
          ) {
            terminalResponse = event.response;
          }
          const chunk = fromOpenAIResponsesStreamEvent(event, toolCalls);
          if (chunk) options.sendChunk!(chunk);
        }
        response = terminalResponse ?? (await stream.finalResponse());
      } else {
        response = await client.responses.create(body, {
          signal: options?.abortSignal,
        });
      }
      const converted = fromOpenAIResponse(
        response,
        request.output?.format === 'json'
      );
      if (options?.streamingRequested && !canStream && options.sendChunk) {
        options.sendChunk({
          content: converted.message?.content ?? [],
        });
      }
      return converted;
    } catch (e) {
      rethrowOpenAIError(e);
    }
  };
}

/**
 * Method to define a new Genkit Model that is served over the OpenAI Responses
 * API.
 *
 * @param params An object containing parameters for defining the model.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.
 * @param params.pluginOptions Options of the plugin that owns the model.
 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAIResponsesModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  client: OpenAI;
  modelRef?: ModelReference<CustomOptions>;
  pluginOptions?: PluginOptions;
  /** Set false for models that reject `stream: true`; defaults to true. */
  streaming?: boolean;
}): ModelAction {
  const { name, client, pluginOptions, modelRef, streaming } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    modelRef?.name ?? `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    openAIResponsesModelRunner(modelName, client, pluginOptions, { streaming })
  );
}

const GENERIC_RESPONSES_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    media: true,
    tools: true,
    toolChoice: true,
    systemRole: true,
    output: ['text', 'json'],
    // Without this genkit wraps the model in simulateConstrainedGeneration,
    // which strips output.schema and appends it to the prompt instead of
    // letting the request carry `text.format`.
    constrained: 'all',
  },
};

/** ModelRef helper for models served over the OpenAI Responses API. */
export function compatOaiResponsesModelRef<
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
    info = GENERIC_RESPONSES_MODEL_INFO,
    configSchema,
    config = undefined,
    namespace,
  } = params;
  return modelRef({
    name,
    configSchema:
      configSchema ?? (ResponsesCommonConfigSchema as unknown as CustomOptions),
    info,
    config,
    namespace,
  });
}
