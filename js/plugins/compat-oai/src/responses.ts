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
import type { ModelAction, ModelInfo } from 'genkit/model';
import { model } from 'genkit/plugin';
import type OpenAI from 'openai';
import type {
  Response as OpenAIResponse,
  ResponseCreateParamsNonStreaming,
  ResponseInput,
  ResponseInputContent,
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
 * Serializes a prior model turn into the text of an assistant input message.
 *
 * Structured output comes back as a `data` part carrying no text, so replaying
 * only the turn's text would send an empty assistant message and lose the
 * model's own previous answer.
 * @param parts The content of the model message.
 * @returns The assistant message text, empty when the turn carries nothing
 * this transport can replay.
 * @throws GenkitError if a part cannot be represented in assistant history.
 */
function fromModelTurn(parts: Part[]): string {
  let text = '';
  for (const part of parts) {
    if (part.text !== undefined) {
      text += part.text;
    } else if (part.data !== undefined) {
      text += JSON.stringify(part.data);
    } else if (part.reasoning !== undefined) {
      // Reasoning is replayed through the encrypted round-trip, which this
      // slice does not implement; the plain summary adds nothing on its own.
      continue;
    } else {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(part)}.`,
      });
    }
  }
  return text;
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
      case 'model': {
        // An assistant message with no content carries nothing and OpenAI
        // rejects it, so a turn that reduces to nothing is left out entirely.
        const content = fromModelTurn(msg.content);
        if (content) input.push({ role: 'assistant', content });
        break;
      }
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
  // Genkit only warns when a model declares `supports.tools: false`, so without
  // this the tools would be dropped and the model would answer as if none had
  // been offered.
  if (request.tools?.length) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Tool calling is not yet supported on the OpenAI Responses API transport (model ${modelName}).`,
    });
  }

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
    // Selects the OpenAI transport; it is a plugin concept, not a wire field.
    transport,
    apiKey,
    ...restOfConfig
  } = request.config ?? {};

  if (transport !== undefined && transport !== 'responses') {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Unsupported transport '${transport}'; this model is served over the Responses API.`,
    });
  }

  const body: ResponseCreateParamsNonStreaming = {
    model: modelVersion ?? modelName,
    input,
    instructions,
    max_output_tokens,
    temperature,
    top_p,
    // The Responses API retains requests and responses server-side by default;
    // pinning it off matches the Chat Completions retention posture.
    store: store ?? false,
    ...restOfConfig,
  };

  const format = request.output?.format;
  if (format === 'json') {
    body.text = {
      format: request.output?.schema
        ? {
            type: 'json_schema',
            name: 'output',
            schema: request.output.schema,
          }
        : { type: 'json_object' },
    };
  } else if (format === 'text') {
    body.text = { format: { type: 'text' } };
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
    case 'failed':
      return {
        finishReason: 'other',
        finishMessage: response.error?.message,
      };
    default:
      return { finishReason: 'unknown' };
  }
}

/**
 * Output items asking the caller to run a tool. Genkit has no way to answer one
 * on this transport yet, and dropping it would leave the caller with an empty
 * response instead of an error.
 */
const TOOL_CALL_ITEM_TYPES = new Set(['function_call', 'custom_tool_call']);

/**
 * Converts an OpenAI Response into Genkit response data.
 * @param response The Response to convert.
 * @param jsonMode Whether the response text is expected to be JSON.
 * @returns The converted Genkit GenerateResponseData object.
 * @throws GenkitError if the response failed or asks for a tool call.
 */
export function fromOpenAIResponse(
  response: OpenAIResponse,
  jsonMode = false
): GenerateResponseData {
  // A failed response carries its cause in `error` and nothing in `output`, so
  // returning it as an empty completion would hide the reason entirely.
  if (response.status === 'failed' && response.error) {
    throw new GenkitError({
      status: 'INTERNAL',
      message: `OpenAI Responses API request failed (${response.error.code}): ${response.error.message}`,
    });
  }

  const content: Part[] = [];
  let refused = false;
  for (const item of response.output ?? []) {
    if (item.type === 'reasoning') {
      for (const summary of item.summary ?? []) {
        if (summary.text) content.push({ reasoning: summary.text });
      }
    } else if (item.type === 'message') {
      for (const contentItem of item.content ?? []) {
        if (contentItem.type === 'output_text') {
          content.push(
            jsonMode
              ? { data: JSON.parse(contentItem.text) }
              : { text: contentItem.text }
          );
        } else if (contentItem.type === 'refusal') {
          refused = true;
          content.push({ text: contentItem.refusal });
        }
      }
    } else if (TOOL_CALL_ITEM_TYPES.has(item.type)) {
      throw new GenkitError({
        status: 'UNIMPLEMENTED',
        message: `Tool calling is not yet supported on the OpenAI Responses API transport; the model returned a '${item.type}' item.`,
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
    },
    raw: response,
  };
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
  pluginOptions?: Omit<PluginOptions, 'apiKey'>
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
    try {
      const response = await client.responses.create(
        toOpenAIResponsesRequestBody(name, request),
        { signal: options?.abortSignal }
      );
      const converted = fromOpenAIResponse(
        response,
        request.output?.format === 'json'
      );
      // The Responses event protocol is not mapped yet, so a streaming caller
      // gets the completed response delivered as a single chunk rather than
      // nothing at all.
      if (options?.streamingRequested && options.sendChunk) {
        options.sendChunk({
          index: 0,
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
}): ModelAction {
  const { name, client, pluginOptions, modelRef } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    modelRef?.name ?? `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    openAIResponsesModelRunner(modelName, client, pluginOptions)
  );
}

const GENERIC_RESPONSES_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    media: true,
    tools: false,
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
