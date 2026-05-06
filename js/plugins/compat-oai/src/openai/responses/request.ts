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

import type { GenerateRequest, MessageData, Part, Role } from 'genkit';
import { Message } from 'genkit';
import type {
  EasyInputMessage,
  FunctionTool,
  ResponseCreateParamsNonStreaming,
  ResponseInputContent,
  ResponseInputItem,
  Tool,
  WebSearchTool,
} from 'openai/resources/responses/responses';
import type { BuiltInToolSpec, OpenAIResponsesConfig } from './types';

/**
 * Map a Genkit role to a Responses API role. The `tool` role is encoded as
 * a `function_call_output` input item rather than a message and is handled
 * separately in {@link chatMessagesToResponsesInput}.
 */
function toResponsesRole(role: Role): EasyInputMessage['role'] {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'assistant';
    case 'system':
      return 'system';
    case 'tool':
      // Should never reach here — tool messages are demuxed to
      // function_call_output items by the caller.
      throw new Error(
        `tool messages must be encoded as function_call_output items`
      );
    default:
      throw new Error(`role ${role} doesn't map to a Responses API role.`);
  }
}

/**
 * Map a Genkit Part (text/media) to a Responses API
 * {@link ResponseInputContent}.
 *
 * Differs from the Chat Completions equivalent in two ways:
 *  - images use `{type:'input_image', image_url: <string>, detail}` instead
 *    of `{type:'image_url', image_url: {url}}`.
 *  - non-image media (PDFs, text files) are sent as `input_file` items.
 */
function toResponsesContent(part: Part): ResponseInputContent {
  if (part.text != null) {
    return { type: 'input_text', text: part.text };
  }
  if (part.media) {
    const url = part.media.url;
    const ct = part.media.contentType ?? '';
    if (ct.startsWith('image/') || (!ct && url.startsWith('data:image/'))) {
      return {
        type: 'input_image',
        image_url: url,
        detail: 'auto',
      };
    }
    // Non-image attachment.
    return {
      type: 'input_file',
      file_data: url,
    };
  }
  throw new Error('Unsupported Part for Responses API: must be text or media');
}

/**
 * Convert a Genkit `MessageData[]` into Responses API `input` items.
 *
 * Tool-result messages are demuxed into `function_call_output` items
 * (Responses API encodes tool outputs as input items, not separate messages).
 * Assistant messages with tool calls are split into a separate
 * `function_call` input item for each call so the model can see prior
 * function-call history during multi-turn agentic flows.
 */
export function chatMessagesToResponsesInput(
  messages: MessageData[]
): ResponseInputItem[] {
  const items: ResponseInputItem[] = [];
  for (const messageData of messages) {
    const message = new Message(messageData);
    switch (message.role) {
      case 'tool': {
        const toolResponses = message.toolResponseParts();
        if (toolResponses.length === 0) {
          throw new Error(
            'tool message must contain at least one toolResponse part'
          );
        }
        for (const part of toolResponses) {
          if (!part.toolResponse.ref) {
            throw new Error(
              `toolResponse for "${part.toolResponse.name}" is missing 'ref' — ` +
                `cannot correlate to its function_call`
            );
          }
          // The Responses API requires `output` to be a string. If the
          // caller's tool returned undefined, JSON.stringify(undefined)
          // would itself return undefined and the field would be
          // dropped from the body, so default to an empty JSON object
          // string in that case.
          items.push({
            type: 'function_call_output',
            call_id: part.toolResponse.ref,
            output:
              typeof part.toolResponse.output === 'string'
                ? part.toolResponse.output
                : JSON.stringify(part.toolResponse.output ?? {}),
          });
        }
        break;
      }
      case 'model': {
        // Surface prior tool calls as separate function_call items so
        // the model can correlate them with subsequent function_call_output.
        const toolRequests = message.content.filter(
          (p): p is Part & { toolRequest: NonNullable<Part['toolRequest']> } =>
            Boolean(p.toolRequest)
        );
        for (const part of toolRequests) {
          if (!part.toolRequest.ref) {
            throw new Error(
              `toolRequest for "${part.toolRequest.name}" is missing 'ref' — ` +
                `Responses API requires a stable call_id`
            );
          }
          items.push({
            type: 'function_call',
            call_id: part.toolRequest.ref,
            name: part.toolRequest.name,
            arguments: JSON.stringify(part.toolRequest.input ?? {}),
          });
        }
        // Plain text from a prior assistant turn.
        const textParts = message.content.filter(
          (p) => p.text != null && !p.toolRequest
        );
        if (textParts.length > 0) {
          // Prior assistant turn replayed as `input_text` items —
          // the Responses API treats anything in `input` as input
          // regardless of role, even when the role is `assistant`.
          items.push({
            type: 'message',
            role: 'assistant',
            content: textParts.map((p) => ({
              type: 'input_text',
              text: p.text!,
            })),
          } as EasyInputMessage);
        }
        break;
      }
      case 'user':
      case 'system': {
        const role = toResponsesRole(message.role);
        const content: string | ResponseInputContent[] =
          message.content.length === 1 && message.content[0].text != null
            ? message.content[0].text!
            : message.content.map(toResponsesContent);
        items.push({
          type: 'message',
          role,
          content,
        } as EasyInputMessage);
        break;
      }
      default:
        throw new Error(`Unsupported role: ${message.role}`);
    }
  }
  return items;
}

/**
 * Translate a {@link BuiltInToolSpec} into the underlying OpenAI Responses
 * tool object. Built-in tools share the `tools[]` array with function tools.
 */
function toOpenAIBuiltInTool(spec: BuiltInToolSpec): Tool {
  switch (spec.type) {
    case 'web_search_preview': {
      const tool: WebSearchTool = { type: 'web_search_preview' };
      if (spec.searchContextSize) {
        tool.search_context_size = spec.searchContextSize;
      }
      if (spec.userLocation) {
        tool.user_location = {
          type: 'approximate',
          city: spec.userLocation.city,
          country: spec.userLocation.country,
          region: spec.userLocation.region,
          timezone: spec.userLocation.timezone,
        };
      }
      return tool;
    }
    case 'file_search': {
      // ranking_options requires score_threshold per the OpenAI API; only
      // emit it when the caller actually provided one.
      const rankingOptions =
        spec.ranker?.scoreThreshold != null
          ? {
              score_threshold: spec.ranker.scoreThreshold,
              ...(spec.ranker.ranker ? { ranker: spec.ranker.ranker } : {}),
            }
          : undefined;
      return {
        type: 'file_search',
        vector_store_ids: spec.vectorStoreIds,
        ...(spec.maxNumResults != null
          ? { max_num_results: spec.maxNumResults }
          : {}),
        ...(rankingOptions ? { ranking_options: rankingOptions } : {}),
      } as Tool;
    }
    case 'code_interpreter': {
      const container = spec.container;
      const resolved: string | { type: 'auto'; file_ids?: string[] } =
        container == null
          ? { type: 'auto' }
          : typeof container === 'string'
            ? container
            : container.fileIds && container.fileIds.length > 0
              ? { type: 'auto', file_ids: container.fileIds }
              : { type: 'auto' };
      return {
        type: 'code_interpreter',
        container: resolved,
      } as Tool;
    }
  }
}

/**
 * Convert a Genkit ToolDefinition list into Responses API `function` tools.
 *
 * Mirrors `toOpenAITool` from `model.ts` but emits the Responses-flavored
 * shape: `{type:'function', name, parameters, strict}` (Chat Completions
 * wraps under a nested `function: {...}` object — Responses API does not).
 */
function toResponsesFunctionTools(request: GenerateRequest): FunctionTool[] {
  return (request.tools ?? []).map(
    (tool): FunctionTool => ({
      type: 'function',
      name: tool.name,
      description: tool.description,
      parameters: (tool.inputSchema ?? null) as Record<string, unknown> | null,
      strict: false,
    })
  );
}

/**
 * Build the Responses API request body from a Genkit
 * {@link GenerateRequest}.
 *
 * Notable behaviours:
 *  - `output.format === 'json'` + `output.schema` ⇒
 *    `text.format = { type: 'json_schema', strict: true, schema }`.
 *  - For models with `supports.systemRole === false` (o1/o3 family),
 *    system messages are extracted into the top-level `instructions`
 *    field and dropped from the input array. Callers can also set
 *    `config.instructions` explicitly to override.
 *  - `config.builtInTools` are appended to `tools[]` after function tools.
 *  - `config.store` defaults to `false` (stateless-by-default — see README).
 */
export function toResponsesRequestBody(
  modelName: string,
  request: GenerateRequest<typeof import('./types').OpenAIResponsesConfigSchema>
): ResponseCreateParamsNonStreaming {
  const config: OpenAIResponsesConfig = (request.config ??
    {}) as OpenAIResponsesConfig;

  let messages = request.messages;
  let instructions = config.instructions;

  // Reasoning models (o1/o3/gpt-5*) ignore `system` role messages; lift
  // them into `instructions`. We avoid changing global plugin behaviour
  // by always doing this when `instructions` is unset and we see a
  // leading system message — Responses API treats the two as roughly
  // equivalent and this preserves caller intent.
  //
  // Lifting is only safe when the system message is text-only. If a
  // system message carries media (rare but valid), we leave it in the
  // input array rather than silently dropping the media.
  if (instructions == null) {
    const systemMessages = messages.filter((m) => m.role === 'system');
    const liftable = systemMessages.filter((m) =>
      m.content.every((p) => p.text != null)
    );
    if (liftable.length > 0) {
      instructions = liftable
        .flatMap((m) => m.content.map((p) => p.text ?? ''))
        .filter(Boolean)
        .join('\n\n');
      const liftedSet = new Set(liftable);
      messages = messages.filter((m) => !liftedSet.has(m));
    }
  }

  const input = chatMessagesToResponsesInput(messages);

  // Compose tools[]: function tools (from request.tools) + builtInTools.
  const tools: Tool[] = [
    ...toResponsesFunctionTools(request),
    ...(config.builtInTools ?? []).map(toOpenAIBuiltInTool),
  ];

  // Resolve text.format. Genkit's high-level output config wins by default;
  // explicit text.format in config overrides everything. The Responses
  // API requires a `name` for json_schema variants — default to `output`.
  let textFormat:
    | NonNullable<
        NonNullable<ResponseCreateParamsNonStreaming['text']>['format']
      >
    | undefined;
  if (config.text?.format) {
    const cf = config.text.format;
    if (cf.type === 'json_schema') {
      textFormat = {
        type: 'json_schema',
        name: cf.name ?? 'output',
        schema: cf.schema,
        ...(cf.strict != null ? { strict: cf.strict } : {}),
        ...(cf.description != null ? { description: cf.description } : {}),
      };
    } else {
      textFormat = cf;
    }
  } else if (request.output?.format === 'json') {
    if (request.output.schema) {
      textFormat = {
        type: 'json_schema',
        name: 'output',
        schema: request.output.schema as Record<string, unknown>,
        strict: true,
      };
    } else {
      textFormat = { type: 'json_object' };
    }
  }

  const body: ResponseCreateParamsNonStreaming = {
    model: config.version ?? modelName,
    input,
    ...(instructions ? { instructions } : {}),
    ...(tools.length > 0 ? { tools } : {}),
    ...(config.temperature != null ? { temperature: config.temperature } : {}),
    ...(config.topP != null ? { top_p: config.topP } : {}),
    ...(config.maxOutputTokens != null
      ? { max_output_tokens: config.maxOutputTokens }
      : {}),
    ...(config.user != null ? { user: config.user } : {}),
    ...(config.previousResponseId != null
      ? { previous_response_id: config.previousResponseId }
      : {}),
    ...(config.metadata ? { metadata: config.metadata } : {}),
    ...(config.parallelToolCalls != null
      ? { parallel_tool_calls: config.parallelToolCalls }
      : {}),
    ...(config.truncation ? { truncation: config.truncation } : {}),
    ...(config.maxToolCalls != null
      ? { max_tool_calls: config.maxToolCalls }
      : {}),
    ...(config.serviceTier != null ? { service_tier: config.serviceTier } : {}),
    ...(textFormat || config.text?.verbosity
      ? {
          text: {
            ...(config.text?.verbosity
              ? { verbosity: config.text.verbosity }
              : {}),
            ...(textFormat ? { format: textFormat } : {}),
          },
        }
      : {}),
    ...(config.reasoning
      ? {
          reasoning: {
            ...(config.reasoning.effort
              ? { effort: config.reasoning.effort }
              : {}),
            ...(config.reasoning.summary
              ? { summary: config.reasoning.summary }
              : {}),
          },
        }
      : {}),
    ...(config.include ? { include: config.include } : {}),
    // Stateless-by-default — see README. Caller can opt into
    // server-side persistence via `config.store: true`.
    store: config.store ?? false,
  };

  return body;
}
