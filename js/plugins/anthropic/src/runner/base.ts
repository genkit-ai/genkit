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

import { Anthropic } from '@anthropic-ai/sdk';
import type { DocumentBlockParam } from '@anthropic-ai/sdk/resources/messages';
import type {
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
  MessageData,
  Part,
  Role,
} from 'genkit';
import { Message as GenkitMessage } from 'genkit';
import type { ToolDefinition } from 'genkit/model';

import {
  AnthropicConfigSchema,
  Media,
  MediaSchema,
  MediaType,
  MediaTypeSchema,
  type ClaudeRunnerParams,
  type ThinkingConfig,
} from '../types.js';

import {
  RunnerContentBlockParam,
  RunnerMessage,
  RunnerMessageParam,
  RunnerRequestBody,
  RunnerStream,
  RunnerStreamEvent,
  RunnerStreamingRequestBody,
  RunnerTool,
  RunnerToolResponseContent,
  RunnerTypes,
} from './types.js';

/**
 * Shared runner logic for Anthropic SDK integrations.
 *
 * Concrete subclasses pass in their SDK-specific type bundle via `RunnerTypes`,
 * letting this base class handle message/tool translation once for both the
 * stable and beta APIs that share the same conceptual surface.
 */
export abstract class BaseRunner<ApiTypes extends RunnerTypes> {
  protected name: string;
  protected client: Anthropic;

  /**
   * Default maximum output tokens for Claude models when not specified in the request.
   */
  protected readonly DEFAULT_MAX_OUTPUT_TOKENS = 4096;

  constructor(params: ClaudeRunnerParams) {
    this.name = params.name;
    this.client = params.client;
  }

  /**
   * Converts a Genkit role to the corresponding Anthropic role.
   */
  protected toAnthropicRole(
    role: Role,
    toolMessageType?: 'tool_use' | 'tool_result'
  ): 'user' | 'assistant' {
    if (role === 'user') {
      return 'user';
    }
    if (role === 'model') {
      return 'assistant';
    }
    if (role === 'tool') {
      return toolMessageType === 'tool_use' ? 'assistant' : 'user';
    }
    throw new Error(`Unsupported genkit role: ${role}`);
  }

  protected isMediaType(value: string): value is MediaType {
    return MediaTypeSchema.safeParse(value).success;
  }

  protected isMediaObject(obj: unknown): obj is Media {
    return MediaSchema.safeParse(obj).success;
  }

  /**
   * Checks if a URL is a data URL (starts with 'data:').
   */
  protected isDataUrl(url: string): boolean {
    return url.startsWith('data:');
  }

  protected extractDataFromBase64Url(
    url: string
  ): { data: string; contentType: string } | null {
    const match = url.match(/^data:([^;]+);base64,(.+)$/);
    return (
      match && {
        contentType: match[1],
        data: match[2],
      }
    );
  }

  /**
   * Both the stable and beta Anthropic SDKs accept the same JSON shape for PDF
   * document sources (either `type: 'base64'` with a base64 payload or `type: 'url'`
   * with a public URL). Even though the return type references the stable SDK
   * union, TypeScript’s structural typing lets the beta runner reuse this helper.
   */
  protected toPdfDocumentSource(media: Media): DocumentBlockParam['source'] {
    if (media.contentType !== 'application/pdf') {
      throw new Error(
        `PDF contentType mismatch: expected application/pdf, got ${media.contentType}`
      );
    }
    const url = media.url;
    if (this.isDataUrl(url)) {
      const extracted = this.extractDataFromBase64Url(url);
      if (!extracted) {
        throw new Error(
          `Invalid PDF data URL format: ${url.substring(0, 50)}...`
        );
      }
      const { data, contentType } = extracted;
      if (contentType !== 'application/pdf') {
        throw new Error(
          `PDF contentType mismatch: expected application/pdf, got ${contentType}`
        );
      }
      return {
        type: 'base64',
        media_type: 'application/pdf',
        data,
      };
    }
    return {
      type: 'url',
      url,
    };
  }

  /**
   * Normalizes Genkit `Media` into either a base64 payload or a remote URL
   * accepted by the Anthropic SDK. Anthropic supports both `data:` URLs (which
   * we forward as base64) and remote `https` URLs without additional handling.
   */
  protected toImageSource(
    media: Media
  ):
    | { kind: 'base64'; data: string; mediaType: MediaType }
    | { kind: 'url'; url: string } {
    if (this.isDataUrl(media.url)) {
      const extracted = this.extractDataFromBase64Url(media.url);
      const { data, contentType } = extracted ?? {};
      if (!data || !contentType) {
        throw new Error(
          `Invalid genkit part media provided to toAnthropicMessageContent: ${JSON.stringify(
            media
          )}.`
        );
      }

      const resolvedMediaType = contentType;
      if (!resolvedMediaType) {
        throw new Error('Media type is required but was not provided');
      }
      if (!this.isMediaType(resolvedMediaType)) {
        // Provide helpful error message for text files
        if (resolvedMediaType === 'text/plain') {
          throw new Error(
            `Unsupported media type: ${resolvedMediaType}. Text files should be sent as text content in the message, not as media. For example, use { text: '...' } instead of { media: { url: '...', contentType: 'text/plain' } }`
          );
        }
        throw new Error(`Unsupported media type: ${resolvedMediaType}`);
      }
      return {
        kind: 'base64',
        data,
        mediaType: resolvedMediaType,
      };
    }

    if (!media.url) {
      throw new Error('Media url is required but was not provided');
    }

    // For non-data URLs, use the provided contentType or default to a generic type
    // Note: Anthropic will validate the actual content when fetching from URL
    if (media.contentType) {
      if (!this.isMediaType(media.contentType)) {
        // Provide helpful error message for text files
        if (media.contentType === 'text/plain') {
          throw new Error(
            `Unsupported media type: ${media.contentType}. Text files should be sent as text content in the message, not as media. For example, use { text: '...' } instead of { media: { url: '...', contentType: 'text/plain' } }`
          );
        }
        throw new Error(`Unsupported media type: ${media.contentType}`);
      }
    }

    return {
      kind: 'url',
      url: media.url,
    };
  }

  /**
   * Converts tool response output to the appropriate Anthropic content format.
   * Handles Media objects, data URLs, strings, and other outputs.
   */
  protected toAnthropicToolResponseContent(
    part: Part
  ): RunnerToolResponseContent<ApiTypes> {
    const output = part.toolResponse?.output ?? {};

    // Handle Media objects (images returned by tools)
    if (this.isMediaObject(output)) {
      const { data, contentType } =
        this.extractDataFromBase64Url(output.url) ?? {};
      if (data && contentType) {
        if (!this.isMediaType(contentType)) {
          // Provide helpful error message for text files
          if (contentType === 'text/plain') {
            throw new Error(
              `Unsupported media type: ${contentType}. Text files should be sent as text content, not as media.`
            );
          }
          throw new Error(`Unsupported media type: ${contentType}`);
        }
        return {
          type: 'image',
          source: {
            type: 'base64',
            data,
            media_type: contentType,
          },
        };
      }
    }

    // Handle string outputs - check if it's a data URL
    if (typeof output === 'string') {
      // Check if string is a data URL (e.g., "data:image/gif;base64,...")
      if (this.isDataUrl(output)) {
        const { data, contentType } =
          this.extractDataFromBase64Url(output) ?? {};
        if (data && contentType) {
          if (!this.isMediaType(contentType)) {
            // Provide helpful error message for text files
            if (contentType === 'text/plain') {
              throw new Error(
                `Unsupported media type: ${contentType}. Text files should be sent as text content, not as media.`
              );
            }
            throw new Error(`Unsupported media type: ${contentType}`);
          }
          return {
            type: 'image',
            source: {
              type: 'base64',
              data,
              media_type: contentType,
            },
          };
        }
      }
      // Regular string output
      return {
        type: 'text',
        text: output,
      };
    }

    // Handle other outputs by stringifying
    return {
      type: 'text',
      text: JSON.stringify(output),
    };
  }

  protected getThinkingSignature(part: Part): string | undefined {
    const metadata = part.metadata as Record<string, unknown> | undefined;
    return typeof metadata?.thoughtSignature === 'string'
      ? metadata.thoughtSignature
      : undefined;
  }

  protected getRedactedThinkingData(part: Part): string | undefined {
    const custom = part.custom as Record<string, unknown> | undefined;
    const redacted = custom?.redactedThinking;
    return typeof redacted === 'string' ? redacted : undefined;
  }

  protected toAnthropicThinkingConfig(
    config: ThinkingConfig | undefined
  ):
    | { type: 'enabled'; budget_tokens: number }
    | { type: 'disabled' }
    | undefined {
    if (!config) return undefined;

    const { enabled, budgetTokens } = config;

    if (enabled === true) {
      if (budgetTokens === undefined) {
        return undefined;
      }
      return { type: 'enabled', budget_tokens: budgetTokens };
    }

    if (enabled === false) {
      return { type: 'disabled' };
    }

    if (budgetTokens !== undefined) {
      return { type: 'enabled', budget_tokens: budgetTokens };
    }

    return undefined;
  }

  /**
   * Converts a Genkit Part to the corresponding Anthropic content block.
   * Each runner implements this to return its specific API type.
   */
  protected abstract toAnthropicMessageContent(
    part: Part
  ): RunnerContentBlockParam<ApiTypes>;

  /**
   * Converts Genkit messages to Anthropic format.
   * Extracts system message and converts remaining messages using the runner's
   * toAnthropicMessageContent implementation.
   */
  protected toAnthropicMessages(messages: MessageData[]): {
    system?: RunnerContentBlockParam<ApiTypes>[];
    messages: RunnerMessageParam<ApiTypes>[];
  } {
    let system: RunnerContentBlockParam<ApiTypes>[] | undefined;

    if (messages[0]?.role === 'system') {
      const systemMessage = messages[0];
      messages = messages.slice(1);

      for (const part of systemMessage.content ?? []) {
        if (part.media || part.toolRequest || part.toolResponse) {
          throw new Error(
            'System messages can only contain text content. Media, tool requests, and tool responses are not supported in system messages.'
          );
        }
      }

      system = systemMessage.content.map((part) =>
        this.toAnthropicMessageContent(part)
      );
    }

    const anthropicMsgs: RunnerMessageParam<ApiTypes>[] = [];

    for (const message of messages) {
      const msg = new GenkitMessage(message);

      // Detect tool message kind from Genkit Parts (no SDK typing needed)
      const hadToolUse = msg.content.some((p) => !!p.toolRequest);
      const hadToolResult = msg.content.some((p) => !!p.toolResponse);

      const toolMessageType = hadToolUse
        ? ('tool_use' as const)
        : hadToolResult
          ? ('tool_result' as const)
          : undefined;

      const role = this.toAnthropicRole(message.role, toolMessageType);

      const content = msg.content.map((part) =>
        this.toAnthropicMessageContent(part)
      );

      anthropicMsgs.push({ role, content });
    }

    return { system, messages: anthropicMsgs };
  }

  /**
   * Converts a Genkit ToolDefinition to an Anthropic Tool object.
   *
   * Anthropic requires `input_schema.type` to be present (usually `"object"`).
   * Genkit's `ToolDefinition` may have an empty schema (e.g. from `z.void()`)
   * which lacks the `type` field. We default to `{ type: "object" }` to
   * prevent 400 errors from the Anthropic API.
   */
  protected toAnthropicTool(tool: ToolDefinition): RunnerTool<ApiTypes> {
    const schema = tool.inputSchema || {};
    const inputSchema =
      'type' in schema ? schema : { ...schema, type: 'object' as const };
    return {
      name: tool.name,
      description: tool.description,
      input_schema: inputSchema,
    } as RunnerTool<ApiTypes>;
  }

  /**
   * Converts an Anthropic request to a non-streaming Anthropic API request body.
   * @param modelName The name of the Anthropic model to use.
   * @param request The Genkit GenerateRequest to convert.
   * @returns The converted Anthropic API non-streaming request body.
   * @throws An error if an unsupported output format is requested.
   */
  protected abstract toAnthropicRequestBody(
    modelName: string,
    request: GenerateRequest<typeof AnthropicConfigSchema>
  ): RunnerRequestBody<ApiTypes>;

  /**
   * Converts an Anthropic request to a streaming Anthropic API request body.
   * @param modelName The name of the Anthropic model to use.
   * @param request The Genkit GenerateRequest to convert.
   * @returns The converted Anthropic API streaming request body.
   * @throws An error if an unsupported output format is requested.
   */
  protected abstract toAnthropicStreamingRequestBody(
    modelName: string,
    request: GenerateRequest<typeof AnthropicConfigSchema>
  ): RunnerStreamingRequestBody<ApiTypes>;

  protected abstract createMessage(
    body: RunnerRequestBody<ApiTypes>,
    abortSignal: AbortSignal
  ): Promise<RunnerMessage<ApiTypes>>;

  protected abstract streamMessages(
    body: RunnerStreamingRequestBody<ApiTypes>,
    abortSignal: AbortSignal
  ): RunnerStream<ApiTypes>;

  protected abstract toGenkitResponse(
    message: RunnerMessage<ApiTypes>
  ): GenerateResponseData;

  protected abstract toGenkitPart(
    event: RunnerStreamEvent<ApiTypes>
  ): Part | undefined;

  public async run(
    request: GenerateRequest<typeof AnthropicConfigSchema>,
    options: {
      streamingRequested: boolean;
      sendChunk: (chunk: GenerateResponseChunkData) => void;
      abortSignal: AbortSignal;
    }
  ): Promise<GenerateResponseData> {
    const { streamingRequested, sendChunk, abortSignal } = options;

    if (streamingRequested) {
      const body = this.toAnthropicStreamingRequestBody(this.name, request);
      const stream = this.streamMessages(body, abortSignal);
      for await (const event of stream) {
        const part = this.toGenkitPart(event);
        if (part) {
          sendChunk({
            index: 0,
            content: [part],
          });
        }
      }
      const finalMessage = await stream.finalMessage();
      return this.toGenkitResponse(finalMessage);
    }

    const body = this.toAnthropicRequestBody(this.name, request);
    const response = await this.createMessage(body, abortSignal);
    return this.toGenkitResponse(response);
  }
}
