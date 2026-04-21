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

import { MessageStream } from '@anthropic-ai/sdk/lib/MessageStream.js';
import type {
  ContentBlock,
  DocumentBlockParam,
  ImageBlockParam,
  Message,
  MessageCreateParams,
  MessageCreateParamsNonStreaming,
  MessageCreateParamsStreaming,
  MessageParam,
  MessageStreamEvent,
  RedactedThinkingBlockParam,
  TextBlockParam,
  ThinkingBlockParam,
  Tool,
  ToolResultBlockParam,
  ToolUseBlockParam,
} from '@anthropic-ai/sdk/resources/messages';
import type {
  GenerateRequest,
  GenerateResponseData,
  ModelResponseData,
  Part,
} from 'genkit';
import { logger } from 'genkit/logging';

import { KNOWN_CLAUDE_MODELS, extractVersion } from '../models.js';
import {
  AnthropicConfigSchema,
  type AnthropicDocumentOptions,
  type ClaudeRunnerParams,
} from '../types.js';
import { removeUndefinedProperties } from '../utils.js';
import { BaseRunner } from './base.js';
import {
  citationsDeltaToPart,
  inputJsonDeltaError,
  redactedThinkingBlockToPart,
  textBlockToPart,
  textDeltaToPart,
  thinkingBlockToPart,
  thinkingDeltaToPart,
  toolUseBlockToPart,
  webSearchToolResultBlockToPart,
} from './converters/shared.js';
import {
  serverToolUseBlockToPart,
  toDocumentBlock,
} from './converters/stable.js';
import { RunnerTypes as BaseRunnerTypes } from './types.js';

interface RunnerTypes extends BaseRunnerTypes {
  Message: Message;
  Stream: MessageStream;
  StreamEvent: MessageStreamEvent;
  RequestBody: MessageCreateParamsNonStreaming;
  StreamingRequestBody: MessageCreateParamsStreaming;
  Tool: Tool;
  MessageParam: MessageParam;
  ToolResponseContent: TextBlockParam | ImageBlockParam;
  ContentBlockParam:
    | TextBlockParam
    | ImageBlockParam
    | DocumentBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | ThinkingBlockParam
    | RedactedThinkingBlockParam;
}

export class Runner extends BaseRunner<RunnerTypes> {
  constructor(params: ClaudeRunnerParams) {
    super(params);
  }

  protected toAnthropicMessageContent(
    part: Part
  ):
    | TextBlockParam
    | ImageBlockParam
    | DocumentBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | ThinkingBlockParam
    | RedactedThinkingBlockParam {
    if (part.reasoning) {
      const signature = this.getThinkingSignature(part);
      if (!signature) {
        throw new Error(
          'Anthropic thinking parts require a signature when sending back to the API. Preserve the `metadata.thoughtSignature` value from the original response.'
        );
      }
      return {
        type: 'thinking',
        thinking: part.reasoning,
        signature,
      };
    }

    const redactedThinking = this.getRedactedThinkingData(part);
    if (redactedThinking !== undefined) {
      return {
        type: 'redacted_thinking',
        data: redactedThinking,
      };
    }

    if (part.text) {
      return {
        type: 'text',
        text: part.text,
        citations: null,
        // This is intentional. `part.metadata?.cache_control` is unknown, and casting it to the relevant type of the property makes it more robust to Anthropic SDK API changes.
        cache_control: part.metadata
          ?.cache_control as TextBlockParam['cache_control'],
      };
    }

    // Custom document (for citations support)
    if (part.custom?.anthropicDocument) {
      return toDocumentBlock(
        part.custom.anthropicDocument as AnthropicDocumentOptions
      );
    }

    if (part.media) {
      if (part.media.contentType === 'application/pdf') {
        return {
          type: 'document',
          source: this.toPdfDocumentSource(part.media),
          cache_control: part.metadata
            ?.cache_control as DocumentBlockParam['cache_control'],
        };
      }

      const source = this.toImageSource(part.media);
      if (source.kind === 'base64') {
        return {
          type: 'image',
          source: {
            type: 'base64',
            data: source.data,
            media_type: source.mediaType,
          },
          cache_control: part.metadata
            ?.cache_control as ImageBlockParam['cache_control'],
        };
      }
      return {
        type: 'image',
        source: {
          type: 'url',
          url: source.url,
        },
        cache_control: part.metadata
          ?.cache_control as ImageBlockParam['cache_control'],
      };
    }

    if (part.toolRequest) {
      if (!part.toolRequest.ref) {
        throw new Error(
          `Tool request ref is required for Anthropic API. Part: ${JSON.stringify(
            part.toolRequest
          )}`
        );
      }
      return {
        type: 'tool_use',
        id: part.toolRequest.ref,
        name: part.toolRequest.name,
        input: part.toolRequest.input,
        cache_control: part.metadata
          ?.cache_control as ToolUseBlockParam['cache_control'],
      };
    }

    if (part.toolResponse) {
      if (!part.toolResponse.ref) {
        throw new Error(
          `Tool response ref is required for Anthropic API. Part: ${JSON.stringify(
            part.toolResponse
          )}`
        );
      }
      return {
        type: 'tool_result',
        tool_use_id: part.toolResponse.ref,
        content: [this.toAnthropicToolResponseContent(part)],
        cache_control: part.metadata
          ?.cache_control as ToolResultBlockParam['cache_control'],
      };
    }

    throw new Error(
      `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(
        part
      )}.`
    );
  }

  protected toAnthropicRequestBody(
    modelName: string,
    request: GenerateRequest<typeof AnthropicConfigSchema>
  ): MessageCreateParamsNonStreaming {
    if (request.output?.format && request.output.format !== 'text') {
      throw new Error(
        `Only text output format is supported for Claude models currently`
      );
    }

    const model = KNOWN_CLAUDE_MODELS[modelName];
    const { system, messages } = this.toAnthropicMessages(request.messages);
    const mappedModelName =
      request.config?.version ?? extractVersion(model, modelName);

    const thinkingConfig = this.toAnthropicThinkingConfig(
      request.config?.thinking
    ) as MessageCreateParams['thinking'] | undefined;

    // Need to extract topP and topK from request.config to avoid duplicate properties being added to the body
    // This happens because topP and topK have different property names (top_p and top_k) in the Anthropic API.
    // Thinking is extracted separately to avoid type issues.
    // ApiVersion is extracted separately as it's not a valid property for the Anthropic API.
    const {
      topP,
      topK,
      apiVersion: _1,
      thinking: _2,
      maxOutputTokens,
      stopSequences,
      version,
      apiKey,
      ...restConfig
    } = request.config ?? {};

    const body: MessageCreateParamsNonStreaming = {
      model: mappedModelName,
      max_tokens: maxOutputTokens ?? this.DEFAULT_MAX_OUTPUT_TOKENS,
      messages,
      system: system as TextBlockParam[],
      stop_sequences: stopSequences,
      temperature: request.config?.temperature,
      top_k: topK,
      top_p: topP,
      tool_choice: request.config?.tool_choice,
      metadata: request.config?.metadata,
      tools: request.tools?.map((tool) => this.toAnthropicTool(tool)),
      thinking: thinkingConfig,
      ...restConfig,
    };

    return removeUndefinedProperties(body);
  }

  protected toAnthropicStreamingRequestBody(
    modelName: string,
    request: GenerateRequest<typeof AnthropicConfigSchema>
  ): MessageCreateParamsStreaming {
    if (request.output?.format && request.output.format !== 'text') {
      throw new Error(
        `Only text output format is supported for Claude models currently`
      );
    }

    const model = KNOWN_CLAUDE_MODELS[modelName];
    const { system, messages } = this.toAnthropicMessages(request.messages);
    const mappedModelName =
      request.config?.version ?? extractVersion(model, modelName);

    const thinkingConfig = this.toAnthropicThinkingConfig(
      request.config?.thinking
    ) as MessageCreateParams['thinking'] | undefined;

    // Need to extract topP and topK from request.config to avoid duplicate properties being added to the body
    // This happens because topP and topK have different property names (top_p and top_k) in the Anthropic API.
    // Thinking is extracted separately to avoid type issues.
    // ApiVersion is extracted separately as it's not a valid property for the Anthropic API.
    const {
      topP,
      topK,
      apiVersion: _1,
      thinking: _2,
      maxOutputTokens,
      stopSequences,
      version,
      apiKey,
      ...restConfig
    } = request.config ?? {};

    const body: MessageCreateParamsStreaming = {
      model: mappedModelName,
      max_tokens:
        request.config?.maxOutputTokens ?? this.DEFAULT_MAX_OUTPUT_TOKENS,
      messages,
      stream: true,
      system: system as TextBlockParam[],
      stop_sequences: request.config?.stopSequences,
      temperature: request.config?.temperature,
      top_k: topK,
      top_p: topP,
      tool_choice: request.config?.tool_choice,
      metadata: request.config?.metadata,
      tools: request.tools?.map((tool) => this.toAnthropicTool(tool)),
      thinking: thinkingConfig,
      ...restConfig,
    };

    return removeUndefinedProperties(body);
  }

  protected async createMessage(
    body: MessageCreateParamsNonStreaming,
    abortSignal: AbortSignal
  ): Promise<Message> {
    return await this.client.messages.create(body, { signal: abortSignal });
  }

  protected streamMessages(
    body: MessageCreateParamsStreaming,
    abortSignal: AbortSignal
  ): MessageStream {
    return this.client.messages.stream(body, { signal: abortSignal });
  }

  protected toGenkitResponse(message: Message): GenerateResponseData {
    return this.fromAnthropicResponse(message);
  }

  protected toGenkitPart(event: MessageStreamEvent): Part | undefined {
    return this.fromAnthropicContentBlockChunk(event);
  }

  protected fromAnthropicContentBlockChunk(
    event: MessageStreamEvent
  ): Part | undefined {
    // Handle content_block_delta events
    if (event.type === 'content_block_delta') {
      const delta = event.delta;

      if (delta.type === 'text_delta') {
        return textDeltaToPart(delta);
      }

      if (delta.type === 'thinking_delta') {
        return thinkingDeltaToPart(delta);
      }

      if (delta.type === 'citations_delta') {
        return citationsDeltaToPart(delta);
      }

      if (delta.type === 'input_json_delta') {
        throw inputJsonDeltaError();
      }

      // signature_delta - ignore
      return undefined;
    }

    // Handle content_block_start events
    if (event.type === 'content_block_start') {
      const block = event.content_block;

      switch (block.type) {
        case 'text':
          return textBlockToPart(block);

        case 'tool_use':
          return toolUseBlockToPart(block);

        case 'thinking':
          return thinkingBlockToPart(block);

        case 'redacted_thinking':
          return redactedThinkingBlockToPart(block);

        case 'server_tool_use':
          return serverToolUseBlockToPart(block);

        case 'web_search_tool_result':
          return webSearchToolResultBlockToPart(block);

        default: {
          const unknownType = (block as { type: string }).type;
          logger.warn(
            `Unexpected Anthropic content block type in stream: ${unknownType}. Returning undefined. Content block: ${JSON.stringify(block)}`
          );
          return undefined;
        }
      }
    }

    // Other event types (message_start, message_delta, etc.) - ignore
    return undefined;
  }

  protected fromAnthropicContentBlock(contentBlock: ContentBlock): Part {
    switch (contentBlock.type) {
      case 'text':
        return textBlockToPart(contentBlock);

      case 'tool_use':
        return toolUseBlockToPart(contentBlock);

      case 'thinking':
        return thinkingBlockToPart(contentBlock);

      case 'redacted_thinking':
        return redactedThinkingBlockToPart(contentBlock);

      case 'server_tool_use':
        return serverToolUseBlockToPart(contentBlock);

      case 'web_search_tool_result':
        return webSearchToolResultBlockToPart(contentBlock);

      default: {
        // Exhaustive check (uncomment when all types are handled):
        // const _exhaustive: never = contentBlock;
        // throw new Error(
        //   `Unhandled block type: ${(_exhaustive as { type: string }).type}`
        // );
        const unknownType = (contentBlock as { type: string }).type;
        logger.warn(
          `Unexpected Anthropic content block type: ${unknownType}. Returning empty text. Content block: ${JSON.stringify(contentBlock)}`
        );
        return { text: '' };
      }
    }
  }

  protected fromAnthropicStopReason(
    reason: Message['stop_reason']
  ): ModelResponseData['finishReason'] {
    switch (reason) {
      case 'max_tokens':
        return 'length';
      case 'end_turn':
      // fall through
      case 'stop_sequence':
      // fall through
      case 'tool_use':
        return 'stop';
      case null:
        return 'unknown';
      default:
        return 'other';
    }
  }

  protected fromAnthropicResponse(response: Message): GenerateResponseData {
    return {
      candidates: [
        {
          index: 0,
          finishReason: this.fromAnthropicStopReason(response.stop_reason),
          message: {
            role: 'model',
            content: response.content.map((block) =>
              this.fromAnthropicContentBlock(block)
            ),
          },
        },
      ],
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        custom: {
          cache_creation_input_tokens:
            response.usage.cache_creation_input_tokens ?? 0,
          cache_read_input_tokens: response.usage.cache_read_input_tokens ?? 0,
          ephemeral_5m_input_tokens:
            response.usage.cache_creation?.ephemeral_5m_input_tokens ?? 0,
          ephemeral_1h_input_tokens:
            response.usage.cache_creation?.ephemeral_1h_input_tokens ?? 0,
        },
      },
      custom: response,
    };
  }
}
