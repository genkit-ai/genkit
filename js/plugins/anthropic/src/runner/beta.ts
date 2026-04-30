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

import { BetaMessageStream } from '@anthropic-ai/sdk/lib/BetaMessageStream.js';
import type {
  BetaContentBlock,
  BetaImageBlockParam,
  BetaMessage,
  MessageCreateParams as BetaMessageCreateParams,
  MessageCreateParamsNonStreaming as BetaMessageCreateParamsNonStreaming,
  MessageCreateParamsStreaming as BetaMessageCreateParamsStreaming,
  BetaMessageParam,
  BetaRawMessageStreamEvent,
  BetaRedactedThinkingBlockParam,
  BetaRequestDocumentBlock,
  BetaStopReason,
  BetaTextBlockParam,
  BetaThinkingBlockParam,
  BetaTool,
  BetaToolResultBlockParam,
  BetaToolUseBlockParam,
} from '@anthropic-ai/sdk/resources/beta/messages';

import type {
  GenerateRequest,
  GenerateResponseData,
  ModelResponseData,
  Part,
} from 'genkit';
import { logger } from 'genkit/logging';

import {
  AnthropicConfigSchema,
  type AnthropicDocumentOptions,
  type ClaudeRunnerParams,
} from '../types.js';
import { checkModelName, removeUndefinedProperties } from '../utils.js';
import { BaseRunner } from './base.js';
import {
  betaServerToolUseBlockToPart,
  toBetaDocumentBlock,
  unsupportedServerToolError,
} from './converters/beta.js';
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
import { RunnerTypes } from './types.js';

const BETA_APIS = [
  // 'message-batches-2024-09-24',
  // 'prompt-caching-2024-07-31',
  // 'computer-use-2025-01-24',
  // 'pdfs-2024-09-25',
  // 'token-counting-2024-11-01',
  // 'token-efficient-tools-2025-02-19',
  // 'output-128k-2025-02-19',
  'files-api-2025-04-14',
  // 'mcp-client-2025-04-04',
  // 'dev-full-thinking-2025-05-14',
  // 'interleaved-thinking-2025-05-14',
  // 'code-execution-2025-05-22',
  // 'extended-cache-ttl-2025-04-11',
  // 'context-1m-2025-08-07',
  // 'context-management-2025-06-27',
  // 'model-context-window-exceeded-2025-08-26',
  // 'skills-2025-10-02',
  'effort-2025-11-24',
  // 'advanced-tool-use-2025-11-20',
  'structured-outputs-2025-11-13',
  'task-budgets-2026-03-13',
];

/**
 * Transforms a JSON schema to be compatible with Anthropic's structured output requirements.
 * Anthropic requires `additionalProperties: false` on all object types.
 * @see https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations
 */
function toAnthropicSchema(
  schema: Record<string, unknown>
): Record<string, unknown> {
  const out = structuredClone(schema);

  // Remove $schema if present
  delete out.$schema;

  // Add additionalProperties: false to objects
  if (out.type === 'object') {
    out.additionalProperties = false;
  }

  // Recursively process nested objects
  for (const key in out) {
    if (typeof out[key] === 'object' && out[key] !== null) {
      out[key] = toAnthropicSchema(out[key] as Record<string, unknown>);
    }
  }

  return out;
}

interface BetaRunnerTypes extends RunnerTypes {
  Message: BetaMessage;
  Stream: BetaMessageStream;
  StreamEvent: BetaRawMessageStreamEvent;
  RequestBody: BetaMessageCreateParamsNonStreaming;
  StreamingRequestBody: BetaMessageCreateParamsStreaming;
  Tool: BetaTool;
  MessageParam: BetaMessageParam;
  ToolResponseContent: BetaTextBlockParam | BetaImageBlockParam;
  ContentBlockParam:
    | BetaTextBlockParam
    | BetaImageBlockParam
    | BetaRequestDocumentBlock
    | BetaToolUseBlockParam
    | BetaToolResultBlockParam
    | BetaThinkingBlockParam
    | BetaRedactedThinkingBlockParam;
}

/**
 * Runner for the Anthropic Beta API.
 */
export class BetaRunner extends BaseRunner<BetaRunnerTypes> {
  constructor(params: ClaudeRunnerParams) {
    super(params);
  }

  /**
   * Map a Genkit Part -> Anthropic beta content block param.
   * Supports: text, images (base64 data URLs), PDFs (document source),
   * tool_use (client tool request), tool_result (client tool response).
   */
  protected toAnthropicMessageContent(
    part: Part
  ):
    | BetaTextBlockParam
    | BetaImageBlockParam
    | BetaRequestDocumentBlock
    | BetaToolUseBlockParam
    | BetaToolResultBlockParam
    | BetaThinkingBlockParam
    | BetaRedactedThinkingBlockParam {
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

    // Text
    if (part.text) {
      return { type: 'text', text: part.text };
    }

    // Custom document (for citations support)
    if (part.custom?.anthropicDocument) {
      return toBetaDocumentBlock(
        part.custom.anthropicDocument as AnthropicDocumentOptions
      );
    }

    // Media
    if (part.media) {
      if (part.media.contentType === 'anthropic/file') {
        return {
          type: 'document',
          source: {
            type: 'file',
            file_id: part.media.url,
          },
        };
      }

      if (part.media.contentType === 'anthropic/image') {
        return {
          type: 'image',
          source: {
            type: 'file',
            file_id: part.media.url,
          },
        };
      }

      if (part.media.contentType === 'application/pdf') {
        return {
          type: 'document',
          source: this.toPdfDocumentSource(part.media),
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
        };
      }
      return {
        type: 'image',
        source: {
          type: 'url',
          url: source.url,
        },
      };
    }

    // Tool request (client tool use)
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
      };
    }

    // Tool response (client tool result)
    if (part.toolResponse) {
      if (!part.toolResponse.ref) {
        throw new Error(
          `Tool response ref is required for Anthropic API. Part: ${JSON.stringify(
            part.toolResponse
          )}`
        );
      }
      const betaResult: BetaToolResultBlockParam = {
        type: 'tool_result',
        tool_use_id: part.toolResponse.ref,
        content: [this.toAnthropicToolResponseContent(part)],
      };
      return betaResult;
    }

    throw new Error(
      `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(
        part
      )}.`
    );
  }

  protected createMessage(
    body: BetaMessageCreateParamsNonStreaming,
    abortSignal: AbortSignal
  ): Promise<BetaMessage> {
    return this.client.beta.messages.create(body, { signal: abortSignal });
  }

  protected streamMessages(
    body: BetaMessageCreateParamsStreaming,
    abortSignal: AbortSignal
  ): BetaMessageStream {
    return this.client.beta.messages.stream(body, { signal: abortSignal });
  }

  /**
   * Build non-streaming request body.
   */
  protected toAnthropicRequestBody(
    modelName: string,
    request: GenerateRequest<typeof AnthropicConfigSchema>
  ): BetaMessageCreateParamsNonStreaming {
    const { system, messages } = this.toAnthropicMessages(request.messages);
    const mappedModelName =
      request.config?.version ?? checkModelName(modelName);

    const thinkingConfig = this.toAnthropicThinkingConfig(
      request.config?.thinking
    ) as BetaMessageCreateParams['thinking'] | undefined;

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

    const body = {
      model: mappedModelName,
      max_tokens: maxOutputTokens ?? this.DEFAULT_MAX_OUTPUT_TOKENS,
      messages,
      system: system as BetaTextBlockParam[],
      stop_sequences: stopSequences,
      temperature: request.config?.temperature,
      top_k: topK,
      top_p: topP,
      tool_choice: request.config?.tool_choice,
      metadata: request.config?.metadata,
      tools: request.tools?.map((tool) => this.toAnthropicTool(tool)),
      thinking: thinkingConfig,
      output_format: this.isStructuredOutputEnabled(request)
        ? {
            type: 'json_schema',
            schema: toAnthropicSchema(request.output!.schema!),
          }
        : undefined,
      betas: Array.isArray(request.config?.betas)
        ? [...(request.config?.betas ?? [])]
        : [...BETA_APIS],
      ...restConfig,
    } as BetaMessageCreateParamsNonStreaming;

    return removeUndefinedProperties(body);
  }

  /**
   * Build streaming request body.
   */
  protected toAnthropicStreamingRequestBody(
    modelName: string,
    request: GenerateRequest<typeof AnthropicConfigSchema>
  ): BetaMessageCreateParamsStreaming {
    const { system, messages } = this.toAnthropicMessages(request.messages);
    const mappedModelName =
      request.config?.version ?? checkModelName(modelName);

    const thinkingConfig = this.toAnthropicThinkingConfig(
      request.config?.thinking
    ) as BetaMessageCreateParams['thinking'] | undefined;

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

    const body = {
      model: mappedModelName,
      max_tokens: maxOutputTokens ?? this.DEFAULT_MAX_OUTPUT_TOKENS,
      messages,
      stream: true,
      system: system as BetaTextBlockParam[],
      stop_sequences: stopSequences,
      temperature: request.config?.temperature,
      top_k: topK,
      top_p: topP,
      tool_choice: request.config?.tool_choice,
      metadata: request.config?.metadata,
      tools: request.tools?.map((tool) => this.toAnthropicTool(tool)),
      thinking: thinkingConfig,
      output_format: this.isStructuredOutputEnabled(request)
        ? {
            type: 'json_schema',
            schema: toAnthropicSchema(request.output!.schema!),
          }
        : undefined,
      betas: Array.isArray(request.config?.betas)
        ? [...(request.config?.betas ?? [])]
        : [...BETA_APIS],
      ...restConfig,
    } as BetaMessageCreateParamsStreaming;

    return removeUndefinedProperties(body);
  }

  protected toGenkitResponse(message: BetaMessage): GenerateResponseData {
    return {
      candidates: [
        {
          index: 0,
          finishReason: this.fromBetaStopReason(message.stop_reason),
          message: {
            role: 'model',
            content: message.content.map((block) =>
              this.fromBetaContentBlock(block)
            ),
          },
        },
      ],
      usage: {
        inputTokens: message.usage.input_tokens,
        outputTokens: message.usage.output_tokens,
      },
      custom: message,
      raw: message,
    };
  }

  protected toGenkitPart(event: BetaRawMessageStreamEvent): Part | undefined {
    if (event.type === 'content_block_start') {
      return this.fromBetaContentBlock(event.content_block);
    }
    if (event.type === 'content_block_delta') {
      if (event.delta.type === 'text_delta') {
        return textDeltaToPart(event.delta);
      }
      if (event.delta.type === 'thinking_delta') {
        return thinkingDeltaToPart(event.delta);
      }
      if (event.delta.type === 'citations_delta') {
        return citationsDeltaToPart(event.delta);
      }
      if (event.delta.type === 'input_json_delta') {
        throw inputJsonDeltaError();
      }
      // signature_delta - ignore
      return undefined;
    }
    return undefined;
  }

  private fromBetaContentBlock(contentBlock: BetaContentBlock): Part {
    switch (contentBlock.type) {
      case 'text':
        return textBlockToPart(contentBlock);

      case 'tool_use':
        // Beta API may have undefined name, fallback to 'unknown_tool'
        return toolUseBlockToPart({
          id: contentBlock.id,
          name: contentBlock.name ?? 'unknown_tool',
          input: contentBlock.input,
        });

      case 'thinking':
        return thinkingBlockToPart(contentBlock);

      case 'redacted_thinking':
        return redactedThinkingBlockToPart(contentBlock);

      case 'server_tool_use':
        return betaServerToolUseBlockToPart(contentBlock);

      case 'web_search_tool_result':
        return webSearchToolResultBlockToPart(contentBlock);

      // Unsupported beta server tool types
      case 'mcp_tool_use':
      case 'mcp_tool_result':
      case 'web_fetch_tool_result':
      case 'code_execution_tool_result':
      case 'bash_code_execution_tool_result':
      case 'text_editor_code_execution_tool_result':
      case 'container_upload':
      case 'tool_search_tool_result':
        throw new Error(unsupportedServerToolError(contentBlock.type));

      default: {
        const unknownType = (contentBlock as { type: string }).type;
        logger.warn(
          `Unexpected Anthropic beta content block type: ${unknownType}. Returning empty text. Content block: ${JSON.stringify(contentBlock)}`
        );
        return { text: '' };
      }
    }
  }

  private fromBetaStopReason(
    reason: BetaStopReason | null
  ): ModelResponseData['finishReason'] {
    switch (reason) {
      case 'max_tokens':
      case 'model_context_window_exceeded':
        return 'length';
      case 'end_turn':
      case 'stop_sequence':
      case 'tool_use':
      case 'pause_turn':
        return 'stop';
      case null:
        return 'unknown';
      case 'refusal':
        return 'other';
      default:
        return 'other';
    }
  }

  private isStructuredOutputEnabled(
    request: GenerateRequest<typeof AnthropicConfigSchema>
  ): boolean {
    return !!(
      request.output?.schema &&
      request.output.constrained &&
      request.output.format === 'json'
    );
  }
}
