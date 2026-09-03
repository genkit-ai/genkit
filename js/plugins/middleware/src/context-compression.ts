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

import {
  generateMiddleware,
  z,
  type GenerateMiddleware,
  type MessageData,
  type Part,
} from 'genkit';
import { AsyncLocalStorage } from 'node:async_hooks';

interface CompressionExecutionState {
  lastInputTokens?: number;
  latestCompressionMeta?: Record<string, unknown> | null;
}

const compressionStorage = new AsyncLocalStorage<CompressionExecutionState>();

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

export const ToolResponsesOptionsSchema = z.object({
  /**
   * Maximum character length for each tool response content.
   * Responses exceeding this will be truncated with a `…[truncated]` marker.
   */
  maxChars: z
    .number()
    .describe(
      'Max chars per tool response. Responses beyond this are truncated.'
    ),

  /**
   * Number of most recent tool responses to leave untouched.
   * @default 2
   */
  preserveRecent: z
    .number()
    .optional()
    .describe("Don't truncate the last N tool responses. Default: 2."),
});

export const ContextCompressionOptionsSchema = z
  .object({
    /**
     * Compression triggers when the previous turn's `inputTokens` exceeds
     * this threshold. On turn 0, token count is estimated from messages.
     */
    maxInputTokens: z
      .number()
      .describe('Compress when token count exceeds this threshold.'),

    /**
     * Number of most recent messages to never compress or drop.
     * @default 4
     */
    preserveRecent: z
      .number()
      .optional()
      .describe('Number of recent messages to always keep intact. Default: 4.'),

    /**
     * Always keep system/instructions messages.
     * @default true
     */
    preserveSystem: z
      .boolean()
      .optional()
      .describe('Always keep system messages. Default: true.'),

    /**
     * Hard cap on individual tool response size in characters.
     * Applied regardless of other toolResponses config as a safety net.
     * Set to `Infinity` to disable.
     * @default 400000
     */
    maxToolResponseChars: z
      .number()
      .optional()
      .describe(
        'Hard cap on any single tool response size. Default: 400000 chars.'
      ),

    /**
     * Truncate tool response content that exceeds a character limit.
     * This is a cheap strategy that requires no LLM call.
     */
    toolResponses: ToolResponsesOptionsSchema.optional().describe(
      'Truncate verbose tool response content.'
    ),

    /**
     * Hard cap on message count. Messages beyond this (oldest first) are
     * dropped, preserving system messages and recent messages.
     */
    maxMessages: z
      .number()
      .optional()
      .describe('Hard cap on message count. Drop oldest beyond this.'),

    /**
     * Insert a notice message when messages are dropped during message
     * truncation, so the model knows context was removed.
     * @default true
     */
    insertTruncationNotice: z
      .boolean()
      .optional()
      .describe('Insert a notice when messages are dropped. Default: true.'),

    /**
     * Custom truncation notice text. Used when messages are dropped.
     */
    truncationNotice: z
      .string()
      .optional()
      .describe('Custom notice text for when messages are dropped.'),
  })
  .passthrough();

export type ContextCompressionOptions = z.infer<
  typeof ContextCompressionOptionsSchema
>;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_MAX_TOOL_RESPONSE_CHARS = 400_000;
const DEFAULT_TOOL_RESPONSE_PRESERVE_RECENT = 2;
const DEFAULT_TRUNCATION_NOTICE =
  '[NOTE] Some earlier messages in this conversation have been removed to stay within ' +
  'context limits. The most recent messages are preserved. Pay close attention to the ' +
  'latest messages and any conversation summary above.';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Stringify tool output, avoiding re-stringifying if already a string.
 */
function stringifyOutput(output: unknown): string {
  return typeof output === 'string' ? output : JSON.stringify(output ?? '');
}

/**
 * Split messages into system and non-system messages.
 */
function partitionMessages(
  messages: MessageData[],
  preserveSystem: boolean
): { systemMessages: MessageData[]; nonSystemMessages: MessageData[] } {
  const systemMessages: MessageData[] = [];
  const nonSystemMessages: MessageData[] = [];

  for (const msg of messages) {
    if (preserveSystem && msg.role === 'system') {
      systemMessages.push(msg);
    } else {
      nonSystemMessages.push(msg);
    }
  }

  return { systemMessages, nonSystemMessages };
}

/**
 * Estimate the total character count across all message content.
 */
function estimateMessageChars(messages: MessageData[]): number {
  return messages.reduce(
    (sum, m) =>
      sum +
      m.content.reduce((pSum, p) => {
        if (p.text) return pSum + p.text.length;
        if (p.media?.url) return pSum + p.media.url.length;
        if (p.toolRequest) return pSum + JSON.stringify(p.toolRequest).length;
        if (p.toolResponse) return pSum + JSON.stringify(p.toolResponse).length;
        return pSum;
      }, 0),
    0
  );
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

export const contextCompression: GenerateMiddleware<
  typeof ContextCompressionOptionsSchema
> = generateMiddleware(
  {
    name: 'contextCompression',
    description:
      'Compresses conversation context when it grows too large, using ' +
      'tool response truncation and message dropping.',
    configSchema: ContextCompressionOptionsSchema,
  },
  ({ config, ai }) => {
    const maxInputTokens = config?.maxInputTokens ?? Infinity;
    const preserveSystem = config?.preserveSystem !== false;
    const maxToolResponseChars =
      config?.maxToolResponseChars ?? DEFAULT_MAX_TOOL_RESPONSE_CHARS;

    const toolResponseConfig = config?.toolResponses;
    const toolMaxChars = toolResponseConfig?.maxChars;
    const toolPreserveRecent =
      toolResponseConfig?.preserveRecent ??
      DEFAULT_TOOL_RESPONSE_PRESERVE_RECENT;

    const maxMessages = config?.maxMessages;
    const insertTruncationNotice = config?.insertTruncationNotice !== false;
    const truncationNoticeText =
      config?.truncationNotice ?? DEFAULT_TRUNCATION_NOTICE;

    function applyToolResponseSafetyCap(messages: MessageData[]): {
      messages: MessageData[];
      capped: number;
    } {
      if (maxToolResponseChars === Infinity) return { messages, capped: 0 };

      let cappedCount = 0;
      const result = messages.map((msg) => {
        if (msg.role !== 'tool') return msg;

        let changed = false;
        const newContent = msg.content.map((part): Part => {
          if (part.toolResponse) {
            const outputStr = stringifyOutput(part.toolResponse.output);
            if (outputStr.length > maxToolResponseChars) {
              cappedCount++;
              changed = true;
              return {
                toolResponse: {
                  ...part.toolResponse,
                  output:
                    outputStr.slice(0, maxToolResponseChars) +
                    `\n\n---\n\n[TRUNCATED: Response was ${outputStr.length} chars ` +
                    `but only first ${maxToolResponseChars} are shown.]`,
                },
              };
            }
          }
          return part;
        });
        return changed ? { ...msg, content: newContent } : msg;
      });

      return { messages: result, capped: cappedCount };
    }

    function applyToolResponseTruncation(messages: MessageData[]): {
      messages: MessageData[];
      truncated: number;
    } {
      if (!toolMaxChars) return { messages, truncated: 0 };

      const toolIndices: number[] = [];
      for (let i = 0; i < messages.length; i++) {
        if (messages[i].role === 'tool') {
          toolIndices.push(i);
        }
      }

      const numToPreserve = Math.min(toolPreserveRecent, toolIndices.length);
      const truncatableIndices = new Set(
        toolIndices.slice(0, toolIndices.length - numToPreserve)
      );

      let truncatedCount = 0;
      const result = messages.map((msg, idx) => {
        if (!truncatableIndices.has(idx)) return msg;

        let changed = false;
        const newContent = msg.content.map((part): Part => {
          if (part.toolResponse) {
            const outputStr = stringifyOutput(part.toolResponse.output);
            if (outputStr.length > toolMaxChars) {
              truncatedCount++;
              changed = true;
              return {
                toolResponse: {
                  ...part.toolResponse,
                  output:
                    outputStr.slice(0, toolMaxChars) +
                    `\n\n---\n\n[TRUNCATED: Tool response was ${outputStr.length} characters long, ` +
                    `only the first ${toolMaxChars} characters are shown above. ` +
                    `Call this tool again if you need the full output.]`,
                },
              };
            }
          }
          return part;
        });

        return changed ? { ...msg, content: newContent } : msg;
      });

      return { messages: result, truncated: truncatedCount };
    }

    function applyMessageTruncation(messages: MessageData[]): {
      messages: MessageData[];
      dropped: number;
      noticeInserted: boolean;
      tailCount: number;
    } {
      if (!maxMessages || messages.length <= maxMessages) {
        return { messages, dropped: 0, noticeInserted: false, tailCount: 0 };
      }

      const { systemMessages, nonSystemMessages } = partitionMessages(
        messages,
        preserveSystem
      );

      const keepCount = Math.max(
        0,
        maxMessages - systemMessages.length - (insertTruncationNotice ? 1 : 0)
      );
      const kept = nonSystemMessages.slice(-keepCount);
      const dropped = nonSystemMessages.length - kept.length;

      let noticeInserted = false;
      if (dropped > 0 && insertTruncationNotice) {
        const notice: MessageData = {
          role: 'model',
          content: [{ text: truncationNoticeText }],
        };
        noticeInserted = true;
        return {
          messages: [...systemMessages, notice, ...kept],
          dropped,
          noticeInserted,
          tailCount: kept.length,
        };
      }

      return {
        messages: [...systemMessages, ...kept],
        dropped,
        noticeInserted,
        tailCount: kept.length,
      };
    }

    return {
      model: async (req, ctx, next) => {
        const result = await next(req, ctx);
        const store = compressionStorage.getStore();
        if (store && result.usage?.inputTokens !== undefined) {
          store.lastInputTokens = result.usage.inputTokens;
        }
        return result;
      },

      generate: async (envelope, ctx, next) => {
        const currentTurn = (envelope as any).currentTurn ?? 0;
        const isTopLevel = currentTurn === 0;

        const executeTurn = async () => {
          const store = compressionStorage.getStore();
          if (isTopLevel && store) {
            store.latestCompressionMeta = null;
            store.lastInputTokens = undefined;
          }

          const rawMessages = envelope.request.messages || [];
          const estimatedTokens = Math.ceil(
            estimateMessageChars(rawMessages) / 3.5
          );
          const effectiveTokens = Math.max(
            store?.lastInputTokens ?? 0,
            estimatedTokens
          );

          const shouldCompress =
            effectiveTokens > maxInputTokens ||
            (maxMessages !== undefined && rawMessages.length > maxMessages);

          if (!shouldCompress) {
            const response = await next(envelope, ctx);
            if (
              response.custom &&
              typeof response.custom === 'object' &&
              'contextCompression' in response.custom
            ) {
              const { contextCompression, ...restCustom } =
                response.custom as Record<string, unknown>;
              return { ...response, custom: restCustom };
            }
            return response;
          }

          const originalCount = rawMessages.length;

          const {
            messages: compressedMessages,
            toolResponsesSafetyCapped,
            toolResponsesTruncated,
            truncationNoticeInserted,
          } = await ai.run('contextCompression', rawMessages, async () => {
            let messages = [...rawMessages];
            let capped = 0;
            let truncated = 0;
            let noticeInserted = false;

            // 1. Safety cap on oversized tool responses
            if (maxToolResponseChars !== Infinity) {
              const capResult = applyToolResponseSafetyCap(messages);
              messages = capResult.messages;
              capped = capResult.capped;
            }

            // 2. Tool response truncation
            if (toolMaxChars) {
              const truncResult = applyToolResponseTruncation(messages);
              messages = truncResult.messages;
              truncated = truncResult.truncated;
            }

            // 3. Message truncation
            if (maxMessages && messages.length > maxMessages) {
              const msgResult = applyMessageTruncation(messages);
              messages = msgResult.messages;
              noticeInserted = msgResult.noticeInserted;
            }

            return {
              messages,
              toolResponsesSafetyCapped: capped,
              toolResponsesTruncated: truncated,
              truncationNoticeInserted: noticeInserted,
            };
          });

          const compressedCount = compressedMessages.length;
          const wasCompressed =
            toolResponsesSafetyCapped > 0 ||
            toolResponsesTruncated > 0 ||
            compressedCount < originalCount ||
            truncationNoticeInserted;

          let turnCompressionMeta: Record<string, unknown> | null = null;
          if (wasCompressed) {
            turnCompressionMeta = {
              triggered: true,
              inputTokensBefore: effectiveTokens,
              messagesOriginal: originalCount,
              messagesAfter: compressedCount,
              toolResponsesSafetyCapped,
              toolResponsesTruncated,
              truncationNoticeInserted,
            };
            if (store) {
              store.latestCompressionMeta = turnCompressionMeta;
            }
          }

          const modifiedEnvelope = {
            ...envelope,
            request: {
              ...envelope.request,
              messages: wasCompressed ? compressedMessages : rawMessages,
            },
          };

          const response = await next(modifiedEnvelope, ctx);

          if (turnCompressionMeta) {
            return {
              ...response,
              custom: {
                ...((response.custom as Record<string, unknown>) ?? {}),
                contextCompression: turnCompressionMeta,
              },
            };
          }

          if (
            response.custom &&
            typeof response.custom === 'object' &&
            'contextCompression' in response.custom
          ) {
            const { contextCompression, ...restCustom } =
              response.custom as Record<string, unknown>;
            return { ...response, custom: restCustom };
          }

          return response;
        };

        if (isTopLevel || !compressionStorage.getStore()) {
          return compressionStorage.run(
            { lastInputTokens: undefined, latestCompressionMeta: null },
            executeTurn
          );
        }
        return executeTurn();
      },
    };
  }
);
