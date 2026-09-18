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

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

export const ToolResponsesOptionsSchema = z.object({
  /**
   * Maximum character length for each tool response content.
   * Responses exceeding this will be truncated with a `[TRUNCATED: ...]` marker.
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

export const ContextCompressionOptionsSchema = z.object({
  /**
   * Compression triggers when the previous turn's `inputTokens` exceeds
   * this threshold. On turn 0, token count is estimated from messages.
   */
  maxInputTokens: z
    .number()
    .optional()
    .describe('Compress when token count exceeds this threshold.'),

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
   * Maximum message count target. Messages beyond this (oldest first) are
   * dropped while preserving system messages. Any leading tool or model messages
   * at the truncation cutoff are also discarded to satisfy LLM API requirements
   * (ensuring history begins with a user turn and avoiding orphaned tool responses).
   * The final message count will be at most `maxMessages`.
   */
  maxMessages: z
    .number()
    .optional()
    .describe(
      'Maximum message count target. Drops older non-system messages, ensuring history begins with a user turn.'
    ),

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
});

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

/**
 * Average character-to-token ratio heuristic across natural language and code payloads (~3.5-4 chars/token).
 */
const CHARS_PER_TOKEN_ESTIMATE = 3.5;

/**
 * Multimodal LLMs (e.g. Gemini) tokenize images at a fixed rate (~258 tokens)
 * regardless of base64 payload size. 1000 chars / 3.5 ≈ 285 tokens prevents
 * multi-megabyte inline data URIs from causing phantom token spikes on turn 0.
 */
const DATA_URI_APPROX_CHARS = 1000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Stringify tool output, avoiding re-stringifying if already a string.
 */
function stringifyOutput(output: unknown): string {
  if (typeof output === 'string') return output;
  try {
    return JSON.stringify(output ?? '');
  } catch {
    return String(output);
  }
}

function hasCompressionFlag(
  msg: MessageData,
  flag: 'truncated' | 'capped' | 'notice'
): boolean {
  const ccMeta = msg.metadata?.contextCompression as
    | Record<string, unknown>
    | undefined;
  return Boolean(ccMeta?.[flag]);
}

function withCompressionMetadata(
  target: { metadata?: Record<string, unknown> },
  fields: Record<string, unknown>
): Record<string, unknown> {
  return {
    ...target.metadata,
    contextCompression: {
      ...((target.metadata?.contextCompression as Record<string, unknown>) ??
        {}),
      ...fields,
    },
  };
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
  return messages.reduce((sum, m) => {
    return (
      sum +
      m.content.reduce((pSum, p) => {
        if (p.text) return pSum + p.text.length;
        if (p.reasoning) return pSum + p.reasoning.length;
        if ('data' in p && p.data !== undefined) {
          return pSum + stringifyOutput(p.data).length;
        }
        if ('custom' in p && p.custom) {
          return pSum + stringifyOutput(p.custom).length;
        }
        if (p.media?.url) {
          // Use a fixed character approximation for inline base64 data URIs
          // to reflect fixed image token billing rather than raw string length.
          const urlLen = p.media.url.startsWith('data:')
            ? DATA_URI_APPROX_CHARS
            : p.media.url.length;
          return pSum + urlLen;
        }
        if (p.toolRequest) return pSum + stringifyOutput(p.toolRequest).length;
        if (p.toolResponse)
          return pSum + stringifyOutput(p.toolResponse).length;
        return pSum;
      }, 0)
    );
  }, 0);
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

    let lastInputTokens: number | undefined;
    let latestCompressionMeta: Record<string, unknown> | null = null;

    const maxMessages = config?.maxMessages;
    const insertTruncationNotice = config?.insertTruncationNotice !== false;
    const truncationNoticeText =
      config?.truncationNotice ?? DEFAULT_TRUNCATION_NOTICE;

    function applyToolLimits(messages: MessageData[]): {
      messages: MessageData[];
      capped: number;
      truncated: number;
    } {
      const toolParts: { msgIdx: number; partIdx: number }[] = [];
      messages.forEach((msg, mIdx) => {
        if (msg.role === 'tool') {
          msg.content.forEach((p, pIdx) => {
            if (p.toolResponse) toolParts.push({ msgIdx: mIdx, partIdx: pIdx });
          });
        }
      });

      const numPreserved = Math.min(toolPreserveRecent, toolParts.length);
      const truncatableParts = new Set(
        toolParts
          .slice(0, toolParts.length - numPreserved)
          .map((tp) => `${tp.msgIdx}:${tp.partIdx}`)
      );

      let capped = 0;
      let truncated = 0;

      const result = messages.map((msg, mIdx) => {
        if (msg.role !== 'tool') return msg;

        // Skip if already compressed to the lowest limit (toolResponses.maxChars)
        if (hasCompressionFlag(msg, 'truncated')) {
          return msg;
        }

        let msgTruncated = false;
        let msgCapped = false;
        let changed = false;

        const newContent = msg.content.map((part, pIdx): Part => {
          if (!part.toolResponse) {
            return part;
          }

          const isTruncatable = truncatableParts.has(`${mIdx}:${pIdx}`);
          const limit =
            isTruncatable && toolMaxChars
              ? Math.min(maxToolResponseChars, toolMaxChars)
              : maxToolResponseChars;

          if (limit === Infinity) return part;
          // Skip if already capped by safety ceiling and still within the safety-cap zone
          if (
            limit === maxToolResponseChars &&
            hasCompressionFlag(msg, 'capped')
          ) {
            return part;
          }

          const outputStr = stringifyOutput(part.toolResponse.output);
          if (outputStr.length <= limit) return part;

          changed = true;

          // If truncatable and clamped to toolMaxChars, it's context-compression truncation.
          // Otherwise, it was clamped by maxToolResponseChars (the hard safety cap).
          if (isTruncatable && limit === toolMaxChars) {
            truncated++;
            msgTruncated = true;
            return {
              ...part,
              toolResponse: {
                ...part.toolResponse,
                output:
                  outputStr.slice(0, limit) +
                  `\n\n---\n\n[TRUNCATED: Tool response was ${outputStr.length} characters long, ` +
                  `only the first ${limit} characters are shown above. ` +
                  `Call this tool again if you need the full output.]`,
              },
            };
          } else {
            capped++;
            msgCapped = true;
            return {
              ...part,
              toolResponse: {
                ...part.toolResponse,
                output:
                  outputStr.slice(0, limit) +
                  `\n\n---\n\n[TRUNCATED: Response was ${outputStr.length} chars ` +
                  `but only first ${limit} are shown.]`,
              },
            };
          }
        });

        if (!changed) return msg;

        return {
          ...msg,
          metadata: withCompressionMetadata(msg, {
            ...(msgTruncated ? { truncated: true } : {}),
            ...(msgCapped ? { capped: true } : {}),
          }),
          content: newContent,
        };
      });

      return { messages: result, capped, truncated };
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

      const noticeConsumesSlot =
        insertTruncationNotice && systemMessages.length === 0;
      const keepCount = Math.max(
        0,
        maxMessages - systemMessages.length - (noticeConsumesSlot ? 1 : 0)
      );
      let kept = keepCount === 0 ? [] : nonSystemMessages.slice(-keepCount);

      // Prevent orphaned tool messages and dangling model turns
      while (
        kept.length > 0 &&
        (kept[0].role === 'tool' || kept[0].role === 'model')
      ) {
        kept.shift();
      }

      const dropped = nonSystemMessages.length - kept.length;

      let noticeInserted = false;
      if (dropped > 0 && insertTruncationNotice) {
        noticeInserted = true;
        if (systemMessages.length > 0) {
          const alreadyHasNotice = systemMessages.some((m) =>
            hasCompressionFlag(m, 'notice')
          );
          const lastIdx = systemMessages.length - 1;
          const updatedSystemMessages = alreadyHasNotice
            ? systemMessages
            : systemMessages.map((msg, idx) =>
                idx === lastIdx
                  ? {
                      ...msg,
                      metadata: withCompressionMetadata(msg, { notice: true }),
                      content: [
                        ...msg.content,
                        { text: `\n\n${truncationNoticeText}` },
                      ],
                    }
                  : msg
              );
          return {
            messages: [...updatedSystemMessages, ...kept],
            dropped,
            noticeInserted,
            tailCount: kept.length,
          };
        } else {
          const notice: MessageData = {
            role: 'system',
            metadata: withCompressionMetadata({}, { notice: true }),
            content: [{ text: truncationNoticeText }],
          };
          return {
            messages: [notice, ...kept],
            dropped,
            noticeInserted,
            tailCount: kept.length,
          };
        }
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
        if (result.usage?.inputTokens !== undefined) {
          lastInputTokens = result.usage.inputTokens;
        }
        return result;
      },

      generate: async (envelope, ctx, next) => {
        const currentTurn = envelope.currentTurn ?? 0;
        const isTopLevel = currentTurn === 0;

        if (isTopLevel) {
          latestCompressionMeta = null;
          lastInputTokens = undefined;
        }

        const rawMessages = envelope.request.messages || [];
        const estimatedTokens = Math.ceil(
          estimateMessageChars(rawMessages) / CHARS_PER_TOKEN_ESTIMATE
        );
        const effectiveTokens = Math.max(lastInputTokens ?? 0, estimatedTokens);

        const shouldCompress =
          effectiveTokens > maxInputTokens ||
          (maxMessages !== undefined && rawMessages.length > maxMessages);

        if (!shouldCompress) {
          const response = await next(envelope, ctx);
          if (isTopLevel && latestCompressionMeta) {
            return {
              ...response,
              custom: {
                ...((response.custom as Record<string, unknown>) ?? {}),
                contextCompression: latestCompressionMeta,
              },
            };
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

          // 1. Tool response limits (Safety cap & Truncation in a single pass)
          const toolResult = applyToolLimits(messages);
          messages = toolResult.messages;
          capped = toolResult.capped;
          truncated = toolResult.truncated;

          // 2. Message truncation
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
          latestCompressionMeta = turnCompressionMeta;
        }

        const modifiedEnvelope = {
          ...envelope,
          request: {
            ...envelope.request,
            messages: wasCompressed ? compressedMessages : rawMessages,
          },
        };

        const response = await next(modifiedEnvelope, ctx);

        if (isTopLevel) {
          const finalMeta = turnCompressionMeta ?? latestCompressionMeta;
          if (finalMeta) {
            return {
              ...response,
              custom: {
                ...((response.custom as Record<string, unknown>) ?? {}),
                contextCompression: finalMeta,
              },
            };
          }
        }

        return response;
      },
    };
  }
);
