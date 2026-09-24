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
   * Responses exceeding this will be truncated with a `[Truncated N characters]` marker.
   */
  maxChars: z
    .number()
    .int()
    .positive()
    .describe(
      'Max chars per tool response. Responses beyond this are truncated.'
    ),

  /**
   * Number of most recent tool response messages to leave untouched.
   * @default 2
   */
  preserveRecent: z
    .number()
    .int()
    .nonnegative()
    .optional()
    .describe("Don't truncate the last N tool response messages. Default: 2."),
});

export const DeduplicateToolResponsesOptionsSchema = z.object({
  /**
   * How to identify duplicates:
   * - `'name-and-input'`: Match by tool name and exact arguments (default).
   * - `'name-only'`: Match by tool name only.
   */
  matchBy: z
    .enum(['name-and-input', 'name-only'])
    .optional()
    .describe(
      'Match by tool name and arguments ("name-and-input") or name only ("name-only"). Default: "name-and-input".'
    ),

  /**
   * Number of most recent responses to leave untouched per tool/args group.
   * Older duplicates are replaced with `notice`.
   * @default 1
   */
  keepRecent: z
    .number()
    .optional()
    .describe('Number of recent duplicates to keep untouched. Default: 1.'),

  /**
   * Replacement text for deduplicated tool responses.
   */
  notice: z
    .string()
    .optional()
    .describe('Replacement text for deduplicated tool responses.'),
});

export const ContextCompressionOptionsSchema = z.object({
  /**
   * Compression triggers when the previous turn's `inputTokens` exceeds
   * this threshold. On turn 0, token count is estimated from messages.
   */
  maxInputTokens: z
    .number()
    .int()
    .positive()
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
   * Set to a negative number (or `Infinity`) to disable.
   * @default 400000
   */
  maxToolResponseChars: z
    .number()
    .optional()
    .describe(
      'Hard cap on any single tool response size. Set negative to disable. Default: 400000 chars.'
    ),

  /**
   * Deduplicate repeated tool calls with the same arguments.
   * Replaces older duplicate outputs with a short notice.
   */
  deduplicateToolResponses:
    DeduplicateToolResponsesOptionsSchema.optional().describe(
      'Deduplicate repeated tool calls with same arguments.'
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
    .int()
    .positive()
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
const DEFAULT_DEDUP_KEEP_RECENT = 1;
const DEFAULT_DEDUP_NOTICE =
  '[Deduplicated: This tool response has been removed to save context. ' +
  'See the most recent call of this tool for current output.]';
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

/**
 * Slice a string to at most `limit` UTF-16 code units without splitting
 * a surrogate pair at the boundary.
 */
function sliceCodePointSafe(str: string, limit: number): string {
  if (str.length <= limit) return str;
  if (limit > 0) {
    const code = str.charCodeAt(limit - 1);
    if (code >= 0xd800 && code <= 0xdbff) {
      return str.slice(0, limit - 1);
    }
  }
  return str.slice(0, limit);
}

function hasCompressionFlag(
  target: { metadata?: Record<string, unknown> },
  flag: 'truncated' | 'capped' | 'deduplicated' | 'notice'
): boolean {
  const ccMeta = target.metadata?.contextCompression as
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
 * Split messages into leading system messages and remaining conversation messages.
 * Only leading system messages (or existing synthetic truncation notice system messages)
 * are extracted so mid-conversation system instructions keep their relative order.
 */
function partitionMessages(
  messages: MessageData[],
  preserveSystem: boolean
): { systemMessages: MessageData[]; nonSystemMessages: MessageData[] } {
  if (!preserveSystem) {
    return {
      systemMessages: [],
      nonSystemMessages: messages.filter(
        (m) => !hasCompressionFlag(m, 'notice')
      ),
    };
  }

  const systemMessages: MessageData[] = [];
  let idx = 0;
  while (idx < messages.length && messages[idx].role === 'system') {
    systemMessages.push(messages[idx]);
    idx++;
  }

  return {
    systemMessages,
    nonSystemMessages: messages.slice(idx),
  };
}

/**
 * Read the `inputTokens` stamp from the newest model message in history, if present.
 */
function lastReportedInputTokens(messages: MessageData[]): number | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (!msg || msg.role !== 'model') continue;
    const ccMeta = msg.metadata?.contextCompression as
      | Record<string, unknown>
      | undefined;
    if (typeof ccMeta?.inputTokens === 'number' && ccMeta.inputTokens > 0) {
      return ccMeta.inputTokens;
    }
    return undefined;
  }
  return undefined;
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
    const rawMaxToolResponseChars =
      config?.maxToolResponseChars ?? DEFAULT_MAX_TOOL_RESPONSE_CHARS;
    const maxToolResponseChars =
      rawMaxToolResponseChars <= 0 ? Infinity : rawMaxToolResponseChars;

    const dedupConfig = config?.deduplicateToolResponses;
    const dedupMatchBy = dedupConfig?.matchBy ?? 'name-and-input';
    const dedupKeepRecent =
      dedupConfig?.keepRecent ?? DEFAULT_DEDUP_KEEP_RECENT;
    const dedupNotice = dedupConfig?.notice ?? DEFAULT_DEDUP_NOTICE;

    const toolResponseConfig = config?.toolResponses;
    const toolMaxChars = toolResponseConfig?.maxChars;
    const toolPreserveRecent =
      toolResponseConfig?.preserveRecent ??
      DEFAULT_TOOL_RESPONSE_PRESERVE_RECENT;

    let lastInputTokens: number | undefined;
    let cumulativeCapped = 0;
    let cumulativeDeduplicated = 0;
    let cumulativeTruncated = 0;
    let latestCompressionMeta: Record<string, unknown> | null = null;

    const maxMessages = config?.maxMessages;
    const insertTruncationNotice = config?.insertTruncationNotice !== false;
    const truncationNoticeText =
      config?.truncationNotice ?? DEFAULT_TRUNCATION_NOTICE;

    function applyToolResponseDeduplication(messages: MessageData[]): {
      messages: MessageData[];
      deduplicated: number;
    } {
      if (!dedupConfig) return { messages, deduplicated: 0 };

      // Map tool call IDs to tool request input across model messages
      const toolInputByRef = new Map<string, unknown>();
      for (const msg of messages) {
        if (msg.role === 'model') {
          for (const part of msg.content) {
            if (part.toolRequest?.ref) {
              toolInputByRef.set(part.toolRequest.ref, part.toolRequest.input);
            }
          }
        }
      }

      const groups = new Map<string, number[]>();
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg.role !== 'tool') continue;

        for (const part of msg.content) {
          if (!part.toolResponse) continue;

          let toolInput = part.toolResponse.ref
            ? toolInputByRef.get(part.toolResponse.ref)
            : undefined;

          // If no ref was matched, check if preceding model message had a matching toolRequest with input
          if (
            toolInput === undefined &&
            i > 0 &&
            messages[i - 1]?.role === 'model'
          ) {
            const reqPart = messages[i - 1].content.find(
              (p) => p.toolRequest?.name === part.toolResponse?.name
            );
            if (reqPart?.toolRequest) {
              toolInput = reqPart.toolRequest.input;
            }
          }

          const key =
            dedupMatchBy === 'name-only'
              ? part.toolResponse.name
              : JSON.stringify({
                  name: part.toolResponse.name,
                  input: toolInput,
                });
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key)!.push(i);
        }
      }

      const indicesToReplace = new Set<number>();
      for (const indices of groups.values()) {
        if (indices.length > dedupKeepRecent) {
          const toRemove = indices.slice(0, indices.length - dedupKeepRecent);
          for (const idx of toRemove) {
            indicesToReplace.add(idx);
          }
        }
      }

      if (indicesToReplace.size === 0) {
        return { messages, deduplicated: 0 };
      }

      let deduplicatedCount = 0;
      const result = messages.map((msg, idx) => {
        if (!indicesToReplace.has(idx)) return msg;
        if (hasCompressionFlag(msg, 'deduplicated')) return msg;

        const newContent = msg.content.map((part): Part => {
          if (part.toolResponse) {
            deduplicatedCount++;
            return {
              toolResponse: {
                ...part.toolResponse,
                output: dedupNotice,
              },
            };
          }
          return part;
        });
        return {
          ...msg,
          metadata: withCompressionMetadata(msg, { deduplicated: true }),
          content: newContent,
        };
      });

      return { messages: result, deduplicated: deduplicatedCount };
    }

    function applyToolLimits(
      messages: MessageData[],
      includeToolTruncation: boolean
    ): {
      messages: MessageData[];
      capped: number;
      truncated: number;
    } {
      const toolMsgIndices: number[] = [];
      messages.forEach((msg, mIdx) => {
        if (msg.role === 'tool') {
          toolMsgIndices.push(mIdx);
        }
      });

      const numPreserved = Math.min(toolPreserveRecent, toolMsgIndices.length);
      const truncatableMsgIndices = new Set(
        toolMsgIndices.slice(0, toolMsgIndices.length - numPreserved)
      );

      let capped = 0;
      let truncated = 0;

      const result = messages.map((msg, mIdx) => {
        if (msg.role !== 'tool') return msg;
        if (hasCompressionFlag(msg, 'deduplicated')) return msg;

        const isTruncatableMsg =
          includeToolTruncation &&
          Boolean(toolMaxChars) &&
          truncatableMsgIndices.has(mIdx);
        let changed = false;

        const newContent = msg.content.map((part): Part => {
          if (!part.toolResponse) {
            return part;
          }

          // Skip if this part was already truncated to toolResponses.maxChars
          if (hasCompressionFlag(part, 'truncated')) {
            return part;
          }

          const limit =
            isTruncatableMsg && toolMaxChars
              ? Math.min(maxToolResponseChars, toolMaxChars)
              : maxToolResponseChars;

          if (limit === Infinity) return part;
          // Skip if already capped by safety ceiling and still within the safety-cap zone
          if (
            limit === maxToolResponseChars &&
            hasCompressionFlag(part, 'capped')
          ) {
            return part;
          }

          const outputStr = stringifyOutput(part.toolResponse.output);
          if (outputStr.length <= limit) return part;

          const sliced = sliceCodePointSafe(outputStr, limit);
          const omitted = outputStr.length - sliced.length;

          // If truncatable and clamped to toolMaxChars, it's context-compression truncation.
          // Otherwise, it was clamped by maxToolResponseChars (the hard safety cap).
          if (isTruncatableMsg && limit === toolMaxChars) {
            const marker = `\n\n[Truncated ${omitted} characters]`;
            changed = true;
            truncated++;
            return {
              ...part,
              metadata: withCompressionMetadata(part, { truncated: true }),
              toolResponse: {
                ...part.toolResponse,
                output: sliced + marker,
              },
            };
          } else {
            const marker =
              `\n\n---\n\n[TRUNCATED: Response was ${outputStr.length} chars ` +
              `but only first ${limit} are shown.]`;
            changed = true;
            capped++;
            return {
              ...part,
              metadata: withCompressionMetadata(part, { capped: true }),
              toolResponse: {
                ...part.toolResponse,
                output: sliced + marker,
              },
            };
          }
        });

        if (!changed) return msg;

        return {
          ...msg,
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
      if (!maxMessages || maxMessages <= 0 || messages.length <= maxMessages) {
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

      // Strip leading orphaned tool messages so a toolResponse is never separated
      // from the model toolRequest that preceded it.
      while (kept.length > 0 && kept[0].role === 'tool') {
        kept.shift();
      }

      // Ensure kept conversation starts with a user message:
      // - If kept contains a later user message, shift leading model/tool messages up to it.
      // - If kept has no user message (e.g. a single-user-prompt multi-turn tool loop),
      //   preserve the initiating user message ahead of the kept [model, tool] turns.
      if (kept.length > 0 && kept[0].role === 'model') {
        const firstUserIdx = kept.findIndex((m) => m.role === 'user');
        if (firstUserIdx >= 0) {
          kept = kept.slice(firstUserIdx);
        } else if (keepCount > 0) {
          const droppedPrefix = nonSystemMessages.slice(
            0,
            nonSystemMessages.length - kept.length
          );
          let anchorUser: MessageData | undefined;
          for (let i = droppedPrefix.length - 1; i >= 0; i--) {
            if (droppedPrefix[i].role === 'user') {
              anchorUser = droppedPrefix[i];
              break;
            }
          }
          if (anchorUser) {
            const tail =
              keepCount > 1 ? nonSystemMessages.slice(-(keepCount - 1)) : [];
            while (tail.length > 0 && tail[0].role === 'tool') {
              tail.shift();
            }
            kept = [anchorUser, ...tail];
          } else {
            kept = [];
          }
        }
      }

      const dropped = nonSystemMessages.length - kept.length;

      let noticeInserted = false;
      if (dropped > 0 && insertTruncationNotice) {
        noticeInserted = true;
        if (systemMessages.length > 0) {
          const alreadyHasNotice = systemMessages.some((m) =>
            hasCompressionFlag(m, 'notice')
          );
          const updatedSystemMessages = alreadyHasNotice
            ? systemMessages
            : systemMessages.map((msg, idx) =>
                idx === 0
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
            metadata: {
              ...withCompressionMetadata({}, { notice: true }),
              agentPreamble: true,
            },
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
          if (result.message && result.usage.inputTokens > 0) {
            result.message = {
              ...result.message,
              metadata: withCompressionMetadata(result.message, {
                inputTokens: result.usage.inputTokens,
              }),
            };
          }
        }
        return result;
      },

      generate: async (envelope, ctx, next) => {
        const currentTurn = envelope.currentTurn ?? 0;
        const isTopLevel = currentTurn === 0;

        if (isTopLevel) {
          latestCompressionMeta = null;
          lastInputTokens = undefined;
          cumulativeCapped = 0;
          cumulativeDeduplicated = 0;
          cumulativeTruncated = 0;
        }

        const rawMessages = envelope.request.messages || [];
        const stampedTokens =
          lastInputTokens ?? lastReportedInputTokens(rawMessages);
        const estimatedTokens =
          maxInputTokens === Infinity
            ? 0
            : Math.ceil(
                estimateMessageChars(rawMessages) / CHARS_PER_TOKEN_ESTIMATE
              );
        const effectiveTokens = Math.max(stampedTokens ?? 0, estimatedTokens);

        const shouldCompress =
          effectiveTokens > maxInputTokens ||
          (maxMessages !== undefined &&
            maxMessages > 0 &&
            rawMessages.length > maxMessages);

        const hasOversizedToolResponse =
          maxToolResponseChars !== Infinity &&
          rawMessages.some(
            (m) =>
              m.role === 'tool' &&
              m.content.some(
                (p) =>
                  p.toolResponse &&
                  !hasCompressionFlag(p, 'capped') &&
                  !hasCompressionFlag(p, 'truncated') &&
                  stringifyOutput(p.toolResponse.output).length >
                    maxToolResponseChars
              )
          );

        if (!shouldCompress && !hasOversizedToolResponse) {
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
          toolResponsesDeduplicated,
          toolResponsesTruncated,
          truncationNoticeInserted,
        } = await ai.run(
          'contextCompression',
          { messageCount: originalCount, effectiveTokens },
          async () => {
            let messages = [...rawMessages];
            let capped = 0;
            let deduplicated = 0;
            let truncated = 0;
            let noticeInserted = false;

            // 1. Tool response deduplication (when shouldCompress is true)
            if (shouldCompress && dedupConfig) {
              const dedupResult = applyToolResponseDeduplication(messages);
              messages = dedupResult.messages;
              deduplicated = dedupResult.deduplicated;
            }

            // 2. Tool response limits (Safety cap always; Truncation when shouldCompress is true)
            const toolResult = applyToolLimits(messages, shouldCompress);
            messages = toolResult.messages;
            capped = toolResult.capped;
            truncated = toolResult.truncated;

            // 3. Message truncation
            if (
              shouldCompress &&
              maxMessages &&
              messages.length > maxMessages
            ) {
              const msgResult = applyMessageTruncation(messages);
              messages = msgResult.messages;
              noticeInserted = msgResult.noticeInserted;
            }

            return {
              messages,
              toolResponsesSafetyCapped: capped,
              toolResponsesDeduplicated: deduplicated,
              toolResponsesTruncated: truncated,
              truncationNoticeInserted: noticeInserted,
            };
          }
        );

        const compressedCount = compressedMessages.length;
        const wasCompressed =
          toolResponsesSafetyCapped > 0 ||
          toolResponsesDeduplicated > 0 ||
          toolResponsesTruncated > 0 ||
          compressedCount < originalCount ||
          truncationNoticeInserted;

        let turnCompressionMeta: Record<string, unknown> | null = null;
        if (wasCompressed) {
          cumulativeCapped += toolResponsesSafetyCapped;
          cumulativeDeduplicated += toolResponsesDeduplicated;
          cumulativeTruncated += toolResponsesTruncated;
          turnCompressionMeta = {
            triggered: true,
            inputTokensBefore: effectiveTokens,
            messagesOriginal: originalCount,
            messagesAfter: compressedCount,
            toolResponsesSafetyCapped: cumulativeCapped,
            toolResponsesDeduplicated: cumulativeDeduplicated,
            toolResponsesTruncated: cumulativeTruncated,
            truncationNoticeInserted:
              truncationNoticeInserted ||
              Boolean(latestCompressionMeta?.truncationNoticeInserted),
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
