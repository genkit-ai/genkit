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
  ModelReferenceSchema,
  z,
  type GenerateMiddleware,
  type MessageData,
  type Part,
} from 'genkit';
import { logger } from 'genkit/logging';
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

export const SummarizeOptionsSchema = z.object({
  /**
   * Model to use for summarization. A model reference, model name string,
   * or ModelAction, e.g. `{ name: 'googleai/gemini-flash-lite-latest' }`.
   */
  model: ModelReferenceSchema.describe('Model to use for summarization.'),

  /**
   * Number of most recent messages to keep un-summarized.
   * Everything before this window is replaced with a summary.
   * @default 6
   */
  preserveRecent: z
    .number()
    .optional()
    .describe('Keep last N messages un-summarized. Default: 6.'),

  /**
   * Custom summarization prompt. The string `{conversation}` will be
   * replaced with a text rendering of the messages to summarize.
   */
  prompt: z
    .string()
    .optional()
    .describe('Custom summarization prompt. Use {conversation} placeholder.'),
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
     * Hard cap on message count. Messages beyond this (oldest first) are
     * dropped, preserving system messages and recent messages.
     */
    maxMessages: z
      .number()
      .optional()
      .describe('Hard cap on message count. Drop oldest beyond this.'),

    /**
     * Use an LLM to summarize older messages into a condensed form.
     * The summary replaces the original messages, preserving recent context.
     */
    summarize: SummarizeOptionsSchema.optional().describe(
      'Summarize older messages using an LLM.'
    ),

    /**
     * If cheap strategies (deduplication + tool truncation) reduce estimated
     * context by at least this fraction, skip the LLM summarization step.
     * Set to `0` to always summarize when configured.
     * @default undefined (always summarize when configured)
     */
    skipSummarizationThreshold: z
      .number()
      .optional()
      .describe(
        'Skip summarization if cheap strategies save at least this fraction of context. E.g. 0.3 = 30%.'
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

    /**
     * Store compressed messages on `message.metadata.compressedHistory` of the
     * last compressed message, keeping original uncompressed messages in
     * `request.messages` and `response.messages`.
     *
     * The middleware automatically resolves `compressedHistory` on subsequent turns.
     * Use `resolveCompressedHistory(messages)` to resolve the active messages yourself.
     *
     * Set to `false` to overwrite `request.messages` directly (destructive).
     * @default true
     */
    preserveOriginalMessages: z
      .boolean()
      .optional()
      .describe(
        'Preserve original messages and store compressed history in metadata. Default: true.'
      ),
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
const DEFAULT_DEDUP_KEEP_RECENT = 1;
const DEFAULT_DEDUP_NOTICE =
  '[Deduplicated: This tool response has been removed to save context. ' +
  'See the most recent call of this tool for current output.]';
const DEFAULT_PRESERVE_RECENT = 4;
const DEFAULT_SUMMARIZE_PRESERVE_RECENT = 6;
const SUMMARY_PREFIX = '[Conversation Summary]';
const DEFAULT_SUMMARIZE_PROMPT = `Summarize the following conversation concisely. Capture key facts, decisions made, tool calls and their results, and the current state of the conversation so that the assistant can continue helping the user effectively.

Conversation:
{conversation}

Summary:`;
const DEFAULT_TRUNCATION_NOTICE =
  '[NOTE] Some earlier messages in this conversation have been removed to stay within ' +
  'context limits. The most recent messages are preserved. Pay close attention to the ' +
  'latest messages and any conversation summary above.';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Resolves active messages by scanning bottom-up for `message.metadata.compressedHistory`.
 * When found, replaces messages from index 0 up to that message with the compressedHistory.
 */
export function resolveCompressedHistory(
  messages: MessageData[]
): MessageData[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    const meta = messages[i]?.metadata;
    if (meta && Array.isArray(meta.compressedHistory)) {
      return [
        ...(meta.compressedHistory as MessageData[]),
        ...messages.slice(i + 1),
      ];
    }
  }
  return messages;
}

/**
 * Stringify tool output, avoiding re-stringifying if already a string.
 */
function stringifyOutput(output: unknown): string {
  return typeof output === 'string' ? output : JSON.stringify(output ?? '');
}

/**
 * Render messages as text for summarization.
 */
function renderMessages(messages: MessageData[]): string {
  return messages
    .map((m) => {
      const parts = m.content
        .map((p) => {
          if (p.text) return p.text;
          if (p.toolRequest)
            return `[Tool call: ${p.toolRequest.name}(${JSON.stringify(p.toolRequest.input)})]`;
          if (p.toolResponse)
            return `[Tool response: ${p.toolResponse.name} → ${JSON.stringify(p.toolResponse.output)}]`;
          return '[other content]';
        })
        .join(' ');
      return `${m.role}: ${parts}`;
    })
    .join('\n');
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
 * Inject conversation summary as a dedicated user message preceding preserved messages.
 */
function buildSummarizedMessages(
  systemMessages: MessageData[],
  summaryText: string,
  toKeep: MessageData[]
): MessageData[] {
  const summaryPrefix = `${SUMMARY_PREFIX}\n${summaryText}`;
  const summaryMessage: MessageData = {
    role: 'user',
    content: [{ text: summaryPrefix }],
  };
  return [...systemMessages, summaryMessage, ...toKeep];
}

/**
 * Adjust preserve windows based on how far over budget we are.
 */
function adjustForOvershoot(
  overshootRatio: number,
  preserveRecent: number,
  summaryPreserveRecent: number
): {
  adjustedPreserveRecent: number;
  adjustedSummaryPreserveRecent: number;
} {
  if (overshootRatio >= 2.0) {
    return {
      adjustedPreserveRecent: Math.min(preserveRecent, 2),
      adjustedSummaryPreserveRecent: Math.min(summaryPreserveRecent, 2),
    };
  }
  if (overshootRatio >= 1.5) {
    return {
      adjustedPreserveRecent: Math.min(
        preserveRecent,
        Math.max(1, Math.floor(preserveRecent / 2))
      ),
      adjustedSummaryPreserveRecent: Math.min(
        summaryPreserveRecent,
        Math.max(1, Math.floor(summaryPreserveRecent / 2))
      ),
    };
  }
  return {
    adjustedPreserveRecent: preserveRecent,
    adjustedSummaryPreserveRecent: summaryPreserveRecent,
  };
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
    const basePreserveRecent =
      config?.preserveRecent ?? DEFAULT_PRESERVE_RECENT;
    const preserveSystem = config?.preserveSystem !== false;
    const preserveOriginalMessages = config?.preserveOriginalMessages !== false;
    const maxToolResponseChars =
      config?.maxToolResponseChars ?? DEFAULT_MAX_TOOL_RESPONSE_CHARS;

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

    const maxMessages = config?.maxMessages;
    const insertTruncationNotice = config?.insertTruncationNotice !== false;
    const truncationNoticeText =
      config?.truncationNotice ?? DEFAULT_TRUNCATION_NOTICE;

    const summarizeConfig = config?.summarize;
    const skipSummarizationThreshold = config?.skipSummarizationThreshold;
    const baseSummaryPreserveRecent =
      summarizeConfig?.preserveRecent ?? DEFAULT_SUMMARIZE_PRESERVE_RECENT;
    const summaryPromptTemplate =
      summarizeConfig?.prompt ?? DEFAULT_SUMMARIZE_PROMPT;
    const summaryModelRef = summarizeConfig?.model;

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
        return { ...msg, content: newContent };
      });

      return { messages: result, deduplicated: deduplicatedCount };
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

    function applyMessageTruncation(
      messages: MessageData[],
      effectiveMaxMessages?: number
    ): {
      messages: MessageData[];
      dropped: number;
      noticeInserted: boolean;
      tailCount: number;
    } {
      const cap = effectiveMaxMessages ?? maxMessages;
      if (!cap || messages.length <= cap) {
        return { messages, dropped: 0, noticeInserted: false, tailCount: 0 };
      }

      const { systemMessages, nonSystemMessages } = partitionMessages(
        messages,
        preserveSystem
      );

      const keepCount = Math.max(
        0,
        cap - systemMessages.length - (insertTruncationNotice ? 1 : 0)
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

    async function applySummarization(
      messages: MessageData[],
      effectiveSummaryPreserveRecent?: number
    ): Promise<{
      messages: MessageData[];
      summarized: boolean;
      tailCount: number;
    }> {
      if (!summaryModelRef)
        return { messages, summarized: false, tailCount: 0 };

      const summaryPreserveRecent =
        effectiveSummaryPreserveRecent ?? baseSummaryPreserveRecent;

      const { systemMessages, nonSystemMessages } = partitionMessages(
        messages,
        preserveSystem
      );

      if (nonSystemMessages.length <= summaryPreserveRecent) {
        return { messages, summarized: false, tailCount: 0 };
      }

      const toSummarize = nonSystemMessages.slice(
        0,
        nonSystemMessages.length - summaryPreserveRecent
      );
      const toKeep = nonSystemMessages.slice(-summaryPreserveRecent);

      try {
        const conversationText = renderMessages(toSummarize);
        const prompt = summaryPromptTemplate.replace(
          '{conversation}',
          conversationText
        );

        const response = await ai.generate({
          model: summaryModelRef as any,
          config: summaryModelRef?.config,
          prompt,
        });

        return {
          messages: buildSummarizedMessages(
            systemMessages,
            response.text,
            toKeep
          ),
          summarized: true,
          tailCount: toKeep.length,
        };
      } catch (e: any) {
        logger.warn(
          `Summarization failed, proceeding without compression: ${
            e instanceof Error ? e.message : String(e)
          }`,
          { 'genkit.middleware.name': 'contextCompression' },
          e
        );
        return { messages, summarized: false, tailCount: 0 };
      }
    }

    return {
      model: async (req, ctx, next) => {
        const resolvedMessages = resolveCompressedHistory(req.messages || []);
        const modifiedReq =
          resolvedMessages !== req.messages
            ? { ...req, messages: resolvedMessages }
            : req;

        const result = await next(modifiedReq, ctx);
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
          const activeMessages = resolveCompressedHistory(rawMessages);
          const estimatedTokens = Math.ceil(
            estimateMessageChars(activeMessages) / 3.5
          );
          const effectiveTokens = Math.max(
            store?.lastInputTokens ?? 0,
            estimatedTokens
          );

          const shouldCompress =
            effectiveTokens > maxInputTokens ||
            (maxMessages !== undefined && activeMessages.length > maxMessages);

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

          const charsBefore = estimateMessageChars(activeMessages);
          const originalCount = activeMessages.length;

          const overshootRatio =
            maxInputTokens !== Infinity && maxInputTokens > 0
              ? effectiveTokens / maxInputTokens
              : 1;

          const { adjustedPreserveRecent, adjustedSummaryPreserveRecent } =
            adjustForOvershoot(
              overshootRatio,
              basePreserveRecent,
              baseSummaryPreserveRecent
            );

          const {
            messages: compressedMessages,
            toolResponsesSafetyCapped,
            toolResponsesDeduplicated,
            toolResponsesTruncated,
            truncationNoticeInserted,
            messagesTruncated,
            truncTailCount,
            summarized,
            summaryTailCount,
            summarizationSkipped,
          } = await ai.run('contextCompression', activeMessages, async () => {
            let messages = [...activeMessages];
            let capped = 0;
            let deduplicated = 0;
            let truncated = 0;
            let noticeInserted = false;
            let msgTruncated = false;
            let mTailCount = 0;
            let isSummarized = false;
            let sTailCount = 0;
            let skippedSummary = false;

            // 1. Safety cap on oversized tool responses
            if (maxToolResponseChars !== Infinity) {
              const capResult = applyToolResponseSafetyCap(messages);
              messages = capResult.messages;
              capped = capResult.capped;
            }

            // 2. Tool response deduplication
            if (dedupConfig) {
              const dedupResult = applyToolResponseDeduplication(messages);
              messages = dedupResult.messages;
              deduplicated = dedupResult.deduplicated;
            }

            // 3. Tool response truncation
            if (toolMaxChars) {
              const truncResult = applyToolResponseTruncation(messages);
              messages = truncResult.messages;
              truncated = truncResult.truncated;
            }

            // 4. Check if cheap strategies saved enough to skip summarization
            const charsAfterCheap = estimateMessageChars(messages);
            const charsSaved = charsBefore - charsAfterCheap;
            const savingsRatio = charsBefore > 0 ? charsSaved / charsBefore : 0;

            const shouldSkipSummarization =
              skipSummarizationThreshold !== undefined &&
              savingsRatio >= skipSummarizationThreshold;

            // 5. Message truncation
            const effectiveMaxMessages = maxMessages
              ? Math.min(
                  maxMessages,
                  Math.max(
                    1,
                    maxMessages - (basePreserveRecent - adjustedPreserveRecent)
                  )
                )
              : undefined;

            if (
              effectiveMaxMessages &&
              messages.length > effectiveMaxMessages
            ) {
              const msgResult = applyMessageTruncation(
                messages,
                effectiveMaxMessages
              );
              messages = msgResult.messages;
              noticeInserted = msgResult.noticeInserted;
              if (msgResult.dropped > 0) {
                msgTruncated = true;
                mTailCount = msgResult.tailCount;
              }
            }

            // 6. Summarization
            if (summaryModelRef) {
              if (shouldSkipSummarization) {
                skippedSummary = true;
              } else {
                const sumResult = await applySummarization(
                  messages,
                  adjustedSummaryPreserveRecent
                );
                messages = sumResult.messages;
                isSummarized = sumResult.summarized;
                if (isSummarized) {
                  sTailCount = sumResult.tailCount;
                }
              }
            }

            return {
              messages,
              toolResponsesSafetyCapped: capped,
              toolResponsesDeduplicated: deduplicated,
              toolResponsesTruncated: truncated,
              truncationNoticeInserted: noticeInserted,
              messagesTruncated: msgTruncated,
              truncTailCount: mTailCount,
              summarized: isSummarized,
              summaryTailCount: sTailCount,
              summarizationSkipped: skippedSummary,
            };
          });

          const compressedCount = compressedMessages.length;
          const wasCompressed =
            toolResponsesSafetyCapped > 0 ||
            toolResponsesDeduplicated > 0 ||
            toolResponsesTruncated > 0 ||
            summarized ||
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
              toolResponsesDeduplicated,
              toolResponsesTruncated,
              truncationNoticeInserted,
              summarized,
              summarizationSkipped,
            };
            if (store) {
              store.latestCompressionMeta = turnCompressionMeta;
            }
          }

          let outgoingMessages: MessageData[];
          if (wasCompressed && preserveOriginalMessages) {
            let tailCount = 0;
            if (summarized) {
              tailCount = summaryTailCount;
            } else if (messagesTruncated) {
              tailCount = truncTailCount;
            } else {
              tailCount = 0;
            }

            const cutIndex =
              tailCount > 0
                ? Math.max(0, rawMessages.length - tailCount - 1)
                : rawMessages.length - 1;

            const compressedPrefix =
              tailCount > 0
                ? compressedMessages.slice(
                    0,
                    compressedMessages.length - tailCount
                  )
                : compressedMessages;

            outgoingMessages = rawMessages.map((m, idx) => {
              if (idx === cutIndex) {
                return {
                  ...m,
                  metadata: {
                    ...m.metadata,
                    compressedHistory: structuredClone(compressedPrefix),
                  },
                };
              }
              if (m.metadata && 'compressedHistory' in m.metadata) {
                const { compressedHistory, ...restMeta } = m.metadata;
                return {
                  ...m,
                  metadata:
                    Object.keys(restMeta).length > 0 ? restMeta : undefined,
                };
              }
              return m;
            });
          } else {
            outgoingMessages = wasCompressed ? compressedMessages : rawMessages;
          }

          const modifiedEnvelope = {
            ...envelope,
            request: {
              ...envelope.request,
              messages: outgoingMessages,
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
