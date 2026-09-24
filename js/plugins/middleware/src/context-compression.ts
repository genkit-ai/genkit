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
   * Number of most recent non-system messages to preserve untouched during
   * dynamic overshoot adjustment.
   * @default 4
   */
  preserveRecent: z
    .number()
    .optional()
    .describe('Number of recent messages to preserve. Default: 4.'),

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
            return `[Tool response: ${p.toolResponse.name} → ${stringifyOutput(p.toolResponse.output)}]`;
          return '[other content]';
        })
        .join(' ');
      return `${m.role}: ${parts}`;
    })
    .join('\n');
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
    const basePreserveRecent =
      config?.preserveRecent ?? DEFAULT_PRESERVE_RECENT;
    const preserveSystem = config?.preserveSystem !== false;
    const preserveOriginalMessages = config?.preserveOriginalMessages !== false;
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

    const summarizeConfig = config?.summarize;
    const skipSummarizationThreshold = config?.skipSummarizationThreshold;
    const baseSummaryPreserveRecent =
      summarizeConfig?.preserveRecent ?? DEFAULT_SUMMARIZE_PRESERVE_RECENT;
    const summaryPromptTemplate =
      summarizeConfig?.prompt ?? DEFAULT_SUMMARIZE_PROMPT;
    const summaryModelRef = summarizeConfig?.model;

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

      const groups = new Map<string, { msgIdx: number; partIdx: number }[]>();
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg.role !== 'tool') continue;

        for (let j = 0; j < msg.content.length; j++) {
          const part = msg.content[j];
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
            const prevParts = messages[i - 1].content;
            const positionalPart =
              prevParts[j]?.toolRequest?.name === part.toolResponse.name
                ? prevParts[j]
                : prevParts.find(
                    (p) => p.toolRequest?.name === part.toolResponse?.name
                  );
            if (positionalPart?.toolRequest) {
              toolInput = positionalPart.toolRequest.input;
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
          groups.get(key)!.push({ msgIdx: i, partIdx: j });
        }
      }

      const partsToReplace = new Set<string>();
      for (const occurrences of groups.values()) {
        if (occurrences.length > dedupKeepRecent) {
          const toRemove = occurrences.slice(
            0,
            occurrences.length - dedupKeepRecent
          );
          for (const occ of toRemove) {
            partsToReplace.add(`${occ.msgIdx}-${occ.partIdx}`);
          }
        }
      }

      if (partsToReplace.size === 0) {
        return { messages, deduplicated: 0 };
      }

      let deduplicatedCount = 0;
      const result = messages.map((msg, i) => {
        if (msg.role !== 'tool') return msg;

        let changed = false;
        const newContent = msg.content.map((part, j): Part => {
          if (
            part.toolResponse &&
            partsToReplace.has(`${i}-${j}`) &&
            !hasCompressionFlag(part, 'deduplicated')
          ) {
            deduplicatedCount++;
            changed = true;
            return {
              ...part,
              metadata: withCompressionMetadata(part, { deduplicated: true }),
              toolResponse: {
                ...part.toolResponse,
                output: dedupNotice,
              },
            };
          }
          return part;
        });
        return changed ? { ...msg, content: newContent } : msg;
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

        const isTruncatableMsg =
          includeToolTruncation &&
          Boolean(toolMaxChars) &&
          truncatableMsgIndices.has(mIdx);
        let changed = false;

        const newContent = msg.content.map((part): Part => {
          if (!part.toolResponse) {
            return part;
          }

          // Skip if this part was already truncated to toolResponses.maxChars or deduplicated
          if (
            hasCompressionFlag(part, 'truncated') ||
            hasCompressionFlag(part, 'deduplicated')
          ) {
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
      if (!cap || cap <= 0 || messages.length <= cap) {
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
        cap - systemMessages.length - (noticeConsumesSlot ? 1 : 0)
      );
      let kept = keepCount === 0 ? [] : nonSystemMessages.slice(-keepCount);

      // Strip leading orphaned tool messages so a toolResponse is never separated
      // from the model toolRequest that preceded it.
      while (kept.length > 0 && kept[0].role === 'tool') {
        kept.shift();
      }

      let tailCount: number | undefined;

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
            tailCount = tail.length;
          } else {
            kept = [];
          }
        }
      }

      const finalTailCount = tailCount ?? kept.length;
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
            tailCount: finalTailCount,
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
            tailCount: finalTailCount,
          };
        }
      }

      return {
        messages: [...systemMessages, ...kept],
        dropped,
        noticeInserted,
        tailCount: finalTailCount,
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
        const prompt = summaryPromptTemplate.replaceAll(
          '{conversation}',
          () => conversationText
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
        const activeMessages = resolveCompressedHistory(rawMessages);
        const stampedTokens =
          lastInputTokens ??
          lastReportedInputTokens(activeMessages) ??
          lastReportedInputTokens(rawMessages);
        const estimatedTokens =
          maxInputTokens === Infinity
            ? 0
            : Math.ceil(
                estimateMessageChars(activeMessages) / CHARS_PER_TOKEN_ESTIMATE
              );
        const effectiveTokens = Math.max(stampedTokens ?? 0, estimatedTokens);

        const shouldCompress =
          effectiveTokens > maxInputTokens ||
          (maxMessages !== undefined &&
            maxMessages > 0 &&
            activeMessages.length > maxMessages);

        const hasOversizedToolResponse =
          maxToolResponseChars !== Infinity &&
          activeMessages.some(
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
        } = await ai.run(
          'contextCompression',
          { messageCount: originalCount, effectiveTokens },
          async () => {
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

            if (shouldCompress) {
              // 3. Check if cheap strategies saved enough to skip summarization
              const charsAfterCheap = estimateMessageChars(messages);
              const charsSaved = charsBefore - charsAfterCheap;
              const savingsRatio =
                charsBefore > 0 ? charsSaved / charsBefore : 0;

              const shouldSkipSummarization =
                skipSummarizationThreshold !== undefined &&
                savingsRatio >= skipSummarizationThreshold;

              // 4. Summarization
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

              // 5. Message truncation (as fallback or hard cap)
              const effectiveMaxMessages = maxMessages
                ? Math.min(
                    maxMessages,
                    Math.max(
                      adjustedPreserveRecent + (insertTruncationNotice ? 1 : 0),
                      maxMessages -
                        (basePreserveRecent - adjustedPreserveRecent)
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
          }
        );

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
            summarized:
              summarized || Boolean(latestCompressionMeta?.summarized),
            summarizationSkipped,
          };
          latestCompressionMeta = turnCompressionMeta;
        }

        let outgoingMessages: MessageData[];
        if (wasCompressed && preserveOriginalMessages) {
          let tailCount = 0;
          if (messagesTruncated) {
            tailCount = truncTailCount;
          } else if (summarized) {
            tailCount = summaryTailCount;
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
                  Math.max(0, compressedMessages.length - tailCount)
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
