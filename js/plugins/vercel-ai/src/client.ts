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

import type { ChatTransport, UIMessage, UIMessageChunk } from 'ai';
// Type-only imports are erased at compile time, so importing from `genkit`
// keeps this module browser-safe (no Node-only runtime code is pulled in)
// while staying in sync with the canonical agent protocol types.
import type { AgentOutput, AgentStreamChunk } from 'genkit/beta';
import { applyPatch, streamFlow } from 'genkit/beta/client';

import {
  asRestartInterrupt,
  currentTurnResolvedTools,
  findLastNonEmptyUserMessage,
  mapUIMessageToGenkit,
} from './mapping.js';

export {
  mapGenkitMessageToUI,
  messagesFromSnapshot,
  restartInterrupt,
} from './mapping.js';
export type { ResolvedToolResult, RestartInterruptOutput } from './mapping.js';

export type { ChatTransport, UIMessage, UIMessageChunk };

// A bare RFC-4122 UUID, used to validate `chatId` (which doubles as the Genkit
// `sessionId`). The agent server requires session ids to be bare UUIDs, so we
// validate client-side too for a clearer, earlier error.
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Derives a human-readable error message from an unknown thrown value,
 * preserving as much context as possible.
 *
 * `streamFlow` throws `Error`s whose `.message` already embeds the HTTP status
 * and any Genkit error status/message/details, so those are returned verbatim.
 * Non-Error throwables are stringified (JSON when possible) instead of being
 * collapsed to a generic label, so detail isn't silently lost.
 */
function errorTextFromUnknown(err: unknown): string {
  if (err instanceof Error) {
    return err.message || 'Agent execution failed';
  }
  if (typeof err === 'string') {
    return err || 'Agent execution failed';
  }
  if (err === undefined || err === null) {
    return 'Agent execution failed';
  }
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

/**
 * Builds a `ReadableStream` that emits a single error inside the standard
 * `start`/`error`/`finish` envelope, matching the shape of a normal stream.
 */
function errorStream(errorText: string): ReadableStream<UIMessageChunk> {
  return new ReadableStream<UIMessageChunk>({
    start(controller) {
      controller.enqueue({
        type: 'start',
        messageId: `msg-${crypto.randomUUID()}`,
      });
      controller.enqueue({ type: 'error', errorText });
      controller.enqueue({ type: 'finish' });
      controller.close();
    },
  });
}

// ---------------------------------------------------------------------------
// GenkitChatTransport
// ---------------------------------------------------------------------------

/**
 * Configuration for {@link GenkitChatTransport}.
 */
export interface GenkitChatTransportConfig {
  /** URL of the Genkit agent endpoint (e.g. `/api/weatherAgent`). */
  url: string;

  /**
   * Additional HTTP headers to include in every request.
   * Can be a static object or a function that returns headers (useful for
   * dynamic auth tokens).
   */
  headers?: Record<string, string> | (() => Record<string, string>);
}

/**
 * A Vercel AI SDK `ChatTransport` implementation that communicates with a
 * Genkit agent using Genkit's native streaming protocol via `streamFlow`.
 *
 * Conversation state is **fully server-managed**: the transport sends the
 * `useChat` `id` (the `chatId`) to the agent as its `sessionId`, and the agent
 * persists per-session state in its configured `SessionStore`. Each turn
 * resumes the session's latest snapshot automatically — there is no
 * client-side snapshot bookkeeping to manage or persist.
 *
 * Because the agent server requires session ids to be bare UUIDs, the
 * `chatId` (the `id` you pass to `useChat`) must be a UUID. Generate one with
 * `crypto.randomUUID()`.
 *
 * Interrupt resumes are derived directly from the `messages` the SDK already
 * holds: when the latest turn's assistant message carries resolved tool
 * outputs (e.g. via `addToolResult`), the transport sends them back to the
 * agent as a `resume` payload.
 *
 * @example
 * ```tsx
 * import { useMemo } from 'react';
 * import { useChat } from '@ai-sdk/react';
 * import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';
 *
 * const chatId = useMemo(() => crypto.randomUUID(), []);
 * const { messages, sendMessage } = useChat({
 *   id: chatId,
 *   transport: new GenkitChatTransport({
 *     url: '/api/weatherAgent',
 *   }),
 * });
 * ```
 */
export class GenkitChatTransport implements ChatTransport<UIMessage> {
  private readonly url: string;
  private readonly headersConfig?:
    | Record<string, string>
    | (() => Record<string, string>);

  constructor(config: GenkitChatTransportConfig) {
    this.url = config.url;
    this.headersConfig = config.headers;
  }

  private resolveHeaders(
    extra?: Record<string, string> | Headers
  ): Record<string, string> | undefined {
    const base =
      typeof this.headersConfig === 'function'
        ? this.headersConfig()
        : this.headersConfig;
    // Normalize Headers objects to plain records.
    let extraRecord: Record<string, string> | undefined;
    if (extra instanceof Headers) {
      extraRecord = {};
      extra.forEach((value, key) => {
        extraRecord![key] = value;
      });
    } else {
      extraRecord = extra;
    }
    if (!base && !extraRecord) return undefined;
    return { ...base, ...extraRecord };
  }

  // The `trigger` parameter is either 'submit-message' (a new user turn) or
  // 'regenerate-message' (re-run the last assistant response). Because state
  // is server-managed by `sessionId`, regeneration is treated as a fresh turn
  // from the current session state (there is no client-side snapshot pointer
  // to rewind to).
  async sendMessages({
    trigger,
    chatId,
    messages,
    abortSignal,
    headers,
  }: {
    trigger: 'submit-message' | 'regenerate-message';
    chatId: string;
    messageId: string | undefined;
    messages: UIMessage[];
    abortSignal: AbortSignal | undefined;
    headers?: Record<string, string> | Headers;
    body?: object;
    metadata?: unknown;
  }): Promise<ReadableStream<UIMessageChunk>> {
    // `chatId` doubles as the Genkit session id and must be a bare UUID.
    if (!UUID_PATTERN.test(chatId)) {
      return errorStream(
        `Invalid chatId '${chatId}': the useChat 'id' is used as the Genkit ` +
          `sessionId and must be a UUID. Generate one with crypto.randomUUID().`
      );
    }

    const isRegenerate = trigger === 'regenerate-message';

    // Detect an interrupt resume from the message history alone: the current
    // turn's assistant message(s) carry resolved tool outputs that have not
    // been followed by a new user message. Regeneration never resumes an
    // interrupt — it replays from the current session state.
    const resolvedTools = isRegenerate
      ? []
      : currentTurnResolvedTools(messages);
    const isResume = resolvedTools.length > 0;

    // The session id is always sent so the agent resumes (or seeds) the
    // server-managed session for this chat.
    const init: Record<string, unknown> = { sessionId: chatId };
    let agentInput: Record<string, unknown>;

    // Tool call ids that were resolved with a user-supplied *response* (not a
    // restart). Their outputs are already on the client, so the transport
    // suppresses re-emitting them. Restarted tools, by contrast, re-execute
    // server-side and produce a *fresh* output that the client must receive.
    const respondedInterruptIds = new Set<string>();

    if (isResume) {
      // Split resolutions into restarts (re-run the tool) and responds
      // (user-supplied output). A single resume turn can carry both.
      const respond: Array<Record<string, unknown>> = [];
      const restart: Array<Record<string, unknown>> = [];
      for (const tr of resolvedTools) {
        const asRestart = asRestartInterrupt(tr.result);
        if (asRestart) {
          // A restart re-runs the tool with the *original* input (the server
          // validates it matches the interrupted request exactly) and passes
          // any metadata through as the tool's `resumed` value.
          restart.push({
            toolRequest: {
              name: tr.toolName,
              ref: tr.toolCallId,
              input: tr.input,
            },
            metadata: { resumed: asRestart.metadata ?? true },
          });
        } else {
          respondedInterruptIds.add(tr.toolCallId);
          respond.push({
            toolResponse: {
              name: tr.toolName,
              ref: tr.toolCallId,
              output: tr.result,
            },
          });
        }
      }
      agentInput = {
        resume: {
          ...(respond.length > 0 && { respond }),
          ...(restart.length > 0 && { restart }),
        },
      };
    } else {
      const lastUserMsg = findLastNonEmptyUserMessage(messages);
      if (!lastUserMsg) {
        return errorStream('No user message found');
      }

      const genkitMsg = mapUIMessageToGenkit(lastUserMsg);
      agentInput = { message: genkitMsg };
    }

    // Call streamFlow — Genkit's native streaming protocol.
    const response = streamFlow<AgentOutput, AgentStreamChunk>({
      url: this.url,
      input: agentInput,
      init,
      headers: this.resolveHeaders(headers),
      abortSignal,
    });

    // Track whether the stream was aborted so we can emit a clean abort chunk.
    let aborted = false;

    return new ReadableStream<UIMessageChunk>({
      start: async (controller) => {
        // Listen for abort to break out of the streaming loop cleanly.
        const onAbort = () => {
          aborted = true;
        };
        abortSignal?.addEventListener('abort', onAbort, { once: true });

        // Streaming state machine.
        let started = false;
        let textBlockId: string | null = null;
        let reasoningBlockId: string | null = null;
        let inStep = false;
        // Locally tracked agent custom state, kept live by applying each
        // streamed RFC 6902 JSON Patch (`chunk.customPatch`). The first patch
        // of a turn is a whole-document replace that re-bases us onto the
        // server's current baseline.
        let customState: unknown;
        const emittedToolInputs = new Set<string>();

        // Deterministic fallback for tool calls that arrive without a `ref`.
        // The model may omit refs, in which case the request and its matching
        // response must still share the same `toolCallId`.
        //
        // We pair each ref-less response with the *oldest* unmatched ref-less
        // request for the same tool name (FIFO by arrival order). Allocating a
        // fresh id per request — rather than reusing one id per tool name —
        // means two ref-less calls to the *same* tool in one turn don't share
        // an id and cross-wire each other's request/response. (In practice
        // Genkit tool requests carry refs, so this is a safety net.)
        const pendingRefByToolName = new Map<string, string[]>();
        const correlateRequestId = (name: string): string => {
          const id = crypto.randomUUID();
          const queue = pendingRefByToolName.get(name) ?? [];
          queue.push(id);
          pendingRefByToolName.set(name, queue);
          return id;
        };
        const correlateResponseId = (name: string): string => {
          const queue = pendingRefByToolName.get(name);
          if (queue && queue.length > 0) {
            return queue.shift()!;
          }
          // No matching request seen (shouldn't happen) — mint a new id.
          return crypto.randomUUID();
        };

        const enqueue = (chunk: UIMessageChunk) => {
          try {
            controller.enqueue(chunk);
          } catch {
            // Controller already closed (e.g. consumer cancelled).
          }
        };

        const ensureStarted = () => {
          if (!started) {
            enqueue({
              type: 'start',
              messageId: `msg-${crypto.randomUUID()}`,
            });
            started = true;
          }
        };

        const ensureStep = () => {
          if (!inStep) {
            enqueue({ type: 'start-step' });
            inStep = true;
          }
        };

        const closeTextBlock = () => {
          if (textBlockId) {
            enqueue({ type: 'text-end', id: textBlockId });
            textBlockId = null;
          }
        };

        const closeReasoningBlock = () => {
          if (reasoningBlockId) {
            enqueue({ type: 'reasoning-end', id: reasoningBlockId });
            reasoningBlockId = null;
          }
        };

        const closeStep = () => {
          if (inStep) {
            enqueue({ type: 'finish-step' });
            inStep = false;
          }
        };

        try {
          for await (const chunk of response.stream) {
            // If aborted during iteration, break out cleanly.
            if (aborted) break;

            // Agent custom-state update. The runtime auto-emits a `customPatch`
            // (RFC 6902 JSON Patch) whenever the session's custom state
            // mutates. Apply it to our locally tracked copy and surface the
            // full, post-patch state as a transient `data-custom` UI chunk so
            // the React app can render live progress. Must run BEFORE the
            // `modelChunk` early-continue below, since custom-only chunks carry
            // no `modelChunk`.
            if (chunk?.customPatch) {
              customState = applyPatch(customState, chunk.customPatch);
              ensureStarted();
              enqueue({
                type: 'data-custom',
                data: customState,
                transient: true,
              } as UIMessageChunk);
            }

            const mc = chunk?.modelChunk;
            if (!mc?.content) continue;

            ensureStarted();

            for (const part of mc.content) {
              // Reasoning delta (model's "thinking"). Genkit reasoning parts
              // use the `reasoning` field; map them to AI SDK reasoning chunks
              // so they can be surfaced separately from the final answer.
              if (typeof part.reasoning === 'string' && part.reasoning !== '') {
                ensureStep();
                if (!reasoningBlockId) {
                  reasoningBlockId = `reasoning-${crypto.randomUUID()}`;
                  enqueue({ type: 'reasoning-start', id: reasoningBlockId });
                }
                enqueue({
                  type: 'reasoning-delta',
                  id: reasoningBlockId,
                  delta: part.reasoning,
                });
              }

              // Text delta
              if (part.text !== undefined && part.text !== '') {
                // Reasoning precedes the answer; close any open reasoning
                // block before streaming visible text.
                closeReasoningBlock();
                ensureStep();
                if (!textBlockId) {
                  textBlockId = `text-${crypto.randomUUID()}`;
                  enqueue({ type: 'text-start', id: textBlockId });
                }
                enqueue({
                  type: 'text-delta',
                  id: textBlockId,
                  delta: part.text,
                });
              }

              // Tool request (tool call from model)
              if (part.toolRequest) {
                const tr = part.toolRequest;
                const toolCallId = tr.ref ?? correlateRequestId(tr.name);
                if (!emittedToolInputs.has(toolCallId)) {
                  emittedToolInputs.add(toolCallId);
                  closeReasoningBlock();
                  closeTextBlock();
                  ensureStep();
                  enqueue({
                    type: 'tool-input-start',
                    toolCallId,
                    toolName: tr.name,
                  });
                  enqueue({
                    type: 'tool-input-available',
                    toolCallId,
                    toolName: tr.name,
                    input: tr.input,
                  });
                }
              }

              // Tool response (tool executed by agent)
              if (part.toolResponse) {
                const tr = part.toolResponse;
                const toolCallId = tr.ref ?? correlateResponseId(tr.name);

                // Skip interrupt responses the client already supplied — it
                // has the result and re-emitting would confuse the UI SDK.
                // Restarted tools are NOT skipped: they re-execute server-side
                // and produce a fresh output the client must receive.
                if (!respondedInterruptIds.has(toolCallId)) {
                  enqueue({
                    type: 'tool-output-available',
                    toolCallId,
                    output: tr.output,
                  });
                  closeStep();
                }
              }
            }

            // Handle turn end signals.
            if (chunk?.turnEnd) {
              closeReasoningBlock();
              closeTextBlock();
              closeStep();
            }
          }

          // If aborted, emit an abort chunk and skip output processing.
          if (aborted) {
            ensureStarted();
            closeReasoningBlock();
            closeTextBlock();
            closeStep();
            enqueue({ type: 'abort', reason: 'Request was aborted' });
            enqueue({ type: 'finish' });
            try {
              controller.close();
            } catch {
              /* already closed */
            }
            return;
          }

          // Wait for final output (drains the stream / surfaces server errors).
          await response.output;
        } catch (err: unknown) {
          // If the error is an AbortError, treat it as an abort, not an error.
          if (err instanceof DOMException && err.name === 'AbortError') {
            ensureStarted();
            closeReasoningBlock();
            closeTextBlock();
            closeStep();
            enqueue({ type: 'abort', reason: 'Request was aborted' });
          } else {
            ensureStarted();
            // `streamFlow` throws plain `Error`s whose message already embeds
            // the HTTP status and any Genkit error status/message/details
            // (e.g. `"Server returned: 500: ..."` or `"<status>: <message>\n
            // <details>"`), so `err.message` preserves that context. For
            // non-Error throwables, stringify rather than collapsing to a
            // generic label so detail isn't lost.
            enqueue({
              type: 'error',
              errorText: errorTextFromUnknown(err),
            });
          }
        } finally {
          closeReasoningBlock();
          closeTextBlock();
          closeStep();
          ensureStarted();
          enqueue({ type: 'finish' });
          abortSignal?.removeEventListener('abort', onAbort);
          try {
            controller.close();
          } catch {
            /* already closed */
          }
        }
      },

      cancel: () => {
        // The ReadableStream consumer cancelled (e.g. useChat unmounted).
        // The abort signal (if provided) will handle the fetch cancellation.
        aborted = true;
      },
    });
  }

  /**
   * Genkit transport does not support stream reconnection.
   *
   * Genkit has a durable streaming feature on the server side; support
   * for reconnection may be added in a future version.
   *
   * @returns Always returns `null`.
   */
  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    return null;
  }
}
