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

import type { ChatTransport, UIMessageChunk } from 'ai';
import { streamFlow } from 'genkit/beta/client';
import {
  extractResolvedToolResults,
  findLastUserMessage,
  mapUIMessageToGenkit,
  type UIMessage,
} from './mapping.js';

export type { ChatTransport, UIMessageChunk };

// ---------------------------------------------------------------------------
// Agent wire-format types
//
// These mirror the server-side AgentStreamChunk / AgentOutput types from
// genkit/ai/src/agent.ts.  Defined locally so this module stays browser-safe
// without importing server-only code.
//
// Derived from genkit@1.x agent protocol.
// ---------------------------------------------------------------------------

/** A single content part within a model message (wire format). */
interface ContentPart {
  text?: string;
  toolRequest?: { name: string; input: unknown; ref?: string };
  toolResponse?: { name: string; output: unknown; ref?: string };
  media?: { url: string; contentType?: string };
}

/** Wire format for model response chunks streamed during agent execution. */
interface ModelChunkWire {
  content: ContentPart[];
}

/** Wire format for agent stream chunks (mirrors `AgentStreamChunk`). */
interface AgentStreamChunkWire {
  modelChunk?: ModelChunkWire;
  turnEnd?: { snapshotId?: string };
  status?: unknown;
  artifact?: unknown;
}

/** Wire format for agent output returned after the stream completes. */
interface AgentOutputWire {
  snapshotId?: string;
  message?: { role: string; content: ContentPart[] };
  state?: unknown;
  artifacts?: unknown[];
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
 * This transport maintains a client-side mapping of `chatId → snapshotId`
 * for multi-turn session continuity, eliminating the need for a server-side
 * `ChatSessionStore`.
 *
 * **Lifecycle:** A single transport instance can manage multiple independent
 * chats (each identified by `chatId`). Internal state per chat is lightweight
 * (two string-keyed Map entries). Call {@link clearChat} to release state for
 * a chat that is no longer needed.
 *
 * @example
 * ```tsx
 * import { useChat } from '@ai-sdk/react';
 * import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';
 *
 * const { messages, sendMessage } = useChat({
 *   transport: new GenkitChatTransport({
 *     url: '/api/weatherAgent',
 *   }),
 * });
 * ```
 */
export class GenkitChatTransport {
  private readonly url: string;
  private readonly headersConfig?:
    | Record<string, string>
    | (() => Record<string, string>);

  /** Client-side chatId → snapshotId mapping. */
  private snapshots = new Map<string, string>();
  /** Tracks whether a chat is in an interrupted state. */
  private interrupted = new Map<string, boolean>();

  constructor(config: GenkitChatTransportConfig) {
    this.url = config.url;
    this.headersConfig = config.headers;
  }

  /**
   * Releases internal state for a given chat.
   *
   * Call this when a chat is discarded (e.g. the user closes a tab or
   * navigates away) to avoid holding stale snapshot references.
   */
  clearChat(chatId: string): void {
    this.snapshots.delete(chatId);
    this.interrupted.delete(chatId);
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

  // Note: The `trigger` parameter accepts both 'submit-message' and
  // 'regenerate-message'. Currently both are handled identically — the
  // agent is stateful (snapshot-based), so a regeneration simply produces
  // a new response from the same conversation state. A future version may
  // add specific regeneration behavior (e.g. rewinding one turn).
  async sendMessages({
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
    const snapshotId = this.snapshots.get(chatId);
    const wasInterrupted = this.interrupted.get(chatId) ?? false;

    // Detect whether this is an interrupt resume or a normal message.
    const resolvedTools = wasInterrupted
      ? extractResolvedToolResults(messages)
      : [];
    const isResume = resolvedTools.length > 0 && !!snapshotId;

    // Build AgentInput + AgentInit.
    let agentInput: Record<string, unknown>;
    let init: Record<string, unknown>;

    if (isResume) {
      agentInput = {
        resume: {
          respond: resolvedTools.map((tr) => ({
            toolResponse: {
              name: tr.toolName,
              ref: tr.toolCallId,
              output: tr.result,
            },
          })),
        },
      };
      init = { snapshotId };
    } else {
      const lastUserMsg = findLastUserMessage(messages);
      if (!lastUserMsg) {
        // Return a stream with an error chunk wrapped in the standard
        // start/finish envelope for consistency with normal streams.
        return new ReadableStream<UIMessageChunk>({
          start(controller) {
            controller.enqueue({
              type: 'start',
              messageId: `msg-${crypto.randomUUID()}`,
            });
            controller.enqueue({
              type: 'error',
              errorText: 'No user message found',
            });
            controller.enqueue({ type: 'finish' });
            controller.close();
          },
        });
      }

      const genkitMsg = mapUIMessageToGenkit(lastUserMsg);
      agentInput = { messages: [genkitMsg] };
      init = snapshotId ? { snapshotId } : {};
    }

    // Call streamFlow — Genkit's native streaming protocol.
    const response = streamFlow<AgentOutputWire, AgentStreamChunkWire>({
      url: this.url,
      input: agentInput,
      init,
      headers: this.resolveHeaders(headers),
      abortSignal: abortSignal ?? undefined,
    });

    // Track resolved interrupt tool IDs so we don't re-emit their outputs.
    const resolvedInterruptIds = new Set(
      resolvedTools.map((tr) => tr.toolCallId)
    );

    // References captured by the ReadableStream closure.
    const snapshots = this.snapshots;
    const interrupted = this.interrupted;

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
        let inStep = false;
        const emittedToolInputs = new Set<string>();

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

            const mc = chunk?.modelChunk;
            if (!mc?.content) continue;

            ensureStarted();

            for (const part of mc.content) {
              // Text delta
              if (part.text !== undefined && part.text !== '') {
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
                const toolCallId = tr.ref || crypto.randomUUID();
                if (!emittedToolInputs.has(toolCallId)) {
                  emittedToolInputs.add(toolCallId);
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
                const toolCallId = tr.ref || crypto.randomUUID();

                // Skip resolved interrupt responses — the client already has
                // the result and re-emitting would confuse the UI SDK.
                if (!resolvedInterruptIds.has(toolCallId)) {
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
              closeTextBlock();
              closeStep();
            }
          }

          // If aborted, emit an abort chunk and skip output processing.
          if (aborted) {
            ensureStarted();
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

          // Wait for final output.
          const output = await response.output;

          // Update client-side chatId → snapshotId mapping.
          if (output?.snapshotId) {
            snapshots.set(chatId, output.snapshotId);
          }

          // Detect interrupts — if the last message has unresolved tool
          // requests, the agent was interrupted.
          let wasAgentInterrupted = false;
          const lastMsg = output?.message;
          if (lastMsg?.content) {
            const pendingToolCalls = lastMsg.content.filter(
              (p) =>
                p.toolRequest &&
                !lastMsg.content.some(
                  (r) => r.toolResponse?.ref === p.toolRequest?.ref
                )
            );
            if (pendingToolCalls.length > 0) {
              wasAgentInterrupted = true;
            }
          }
          interrupted.set(chatId, wasAgentInterrupted);
        } catch (err: unknown) {
          // If the error is an AbortError, treat it as an abort, not an error.
          if (
            err instanceof DOMException &&
            err.name === 'AbortError'
          ) {
            ensureStarted();
            closeTextBlock();
            closeStep();
            enqueue({ type: 'abort', reason: 'Request was aborted' });
          } else {
            ensureStarted();
            enqueue({
              type: 'error',
              errorText:
                err instanceof Error
                  ? err.message
                  : 'Agent execution failed',
            });
          }
        } finally {
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
