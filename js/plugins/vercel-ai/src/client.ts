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
import { streamFlow } from 'genkit/beta/client';
import {
  asRestartInterrupt,
  extractResolvedToolResults,
  findLastUserMessage,
  mapUIMessageToGenkit,
} from './mapping.js';
import type { PendingInterrupt } from './store.js';
import { InMemorySnapshotStore, type SnapshotStore } from './store.js';

export { restartInterrupt } from './mapping.js';
export type { ResolvedToolResult, RestartInterruptOutput } from './mapping.js';
export {
  InMemorySnapshotStore,
  LocalStorageSnapshotStore,
  type ChatSnapshot,
  type PendingInterrupt,
  type SnapshotStore,
} from './store.js';
export type { ChatTransport, UIMessage, UIMessageChunk };

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

  /**
   * Storage for per-chat snapshot state (snapshot ids + interrupt flag).
   *
   * Defaults to {@link InMemorySnapshotStore}, which keeps state for the
   * lifetime of the transport instance and loses it on page reload. Pass
   * {@link LocalStorageSnapshotStore} (or a custom {@link SnapshotStore}) to
   * persist multi-turn continuity across reloads.
   */
  store?: SnapshotStore;
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
 * chats (each identified by `chatId`). Per-chat snapshot state is held in a
 * pluggable {@link SnapshotStore} (in-memory by default). Call
 * {@link clearChat} to release state for a chat that is no longer needed.
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
export class GenkitChatTransport implements ChatTransport<UIMessage> {
  private readonly url: string;
  private readonly headersConfig?:
    | Record<string, string>
    | (() => Record<string, string>);

  /** Pluggable per-chat snapshot state (snapshot ids + interrupt flag). */
  private readonly store: SnapshotStore;

  constructor(config: GenkitChatTransportConfig) {
    this.url = config.url;
    this.headersConfig = config.headers;
    this.store = config.store ?? new InMemorySnapshotStore();
  }

  /**
   * Releases internal state for a given chat.
   *
   * Call this when a chat is discarded (e.g. the user closes a tab or
   * navigates away) to avoid holding stale snapshot references.
   */
  async clearChat(chatId: string): Promise<void> {
    await this.store.delete(chatId);
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
  // 'regenerate-message' (re-run the last assistant response). Because the
  // agent is stateful (snapshot-based), regeneration re-runs from the
  // *previous* snapshot — the conversation state from before the last turn —
  // so the final turn is produced again instead of appending a new one.
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
    const chatSnapshot = (await this.store.get(chatId)) ?? {};
    const isRegenerate = trigger === 'regenerate-message';

    // For a regeneration, resume from the snapshot *before* the last turn so
    // the final response is produced again from the prior state. Otherwise
    // continue from the current snapshot.
    const snapshotId = isRegenerate
      ? chatSnapshot.previousSnapshotId
      : chatSnapshot.snapshotId;
    const wasInterrupted = chatSnapshot.interrupted ?? false;
    const pendingInterrupts = chatSnapshot.pendingInterrupts ?? [];

    // Detect whether this is an interrupt resume or a normal message.
    // Regeneration never resumes an interrupt — it replays a completed turn.
    //
    // We only treat resolved tool results as interrupt resolutions when their
    // `toolCallId` matches a tool request the agent actually paused on. This
    // prevents an auto-executed tool's output (also `output-available`) from
    // being misclassified as an interrupt response. If we have no specific
    // pending refs (older snapshots), fall back to all resolved results.
    const pendingIds = new Set(pendingInterrupts.map((p) => p.toolCallId));
    const allResolved =
      wasInterrupted && !isRegenerate
        ? extractResolvedToolResults(messages)
        : [];
    const resolvedTools =
      pendingIds.size > 0
        ? allResolved.filter((tr) => pendingIds.has(tr.toolCallId))
        : allResolved;
    const isResume = resolvedTools.length > 0 && !!snapshotId;

    // Build AgentInput + AgentInit.
    let agentInput: Record<string, unknown>;
    let init: Record<string, unknown>;

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
    const response = streamFlow<AgentOutput, AgentStreamChunk>({
      url: this.url,
      input: agentInput,
      init,
      headers: this.resolveHeaders(headers),
      abortSignal,
    });

    // References captured by the ReadableStream closure.
    const store = this.store;

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
        const emittedToolInputs = new Set<string>();
        // Deterministic fallback for tool calls that arrive without a `ref`.
        // The model may omit refs, in which case the request and its matching
        // response must still share the same `toolCallId`. We assign one id
        // per tool name (in arrival order) so the response can correlate.
        const refByToolName = new Map<string, string>();
        const correlateToolCallId = (name: string, ref?: string): string => {
          if (ref) return ref;
          let id = refByToolName.get(name);
          if (!id) {
            id = crypto.randomUUID();
            refByToolName.set(name, id);
          }
          return id;
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
                const toolCallId = correlateToolCallId(tr.name, tr.ref);
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
                const toolCallId = correlateToolCallId(tr.name, tr.ref);

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

          // Wait for final output.
          const output = await response.output;

          // Detect interrupts — if the last message has unresolved tool
          // requests, the agent was interrupted. Capture the specific pending
          // tool requests (ref + name) so the next turn can build a precise
          // resume payload and ignore unrelated tool outputs in the history.
          const newPendingInterrupts: PendingInterrupt[] = [];
          const lastMsg = output?.message;
          if (lastMsg?.content) {
            for (const p of lastMsg.content) {
              const ref = p.toolRequest?.ref;
              const isResolved = lastMsg.content.some(
                (r) => r.toolResponse?.ref === p.toolRequest?.ref
              );
              if (p.toolRequest && ref && !isResolved) {
                newPendingInterrupts.push({
                  toolCallId: ref,
                  toolName: p.toolRequest.name,
                });
              }
            }
          }
          const wasAgentInterrupted = newPendingInterrupts.length > 0;

          // Persist snapshot bookkeeping for the next turn. On a normal turn
          // the prior `snapshotId` becomes `previousSnapshotId` so a later
          // `regenerate-message` can re-run from before this turn. On a
          // regeneration we replace the current snapshot but keep the same
          // `previousSnapshotId` so it stays repeatable.
          if (output?.snapshotId) {
            await store.set(chatId, {
              snapshotId: output.snapshotId,
              previousSnapshotId: isRegenerate
                ? chatSnapshot.previousSnapshotId
                : chatSnapshot.snapshotId,
              interrupted: wasAgentInterrupted,
              pendingInterrupts: wasAgentInterrupted
                ? newPendingInterrupts
                : [],
            });
          } else {
            // No new snapshot (e.g. interrupt without a fresh id) — preserve
            // existing snapshot ids but update the interrupt flag.
            await store.set(chatId, {
              ...chatSnapshot,
              interrupted: wasAgentInterrupted,
              pendingInterrupts: wasAgentInterrupted
                ? newPendingInterrupts
                : [],
            });
          }
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
            enqueue({
              type: 'error',
              errorText:
                err instanceof Error ? err.message : 'Agent execution failed',
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
