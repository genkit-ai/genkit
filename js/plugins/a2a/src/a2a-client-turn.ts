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

import type {
  Message as A2AMessage,
  Part as A2APart,
  AgentCard,
  MessageSendParams,
  Task,
  TaskArtifactUpdateEvent,
  TaskStatusUpdateEvent,
} from '@a2a-js/sdk';
import { A2AClient } from '@a2a-js/sdk/client';
import type {
  AgentFinishReason,
  AgentInput,
  MessageData,
  Part,
} from 'genkit/beta';

import {
  a2aMessageToGenkit,
  a2aPartsToGenkit,
  a2aStateToFinishReason,
  genkitMessageToA2AParts,
  genkitResumeToA2AParts,
} from './mapping.js';

/**
 * Connection options for talking to a remote A2A agent. Shared by
 * {@link createA2aClient} and, in the future, a `remoteA2aAgent` transport, so
 * both the registered-agent (`defineA2aAgent`) and pure-client entry points
 * accept the same shape.
 */
export interface A2aConnectionOptions {
  /**
   * Base URL of the remote A2A agent. Its agent card is resolved from
   * `${url}/.well-known/agent-card.json` (unless `agentCardPath` overrides it).
   */
  url?: string;
  /** A pre-fetched {@link AgentCard} to construct the client from directly. */
  card?: AgentCard;
  /** Path to the agent card, relative to `url`. Defaults to the well-known path. */
  agentCardPath?: string;
  /**
   * Extra HTTP headers for every request (e.g. auth). A function is invoked per
   * request so short-lived tokens can be refreshed.
   */
  headers?:
    | Record<string, string>
    | (() => Record<string, string> | Promise<Record<string, string>>);
  /**
   * Custom fetch implementation. When omitted the global `fetch` is used
   * (wrapped to inject `headers`).
   */
  fetchImpl?: typeof fetch;
}

/**
 * The outcome of a single remote A2A turn, as consumed by both
 * `defineA2aAgent` and a future `remoteA2aAgent`.
 */
export interface A2aTurnResult {
  /** The agent's final message for this turn (mapped to Genkit). */
  message?: MessageData;
  /** The Genkit finish reason derived from the terminal A2A task state. */
  finishReason: AgentFinishReason;
  /**
   * The A2A task this turn produced/updated. The caller persists it per
   * context so a follow-up turn can resume an `input-required` task.
   */
  taskId?: string;
  /** Set when `finishReason` is `failed`; carries the error text. */
  error?: { message: string };
}

/**
 * Resolves `headers` (static or a function) to a plain object, or `undefined`.
 */
async function resolveHeaders(
  headers: A2aConnectionOptions['headers']
): Promise<Record<string, string> | undefined> {
  if (!headers) return undefined;
  if (typeof headers === 'function') return headers();
  return headers;
}

/**
 * Wraps a fetch implementation so it injects (freshly resolved) `headers` on
 * every request. Returns the base implementation unchanged when there are no
 * headers to inject.
 */
function withHeaders(
  base: typeof fetch,
  headers: A2aConnectionOptions['headers']
): typeof fetch {
  if (!headers) return base;
  return (async (input: any, init?: any) => {
    const extra = await resolveHeaders(headers);
    const merged = new Headers(init?.headers as HeadersInit | undefined);
    for (const [k, v] of Object.entries(extra ?? {})) {
      merged.set(k, v);
    }
    return base(input, { ...init, headers: merged });
  }) as typeof fetch;
}

/**
 * Creates an {@link A2AClient} for a remote A2A agent and returns it alongside
 * the resolved {@link AgentCard} (useful for deriving the Genkit agent's name /
 * description).
 */
export async function createA2aClient(
  options: A2aConnectionOptions
): Promise<{ client: A2AClient; card: AgentCard }> {
  const baseFetch = options.fetchImpl ?? fetch;
  const fetchImpl = withHeaders(baseFetch, options.headers);

  if (options.card) {
    const client = new A2AClient(options.card, { fetchImpl });
    return { client, card: options.card };
  }

  if (!options.url) {
    throw new Error(
      'defineA2aAgent requires either `url` (the remote A2A agent endpoint) or `card`.'
    );
  }

  const client = new A2AClient(options.url, {
    fetchImpl,
    ...(options.agentCardPath && { agentCardPath: options.agentCardPath }),
  });
  const card = await client.getAgentCard();
  return { client, card };
}

type A2AStreamEvent =
  | A2AMessage
  | Task
  | TaskStatusUpdateEvent
  | TaskArtifactUpdateEvent;

/**
 * Builds the A2A {@link MessageSendParams} for a Genkit {@link AgentInput},
 * setting `contextId` (from the Genkit sessionId) and, when resuming a paused
 * task, `taskId`.
 */
function buildSendParams(
  input: AgentInput,
  ctx: { contextId: string; taskId?: string }
): MessageSendParams {
  let parts: A2APart[];
  if (input.resume) {
    parts = genkitResumeToA2AParts(input.resume);
  } else if (input.message) {
    parts = genkitMessageToA2AParts(input.message);
  } else {
    parts = [];
  }

  const message: A2AMessage = {
    kind: 'message',
    messageId: crypto.randomUUID(),
    role: 'user',
    parts,
    contextId: ctx.contextId,
    ...(ctx.taskId && { taskId: ctx.taskId }),
  };

  return { message };
}

/**
 * Extracts the interrupt/final message parts from a terminal A2A status update
 * (`status.message`), mapping them back to Genkit parts.
 */
function statusMessageToGenkit(
  status: Task['status']
): MessageData | undefined {
  if (!status.message) return undefined;
  const content = a2aPartsToGenkit(status.message.parts);
  if (content.length === 0) return undefined;
  return { role: 'model', content };
}

/**
 * Drives a single turn against a remote A2A agent and translates its event
 * stream into Genkit terms.
 *
 * This is the transport-neutral core shared by `defineA2aAgent` (which adapts
 * it into a `defineCustomAgent` handler) and any future `remoteA2aAgent`
 * (which would adapt it into an `AgentTransport`). It:
 *
 * 1. builds the A2A `MessageSendParams` from the Genkit {@link AgentInput}
 *    (fresh message or resume payload), keyed by `ctx.contextId`;
 * 2. yields a Genkit {@link Part} array per streamed `artifact-update` (the
 *    caller emits these as `modelChunk`s); and
 * 3. returns an {@link A2aTurnResult} derived from the terminal status update.
 */
export async function* runA2aTurn(
  client: A2AClient,
  input: AgentInput,
  ctx: { contextId: string; taskId?: string; abortSignal?: AbortSignal }
): AsyncGenerator<Part[], A2aTurnResult, void> {
  const params = buildSendParams(input, ctx);

  let taskId = ctx.taskId;
  // Accumulates streamed artifact parts as the fallback final message when the
  // terminal status carries no message of its own.
  const accumulated: Part[] = [];
  let result: A2aTurnResult | undefined;

  const stream = client.sendMessageStream(params) as AsyncGenerator<
    A2AStreamEvent,
    void,
    undefined
  >;

  for await (const event of stream) {
    if (ctx.abortSignal?.aborted) {
      return { finishReason: 'aborted', taskId };
    }

    switch (event.kind) {
      case 'task': {
        taskId = event.id;
        break;
      }
      case 'artifact-update': {
        const parts = a2aPartsToGenkit(event.artifact.parts);
        if (parts.length > 0) {
          accumulated.push(...parts);
          yield parts;
        }
        break;
      }
      case 'status-update': {
        if (event.taskId) taskId = event.taskId;
        if (event.final) {
          const finishReason = a2aStateToFinishReason(event.status.state);
          const message =
            statusMessageToGenkit(event.status) ??
            (accumulated.length > 0
              ? { role: 'model' as const, content: accumulated }
              : undefined);
          result = {
            finishReason,
            taskId,
            ...(message && { message }),
            ...(finishReason === 'failed' && {
              error: {
                message:
                  textOf(event.status.message) ?? 'Remote A2A agent failed.',
              },
            }),
          };
        }
        break;
      }
      case 'message': {
        // Non-streaming style result: a direct agent message. Treat as the
        // completed final message.
        const message = a2aMessageToGenkit(event);
        result = { finishReason: 'stop', taskId, message };
        break;
      }
      default:
        break;
    }
  }

  if (result) return result;

  // Stream ended without a terminal status. Fall back to whatever we streamed.
  return {
    finishReason: 'stop',
    taskId,
    ...(accumulated.length > 0 && {
      message: { role: 'model', content: accumulated },
    }),
  };
}

/**
 * Joins the text of an A2A message's parts (used for error text extraction).
 */
function textOf(message?: A2AMessage): string | undefined {
  if (!message) return undefined;
  const text = message.parts
    .map((p) => (p.kind === 'text' ? p.text : ''))
    .join('')
    .trim();
  return text.length > 0 ? text : undefined;
}
