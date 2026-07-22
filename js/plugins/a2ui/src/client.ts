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

/**
 * Browser-safe client helpers for consuming an A2UI-enabled Genkit agent.
 *
 * This module has NO Node dependencies. It builds directly on Genkit's native
 * streaming HTTP protocol via `streamFlow` from `genkit/beta/client`, filters
 * A2UI parts out of the agent stream, and hands whole envelopes to whatever
 * renderer you use (e.g. `@a2ui/web_core`'s `MessageProcessor`).
 *
 * @module
 */

// Type-only import keeps this module browser-safe (erased at compile time).
import type { AgentOutput, AgentStreamChunk } from 'genkit/beta';
import { streamFlow } from 'genkit/beta/client';
import { a2uiEnvelopes } from './part.js';
import type { A2uiClientAction, A2uiEnvelope } from './types.js';

export { a2uiEnvelopes, a2uiPart, isA2uiPart } from './part.js';
export {
  A2UI_MIME_TYPE,
  A2UI_VERSION,
  BASIC_CATALOG_ID,
  type A2uiClientAction,
  type A2uiComponent,
  type A2uiEnvelope,
  type A2uiPart,
} from './types.js';

/** A single event yielded while streaming an A2UI agent turn. */
export type A2uiStreamEvent =
  | { type: 'text'; text: string }
  | { type: 'envelopes'; envelopes: A2uiEnvelope[] };

/** Options for {@link streamA2uiAgent}. */
export interface StreamA2uiAgentOptions {
  /** URL of the Genkit agent/flow endpoint (e.g. `/uiAgent`). */
  url: string;
  /** The user message to send (Genkit message data or a plain string). */
  message: string | Record<string, unknown>;
  /** Server-managed session id (a UUID). */
  sessionId?: string;
  /** Extra HTTP headers. */
  headers?: Record<string, string>;
  /** Abort signal to cancel the request. */
  abortSignal?: AbortSignal;
}

/** Normalizes a string message into a Genkit user message. */
function toMessage(message: string | Record<string, unknown>) {
  if (typeof message === 'string') {
    return { role: 'user', content: [{ text: message }] };
  }
  return message;
}

/**
 * Streams an A2UI-enabled Genkit agent, yielding prose text deltas and complete
 * A2UI envelope batches in the order they arrive. Feed the envelopes to your
 * renderer's message processor.
 *
 * @example
 * ```ts
 * import { MessageProcessor } from '@a2ui/web_core/v0_9';
 * import { basicCatalog } from '@a2ui/lit/v0_9';
 * import { streamA2uiAgent } from '@genkit-ai/a2ui/client';
 *
 * const processor = new MessageProcessor([basicCatalog]);
 * for await (const ev of streamA2uiAgent({ url: '/uiAgent', message: 'weather in Tokyo' })) {
 *   if (ev.type === 'text') appendProse(ev.text);
 *   else processor.processMessages(ev.envelopes);
 * }
 * ```
 */
export async function* streamA2uiAgent(
  options: StreamA2uiAgentOptions
): AsyncGenerator<A2uiStreamEvent, void, unknown> {
  // The agent reads state-management (the session id) from `init`, not the
  // message payload; `message` carries the actual turn input.
  const init: Record<string, unknown> | undefined = options.sessionId
    ? { sessionId: options.sessionId }
    : undefined;

  const response = streamFlow<AgentOutput, AgentStreamChunk>({
    url: options.url,
    input: { message: toMessage(options.message) },
    init,
    headers: options.headers,
    abortSignal: options.abortSignal,
  });

  for await (const chunk of response.stream) {
    const mc = (chunk as any)?.modelChunk;
    if (!mc?.content) continue;
    for (const part of mc.content) {
      if (typeof part.text === 'string' && part.text !== '') {
        yield { type: 'text', text: part.text };
      }
    }
    const envelopes = a2uiEnvelopes(chunk);
    if (envelopes.length > 0) {
      yield { type: 'envelopes', envelopes };
    }
  }
  // Surface any server error / drain the stream.
  await response.output;
}

/**
 * Builds the agent input for sending a rendered surface's user action back to
 * the agent as the next turn. The action's `name` becomes the user message so
 * the agent can react to it; the full action object is attached as an a2ui data
 * part for richer handling.
 */
export function actionToMessage(action: A2uiClientAction) {
  const summary =
    `User interacted with the UI (surface "${action.surfaceId}"): ` +
    `action "${action.name}"` +
    (action.context && Object.keys(action.context).length
      ? ` with context ${JSON.stringify(action.context)}`
      : '') +
    '.';
  return {
    role: 'user',
    content: [
      { text: summary },
      {
        data: [{ action }],
        metadata: { mimeType: 'application/a2ui+json' },
      },
    ],
  };
}
