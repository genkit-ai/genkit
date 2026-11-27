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
 * Standalone script that builds an agent client over a custom
 * {@link AgentTransport} with {@link createAgentAPI}, and runs it against the
 * agents Express server (see `index.ts`).
 *
 * The transport here still speaks HTTP (via `streamFlow`/`runFlow`) to keep
 * the sample runnable, but the same shape works for anything: WebSocket, a
 * callable function, a message queue, etc. It also declares a typed per-call
 * option (`requestId`), which shows up on `chat`/`send`/`loadChat`/... .
 *
 * Usage:
 *   1. In one terminal: `pnpm start` (starts the Express server on :8080).
 *   2. In another:      `npx tsx src/custom-transport-client.ts`
 */

import { randomUUID } from 'crypto';
import {
  createAgentAPI,
  runFlow,
  streamFlow,
  type AgentInit,
  type AgentOutput,
  type AgentStreamChunk,
  type AgentTransport,
  type SessionSnapshot,
} from 'genkit/beta/client';

const BASE = process.env.AGENT_BASE_URL ?? 'http://localhost:8080';

/** Transport-specific call options, bound per chat or passed per call. */
interface TracedOptions {
  /** Sent as `x-request-id` so server logs can be correlated. */
  requestId?: string;
}

/**
 * A minimal custom transport. Contract highlights (see `AgentTransport`):
 * - `runTurn` returns the chunk stream plus the final output; agent failures
 *   resolve `output` with `finishReason: 'failed'`, transport errors reject.
 * - `abortSignal` is always provided; forward it so aborts cancel the request.
 * - chunks must arrive in order (custom-state patches are applied in order).
 */
function tracedHttpTransport(url: string): AgentTransport<TracedOptions> {
  const headersFor = (opts?: TracedOptions) =>
    opts?.requestId ? { 'x-request-id': opts.requestId } : undefined;

  return {
    // Advisory only; `weatherAgent` has a server-side store.
    stateManagement: 'server',

    runTurn(input, init, { abortSignal, requestId }) {
      console.log(
        `[transport] turn requestId=${requestId ?? '-'} init=${JSON.stringify(init)}`
      );
      const { stream, output } = streamFlow<
        AgentOutput,
        AgentStreamChunk,
        AgentInit
      >({
        url,
        input,
        init,
        headers: headersFor({ requestId }),
        abortSignal,
      });
      return { stream, output };
    },

    getSnapshot(lookup, opts) {
      return runFlow<SessionSnapshot | undefined>({
        url: `${url}/getSnapshot`,
        input: lookup,
        headers: headersFor(opts),
      });
    },

    async abort(snapshotId, opts) {
      const res = await runFlow<{ status?: SessionSnapshot['status'] }>({
        url: `${url}/abort`,
        input: { snapshotId },
        headers: headersFor(opts),
      });
      return res?.status;
    },
  };
}

async function main() {
  // Pass both type arguments: `createAgentAPI<State>(t)` alone would reset the
  // options type to `never` and reject `requestId`.
  const weather = createAgentAPI<unknown, TracedOptions>(
    tracedHttpTransport(`${BASE}/api/weatherAgent`)
  );

  console.log('\n=== custom transport: streaming turn (bound requestId) ===');
  const chat = weather.chat(
    { sessionId: randomUUID() },
    { requestId: 'chat-1' }
  );
  const turn = chat.sendStream('What is the weather like in Tokyo?');
  for await (const chunk of turn.stream) {
    if (chunk.text) process.stdout.write(chunk.text);
  }
  const res = await turn.response;
  console.log(
    '\nfinishReason:',
    res.finishReason,
    'snapshotId:',
    res.snapshotId
  );

  console.log('\n=== custom transport: per-call requestId override ===');
  const res2 = await chat.send('What about Paris?', { requestId: 'chat-1-b' });
  console.log('text:', res2.text);

  if (chat.snapshotId) {
    console.log('\n=== custom transport: load chat from snapshot ===');
    const loaded = await weather.loadChat(
      { snapshotId: chat.snapshotId },
      { requestId: 'load-1' }
    );
    console.log('restored messages:', loaded.messages.length);
    const res3 = await loaded.send('And London?');
    console.log('text:', res3.text);
  }

  console.log('\nDone.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
