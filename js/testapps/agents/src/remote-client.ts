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
 * Standalone script that exercises the {@link remoteAgent} client against the
 * agents Express server (see `index.ts`). Validates the client ergonomics end
 * to end over real HTTP.
 *
 * Usage:
 *   1. In one terminal: `pnpm start` (starts the Express server on :8080).
 *   2. In another:      `npx tsx src/remote-client.ts`
 */

import { randomUUID } from 'crypto';
import { AgentError, remoteAgent } from 'genkit/beta/client';

const BASE = process.env.AGENT_BASE_URL ?? 'http://localhost:8080';

async function main() {
  // ── A server-managed agent (weatherAgent has a FileSessionStore). ────────
  const weather = remoteAgent({
    url: `${BASE}/api/weatherAgent`,
    getSnapshotUrl: `${BASE}/api/weatherAgent/state`,
    abortUrl: `${BASE}/api/weatherAgent/abort`,
  });

  console.log('\n=== weatherAgent: streaming turn ===');

  const chat = weather.chat({ sessionId: randomUUID() });
  const turn = chat.sendStream('What is the weather like in Tokyo?');
  for await (const chunk of turn.stream) {
    if (chunk.text) process.stdout.write(chunk.text);
  }
  const res = await turn.response;
  console.log('\n--- response ---');
  console.log('text:', res.text);
  console.log('finishReason:', res.finishReason);
  console.log('snapshotId:', res.snapshotId);
  console.log('chat.snapshotId:', chat.snapshotId);

  console.log('\n=== weatherAgent: multi-turn (state auto-carried) ===');
  const res2 = await chat.send('What about Paris?'); // non-streaming send
  console.log('text:', res2.text);
  console.log('chat.snapshotId:', chat.snapshotId);
  console.log('chat.state:', JSON.stringify(chat.state, null, 2));

  // ── Load a chat from the latest snapshot. ────────────────────────────────
  if (chat.snapshotId) {
    console.log('\n=== weatherAgent: load chat from snapshot ===');
    const loaded = await weather.loadChat({ snapshotId: chat.snapshotId });
    console.log('restored messages:', loaded.messages.length);
    console.log('loaded.state:', JSON.stringify(loaded.state, null, 2));
    const res3 = await loaded.send('And London?');
    console.log('text:', res3.text);
  }

  // ── Error handling demonstration. ────────────────────────────────────────
  console.log('\n=== error handling ===');
  try {
    const bad = remoteAgent({ url: `${BASE}/api/does-not-exist` });
    await bad.chat().send('hello');
  } catch (err) {
    if (err instanceof AgentError) {
      console.log('caught AgentError, status:', err.status);
    } else {
      console.log('caught error:', (err as Error).message);
    }
  }

  console.log('\nDone.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
