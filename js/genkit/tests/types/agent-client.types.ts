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

// Type-level assertions for the public `genkit/beta/client` agent surface.
// Never executed: type-checked by `tsconfig.types-test.json` (part of
// `pnpm check`), since the runtime test runner strips types without checking.

import {
  createAgentAPI,
  remoteAgent,
  type AgentTransport,
} from '../../src/client/index';

declare const signal: AbortSignal;

// `remoteAgent` call options carry request headers, but not `context`: it is
// derived server-side.
export function remoteAgentRejectsContext() {
  const agent = remoteAgent<{ count: number }>({ url: '/api/agent' });
  // @ts-expect-error `context` is not a remote option.
  agent.chat({}, { context: {} });
  const chat = agent.chat();
  // @ts-expect-error `context` is not a remote option.
  void chat.send('hi', { context: {} });
  void chat.send('hi', { abortSignal: signal });
}

export async function remoteAgentAcceptsHeaders() {
  const agent = remoteAgent<{ count: number }>({ url: '/api/agent' });
  const chat = agent.chat({}, { headers: { Authorization: 'Bearer t' } });
  await chat.send('hi', {
    headers: { 'x-request-id': 'r' },
    abortSignal: signal,
  });
  await agent.abort('id', { headers: { 'x-request-id': 'r' } });
  // @ts-expect-error headers must be strings.
  agent.chat({}, { headers: { n: 1 } });
}

// Custom transports get exactly the options they declare, plus `abortSignal`.
export async function customTransportOptions(
  transport: AgentTransport<{ tag?: string }>
) {
  const agent = createAgentAPI<{ count: number }, { tag?: string }>(transport);
  // @ts-expect-error `context` is not declared by this transport.
  agent.chat({}, { context: {} });
  const chat = agent.chat({}, { tag: 'bound' });
  await chat.send('hi', { tag: 'x', abortSignal: signal });
  const task = await chat.detach('job', { tag: 'y' });
  await task.abort({ tag: 'operator' });
}
