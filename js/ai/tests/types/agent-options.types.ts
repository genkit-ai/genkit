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

// Type-level assertions for agent call options. Never executed: type-checked
// by `tsconfig.types-test.json` (part of `pnpm check`), since the runtime test
// runner strips types without checking them.

import type { ActionContext } from '@genkit-ai/core';
import {
  createAgentAPI,
  type AgentAPI,
  type AgentTransport,
} from '../../src/agent-core.js';
import type { Agent, LocalAgentOptions } from '../../src/agent.js';

declare const alice: ActionContext;
declare const signal: AbortSignal;

// A transport with the default (no) options, like `remoteAgent`, must reject
// options it cannot honor. Relies on `Opts` defaulting to `never` (not `{}`,
// which would skip excess-property checks).
export function noOptsTransport(transport: AgentTransport) {
  const api = createAgentAPI(transport);
  // @ts-expect-error `context` is not an option of this transport.
  api.chat({}, { context: alice });
  // @ts-expect-error `context` is not an option of this transport.
  void api.loadChat({ snapshotId: 'id' }, { context: alice });
  // @ts-expect-error `context` is not an option of this transport.
  void api.getSnapshot('id', { context: alice });
  // @ts-expect-error `context` is not an option of this transport.
  void api.abort('id', { context: alice });

  const chat = api.chat();
  // @ts-expect-error `context` is not an option of this transport.
  void chat.send('hi', { context: alice });
  // @ts-expect-error `context` is not an option of this transport.
  void chat.detach('hi', { context: alice });
  // `abortSignal` is always accepted.
  void chat.send('hi', { abortSignal: signal });
}

// Passing only `State` resets `Opts` to `never`, so the transport's own
// options are rejected at call sites. Passing both type args keeps them.
export function partialTypeArgs(transport: AgentTransport<{ tag?: string }>) {
  const partial = createAgentAPI<{ count: number }>(transport);
  // @ts-expect-error `createAgentAPI<State>` means `Opts = never`.
  partial.chat({}, { tag: 'x' });

  const full = createAgentAPI<{ count: number }, { tag?: string }>(transport);
  void full
    .chat({}, { tag: 'x' })
    .send('hi', { tag: 'y', abortSignal: signal });
}

// In-process agents accept `context` everywhere, including on detached tasks.
export async function localAgentOptions(agent: Agent<{ count: number }>) {
  const api: AgentAPI<{ count: number }, LocalAgentOptions> = agent;
  const chat = api.chat({}, { context: alice });
  await chat.send('hi', { context: alice, abortSignal: signal });
  const task = await chat.detach('job', { context: alice });
  await task.abort({ context: alice });
  // @ts-expect-error unknown option.
  await task.abort({ tag: 'x' });
}
