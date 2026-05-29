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

import { DestroyRef, Signal, inject, signal } from '@angular/core';
import {
  AgentSession,
  type AgentContinuation,
  type AgentInputBody,
  type AgentSessionOptions,
  type AgentSessionState,
  type AgentVariant,
} from 'genkit/beta/client';

export type { AgentContinuation } from 'genkit/beta/client';

export interface GenkitAgentHandle<S = unknown, TStatus = unknown> {
  /** Reactive state — read in templates with `state().messages`, etc. */
  state: Signal<AgentSessionState<S, TStatus>>;
  submit: (input: AgentInputBody) => void;
  abort: () => Promise<void>;
  reset: () => void;
  respondToInterrupt: (output: unknown) => void;
  restartInterrupt: (metadata?: unknown) => void;
  runVariants: (
    input: AgentInputBody,
    count?: number
  ) => Promise<AgentVariant<S>[]>;
  continueFrom: (
    continuationOrSnapshotId: AgentContinuation | string
  ) => Promise<void>;
}

/**
 * `injectGenkitAgent` — Angular adapter for `AgentSession`.
 *
 * Construct a session, expose its state as a signal, and tear it down
 * automatically when the host component is destroyed. Must be called
 * from an injection context (component / directive constructor or
 * a factory provider).
 *
 * @example
 * ```ts
 * @Component({ ... })
 * export class WeatherChatComponent {
 *   readonly agent = injectGenkitAgent({ url: '/api/weatherAgent' });
 *   send(text: string) {
 *     this.agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
 *   }
 * }
 * ```
 */
export function injectGenkitAgent<S = unknown, TStatus = unknown>(
  options: AgentSessionOptions
): GenkitAgentHandle<S, TStatus> {
  const session = new AgentSession<S, TStatus>(options);
  const state = signal<AgentSessionState<S, TStatus>>(session.getState());
  const unsubscribe = session.subscribe(() => state.set(session.getState()));
  inject(DestroyRef).onDestroy(() => {
    unsubscribe();
    session.dispose();
  });
  return {
    state: state.asReadonly(),
    submit: session.submit.bind(session),
    abort: session.abort.bind(session),
    reset: session.reset.bind(session),
    respondToInterrupt: session.respondToInterrupt.bind(session),
    restartInterrupt: session.restartInterrupt.bind(session),
    runVariants: session.runVariants.bind(session),
    continueFrom: session.continueFrom.bind(session),
  };
}
