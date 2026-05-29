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
 * `useGenkitAgent` — React adapter for `AgentSession`.
 *
 * Thin shim over `AgentSession` from `genkit/beta/client`:
 * `useSyncExternalStore` for state, bound methods for actions. The
 * session's long-running internals (background poll, in-flight stream)
 * self-terminate when no listeners remain, so the adapter doesn't need
 * a dispose hook — the lazy init via `useRef` works under React
 * StrictMode without any teardown gymnastics.
 */
import {
  AgentSession,
  type AgentInputBody,
  type AgentMessage,
  type AgentPhase,
  type AgentSessionOptions,
  type AgentSessionState,
  type AgentVariant,
  type PendingInterrupt,
  type ToolCall,
  type ToolCallState,
} from 'genkit/beta/client';
import { useMemo, useRef, useSyncExternalStore } from 'react';

export type {
  AgentInputBody,
  AgentMessage,
  AgentPhase,
  AgentVariant,
  PendingInterrupt,
  ToolCall,
  ToolCallState,
};

export type UseGenkitAgentOptions = AgentSessionOptions;

export interface UseGenkitAgentResult<S = unknown>
  extends AgentSessionState<S> {
  submit: (input: AgentInputBody) => void;
  abort: () => Promise<void>;
  reset: () => void;
  respondToInterrupt: (output: unknown) => void;
  restartInterrupt: (metadata?: unknown) => void;
  runVariants: (
    input: AgentInputBody,
    count?: number
  ) => Promise<AgentVariant<S>[]>;
  continueFrom: (continuationOrSnapshotId: string) => Promise<void>;
}

export function useGenkitAgent<S = unknown>(
  options: UseGenkitAgentOptions
): UseGenkitAgentResult<S> {
  // One session per component instance. `useRef` keeps it stable across
  // re-renders and StrictMode's double-mount cycle.
  const sessionRef = useRef<AgentSession<S> | null>(null);
  if (sessionRef.current === null) {
    sessionRef.current = new AgentSession<S>(options);
  }
  const session = sessionRef.current;

  // Latest-options ref pattern: the session reads `url` / `headers` /
  // etc. from here on every request, so consumers can re-render freely
  // without re-creating any callbacks.
  session.setOptions(options);

  const subscribe = useMemo(() => session.subscribe.bind(session), [session]);
  const getSnapshot = useMemo(() => session.getState.bind(session), [session]);
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  // Methods are bound to the session instance — permanently stable
  // identity, safe to depend on directly.
  const actions = useMemo(
    () => ({
      submit: session.submit.bind(session),
      abort: session.abort.bind(session),
      reset: session.reset.bind(session),
      respondToInterrupt: session.respondToInterrupt.bind(session),
      restartInterrupt: session.restartInterrupt.bind(session),
      runVariants: session.runVariants.bind(session),
      continueFrom: session.continueFrom.bind(session),
    }),
    [session]
  );

  return useMemo(() => ({ ...state, ...actions }), [state, actions]);
}
