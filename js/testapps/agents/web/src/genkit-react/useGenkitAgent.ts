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
 * The hook itself is a thin shim: `useSyncExternalStore` to surface the
 * session's state, plus the session's action methods passed through with
 * permanently stable identity. All conversation logic — chunk dispatch,
 * continuation round-trip, interrupt detection, background polling,
 * snapshot rehydration — lives in `AgentSession` from
 * `genkit/beta/client`, framework-agnostic.
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
import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react';

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
  // One session instance for the component's lifetime. Constructed
  // lazily so the resume effect (if any) fires at mount time.
  const sessionRef = useRef<AgentSession<S> | null>(null);
  if (sessionRef.current === null) {
    sessionRef.current = new AgentSession<S>(options);
  }
  const session = sessionRef.current;

  // Keep the session's options pointed at the latest values from the
  // parent component (matches the latest-options ref pattern that lets
  // adapters re-render freely without recreating callbacks).
  session.setOptions(options);

  const subscribe = useMemo(() => session.subscribe.bind(session), [session]);
  const getSnapshot = useMemo(() => session.getState.bind(session), [session]);
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  // Action methods are bound to the session instance so their identity is
  // stable for the hook's lifetime — safe to depend on directly.
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

  // Dispose on real unmount. We don't tear down on the StrictMode
  // remount cycle — the session has a single lifetime tied to the
  // component instance, and the lazy init via `useRef` means we'd
  // otherwise dispose a freshly-built session before the user could
  // interact with it. `useSyncExternalStore` already unsubscribes its
  // own listener on remount, so a brief "no listeners attached" gap
  // does not leak.
  useEffect(() => {
    return () => {
      // Defer to a microtask so that an immediate StrictMode remount
      // can claim the session before disposal fires.
      const s = session;
      Promise.resolve().then(() => {
        // Only dispose if no later mount re-claimed this ref. We track
        // claims via a counter on the session.
        if (s === sessionRef.current && !s.hasActiveSubscribers()) {
          s.dispose();
        }
      });
    };
  }, [session]);

  return useMemo(
    () => ({ ...state, ...actions }),
    [state, actions]
  );
}
