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
 * `useGenkitRunFlow` — generic React hook for non-streaming Genkit flow
 * invocations. Wraps `runFlow` from `genkit/beta/client` with reactive
 * `data` / `error` / `status` state and a manual `run()` trigger. Useful
 * for the sibling endpoints around an agent (workspace listings, file
 * reads, snapshot data, etc.) so pages don't have to import `runFlow`
 * directly.
 *
 * For agent invocations, prefer `useGenkitAgent`. For arbitrary streaming
 * (non-agent) flows, prefer `useGenkitStream`.
 */
import { runFlow } from 'genkit/beta/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export type RunFlowStatus = 'idle' | 'pending' | 'done' | 'error';

export interface UseGenkitRunFlowOptions {
  url: string;
  headers?: Record<string, string>;
}

export interface UseGenkitRunFlowResult<I = unknown, O = unknown> {
  data: O | null;
  status: RunFlowStatus;
  error: Error | null;
  /** Fire the flow. Returns the resolved output (or throws). */
  run: (input: I) => Promise<O>;
  reset: () => void;
}

export function useGenkitRunFlow<I = unknown, O = unknown>(
  options: UseGenkitRunFlowOptions
): UseGenkitRunFlowResult<I, O> {
  // Latest-options ref: `run` reads `url`/`headers` from here instead of
  // closing over them, so its identity stays stable for the hook's
  // lifetime. Consumers can include `run` in `useEffect` /
  // `useCallback` dep arrays without worrying about whether the parent
  // memoized the options object — a missed memo previously caused an
  // 81k-error re-render loop in CodingAgent.
  const optsRef = useRef(options);
  optsRef.current = options;

  const [data, setData] = useState<O | null>(null);
  const [status, setStatus] = useState<RunFlowStatus>('idle');
  const [error, setError] = useState<Error | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setData(null);
    setStatus('idle');
    setError(null);
  }, []);

  const run = useCallback(async (input: I): Promise<O> => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus('pending');
    setError(null);
    try {
      const opts = optsRef.current;
      const result = await runFlow<O>({
        url: opts.url,
        input,
        headers: opts.headers,
        abortSignal: controller.signal,
      });
      if (!controller.signal.aborted) {
        setData(result);
        setStatus('done');
      }
      return result;
    } catch (e) {
      if (!controller.signal.aborted) {
        const err = e instanceof Error ? e : new Error(String(e));
        setError(err);
        setStatus('error');
      }
      throw e;
    }
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Memoize so the returned object reference is stable across renders
  // when nothing changes. `run` and `reset` now have permanent identity
  // (empty deps + latest-options ref), so consumers can safely depend
  // on them directly without destructuring through the wrapper.
  return useMemo(
    () => ({ data, status, error, run, reset }),
    [data, status, error, run, reset]
  );
}
