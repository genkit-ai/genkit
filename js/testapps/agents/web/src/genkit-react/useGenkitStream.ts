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
 * Generic React hook over `streamFlow`. Makes no assumptions about chunk
 * shape — works with any flow's `streamSchema`. For agents specifically,
 * use `useGenkitAgent` which derives `text`, `toolCalls`, `interrupts`,
 * `phase`, etc. on top of this.
 */
import { streamFlow } from 'genkit/beta/client';
import { useCallback, useEffect, useRef, useState } from 'react';

export type StreamStatus = 'idle' | 'streaming' | 'done' | 'error';

export interface UseGenkitStreamOptions {
  url: string;
  headers?: Record<string, string>;
}

export interface UseGenkitStreamResult<
  I = unknown,
  O = unknown,
  S = unknown,
  Init = unknown,
> {
  output: O | null;
  chunks: S[];
  status: StreamStatus;
  error: Error | null;
  submit: (input: I, init?: Init) => void;
  abort: () => void;
  reset: () => void;
}

export function useGenkitStream<
  I = unknown,
  O = unknown,
  S = unknown,
  Init = unknown,
>(options: UseGenkitStreamOptions): UseGenkitStreamResult<I, O, S, Init> {
  const { url, headers } = options;
  const [output, setOutput] = useState<O | null>(null);
  const [chunks, setChunks] = useState<S[]>([]);
  const [status, setStatus] = useState<StreamStatus>('idle');
  const [error, setError] = useState<Error | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setOutput(null);
    setChunks([]);
    setStatus('idle');
    setError(null);
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus((s) => (s === 'streaming' ? 'idle' : s));
  }, []);

  const submit = useCallback(
    (input: I, init?: Init) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setOutput(null);
      setChunks([]);
      setError(null);
      setStatus('streaming');
      const { output: outputPromise, stream } = streamFlow<O, S, Init>({
        url,
        input,
        init: init as Init,
        headers,
        abortSignal: controller.signal,
      });
      (async () => {
        try {
          for await (const chunk of stream) {
            setChunks((prev) => [...prev, chunk]);
          }
          const finalOutput = await outputPromise;
          setOutput(finalOutput);
          setStatus('done');
        } catch (err) {
          if (controller.signal.aborted) return;
          setError(err instanceof Error ? err : new Error(String(err)));
          setStatus('error');
        }
      })();
    },
    [url, headers]
  );

  useEffect(() => () => abortRef.current?.abort(), []);
  return { output, chunks, status, error, submit, abort, reset };
}
