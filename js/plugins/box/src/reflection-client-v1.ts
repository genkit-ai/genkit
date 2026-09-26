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

import type { ActionMetadata } from 'genkit';
import { logger } from 'genkit/logging';
import { REFLECTION_SECRET_HEADER } from './reflection-auth.js';
import type {
  BoxConnection,
  RunActionRequest,
  RunActionResult,
  RunOptions,
} from './types.js';

/** An error carrying the structured `Status` payload from a V1 reflection error. */
export class BoxV1RuntimeError extends Error {
  constructor(
    message: string,
    readonly data?: unknown
  ) {
    super(message);
    this.name = 'BoxV1RuntimeError';
  }
}

/** Shape of the `{ result, error, telemetry }` envelope V1 returns. */
interface RunActionEnvelope<O> {
  result?: O;
  error?: { message?: string; [k: string]: unknown };
  telemetry?: { traceId?: string };
}

/** Options for {@link ReflectionClientV1}. */
export interface ReflectionClientV1Options {
  /**
   * Reflection secret the runtime was started with
   * (`GENKIT_REFLECTION_SECRET_TOKEN`). Sent on every request.
   */
  secret?: string;
  /**
   * Extra headers sent with every request, e.g. to address a box behind a
   * shared gateway that routes on a header.
   */
  headers?: Record<string, string>;
}

/**
 * Talks to a Genkit **reflection V1** server over HTTP.
 *
 * V1 inverts V2's direction: the runtime is the server and we are the client.
 * That is what makes containers workable, since `podman run -p` forwards
 * host->container, matching a dial-in client. There is no dial-back URL for the
 * box to resolve, so no container-to-host networking is needed at all.
 *
 * Only the subset {@link BoxConnection} needs is implemented. Input streaming
 * (V2's `streamInput`) has no V1 equivalent, but box never uses it.
 */
export class ReflectionClientV1 implements BoxConnection {
  private readonly headers: Record<string, string>;

  /** @param baseUrl e.g. `http://127.0.0.1:54321` (no trailing slash). */
  constructor(
    private readonly baseUrl: string,
    options?: ReflectionClientV1Options
  ) {
    this.headers = {
      ...options?.headers,
      ...(options?.secret
        ? { [REFLECTION_SECRET_HEADER]: options.secret }
        : {}),
    };
  }

  /**
   * Polls `/api/__health` until the runtime answers. Replaces V2's
   * `waitForRuntime`: nothing dials us, so readiness has to be observed.
   */
  async waitForReady(timeoutMs = 30_000, signal?: AbortSignal): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    let lastErr: unknown;
    while (Date.now() < deadline) {
      if (signal?.aborted) throw new Error('Aborted while waiting for box.');
      try {
        const res = await fetch(`${this.baseUrl}/api/__health`, {
          headers: this.headers,
          signal: AbortSignal.timeout(2_000),
        });
        if (res.ok) return;
        lastErr = new Error(`health returned ${res.status}`);
      } catch (e) {
        // Connection refused until the server binds; keep polling.
        lastErr = e;
      }
      await new Promise((r) => setTimeout(r, 150));
    }
    throw new Error(
      `Timed out waiting for box reflection server at ${this.baseUrl}` +
        (lastErr ? ` (last error: ${lastErr})` : '')
    );
  }

  async listActions(): Promise<Record<string, ActionMetadata>> {
    const res = await fetch(`${this.baseUrl}/api/actions`, {
      headers: this.headers,
    });
    if (!res.ok) {
      throw new Error(`Box listActions failed: ${res.status}`);
    }
    return (await res.json()) as Record<string, ActionMetadata>;
  }

  async runAction<O = unknown>(
    req: RunActionRequest,
    opts?: RunOptions
  ): Promise<RunActionResult<O>> {
    const stream = !!opts?.onChunk;
    const url = `${this.baseUrl}/api/runAction${stream ? '?stream=true' : ''}`;

    // V1 has no in-band cancel; it is a separate call keyed by trace id, which
    // we only learn from the response header. Cancels arriving before that are
    // replayed once the id shows up.
    let traceId: string | undefined;
    let abortPending = false;
    const onAbort = () => {
      if (traceId) {
        this.cancelAction(traceId).catch((e) =>
          logger.debug(`Box cancel failed: ${e}`)
        );
      } else {
        abortPending = true;
      }
    };
    if (opts?.abortSignal) {
      if (opts.abortSignal.aborted) onAbort();
      else opts.abortSignal.addEventListener('abort', onAbort);
    }

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { ...this.headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key: req.key,
          input: req.input,
          init: req.init,
          context: req.context,
          telemetryLabels: req.telemetryLabels,
        }),
      });

      // Headers are flushed early (onTraceStart), so the trace id lands before
      // the first chunk.
      traceId = res.headers.get('x-genkit-trace-id') ?? undefined;
      if (traceId) {
        opts?.onTraceId?.(traceId);
        if (abortPending) onAbort();
      }

      if (!res.ok) {
        throw new BoxV1RuntimeError(
          `Box runAction failed: ${res.status} ${await res.text()}`
        );
      }

      const envelope = stream
        ? await this.readStream<O>(res, opts!.onChunk!)
        : ((await res.json()) as RunActionEnvelope<O>);

      if (envelope.error) {
        throw new BoxV1RuntimeError(
          envelope.error.message ?? 'Box action failed',
          envelope.error
        );
      }
      return {
        result: envelope.result,
        telemetry: envelope.telemetry ?? (traceId ? { traceId } : undefined),
      };
    } finally {
      opts?.abortSignal?.removeEventListener('abort', onAbort);
    }
  }

  /**
   * Reads a streaming runAction body: newline-delimited JSON chunks, where the
   * final line is the `{ result }`/`{ error }` envelope rather than a chunk.
   * Chunks are only dispatched once the *next* line arrives, which is how we
   * tell a trailing envelope from a chunk without lookahead.
   */
  private async readStream<O>(
    res: Response,
    onChunk: (chunk: unknown) => void
  ): Promise<RunActionEnvelope<O>> {
    if (!res.body) throw new Error('Box streaming response had no body.');
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffered = '';
    let pending: string | undefined;

    const flushPending = () => {
      if (pending === undefined) return;
      try {
        onChunk(JSON.parse(pending));
      } catch {
        logger.debug('Box dropped an unparseable stream chunk.');
      }
      pending = undefined;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffered.indexOf('\n')) !== -1) {
        const line = buffered.slice(0, nl).trim();
        buffered = buffered.slice(nl + 1);
        if (!line) continue;
        flushPending();
        pending = line;
      }
    }
    const tail = buffered.trim();
    if (tail) {
      flushPending();
      pending = tail;
    }
    if (pending === undefined) {
      throw new Error('Box stream ended without a result envelope.');
    }
    return JSON.parse(pending) as RunActionEnvelope<O>;
  }

  /** Cancels an in-flight action by trace id. */
  async cancelAction(traceId: string): Promise<void> {
    await fetch(`${this.baseUrl}/api/cancelAction`, {
      method: 'POST',
      headers: { ...this.headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({ traceId }),
    });
  }
}
