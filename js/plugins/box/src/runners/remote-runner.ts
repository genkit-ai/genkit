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

import { ReflectionClientV1 } from '../reflection-client-v1.js';
import type { BoxConnection, BoxRunner } from '../types.js';
import { abortable } from './util.js';

/** Options for {@link remoteRunner}. */
export interface RemoteRunnerOptions {
  /** Base URL of a runtime serving reflection v1, e.g. `http://127.0.0.1:3100`. */
  url: string;
  /** The runtime's reflection secret (`GENKIT_REFLECTION_SECRET_TOKEN`). */
  secret?: string;
  /** Extra headers sent with every request (e.g. for a gateway). */
  headers?: Record<string, string>;
  /** How long to wait for the runtime's health check. Defaults to 30s. */
  readyTimeoutMs?: number;
}

/**
 * A {@link BoxRunner} over a runtime someone else started: a sidecar
 * container, a Cloud Run service, another machine. There is no lifecycle to
 * own, so every routing key maps to the same runtime, and `release`/`close`
 * do nothing. Route and retention still run (leases, `sessionRoute` memory),
 * they just don't start or stop anything.
 *
 * Keys are accepted rather than rejected so a config can swap runners (say,
 * `execRunner` locally, `remoteRunner` in prod) without changing its route.
 */
export class RemoteRunner implements BoxRunner {
  readonly name = 'remote';
  private readonly client: ReflectionClientV1;
  private ready?: Promise<void>;

  constructor(private readonly options: RemoteRunnerOptions) {
    this.client = new ReflectionClientV1(options.url.replace(/\/+$/, ''), {
      secret: options.secret,
      headers: options.headers,
    });
  }

  async acquire(_key: string, signal?: AbortSignal): Promise<BoxConnection> {
    // One readiness check, shared by every caller, so it must not depend on
    // any one caller's signal: each caller races its own instead. A failed
    // check is retried on the next acquire.
    this.ready ??= this.client
      .waitForReady(this.options.readyTimeoutMs ?? 30_000)
      .catch((e: unknown) => {
        this.ready = undefined;
        throw e;
      });
    await abortable(this.ready, signal);
    return this.client;
  }

  async release(): Promise<void> {}

  async close(): Promise<void> {}
}

/** Creates a {@link RemoteRunner}. */
export function remoteRunner(options: RemoteRunnerOptions): RemoteRunner {
  return new RemoteRunner(options);
}
