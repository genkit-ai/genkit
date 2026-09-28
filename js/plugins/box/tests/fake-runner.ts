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

import type {
  BoxConnection,
  BoxRunner,
  RunActionRequest,
  RunOptions,
} from '../src/types.js';

/** One call the fake runner served, with the routing key it landed on. */
export interface RecordedCall {
  routeKey: string;
  req: RunActionRequest;
}

/** Answers a call inside the fake box. Emits chunks via `opts.onChunk`. */
export type FakeHandler = (
  req: RunActionRequest,
  opts: RunOptions | undefined
) => unknown | Promise<unknown>;

/**
 * An in-memory {@link BoxRunner}: no processes, no reflection. Records every
 * acquire/release/call so tests can assert on routing and lifecycle.
 */
export class FakeRunner implements BoxRunner {
  readonly name = 'fake';
  readonly calls: RecordedCall[] = [];
  readonly acquired: string[] = [];
  readonly released: string[] = [];
  attachedTo?: string;
  closed = false;

  constructor(private readonly handler: FakeHandler = () => undefined) {}

  attach(box: { readonly id: string }): void {
    this.attachedTo = box.id;
  }

  async acquire(routeKey: string): Promise<BoxConnection> {
    this.acquired.push(routeKey);
    return {
      runAction: async <O>(req: RunActionRequest, opts?: RunOptions) => {
        this.calls.push({ routeKey, req });
        const result = (await this.handler(req, opts)) as O;
        return { result, telemetry: { traceId: `box-trace-${routeKey}` } };
      },
      listActions: async () => ({}),
    };
  }

  async release(routeKey: string): Promise<void> {
    this.released.push(routeKey);
  }

  async close(): Promise<void> {
    this.closed = true;
  }
}
