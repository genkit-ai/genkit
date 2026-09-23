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

import * as assert from 'assert';
import { describe, it } from 'node:test';
import { GenkitError, getCallableJSON } from '../src/error.js';

describe('GenkitError', () => {
  it('exposes the wrapped error as cause', () => {
    const cause = new Error('db exploded');
    const err = new GenkitError({
      status: 'INTERNAL',
      message: 'tool failed: db exploded',
      cause,
    });
    assert.strictEqual(err.cause, cause);
    assert.strictEqual(err.message, 'INTERNAL: tool failed: db exploded');
    assert.strictEqual(err.originalMessage, 'tool failed: db exploded');
  });

  it('has no cause unless one was given', () => {
    const err = new GenkitError({ status: 'INTERNAL', message: 'boom' });
    assert.strictEqual('cause' in err, false);
  });

  it('sends the public message to a client when one is set', () => {
    const err = new GenkitError({
      status: 'INTERNAL',
      message: 'tool "x" failed: db password rejected',
      publicMessage: 'tool "x" failed',
    });
    assert.deepStrictEqual(err.toJSON(), {
      status: 'INTERNAL',
      message: 'tool "x" failed',
    });
    assert.deepStrictEqual(getCallableJSON(err), {
      status: 'INTERNAL',
      message: 'tool "x" failed',
    });
    // The full text stays in-process.
    assert.strictEqual(
      err.originalMessage,
      'tool "x" failed: db password rejected'
    );
  });

  it('keeps the span markers tracing stamped on the cause', () => {
    const cause = new Error('boom') as Error & {
      ignoreFailedSpan?: boolean;
      traceId?: string;
    };
    cause.ignoreFailedSpan = true;
    cause.traceId = 'trace-1';
    const err = new GenkitError({
      status: 'INTERNAL',
      message: 'wrapped',
      cause,
    }) as GenkitError & { ignoreFailedSpan?: boolean; traceId?: string };
    assert.strictEqual(err.ignoreFailedSpan, true);
    assert.strictEqual(err.traceId, 'trace-1');
  });

  it('claims no span markers from an unmarked cause', () => {
    const err = new GenkitError({
      status: 'INTERNAL',
      message: 'wrapped',
      cause: new Error('boom'),
    }) as GenkitError & { ignoreFailedSpan?: boolean; traceId?: string };
    assert.strictEqual(err.ignoreFailedSpan, undefined);
    assert.strictEqual(err.traceId, undefined);
  });
});
