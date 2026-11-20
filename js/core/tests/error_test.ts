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
import {
  GenkitError,
  UserFacingError,
  getCallableJSON,
  getHttpStatus,
  isGenkitError,
  type SpanFailureMarks,
} from '../src/error.js';

/** A stand-in for `GenkitError` from another copy of `@genkit-ai/core`. */
class ForeignGenkitError extends Error {
  status = 'PERMISSION_DENIED' as const;
  code = 403;
  toJSON() {
    return { status: this.status, message: 'nope' };
  }
}
Object.defineProperty(
  ForeignGenkitError.prototype,
  Symbol.for('genkit.error'),
  {
    value: true,
  }
);

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
    const cause = new Error('boom') as Error & SpanFailureMarks;
    cause.ignoreFailedSpan = true;
    cause.traceId = 'trace-1';
    const err = new GenkitError({
      status: 'INTERNAL',
      message: 'wrapped',
      cause,
    }) as GenkitError & SpanFailureMarks;
    assert.strictEqual(err.ignoreFailedSpan, true);
    assert.strictEqual(err.traceId, 'trace-1');
  });

  it('claims no span markers from an unmarked cause', () => {
    const err = new GenkitError({
      status: 'INTERNAL',
      message: 'wrapped',
      cause: new Error('boom'),
    }) as GenkitError & SpanFailureMarks;
    assert.strictEqual(err.ignoreFailedSpan, undefined);
    assert.strictEqual(err.traceId, undefined);
  });
});

describe('isGenkitError', () => {
  it('recognises GenkitError and its subclasses', () => {
    assert.strictEqual(
      isGenkitError(new GenkitError({ status: 'INTERNAL', message: 'x' })),
      true
    );
    assert.strictEqual(
      isGenkitError(new UserFacingError('PERMISSION_DENIED', 'x')),
      true
    );
  });

  it('recognises a GenkitError from another copy', () => {
    assert.strictEqual(isGenkitError(new ForeignGenkitError()), true);
  });

  it('rejects values that are not a GenkitError', () => {
    for (const v of [
      new Error('x'),
      { status: 'INTERNAL', message: 'x' },
      { [Symbol.for('genkit.error')]: 'yes' },
      'x',
      null,
      undefined,
    ]) {
      assert.strictEqual(isGenkitError(v), false);
    }
  });

  it('adds nothing to the own properties or JSON of an error', () => {
    const err = new GenkitError({
      status: 'NOT_FOUND',
      message: 'gone',
      detail: { id: 1 },
    });
    assert.deepStrictEqual(Object.getOwnPropertySymbols(err), []);
    assert.deepStrictEqual(Object.keys(err), [
      'source',
      'status',
      'detail',
      'code',
      'responseMetadata',
      'originalMessage',
      'publicMessage',
      'name',
    ]);
    assert.strictEqual(
      JSON.stringify(err),
      '{"details":{"id":1},"status":"NOT_FOUND","message":"gone"}'
    );
  });
});

describe('GenkitError instanceof', () => {
  it('matches a GenkitError from another copy', () => {
    assert.strictEqual(new ForeignGenkitError() instanceof GenkitError, true);
  });

  it('keeps the prototype check for subclasses', () => {
    const base = new GenkitError({ status: 'INTERNAL', message: 'x' });
    assert.strictEqual(base instanceof UserFacingError, false);
    assert.strictEqual(
      new ForeignGenkitError() instanceof UserFacingError,
      false
    );

    const userFacing = new UserFacingError('PERMISSION_DENIED', 'x');
    assert.strictEqual(userFacing instanceof UserFacingError, true);
    assert.strictEqual(userFacing instanceof GenkitError, true);
    assert.strictEqual(userFacing instanceof Error, true);
  });

  it('rejects values that are not a GenkitError', () => {
    assert.strictEqual(new Error('x') instanceof GenkitError, false);
    const nothing: unknown = null;
    assert.strictEqual(nothing instanceof GenkitError, false);
  });
});

describe('getHttpStatus and getCallableJSON', () => {
  it('map a GenkitError from another copy to its status and message', () => {
    const err = new ForeignGenkitError();
    assert.strictEqual(getHttpStatus(err), 403);
    assert.deepStrictEqual(getCallableJSON(err), {
      status: 'PERMISSION_DENIED',
      message: 'nope',
    });
  });

  it('map a branded value without toJSON to a generic internal error', () => {
    const err = { [Symbol.for('genkit.error')]: true, status: 'NOT_FOUND' };
    assert.strictEqual(isGenkitError(err), false);
    assert.strictEqual(getHttpStatus(err), 500);
    assert.deepStrictEqual(getCallableJSON(err), {
      message: 'Internal Error',
      status: 'INTERNAL',
    });
  });

  it('map other errors to a generic internal error', () => {
    assert.strictEqual(getHttpStatus(new Error('secret')), 500);
    assert.deepStrictEqual(getCallableJSON(new Error('secret')), {
      message: 'Internal Error',
      status: 'INTERNAL',
    });
  });
});
