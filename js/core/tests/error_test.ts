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
  GENKIT_ERROR_SYMBOL,
  GenkitError,
  UserFacingError,
  getCallableJSON,
  getHttpStatus,
  isGenkitError,
  type SpanFailureMarks,
} from '../src/error.js';

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

  describe('cross-copy branding and instanceof compatibility', () => {
    it('brands GenkitError with GENKIT_ERROR_SYMBOL', () => {
      const err = new GenkitError({
        status: 'NOT_FOUND',
        message: 'not found',
      });
      assert.strictEqual(isGenkitError(err), true);
      assert.strictEqual(err[GENKIT_ERROR_SYMBOL], true);
      assert.strictEqual(isGenkitError(new Error('standard error')), false);
      assert.strictEqual(isGenkitError(null), false);
      assert.strictEqual(isGenkitError('error string'), false);
    });

    it('recognizes simulated foreign GenkitError across package copies', () => {
      const foreignErr = Object.assign(new Error('foreign copy error'), {
        [Symbol.for('genkit.error')]: true,
        name: 'GenkitError',
        status: 'PERMISSION_DENIED' as const,
        code: 403,
        originalMessage: 'permission denied from cjs copy',
        toJSON() {
          return {
            status: 'PERMISSION_DENIED' as const,
            message: 'permission denied from cjs copy',
          };
        },
      });

      assert.strictEqual(isGenkitError(foreignErr), true);
      assert.strictEqual(foreignErr instanceof GenkitError, true);
      assert.strictEqual(getHttpStatus(foreignErr), 403);
      assert.deepStrictEqual(getCallableJSON(foreignErr), {
        status: 'PERMISSION_DENIED',
        message: 'permission denied from cjs copy',
      });
    });

    it('recognizes foreign UserFacingError with proper subclass differentiation', () => {
      const foreignUserFacing = Object.assign(new Error('user facing'), {
        [Symbol.for('genkit.error')]: true,
        name: 'UserFacingError',
        status: 'UNAUTHENTICATED' as const,
        code: 401,
        originalMessage: 'unauthenticated',
      });

      const foreignGenkitOnly = Object.assign(new Error('plain genkit error'), {
        [Symbol.for('genkit.error')]: true,
        name: 'GenkitError',
        status: 'INTERNAL' as const,
        code: 500,
        originalMessage: 'internal',
      });

      assert.strictEqual(foreignUserFacing instanceof GenkitError, true);
      assert.strictEqual(foreignUserFacing instanceof UserFacingError, true);
      assert.strictEqual(foreignGenkitOnly instanceof GenkitError, true);
      assert.strictEqual(foreignGenkitOnly instanceof UserFacingError, false);
      assert.strictEqual(getHttpStatus(foreignUserFacing), 401);
      assert.deepStrictEqual(getCallableJSON(foreignUserFacing), {
        status: 'UNAUTHENTICATED',
        message: 'unauthenticated',
      });
    });
  });
});
