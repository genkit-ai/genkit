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
import { GenkitError, UserFacingError, isGenkitError } from '../src/error.js';
import {
  StreamNotFoundError,
  isStreamNotFoundError,
} from '../src/streaming.js';

/** A stand-in for `StreamNotFoundError` from another copy of `@genkit-ai/core`. */
class ForeignStreamNotFoundError extends Error {
  toJSON() {
    return { status: 'NOT_FOUND', message: this.message };
  }
}
Object.defineProperty(
  ForeignStreamNotFoundError.prototype,
  Symbol.for('genkit.error'),
  { value: true }
);
Object.defineProperty(
  ForeignStreamNotFoundError.prototype,
  Symbol.for('genkit.streamNotFoundError'),
  { value: true }
);

describe('isStreamNotFoundError', () => {
  it('recognises a StreamNotFoundError from this or another copy', () => {
    assert.strictEqual(
      isStreamNotFoundError(new StreamNotFoundError('x')),
      true
    );
    assert.strictEqual(
      isStreamNotFoundError(new ForeignStreamNotFoundError()),
      true
    );
    assert.strictEqual(isGenkitError(new StreamNotFoundError('x')), true);
  });

  it('rejects a value that carries only the stream brand', () => {
    const err = Object.assign(new Error('x'), {
      [Symbol.for('genkit.streamNotFoundError')]: true,
    });
    assert.strictEqual(isStreamNotFoundError(err), false);
    assert.strictEqual(err instanceof StreamNotFoundError, false);
  });

  it('rejects other errors', () => {
    for (const v of [
      new GenkitError({ status: 'NOT_FOUND', message: 'x' }),
      new UserFacingError('NOT_FOUND', 'x'),
      new Error('x'),
      null,
    ]) {
      assert.strictEqual(isStreamNotFoundError(v), false);
    }
  });
});

describe('StreamNotFoundError instanceof', () => {
  it('matches a StreamNotFoundError from another copy', () => {
    const foreign = new ForeignStreamNotFoundError();
    assert.strictEqual(foreign instanceof StreamNotFoundError, true);
    assert.strictEqual(foreign instanceof GenkitError, true);
  });

  it('keeps the prototype check for other classes', () => {
    const base = new GenkitError({ status: 'NOT_FOUND', message: 'x' });
    assert.strictEqual(base instanceof StreamNotFoundError, false);
    assert.strictEqual(
      new StreamNotFoundError('x') instanceof UserFacingError,
      false
    );
    assert.strictEqual(
      new StreamNotFoundError('x') instanceof GenkitError,
      true
    );
  });
});
