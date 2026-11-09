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
import { before, describe, it } from 'node:test';
import {
  GenkitError,
  UserFacingError,
  getCallableJSON,
  getHttpStatus,
} from '../src/error.js';

// A query string makes Node load a second, independent instance of the module.
const errorSpecifier = '../src/error.js?copy=2';
let other: typeof import('../src/error.js');

before(async () => {
  other = await import(errorSpecifier);
});

describe('GenkitError from another copy of the module', () => {
  it('is a GenkitError here, but not a UserFacingError from here', () => {
    assert.notStrictEqual(other.GenkitError, GenkitError);
    const err = new other.UserFacingError('PERMISSION_DENIED', 'nope');
    assert.ok(err instanceof GenkitError);
    assert.ok(!(err instanceof UserFacingError));
    assert.ok(
      new other.GenkitError({ status: 'INTERNAL', message: 'x' }) instanceof
        GenkitError
    );
  });

  it('is recognised the same way in the other direction', () => {
    assert.notStrictEqual(other.UserFacingError, UserFacingError);
    const err = new UserFacingError('PERMISSION_DENIED', 'nope');
    assert.ok(err instanceof other.GenkitError);
    assert.ok(!(err instanceof other.UserFacingError));
    assert.ok(
      new GenkitError({ status: 'INTERNAL', message: 'x' }) instanceof
        other.GenkitError
    );
  });

  it('maps to its HTTP status and wire format', () => {
    const err = new other.UserFacingError('PERMISSION_DENIED', 'nope');
    assert.strictEqual(getHttpStatus(err), 403);
    assert.deepStrictEqual(getCallableJSON(err), {
      status: 'PERMISSION_DENIED',
      message: 'nope',
    });
  });
});
