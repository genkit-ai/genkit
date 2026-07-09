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
  StatusCodes,
  httpStatusCode,
  statusNameToCode,
} from '../src/statusTypes.js';

describe('statusTypes', () => {
  describe('statusNameToCode', () => {
    it('maps valid status names to numeric StatusCodes', () => {
      assert.strictEqual(statusNameToCode('OK'), StatusCodes.OK);
      assert.strictEqual(statusNameToCode('NOT_FOUND'), StatusCodes.NOT_FOUND);
      assert.strictEqual(statusNameToCode('INTERNAL'), StatusCodes.INTERNAL);
      assert.strictEqual(
        statusNameToCode('INVALID_ARGUMENT'),
        StatusCodes.INVALID_ARGUMENT
      );
    });

    it('returns undefined for reverse-mapping numeric string keys or invalid names', () => {
      assert.strictEqual(statusNameToCode('5'), undefined);
      assert.strictEqual(statusNameToCode('999'), undefined);
      assert.strictEqual(statusNameToCode('NOT_A_STATUS'), undefined);
      assert.strictEqual(statusNameToCode(undefined), undefined);
    });
  });

  describe('httpStatusCode', () => {
    it('maps status names to HTTP status codes', () => {
      assert.strictEqual(httpStatusCode('OK'), 200);
      assert.strictEqual(httpStatusCode('NOT_FOUND'), 404);
      assert.strictEqual(httpStatusCode('INTERNAL'), 500);
    });
  });
});
