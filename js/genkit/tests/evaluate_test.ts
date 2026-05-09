/**
 * Copyright 2024 Google LLC
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
import { beforeEach, describe, it } from 'node:test';
import { genkit, type Genkit } from '../src/index.js';
import { bonknessEvaluator } from './helpers';

describe('evaluate', () => {
  describe('default model', () => {
    let ai: Genkit;

    beforeEach(() => {
      ai = genkit({});
      bonknessEvaluator(ai);
    });

    it('calls evaluator', async () => {
      const response = await ai.evaluate({
        evaluator: 'bonkness',
        dataset: [
          {
            testCaseId: 'test-case-1',
            input: 'Why did the chicken cross the road?',
            output: 'To get to the other side',
          },
        ],
        evalRunId: 'my-dog-eval',
      });

      assert.strictEqual(response.length, 1);
      assert.strictEqual(response[0].evaluation.score, 'Much bonk');
      assert.strictEqual(
        response[0].evaluation.details?.reasoning,
        'Because I said so!'
      );
    });
  });
});
