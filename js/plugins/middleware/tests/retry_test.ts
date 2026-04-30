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
import { GenkitError, genkit } from 'genkit';
import { describe, it } from 'node:test';
import { TEST_ONLY, retry } from '../src/retry.js';

describe('retry', () => {
  it('should not retry on success', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [retry()],
    });

    assert.strictEqual(requestCount, 1);
    assert.strictEqual(result.text, 'success');
  });

  it('should retry on a retryable GenkitError', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      if (requestCount < 3) {
        throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
      }
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    TEST_ONLY.setRetryTimeout((callback, ms) => {
      callback();
      return 0 as any;
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [retry({ maxRetries: 3 })],
    });

    assert.strictEqual(requestCount, 3);
    assert.strictEqual(result.text, 'success');
  });

  it('should retry on a non-GenkitError', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      if (requestCount < 2) {
        throw new Error('generic error');
      }
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    TEST_ONLY.setRetryTimeout((callback, ms) => {
      callback();
      return 0 as any;
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [retry({ maxRetries: 2 })],
    });

    assert.strictEqual(requestCount, 2);
    assert.strictEqual(result.text, 'success');
  });

  it('should throw after exhausting retries', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    });

    TEST_ONLY.setRetryTimeout((callback, ms) => {
      callback();
      return 0 as any;
    });

    await assert.rejects(
      ai.generate({
        model: pm,
        prompt: 'test',
        use: [retry({ maxRetries: 2 })],
      }),
      /UNAVAILABLE: test/
    );

    assert.strictEqual(requestCount, 3);
  });

  it('should not retry on non-retryable status', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      throw new GenkitError({ status: 'INVALID_ARGUMENT', message: 'test' });
    });

    await assert.rejects(
      ai.generate({
        model: pm,
        prompt: 'test',
        use: [retry({ maxRetries: 2 })],
      }),
      /INVALID_ARGUMENT: test/
    );

    assert.strictEqual(requestCount, 1);
  });

  it('should respect initial delay', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      if (requestCount < 2) {
        throw new Error('generic error');
      }
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    let totalDelay = 0;
    TEST_ONLY.setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [retry({ maxRetries: 2, initialDelayMs: 50, noJitter: true })],
    });

    assert.strictEqual(requestCount, 2);
    assert.strictEqual(result.text, 'success');
    assert.strictEqual(totalDelay, 50);
  });

  it('should respect backoff factor', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      if (requestCount < 3) {
        throw new Error('generic error');
      }
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    let totalDelay = 0;
    TEST_ONLY.setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [
        retry({
          maxRetries: 3,
          initialDelayMs: 20,
          backoffFactor: 2,
          noJitter: true,
        }),
      ],
    });

    assert.strictEqual(requestCount, 3);
    assert.strictEqual(result.text, 'success');
    assert.strictEqual(totalDelay, 20 + 40);
  });

  it('should apply jitter', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      if (requestCount < 2) {
        throw new Error('generic error');
      }
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    let totalDelay = 0;
    TEST_ONLY.setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [
        retry({
          maxRetries: 2,
          initialDelayMs: 50,
          noJitter: false, // do jitter
        }),
      ],
    });

    assert.strictEqual(requestCount, 2);
    assert.strictEqual(result.text, 'success');
    assert.ok(totalDelay >= 50);
    assert.ok(totalDelay <= 1050);
  });

  it('should respect max delay', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      if (requestCount < 3) {
        throw new Error('generic error');
      }
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    let totalDelay = 0;
    TEST_ONLY.setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [
        retry({
          maxRetries: 3,
          initialDelayMs: 50,
          maxDelayMs: 60,
          backoffFactor: 2,
          noJitter: true,
        }),
      ],
    });

    assert.strictEqual(requestCount, 3);
    assert.strictEqual(result.text, 'success');
    assert.strictEqual(totalDelay, 50 + 60);
  });
});
