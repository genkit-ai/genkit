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
import { fallback } from '../src/fallback.js';

describe('fallback', () => {
  it('should not fallback on success', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      return { message: { role: 'model', content: [{ text: 'success' }] } };
    });

    let pmFallbackRequestCount = 0;
    const pmFallback = ai.defineModel(
      { name: 'programmableModelFallback' },
      async (req) => {
        pmFallbackRequestCount++;
        return {
          message: { role: 'model', content: [{ text: 'fallback success' }] },
        };
      }
    );

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [fallback({ models: [pmFallback] })],
    });

    assert.strictEqual(requestCount, 1);
    assert.strictEqual(pmFallbackRequestCount, 0);
    assert.strictEqual(result.text, 'success');
  });

  it('should fallback on a fallbackable error', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      throw new GenkitError({ status: 'RESOURCE_EXHAUSTED', message: 'test' });
    });

    let pmFallbackRequestCount = 0;
    ai.defineModel({ name: 'programmableModelFallback' }, async (req) => {
      pmFallbackRequestCount++;
      return {
        message: { role: 'model', content: [{ text: 'fallback success' }] },
      };
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'test',
      use: [
        fallback({
          models: [{ name: 'programmableModelFallback' }],
          statuses: ['RESOURCE_EXHAUSTED'],
        }),
      ],
    });

    assert.strictEqual(requestCount, 1);
    assert.strictEqual(pmFallbackRequestCount, 1);
    assert.strictEqual(result.text, 'fallback success');
  });

  it('should throw after all fallbacks fail', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    });

    let pmFallbackRequestCount = 0;
    ai.defineModel({ name: 'programmableModelFallback' }, async (req) => {
      pmFallbackRequestCount++;
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test2' });
    });

    await assert.rejects(
      ai.generate({
        model: pm,
        prompt: 'test',
        use: [
          fallback({
            models: [{ name: 'programmableModelFallback' }],
          }),
        ],
      }),
      /UNAVAILABLE: test2/
    );

    assert.strictEqual(requestCount, 1);
    assert.strictEqual(pmFallbackRequestCount, 1);
  });

  it('should not fallback on non-fallbackable error', async () => {
    let requestCount = 0;
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      requestCount++;
      throw new GenkitError({ status: 'INVALID_ARGUMENT', message: 'test' });
    });

    let pmFallbackRequestCount = 0;
    ai.defineModel({ name: 'programmableModelFallback' }, async (req) => {
      pmFallbackRequestCount++;
      return {
        message: { role: 'model', content: [{ text: 'fallback success' }] },
      };
    });

    await assert.rejects(
      ai.generate({
        model: pm,
        prompt: 'test',
        use: [
          fallback({
            models: [{ name: 'programmableModelFallback' }],
            statuses: ['RESOURCE_EXHAUSTED'],
          }),
        ],
      }),
      /INVALID_ARGUMENT: test/
    );

    assert.strictEqual(requestCount, 1);
    assert.strictEqual(pmFallbackRequestCount, 0);
  });

  it('should not inherit config when isolateConfig is true', async () => {
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    });

    let receivedConfig: any = null;
    ai.defineModel({ name: 'programmableModelFallback' }, async (req) => {
      receivedConfig = req.config;
      return {
        message: { role: 'model', content: [{ text: 'fallback success' }] },
      };
    });

    await ai.generate({
      model: pm,
      prompt: 'test',
      config: { temperature: 0.5 },
      use: [
        fallback({
          models: [{ name: 'programmableModelFallback' }],
          isolateConfig: true,
        }),
      ],
    });

    assert.strictEqual(receivedConfig, undefined);
  });

  it('should inherit config when isolateConfig is false (default)', async () => {
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'programmableModel' }, async (req) => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    });

    let receivedConfig: any = null;
    ai.defineModel({ name: 'programmableModelFallback' }, async (req) => {
      receivedConfig = req.config;
      return {
        message: { role: 'model', content: [{ text: 'fallback success' }] },
      };
    });

    await ai.generate({
      model: pm,
      prompt: 'test',
      config: { temperature: 0.5 },
      use: [
        fallback({
          models: [{ name: 'programmableModelFallback' }],
        }),
      ],
    });

    assert.strictEqual(receivedConfig.temperature, 0.5);
  });
});
