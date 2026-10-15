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

import type Anthropic from '@anthropic-ai/sdk';
import * as assert from 'assert';
import { genkit } from 'genkit';
import { describe, test } from 'node:test';
import { anthropic } from '../src/index.js';
import { __testClient } from '../src/types.js';
import {
  createMockAnthropicClient,
  createMockAnthropicMessage,
} from './mocks/anthropic-client.js';

const MODEL = 'anthropic/claude-sonnet-4-5';

function setup(apiVersion?: 'stable' | 'beta') {
  const mockClient = createMockAnthropicClient({
    messageResponse: createMockAnthropicMessage({ text: 'ok' }),
  }) as Anthropic;
  const ai = genkit({
    plugins: [
      anthropic({
        apiVersion,
        // @ts-ignore
        [__testClient]: mockClient,
      }),
    ],
  });
  return { ai, mockClient };
}

describe('betas config', () => {
  test('uses the beta API when betas is set and the plugin default is stable', async () => {
    const { ai, mockClient } = setup('stable');

    await ai.generate({
      model: MODEL,
      prompt: 'hi',
      config: { betas: ['token-efficient-tools-2025-02-19'] },
    });

    const beta = (mockClient.beta.messages.create as any).mock.calls;
    const stable = (mockClient.messages.create as any).mock.calls;
    assert.strictEqual(beta.length, 1);
    assert.strictEqual(stable.length, 0);
    assert.deepStrictEqual(beta[0].arguments[0].betas, [
      'token-efficient-tools-2025-02-19',
    ]);
  });

  test('does not leak an empty betas array into the stable request body', async () => {
    const { ai, mockClient } = setup();

    await ai.generate({ model: MODEL, prompt: 'hi', config: { betas: [] } });

    const stable = (mockClient.messages.create as any).mock.calls;
    assert.strictEqual(stable.length, 1);
    assert.ok(!('betas' in stable[0].arguments[0]));
  });

  test('rejects betas with an explicit stable apiVersion through the model', async () => {
    const { ai, mockClient } = setup();

    await assert.rejects(
      ai.generate({
        model: MODEL,
        prompt: 'hi',
        config: { apiVersion: 'stable', betas: ['x'] },
      }),
      (err: any) => err.status === 'INVALID_ARGUMENT'
    );
    assert.strictEqual(
      (mockClient.messages.create as any).mock.calls.length,
      0
    );
    assert.strictEqual(
      (mockClient.beta.messages.create as any).mock.calls.length,
      0
    );
  });
});
