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
import { genkit, type Genkit } from 'genkit';
import type { AuthClient } from 'google-auth-library';
import { beforeEach, describe, it } from 'node:test';
import { ChecksEvaluationMetricType, checksMiddleware } from '../src/index.js';

/** Flags any text containing `violative` as DANGEROUS_CONTENT. */
function fakeAuthClient(requests: { content: string; project?: string }[]) {
  const client = {
    quotaProjectId: undefined,
    async request(opts: { body: string; headers: Record<string, string> }) {
      const content: string = JSON.parse(opts.body).input.text_input.content;
      requests.push({ content, project: opts.headers['x-goog-user-project'] });
      return {
        data: {
          policyResults: [
            {
              policyType: 'DANGEROUS_CONTENT',
              violationResult: content.includes('violative')
                ? 'VIOLATIVE'
                : 'NON_VIOLATIVE',
            },
          ],
        },
      };
    },
  };
  // Only `request` and `quotaProjectId` are used by the plugin.
  return client as unknown as AuthClient;
}

describe('checksMiddleware', () => {
  let ai: Genkit;
  let requests: { content: string; project?: string }[];

  beforeEach(() => {
    requests = [];
    ai = genkit({
      plugins: [
        checksMiddleware.plugin({
          authOptions: {
            authClient: fakeAuthClient(requests),
            projectId: 'my-project',
          },
        }),
      ],
    });
    ai.defineModel({ name: 'echo' }, async (req) => ({
      message: {
        role: 'model',
        content: [{ text: `echo: ${req.messages.at(-1)?.content[0].text}` }],
      },
      finishReason: 'stop',
    }));
  });

  const use = [
    checksMiddleware({
      metrics: [ChecksEvaluationMetricType.DANGEROUS_CONTENT],
    }),
  ];

  it('passes through when nothing is violated', async () => {
    const response = await ai.generate({ model: 'echo', prompt: 'hi', use });

    assert.strictEqual(response.text, 'echo: hi');
    assert.deepStrictEqual(requests, [
      { content: 'hi', project: 'my-project' },
      { content: 'echo: hi', project: 'my-project' },
    ]);
  });

  it('blocks violating input before calling the model', async () => {
    const response = await ai.generate({
      model: 'echo',
      prompt: 'violative',
      use,
      throwOnError: false,
    });

    assert.strictEqual(response.finishReason, 'blocked');
    assert.match(response.finishMessage!, /Model input violated/);
    assert.deepStrictEqual(
      requests.map((r) => r.content),
      ['violative']
    );
  });

  it('blocks violating output', async () => {
    ai.defineModel({ name: 'bad' }, async () => ({
      message: { role: 'model', content: [{ text: 'violative reply' }] },
      finishReason: 'stop',
    }));

    const response = await ai.generate({
      model: 'bad',
      prompt: 'hi',
      use,
      throwOnError: false,
    });

    assert.strictEqual(response.finishReason, 'blocked');
    assert.match(response.finishMessage!, /Model output violated/);
  });

  it('keeps auth options out of the traced middleware config', async () => {
    const ref = checksMiddleware({
      metrics: [ChecksEvaluationMetricType.DANGEROUS_CONTENT],
    });
    assert.deepStrictEqual(JSON.parse(JSON.stringify(ref)), {
      name: 'checks',
      config: { metrics: ['DANGEROUS_CONTENT'] },
    });
  });
});
