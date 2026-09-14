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

import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { z } from 'zod';
import { action } from '../src/action.js';
import { initNodeFeatures } from '../src/node.js';
import { enableTelemetry } from '../src/tracing.js';
import { TestSpanExporter } from './utils.js';

initNodeFeatures();

const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

const CREDENTIAL = 'test-credential-value';

/** Runs an action that echoes its input back, so input and output both carry it. */
async function runEchoAction(input: unknown) {
  const act = action(
    {
      name: 'echoAction',
      inputSchema: z.any(),
      outputSchema: z.any(),
      actionType: 'custom',
    },
    async (input) => input
  );
  await act.run(input);
  const span = spanExporter.exportedSpans.find(
    (s) => s.displayName === 'echoAction'
  );
  assert.ok(span, 'echoAction span should be exported');
  return span;
}

describe('span attribute credential redaction', () => {
  beforeEach(() => {
    spanExporter.exportedSpans = [];
  });

  it('redacts credential-shaped keys from genkit:input and genkit:output', async () => {
    const span = await runEchoAction({
      config: {
        apiKey: CREDENTIAL,
        api_key: CREDENTIAL,
        accessToken: CREDENTIAL,
        clientSecret: CREDENTIAL,
        password: CREDENTIAL,
        authorization: CREDENTIAL,
        temperature: 0.5,
      },
    });

    for (const attr of ['genkit:input', 'genkit:output']) {
      const raw = span.attributes[attr] as string;
      assert.ok(raw, `${attr} should be exported`);
      assert.ok(
        !raw.includes(CREDENTIAL),
        `${attr} should not contain the credential value: ${raw}`
      );
      const parsed = JSON.parse(raw);
      // Keys are preserved with a redacted value, not dropped.
      for (const key of [
        'apiKey',
        'api_key',
        'accessToken',
        'clientSecret',
        'password',
        'authorization',
      ]) {
        assert.strictEqual(
          parsed.config[key],
          '<redacted>',
          `${attr}: ${key} should be '<redacted>'`
        );
      }
      assert.strictEqual(
        parsed.config.temperature,
        0.5,
        `${attr}: non-sensitive config should be preserved`
      );
    }
  });

  it('redacts arbitrary passthrough credential fields nested in arrays', async () => {
    const span = await runEchoAction({
      items: [{ sessionToken: CREDENTIAL }, { nested: { secret: CREDENTIAL } }],
    });

    const raw = span.attributes['genkit:input'] as string;
    assert.ok(!raw.includes(CREDENTIAL), raw);
    const parsed = JSON.parse(raw);
    assert.strictEqual(parsed.items[0].sessionToken, '<redacted>');
    assert.strictEqual(parsed.items[1].nested.secret, '<redacted>');
  });

  it('does not redact token-count fields that trace consumers read', async () => {
    const span = await runEchoAction({
      config: { maxOutputTokens: 100 },
      usage: { inputTokens: 7, outputTokens: 11, totalTokens: 18 },
    });

    const parsed = JSON.parse(span.attributes['genkit:input'] as string);
    assert.strictEqual(parsed.config.maxOutputTokens, 100);
    assert.deepStrictEqual(parsed.usage, {
      inputTokens: 7,
      outputTokens: 11,
      totalTokens: 18,
    });
  });

  it('leaves values without credential-shaped keys byte-identical', async () => {
    const input = { messages: [{ role: 'user', content: [{ text: 'hi' }] }] };
    const span = await runEchoAction(input);

    assert.strictEqual(
      span.attributes['genkit:input'],
      JSON.stringify(input),
      'unaffected inputs must serialize exactly as before'
    );
  });
});
