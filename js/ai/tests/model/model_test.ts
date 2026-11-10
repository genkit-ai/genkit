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

import {
  ActionFnArg,
  sentinelNoopStreamingCallback,
  StreamingCallback,
} from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import assert from 'assert';
import { Registry } from 'genkit/registry';
import { describe, it } from 'node:test';
import { z } from 'zod';
import { z as z3 } from 'zod/v3';
import {
  defineModel,
  GenerateResponseChunkData,
  GenerateResponseData,
  GenerationCommonConfigSchema,
  model,
} from '../../src/model.js';

initNodeFeatures();

const GENERATE_RESPONSE = {
  finishReason: 'stop',
  message: {
    role: 'model',
    content: [{ text: 'hi' }],
  },
} as GenerateResponseData;

describe('model', () => {
  describe('v1', () => {
    it('defines a model', async () => {
      const registry = new Registry();
      let calledWithStreamingCallback:
        | StreamingCallback<GenerateResponseChunkData>
        | undefined;
      const model = defineModel(
        registry,
        {
          name: 'foo',
        },
        async (_, streamingCallback) => {
          calledWithStreamingCallback = streamingCallback;
          return GENERATE_RESPONSE;
        }
      );
      let response = await model({ messages: [] });

      assert.strictEqual(calledWithStreamingCallback, undefined);
      delete response.latencyMs;
      assert.deepStrictEqual(response, GENERATE_RESPONSE);

      const { output } = model.stream({ messages: [] });
      response = await output;

      assert.notStrictEqual(calledWithStreamingCallback, undefined);
      delete response.latencyMs;
      assert.deepStrictEqual(response, GENERATE_RESPONSE);
    });
  });

  describe('v2', () => {
    it('defines a model', async () => {
      const registry = new Registry();
      let calledWithOptions: ActionFnArg<GenerateResponseChunkData> | undefined;
      const model = defineModel(
        registry,
        {
          apiVersion: 'v2',
          name: 'foo',
        },
        async (_, opts) => {
          calledWithOptions = opts;
          opts.sendChunk({
            content: [{ text: 'success' }],
          });
          return GENERATE_RESPONSE;
        }
      );
      let response = await model({ messages: [] });

      assert.ok(calledWithOptions!);
      assert.strictEqual(
        calledWithOptions.sendChunk,
        sentinelNoopStreamingCallback
      );
      assert.strictEqual(calledWithOptions.streamingRequested, false);
      delete response.latencyMs;
      assert.deepStrictEqual(response, GENERATE_RESPONSE);

      const { output, stream } = model.stream({ messages: [] });

      const chunks = [] as GenerateResponseChunkData[];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      response = await output;

      assert.ok(calledWithOptions!);
      assert.ok(calledWithOptions.sendChunk);
      assert.notStrictEqual(
        calledWithOptions.sendChunk,
        sentinelNoopStreamingCallback
      );
      assert.strictEqual(calledWithOptions.streamingRequested, true);
      delete response.latencyMs;
      assert.deepStrictEqual(response, GENERATE_RESPONSE);
      assert.deepStrictEqual(chunks, [{ content: [{ text: 'success' }] }]);
    });

    it('defines a dynamic model', async () => {
      let calledWithOptions: ActionFnArg<GenerateResponseChunkData> | undefined;
      const testModel = model(
        {
          name: 'foo',
        },
        async (_, opts) => {
          calledWithOptions = opts;
          opts.sendChunk({
            content: [{ text: 'success' }],
          });
          return GENERATE_RESPONSE;
        }
      );
      let response = await testModel({ messages: [] });

      assert.ok(calledWithOptions!);
      assert.strictEqual(
        calledWithOptions.sendChunk,
        sentinelNoopStreamingCallback
      );
      assert.strictEqual(calledWithOptions.streamingRequested, false);
      delete response.latencyMs;
      assert.deepStrictEqual(response, GENERATE_RESPONSE);

      const { output, stream } = testModel.stream({ messages: [] });

      const chunks = [] as GenerateResponseChunkData[];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      response = await output;

      assert.ok(calledWithOptions!);
      assert.ok(calledWithOptions.sendChunk);
      assert.notStrictEqual(
        calledWithOptions.sendChunk,
        sentinelNoopStreamingCallback
      );
      assert.strictEqual(calledWithOptions.streamingRequested, true);
      delete response.latencyMs;
      assert.deepStrictEqual(response, GENERATE_RESPONSE);
      assert.deepStrictEqual(chunks, [{ content: [{ text: 'success' }] }]);
    });
  });
});

describe('model configSchema checks', () => {
  it('names the model when its config schema mixes in zod v3', () => {
    const registry = new Registry();
    assert.throws(
      () =>
        defineModel(
          registry,
          {
            name: 'legacy',
            // A plugin extending a genkit schema with its own zod v3.
            configSchema: GenerationCommonConfigSchema.extend({
              seed: z3
                .number()
                .optional() as unknown as z.ZodOptional<z.ZodNumber>,
            }),
          },
          async () => GENERATE_RESPONSE
        ),
      /configSchema of model "legacy" is not supported: found a zod v3 schema/
    );
  });
});

describe('model wire schemas', () => {
  // Request and response are parsed with their schemas. Keys the schemas
  // don't know (from a plugin, or a newer runtime) must survive, at every
  // level.
  it('keeps unknown keys in the request and the response', async () => {
    const registry = new Registry();
    const request = {
      messages: [
        {
          role: 'user',
          content: [
            { text: 'hi', extraPartKey: 1 },
            { media: { url: 'http://x', extraMediaKey: 1 } },
            {
              toolResponse: { name: 't', output: {}, extraToolResponseKey: 1 },
            },
          ],
          extraMessageKey: 1,
        },
      ],
      tools: [{ name: 't', description: 'd', extraToolKey: 1 }],
      output: { format: 'json', extraOutputKey: 1 },
      docs: [{ content: [{ text: 'doc' }], extraDocKey: 1 }],
      extraRequestKey: 1,
    };
    const response = {
      finishReason: 'stop',
      message: {
        role: 'model',
        content: [
          { text: 'hi', extraPartKey: 1 },
          { toolRequest: { name: 't', input: {}, extraToolRequestKey: 1 } },
          { futurePartKind: { x: 1 } },
        ],
        extraMessageKey: 1,
      },
      usage: { inputTokens: 1, extraUsageKey: 1 },
      extraResponseKey: 1,
    };
    let seenRequest: unknown;
    const echo = defineModel(registry, { name: 'echo' }, async (req) => {
      seenRequest = req;
      // Deliberately not a GenerateResponseData: it carries keys the type
      // doesn't declare.
      return response as GenerateResponseData;
    });

    const { latencyMs: _, ...result } = await echo(request);

    assert.deepStrictEqual(seenRequest, request);
    assert.deepStrictEqual(result, response);
  });
});
