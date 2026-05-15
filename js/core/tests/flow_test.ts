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
import { defineFlow, run } from '../src/flow.js';
import { defineAction, getContext, z } from '../src/index.js';
import { initNodeFeatures } from '../src/node.js';
import { Registry } from '../src/registry.js';
import { enableTelemetry } from '../src/tracing.js';
import { TestSpanExporter } from './utils.js';

initNodeFeatures();

const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

function createTestFlow(registry: Registry) {
  return defineFlow(
    registry,
    {
      name: 'testFlow',
      inputSchema: z.string(),
      outputSchema: z.string(),
    },
    async (input) => {
      return `bar ${input}`;
    }
  );
}

describe('flow', () => {
  let registry: Registry;

  beforeEach(() => {
    // Skips starting reflection server.
    delete process.env.GENKIT_ENV;
    registry = new Registry();
  });

  describe('runFlow', () => {
    it('should run the flow', async () => {
      const testFlow = createTestFlow(registry);

      const result = await testFlow('foo');

      assert.equal(result, 'bar foo');
    });

    it('should set metadata on the flow action', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
          metadata: { foo: 'bar' },
        },
        async (input) => {
          return `bar ${input}`;
        }
      );

      assert.deepStrictEqual(testFlow.__action.metadata, { foo: 'bar' });
    });

    it('should run simple sync flow', async () => {
      const testFlow = defineFlow(registry, 'testFlow', (input) => {
        return `bar ${input}`;
      });

      const result = await testFlow('foo');

      assert.equal(result, 'bar foo');
    });

    it('should include trace info in the context', async () => {
      const testFlow = defineFlow(registry, 'testFlow', (_, ctx) => {
        return `traceId=${!!ctx.trace.traceId} spanId=${!!ctx.trace.spanId}`;
      });

      const result = await testFlow('foo');

      assert.equal(result, 'traceId=true spanId=true');
    });

    it('should rethrow the error', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'throwing',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input) => {
          throw new Error(`bad happened: ${input}`);
        }
      );

      await assert.rejects(() => testFlow('foo'), {
        name: 'Error',
        message: 'bad happened: foo',
      });
    });

    it('should validate input', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'validating',
          inputSchema: z.object({ foo: z.string(), bar: z.number() }),
          outputSchema: z.string(),
        },
        async (input) => {
          return `ok ${input}`;
        }
      );

      await assert.rejects(
        async () => await testFlow({ foo: 'foo', bar: 'bar' } as any),
        (err: Error) => {
          return (
            err.name === 'GenkitError' &&
            err.message.includes('Schema validation failed')
          );
        }
      );
    });
  });

  describe('getContext', () => {
    let registry: Registry;

    beforeEach(() => {
      registry = new Registry();
    });

    it('should run the flow', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input, ctx) => {
          return `bar ${input} ${JSON.stringify(ctx.context)}`;
        }
      );

      const response = await testFlow('foo', {
        context: { user: 'test-user' },
      });

      assert.equal(response, 'bar foo {"user":"test-user"}');
    });

    it('should streams the flow', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.number(),
          outputSchema: z.string(),
          streamSchema: z.object({ count: z.number() }),
        },
        async (input, streamingCallback) => {
          if (streamingCallback) {
            for (let i = 0; i < input; i++) {
              streamingCallback({ count: i });
            }
          }
          return `bar ${input} ${!!streamingCallback} ${JSON.stringify(getContext())}`;
        }
      );

      const response = testFlow.stream(3, {
        context: { user: 'test-user' },
      });

      const gotChunks: any[] = [];
      for await (const chunk of response.stream) {
        gotChunks.push(chunk);
      }

      assert.equal(await response.output, 'bar 3 true {"user":"test-user"}');
      assert.deepEqual(gotChunks, [{ count: 0 }, { count: 1 }, { count: 2 }]);
    });
  });

  describe('context', () => {
    it('should run the flow with context (old way)', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input) => {
          return `bar ${input} ${JSON.stringify(getContext())}`;
        }
      );

      const response = await testFlow('foo', {
        context: { user: 'test-user' },
      });

      assert.equal(response, 'bar foo {"user":"test-user"}');
    });

    it('should run the flow with context (new way)', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input, { context }) => {
          return `bar ${input} ${JSON.stringify(context)}`;
        }
      );

      const response = await testFlow('foo', {
        context: { user: 'test-user' },
      });

      assert.equal(response, 'bar foo {"user":"test-user"}');
    });

    it('should inherit context from the parent', async () => {
      const childFlow = defineFlow(
        registry,
        {
          name: 'childFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input, { context }) => {
          return `bar ${input} ${JSON.stringify(context)}`;
        }
      );

      const parentFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input) => {
          return childFlow(input);
        }
      );

      const response = await parentFlow('foo', {
        context: { user: 'test-user' },
      });

      assert.equal(response, 'bar foo {"user":"test-user"}');
    });

    it('should streams the flow with context', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.number(),
          outputSchema: z.string(),
          streamSchema: z.object({ count: z.number() }),
        },
        async (input, { sendChunk, context }) => {
          for (let i = 0; i < input; i++) {
            sendChunk({ count: i });
          }
          return `bar ${input} ${!!sendChunk} ${JSON.stringify(context)}`;
        }
      );

      const response = testFlow.stream(3, {
        context: { user: 'test-user' },
      });

      const gotChunks: any[] = [];
      for await (const chunk of response.stream) {
        gotChunks.push(chunk);
      }

      assert.equal(await response.output, 'bar 3 true {"user":"test-user"}');
      assert.deepEqual(gotChunks, [{ count: 0 }, { count: 1 }, { count: 2 }]);
    });

    it('aborts flow via signal', async () => {
      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input, { abortSignal }) => {
          let done = false;
          abortSignal.onabort = () => {
            done = true;
          };
          while (!done) {
            await new Promise((r) => setTimeout(r, 1));
          }
          return 'done ' + input;
        }
      );

      const ctrl = new AbortController();
      const responsePromise = testFlow('foo', {
        abortSignal: ctrl.signal,
      });

      await new Promise((r) => setTimeout(r, 4));
      console.log('abort');
      ctrl.abort();

      assert.equal(await responsePromise, 'done foo');
    });
  });

  describe('run', () => {
    it('should run the action', async () => {
      const act = async () => 'bar';

      const result = await run('action', act, registry);

      assert.equal(result, 'bar');
    });

    it('should run the action with input', async () => {
      const act = async (input: string) => `${input} bar`;

      const result = await run('action', 'foo', act, registry);

      assert.equal(result, 'foo bar');
    });
  });

  describe('telemetry', async () => {
    beforeEach(() => {
      spanExporter.exportedSpans = [];
    });

    it('should create a trace', async () => {
      const testFlow = createTestFlow(registry);

      const result = await testFlow('foo', {
        telemetryLabels: { custom: 'label' },
      });

      assert.equal(result, 'bar foo');
      assert.strictEqual(spanExporter.exportedSpans.length, 1);
      assert.strictEqual(spanExporter.exportedSpans[0].displayName, 'testFlow');
      assert.deepStrictEqual(spanExporter.exportedSpans[0].attributes, {
        'genkit:input': '"foo"',
        'genkit:isRoot': true,
        'genkit:key': '/flow/testFlow',
        'genkit:metadata:subtype': 'flow',
        'genkit:name': 'testFlow',
        'genkit:output': '"bar foo"',
        'genkit:path': '/{testFlow,t:flow}',
        'genkit:state': 'success',
        'genkit:type': 'action',
        custom: 'label',
      });
    });

    it('should create a trace when streaming', async () => {
      const testFlow = createTestFlow(registry);

      const { output } = testFlow.stream('foo', {
        telemetryLabels: { custom: 'label' },
      });
      const result = await output;

      assert.equal(result, 'bar foo');
      assert.strictEqual(spanExporter.exportedSpans.length, 1);
      assert.strictEqual(spanExporter.exportedSpans[0].displayName, 'testFlow');
      assert.deepStrictEqual(spanExporter.exportedSpans[0].attributes, {
        'genkit:input': '"foo"',
        'genkit:isRoot': true,
        'genkit:key': '/flow/testFlow',
        'genkit:metadata:subtype': 'flow',
        'genkit:metadata:context': '{}',
        'genkit:name': 'testFlow',
        'genkit:output': '"bar foo"',
        'genkit:path': '/{testFlow,t:flow}',
        'genkit:state': 'success',
        'genkit:type': 'action',
        custom: 'label',
      });
    });

    it('records traces of nested actions', async () => {
      const testAction = defineAction(
        registry,
        {
          name: 'testAction',
          actionType: 'tool',
          metadata: { type: 'tool' },
        },
        async (i) => {
          return 'bar';
        }
      );

      const testFlow = defineFlow(
        registry,
        {
          name: 'testFlow',
          inputSchema: z.string(),
          outputSchema: z.string(),
        },
        async (input) => {
          return run(
            'custom',
            async () => {
              return 'foo ' + (await testAction(undefined));
            },
            registry
          );
        }
      );
      const result = await testFlow('foo', {
        context: { user: 'pavel' },
      });

      assert.equal(result, 'foo bar');
      assert.strictEqual(spanExporter.exportedSpans.length, 3);

      assert.strictEqual(
        spanExporter.exportedSpans[0].displayName,
        'testAction'
      );
      assert.deepStrictEqual(spanExporter.exportedSpans[0].attributes, {
        'genkit:key': '/tool/testAction',
        'genkit:metadata:subtype': 'tool',
        'genkit:name': 'testAction',
        'genkit:output': '"bar"',
        'genkit:path':
          '/{testFlow,t:flow}/{custom,t:flowStep}/{testAction,t:action,s:tool}',
        'genkit:state': 'success',
        'genkit:type': 'action',
      });

      assert.strictEqual(spanExporter.exportedSpans[1].displayName, 'custom');
      assert.deepStrictEqual(spanExporter.exportedSpans[1].attributes, {
        'genkit:name': 'custom',
        'genkit:output': '"foo bar"',
        'genkit:path': '/{testFlow,t:flow}/{custom,t:flowStep}',
        'genkit:state': 'success',
        'genkit:type': 'flowStep',
      });

      assert.strictEqual(spanExporter.exportedSpans[2].displayName, 'testFlow');
      assert.deepStrictEqual(spanExporter.exportedSpans[2].attributes, {
        'genkit:input': '"foo"',
        'genkit:isRoot': true,
        'genkit:key': '/flow/testFlow',
        'genkit:metadata:subtype': 'flow',
        'genkit:metadata:context': '{"user":"pavel"}',
        'genkit:name': 'testFlow',
        'genkit:output': '"foo bar"',
        'genkit:path': '/{testFlow,t:flow}',
        'genkit:state': 'success',
        'genkit:type': 'action',
      });
    });
  });
});
