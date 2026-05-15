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
import { action, defineAction, defineActionAsync } from '../src/action.js';
import { initNodeFeatures } from '../src/node.js';
import { Registry } from '../src/registry.js';
import { enableTelemetry } from '../src/tracing.js';
import { TestSpanExporter } from './utils.js';

initNodeFeatures();

const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

describe('action', () => {
  var registry: Registry;
  beforeEach(() => {
    registry = new Registry();
    spanExporter.exportedSpans = [];
  });

  it('applies middleware', async () => {
    const act = action(
      {
        name: 'foo',
        inputSchema: z.string(),
        outputSchema: z.number(),
        use: [
          async (input, next) => (await next(input + 'middle1')) + 1,
          async (input, opts, next) =>
            (await next(input + 'middle2', opts)) + 2,
        ],
        actionType: 'util',
      },
      async (input) => {
        return input.length;
      }
    );

    assert.strictEqual(
      await act('foo'),
      20 // "foomiddle1middle2".length + 1 + 2
    );
  });

  it('returns telemetry info', async () => {
    const act = action(
      {
        name: 'foo',
        inputSchema: z.string(),
        outputSchema: z.number(),
        use: [
          async (input, next) => (await next(input + 'middle1')) + 1,
          async (input, opts, next) =>
            (await next(input + 'middle2', opts)) + 2,
        ],
        actionType: 'util',
      },
      async (input) => {
        return input.length;
      }
    );

    const result = await act.run('foo');
    assert.strictEqual(
      result.result,
      20 // "foomiddle1middle2".length + 1 + 2
    );
    assert.strictEqual(result.telemetry !== null, true);
    assert.strictEqual(
      result.telemetry.traceId !== null && result.telemetry.traceId.length > 0,
      true
    );
    assert.strictEqual(
      result.telemetry.spanId !== null && result.telemetry.spanId.length > 0,
      true
    );
  });

  it('run the action with options', async () => {
    let passedContext;
    const act = action(
      {
        name: 'foo',
        inputSchema: z.string(),
        outputSchema: z.number(),
        actionType: 'util',
      },
      async (input, { sendChunk, context }) => {
        passedContext = context;
        sendChunk(1);
        sendChunk(2);
        sendChunk(3);
        return input.length;
      }
    );

    const chunks: any[] = [];
    await act.run('1234', {
      context: { foo: 'bar' },
      onChunk: (c) => chunks.push(c),
    });

    assert.deepStrictEqual(passedContext, {
      foo: 'bar',
    });

    assert.deepStrictEqual(chunks, [1, 2, 3]);
  });

  it('runs the action with context plus registry global context', async () => {
    let passedContext;
    let calledWithStreamingRequestedValue;
    const act = action(
      {
        name: 'foo',
        inputSchema: z.string(),
        outputSchema: z.number(),
        actionType: 'util',
      },
      async (input, { sendChunk, context, streamingRequested }) => {
        calledWithStreamingRequestedValue = streamingRequested;
        passedContext = context;
        sendChunk(1);
        sendChunk(2);
        sendChunk(3);
        return input.length;
      }
    );

    registry.context = { bar: 'baz' };
    act.__registry = registry;

    await act.run('1234', {
      context: { foo: 'bar' },
    });

    assert.strictEqual(calledWithStreamingRequestedValue, false);
    assert.deepStrictEqual(passedContext, {
      foo: 'bar',
      bar: 'baz', // these come from glboal registry context
    });

    registry.context = { bar2: 'baz2' };
    const { output } = act.stream('1234', {
      context: { foo2: 'bar2' },
    });
    await output;

    assert.strictEqual(calledWithStreamingRequestedValue, true);
    assert.deepStrictEqual(passedContext, {
      foo2: 'bar2',
      bar2: 'baz2', // these come from glboal registry context
    });
  });

  it('should stream the response', async () => {
    const action = defineAction(
      registry,
      { name: 'hello', actionType: 'custom' },
      async (input, { sendChunk, streamingRequested }) => {
        sendChunk({ count: 1 });
        sendChunk({ count: 2 });
        sendChunk({ count: 3 });
        return `hi ${input}`;
      }
    );

    const response = action.stream('Pavel');

    const gotChunks: any[] = [];
    for await (const chunk of response.stream) {
      gotChunks.push(chunk);
    }

    assert.equal(await response.output, 'hi Pavel');
    assert.deepStrictEqual(gotChunks, [
      { count: 1 },
      { count: 2 },
      { count: 3 },
    ]);
  });

  it('should inherit context from parent action invocation', async () => {
    const child = defineAction(
      registry,
      { name: 'child', actionType: 'custom' },
      async (_, { context }) => {
        return `hi ${context?.auth?.email}`;
      }
    );
    const parent = defineAction(
      registry,
      { name: 'parent', actionType: 'custom' },
      async () => {
        return child();
      }
    );

    const response = await parent(undefined, {
      context: { auth: { email: 'a@b.c' } },
    });

    assert.strictEqual(response, 'hi a@b.c');
  });

  it('should include trace info in the context', async () => {
    const act = defineAction(
      registry,
      { name: 'child', actionType: 'custom' },
      async (_, ctx) => {
        return `traceId=${!!ctx.trace.traceId} spanId=${!!ctx.trace.spanId}`;
      }
    );

    const response = await act(undefined);

    assert.strictEqual(response, 'traceId=true spanId=true');
  });

  it('passes through the abort signal', async () => {
    var gotAbortSignal;
    const act = defineAction(
      registry,
      { name: 'child', actionType: 'custom' },
      async (_, ctx) => {
        gotAbortSignal = ctx.abortSignal;
        return `traceId=${!!ctx.trace.traceId} spanId=${!!ctx.trace.spanId}`;
      }
    );

    const signal = new AbortController().signal;
    await act(undefined, { abortSignal: signal });

    assert.strictEqual(gotAbortSignal, signal);
  });

  it('passes through the abort signal with middleware', async () => {
    var gotAbortSignal;
    const act = defineAction(
      registry,
      {
        name: 'child',
        actionType: 'custom',
        use: [async (input, next) => (await next(input + 'middle1')) + 1],
      },
      async (_, ctx) => {
        gotAbortSignal = ctx.abortSignal;
        return `traceId=${!!ctx.trace.traceId} spanId=${!!ctx.trace.spanId}`;
      }
    );

    const signal = new AbortController().signal;
    await act(undefined, { abortSignal: signal });

    assert.strictEqual(gotAbortSignal, signal);
  });

  it('includes genkit:key in telemetry', async () => {
    const act = defineAction(
      registry,
      {
        name: 'keyedAction',
        actionType: 'custom',
      },
      async () => {
        return 'success';
      }
    );

    await act();

    assert.strictEqual(spanExporter.exportedSpans.length, 1);
    assert.strictEqual(
      spanExporter.exportedSpans[0].attributes['genkit:key'],
      '/custom/keyedAction'
    );
  });

  it('sets genkit:key on action metadata when using defineActionAsync', async () => {
    const actPromise = defineActionAsync(
      registry,
      'custom',
      'asyncKeyedAction',
      Promise.resolve({
        name: 'asyncKeyedAction',
        actionType: 'custom',
        fn: async () => 'success',
      })
    );

    const act = await actPromise;

    assert.strictEqual(act.__action.key, '/custom/asyncKeyedAction');
  });
});
