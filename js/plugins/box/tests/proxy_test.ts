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
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { describe, it } from 'node:test';
import { box } from '../src/box.js';
import { createProxyAction, type ProxyDispatcher } from '../src/proxy.js';
import { FakeRunner } from './fake-runner.js';

const unusedDispatcher: ProxyDispatcher = {
  acquire: async () => {
    throw new Error('not used in these tests');
  },
};

describe('proxy actions', () => {
  it('carries actionType so the registry accepts it', () => {
    // The registry compares `__action.actionType` against the type it is
    // registered under; without it, `define*` throws INVALID_ARGUMENT.
    const proxy = createProxyAction(unusedDispatcher, 'tool', 'runShell');
    assert.strictEqual(proxy.__action.actionType, 'tool');
    assert.strictEqual(proxy.__action.metadata?.type, 'tool');
    assert.strictEqual(proxy.__action.metadata?.box, true);
  });

  it('registers a spec-defined tool proxy in the Dev UI registry', async () => {
    const ai = genkit({});
    const myBox = box(ai, { runner: new FakeRunner() });
    myBox.defineTool({
      name: 'runShell',
      inputSchema: z.object({ cmd: z.string() }),
      outputSchema: z.object({ stdout: z.string().optional() }),
    });
    const actions = await ai.registry.listActions();
    assert.ok(actions['/tool/runShell'], 'proxy should be registered');
    await myBox.close();
  });

  it('forwards input, context and streamed chunks to the box', async () => {
    const runner = new FakeRunner((req, opts) => {
      opts?.onChunk?.({ i: 1 });
      opts?.onChunk?.({ i: 2 });
      return { echoed: req.input };
    });
    const ai = genkit({});
    const countTo = box(ai, { runner }).flow({
      name: 'countTo',
      streamSchema: z.object({ i: z.number() }),
    });

    const { stream, output } = countTo.stream(
      { n: 2 },
      { context: { auth: { uid: 'u1' } } }
    );
    const chunks: Array<{ i: number }> = [];
    for await (const c of stream) chunks.push(c);

    assert.deepStrictEqual(await output, { echoed: { n: 2 } });
    assert.deepStrictEqual(chunks, [{ i: 1 }, { i: 2 }]);
    assert.deepStrictEqual(runner.calls[0].req, {
      key: '/flow/countTo',
      input: { n: 2 },
      init: undefined,
      context: { auth: { uid: 'u1' } },
    });
  });

  it('reports the caller-side span, not the box trace', async () => {
    const ai = genkit({});
    const shout = box(ai, { runner: new FakeRunner(() => 'OK') }).tool({
      name: 'shout',
    });

    let started: { traceId: string; spanId: string } | undefined;
    const res = await shout.run('ok', { onTraceStart: (t) => (started = t) });

    assert.ok(started, 'onTraceStart should fire before the result');
    assert.deepStrictEqual(res.telemetry, started);
    assert.notStrictEqual(res.telemetry.traceId, 'box-trace-__singleton__');
    assert.strictEqual(res.result, 'OK');
  });
});
