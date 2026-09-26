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
import { afterEach, describe, it } from 'node:test';
import { box } from '../src/box.js';
import { BOX_SELF_ID_ENV } from '../src/env.js';
import { SINGLETON_KEY, perRequest } from '../src/route.js';
import { FakeRunner } from './fake-runner.js';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe('box routing', () => {
  it('attaches the runner to the box', () => {
    const runner = new FakeRunner();
    const myBox = box(genkit({}), { runner, name: 'tools' });
    assert.strictEqual(runner.attachedTo, 'tools');
    assert.strictEqual(myBox.id, 'tools');
  });

  it('routes every call to one box by default, never releasing it', async () => {
    const runner = new FakeRunner();
    const shout = box(genkit({}), { runner }).tool({ name: 'shout' });
    await shout('a');
    await shout('b');
    assert.deepStrictEqual(runner.acquired, [SINGLETON_KEY, SINGLETON_KEY]);
    assert.deepStrictEqual(runner.released, []);
  });

  it('perRequest gives each call a fresh box and releases it', async () => {
    const runner = new FakeRunner();
    const shout = box(genkit({}), { runner, route: perRequest }).tool({
      name: 'shout',
    });
    await shout('a');
    await shout('b');
    assert.strictEqual(new Set(runner.acquired).size, 2);
    assert.deepStrictEqual(runner.released, runner.acquired);
  });

  it('passes the request and its context to the route', async () => {
    const runner = new FakeRunner();
    const seen: Array<{ key: string; ctx: unknown }> = [];
    const shout = box(genkit({}), {
      runner,
      route: (req, ctx) => {
        seen.push({ key: req.key, ctx });
        return String(ctx?.tenant ?? 'none');
      },
    }).tool({ name: 'shout' });

    await shout('a', { context: { tenant: 't1' } });
    assert.deepStrictEqual(seen, [
      { key: '/tool/shout', ctx: { tenant: 't1' } },
    ]);
    assert.deepStrictEqual(runner.acquired, ['t1']);
  });

  it('routes agent calls by the proxy context', async () => {
    const runner = new FakeRunner((req) =>
      req.key.startsWith('/agent/')
        ? { message: { role: 'model', content: [{ text: 'ok' }] } }
        : { snapshotId: 's1', status: 'done' }
    );
    const myBox = box(genkit({}), {
      runner,
      route: (_req, ctx) => String(ctx?.sessionId ?? 'default'),
    });

    await myBox
      .agent({ name: 'a', context: { sessionId: 's-1' } })
      .chat()
      .send('hi');
    await myBox
      .agent({ name: 'a', context: { sessionId: 's-2' } })
      .chat()
      .send('hi');
    await myBox.agent({ name: 'a' }).getSnapshot('snap');

    assert.deepStrictEqual(
      runner.calls.map((c) => [c.routeKey, c.req.key]),
      [
        ['s-1', '/agent/a'],
        ['s-2', '/agent/a'],
        ['default', '/agent-snapshot/a'],
      ]
    );
    // The boxed agent sees the same context.
    assert.deepStrictEqual(runner.calls[0].req.context, { sessionId: 's-1' });
  });
});

describe('box lifecycle', () => {
  afterEach(() => {
    delete process.env[BOX_SELF_ID_ENV];
  });

  it('releases an idle box after the idle window', async () => {
    const runner = new FakeRunner();
    const shout = box(genkit({}), {
      runner,
      route: () => 'k',
      retention: { idle: 20 },
    }).tool({ name: 'shout' });

    await shout('a');
    assert.deepStrictEqual(runner.released, []);
    await sleep(40);
    assert.deepStrictEqual(runner.released, ['k']);
  });

  it('a call inside the idle window keeps the box', async () => {
    const runner = new FakeRunner();
    const shout = box(genkit({}), {
      runner,
      route: () => 'k',
      retention: { idle: 40 },
    }).tool({ name: 'shout' });

    await shout('a');
    await sleep(25);
    await shout('b'); // resets the window
    await sleep(25);
    assert.deepStrictEqual(runner.released, []);
    await sleep(40);
    assert.deepStrictEqual(runner.released, ['k']);
  });

  it('does not release a box while another call on it is in flight', async () => {
    let finishSlow!: () => void;
    const runner = new FakeRunner((req) =>
      req.input === 'slow'
        ? new Promise<string>((r) => (finishSlow = () => r('done')))
        : 'fast'
    );
    const shout = box(genkit({}), {
      runner,
      route: () => 'k',
      retention: { idle: 0 },
    }).tool({ name: 'shout' });

    const slow = shout('slow');
    await shout('fast');
    assert.deepStrictEqual(runner.released, []);
    finishSlow();
    await slow;
    assert.deepStrictEqual(runner.released, ['k']);
  });

  it('refuses proxy calls from inside its own runtime (self mode)', async () => {
    process.env[BOX_SELF_ID_ENV] = 'tools';
    const runner = new FakeRunner();
    const myBox = box(genkit({}), { runner, name: 'tools' });
    assert.strictEqual(myBox.isSelfRuntime, true);
    await assert.rejects(
      () => myBox.tool({ name: 'shout' })('a'),
      /from within the box's own runtime/
    );
    assert.deepStrictEqual(runner.acquired, []);
  });

  it('defineFromTool must rename the proxy', () => {
    const ai = genkit({});
    const shout = ai.defineTool(
      { name: 'shout', description: 'Shouts', inputSchema: z.string() },
      async (s) => s.toUpperCase()
    );
    const myBox = box(ai, { runner: new FakeRunner() });
    assert.throws(
      () => myBox.defineFromTool(shout, { name: 'shout' }),
      /must be renamed/
    );
    const boxed = myBox.defineFromTool(shout, { name: 'boxedShout' });
    assert.strictEqual(boxed.__action.name, 'boxedShout');
    assert.strictEqual(boxed.__action.key, '/tool/boxedShout');
    assert.strictEqual(boxed.__action.description, 'Shouts');
  });

  it('a renamed proxy still calls the original action in the box', async () => {
    const ai = genkit({});
    const shout = ai.defineTool(
      { name: 'shout', description: 'Shouts', inputSchema: z.string() },
      async (s) => s.toUpperCase()
    );
    const runner = new FakeRunner(() => 'HI');
    const boxed = box(ai, { runner }).defineFromTool(shout, {
      name: 'boxedShout',
    });
    await boxed('hi');
    assert.strictEqual(runner.calls[0].req.key, '/tool/shout');
    const actions = await ai.registry.listActions();
    assert.ok(actions['/tool/boxedShout'], 'registered under the new name');
  });

  it('close() closes the runner', async () => {
    const runner = new FakeRunner();
    await box(genkit({}), { runner }).close();
    assert.strictEqual(runner.closed, true);
  });
});
