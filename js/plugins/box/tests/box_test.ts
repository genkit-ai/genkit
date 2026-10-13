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
import { z, type JSONSchema7 } from 'genkit';
import { genkit } from 'genkit/beta';
import { afterEach, describe, it } from 'node:test';
import { box } from '../src/box.js';
import { BOX_SELF_ID_ENV } from '../src/env.js';
import {
  DEFAULT_SESSION_IDLE_MS,
  SHARED_KEY,
  SINGLETON_KEY,
  perRequest,
  resolveRetention,
  sessionIdOf,
  sessionRoute,
} from '../src/route.js';
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

/** A boxed server-managed agent: echoes the session, a new snapshot per turn. */
function sessionAgent() {
  let turns = 0;
  return new FakeRunner((req) => {
    if (req.key.startsWith('/agent/')) {
      turns++;
      const init = (req.init ?? {}) as { sessionId?: string };
      return {
        sessionId: init.sessionId ?? `minted-${turns}`,
        snapshotId: `snap-${turns}`,
        message: { role: 'model', content: [{ text: 'ok' }] },
      };
    }
    if (req.key.startsWith('/agent-snapshot/')) {
      return { snapshotId: 'snap-1', status: 'done' };
    }
    return 'flow-result';
  });
}

describe('sessionRoute', () => {
  it('routes a whole conversation by the context session', async () => {
    const runner = sessionAgent();
    const myBox = box(genkit({}), { runner, route: sessionRoute });
    const agent = myBox.agent({
      name: 'a',
      context: { sessionId: 'thread-1' },
    });

    const chat = agent.chat({ sessionId: 'thread-1' });
    await chat.send('hi');
    await chat.send('again'); // resumes by snapshotId alone
    await agent.getSnapshot('snap-2');
    await agent.abort('snap-2');

    assert.deepStrictEqual(
      runner.calls.map((c) => c.routeKey),
      ['thread-1', 'thread-1', 'thread-1', 'thread-1']
    );
  });

  it('routes a client-picked session id on the first turn', async () => {
    const runner = sessionAgent();
    const agent = box(genkit({}), { runner, route: sessionRoute }).agent({
      name: 'a',
    });
    await agent.chat({ sessionId: 's-1' }).send('hi');
    await agent.chat({ sessionId: 's-2' }).send('hi');
    await agent.getSnapshot({ sessionId: 's-1' });
    assert.deepStrictEqual(
      runner.calls.map((c) => c.routeKey),
      ['s-1', 's-2', 's-1']
    );
  });

  it('sends calls without a session to one shared box', async () => {
    const runner = sessionAgent();
    const myBox = box(genkit({}), { runner, route: sessionRoute });
    await myBox.flow({ name: 'summarize' })('a');
    await myBox.agent({ name: 'a' }).chat().send('hi');
    await myBox.agent({ name: 'a' }).getSnapshot('snap-9');
    assert.deepStrictEqual(runner.acquired, [
      SHARED_KEY,
      SHARED_KEY,
      SHARED_KEY,
    ]);
  });

  it('is stateless: the box keeps no session memory', async () => {
    const runner = sessionAgent();
    const myBox = box(genkit({}), { runner, route: sessionRoute });
    const chat = myBox.agent({ name: 'a' }).chat({ sessionId: 's-1' });
    await chat.send('hi');
    // Without a context the second turn carries only a snapshotId; nothing
    // learned from the first turn routes it back.
    await chat.send('again');
    assert.deepStrictEqual(
      runner.calls.map((c) => c.routeKey),
      ['s-1', SHARED_KEY]
    );
  });

  it('reclaims idle session boxes by default', () => {
    assert.deepStrictEqual(resolveRetention(sessionRoute), {
      idle: DEFAULT_SESSION_IDLE_MS,
    });
  });
});

describe('custom routes', () => {
  it('may be async (e.g. a lookup in a shared store)', async () => {
    const runner = sessionAgent();
    const placements = new Map([['s-1', 'sandbox-42']]);
    const myBox = box(genkit({}), {
      runner,
      route: async (req, ctx) => {
        await sleep(1);
        const sid = String(ctx?.sessionId ?? sessionIdOf(req) ?? '');
        return placements.get(sid) ?? 'shared';
      },
    });
    await myBox
      .agent({ name: 'a', context: { sessionId: 's-1' } })
      .chat()
      .send('hi');
    await myBox.flow({ name: 'summarize' })('x');
    assert.deepStrictEqual(runner.acquired, ['sandbox-42', 'shared']);
  });

  it('a rejected route fails the call without opening a lease', async () => {
    const runner = new FakeRunner();
    const shout = box(genkit({}), {
      runner,
      route: async () => {
        throw new Error('store unavailable');
      },
    }).tool({ name: 'shout' });
    await assert.rejects(() => shout('a'), /store unavailable/);
    assert.deepStrictEqual(runner.acquired, []);
  });
});

describe('sessionIdOf', () => {
  it('reads the session from turns and lookups', () => {
    assert.strictEqual(
      sessionIdOf({ key: '/agent/a', init: { sessionId: 's1' } }),
      's1'
    );
    assert.strictEqual(
      sessionIdOf({ key: '/agent/a', init: { state: { sessionId: 's2' } } }),
      's2'
    );
    assert.strictEqual(
      sessionIdOf({ key: '/agent-snapshot/a', input: { sessionId: 's3' } }),
      's3'
    );
    assert.strictEqual(
      sessionIdOf({ key: '/agent/a', init: { snapshotId: 'p1' } }),
      undefined
    );
    assert.strictEqual(sessionIdOf({ key: '/tool/t', input: {} }), undefined);
  });
});

describe('warm and listActions', () => {
  it('warm() starts the singleton box without running anything', async () => {
    const runner = new FakeRunner();
    await box(genkit({}), { runner }).warm();
    assert.deepStrictEqual(runner.acquired, [SINGLETON_KEY]);
    assert.deepStrictEqual(runner.calls, []);
    assert.deepStrictEqual(runner.released, []);
  });

  it('a warmed box is reclaimed after the idle window if unused', async () => {
    const runner = new FakeRunner();
    const myBox = box(genkit({}), {
      runner,
      route: () => 'k',
      retention: { idle: 20 },
    });
    await myBox.warm('session-1');
    assert.deepStrictEqual(runner.acquired, ['session-1']);
    await sleep(40);
    assert.deepStrictEqual(runner.released, ['session-1']);
  });

  it('inside its own runtime, warm() is a no-op and listActions() refuses', async () => {
    process.env[BOX_SELF_ID_ENV] = 'self';
    try {
      const runner = new FakeRunner();
      const myBox = box(genkit({}), { runner, name: 'self' });
      await myBox.warm();
      await assert.rejects(myBox.listActions(), /own runtime/);
      assert.deepStrictEqual(runner.acquired, []);
    } finally {
      delete process.env[BOX_SELF_ID_ENV];
    }
  });

  it('listActions() asks the box for a key and ends its lease', async () => {
    const runner = new FakeRunner();
    const myBox = box(genkit({}), { runner, retention: { idle: 0 } });
    await myBox.listActions();
    await myBox.listActions('k');
    assert.deepStrictEqual(runner.listed, [SINGLETON_KEY, 'k']);
    assert.deepStrictEqual(runner.released, [SINGLETON_KEY, 'k']);
  });
});

describe('proxy specs', () => {
  it('accepts JSON Schema when there is no zod schema', () => {
    const input: JSONSchema7 = {
      type: 'object',
      properties: { q: { type: 'string' } },
    };
    const output: JSONSchema7 = { type: 'string' };
    const search = box(genkit({}), { runner: new FakeRunner() }).tool({
      name: 'search',
      inputJsonSchema: input,
      outputJsonSchema: output,
    });
    assert.deepStrictEqual(search.__action.inputJsonSchema, input);
    assert.deepStrictEqual(search.__action.outputJsonSchema, output);
  });

  it('prefers the zod schema when both are given', () => {
    const flow = box(genkit({}), { runner: new FakeRunner() }).flow({
      name: 'f',
      inputSchema: z.string(),
      inputJsonSchema: { type: 'number' },
    });
    assert.ok(flow.__action.inputSchema);
    assert.strictEqual(flow.__action.inputJsonSchema, undefined);
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
