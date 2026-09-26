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
import { ReflectionServer, z } from 'genkit';
import {
  InMemorySessionStore,
  genkit,
  type AgentInit,
  type AgentOutput,
} from 'genkit/beta';
import { createServer } from 'node:net';
import { describe, it } from 'node:test';
import { box } from '../src/box.js';
import { sessionIdOf } from '../src/route.js';
import type { RunActionRequest } from '../src/types.js';
import { FakeRunner } from './fake-runner.js';

/**
 * A fake boxed agent: numbers snapshots and echoes input. Like a real agent, a
 * resumed snapshot keeps its session id.
 */
function fakeAgent(opts: { finish?: (turn: number) => string } = {}) {
  let turn = 0;
  const sessionOf = new Map<string, string>();
  return (req: RunActionRequest, run?: { onChunk?: (c: unknown) => void }) => {
    if (req.key.startsWith('/agent-snapshot/')) {
      return { snapshotId: 'snap-1', sessionId: 'sess-1', status: 'done' };
    }
    if (req.key.startsWith('/agent-abort/')) {
      return { snapshotId: 'snap-1', status: 'aborted' };
    }
    turn++;
    const init = (req.init ?? {}) as AgentInit;
    const text = String(
      (req.input as { message?: { content: { text?: string }[] } })?.message
        ?.content[0]?.text
    );
    run?.onChunk?.({ modelChunk: { content: [{ text: `chunk:${text}` }] } });
    const sessionId =
      init.sessionId ??
      (init.snapshotId ? sessionOf.get(init.snapshotId) : undefined) ??
      'sess-1';
    sessionOf.set(`snap-${turn}`, sessionId);
    const output: AgentOutput = {
      sessionId,
      snapshotId: `snap-${turn}`,
      message: { role: 'model', content: [{ text: `echo:${text}` }] },
      finishReason: (opts.finish?.(turn) ??
        'stop') as AgentOutput['finishReason'],
    };
    return output;
  };
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      srv.close(() => resolve(port));
    });
  });
}

const user = (text: string) => ({
  message: { role: 'user' as const, content: [{ text }] },
});

describe('defineAgent', () => {
  it('registers the agent and its companions with Dev UI metadata', async () => {
    const ai = genkit({});
    box(ai, { runner: new FakeRunner() }).defineAgent({
      name: 'coder',
      stateManagement: 'server',
      abortable: true,
      stateSchema: z.object({ plan: z.string() }),
    });
    const actions = await ai.registry.listActions();
    const agent = actions['/agent/coder'];
    assert.ok(agent, 'agent registered');
    assert.ok(actions['/agent-snapshot/coder'], 'snapshot companion');
    assert.ok(actions['/agent-abort/coder'], 'abort companion');
    const meta = agent.__action.metadata?.agent as Record<string, unknown>;
    assert.strictEqual(meta.stateManagement, 'server');
    assert.strictEqual(meta.abortable, true);
    assert.ok(meta.stateSchema, 'state schema as JSON schema');
    assert.strictEqual(agent.__action.metadata?.bidi, true);
  });

  it('chats in-process, threading the snapshot between turns', async () => {
    const runner = new FakeRunner(fakeAgent());
    const coder = box(genkit({}), { runner }).defineAgent({
      name: 'coder',
      stateManagement: 'server',
    });
    const chat = coder.chat();
    const first = await chat.sendStream('hi');
    const chunks: unknown[] = [];
    for await (const c of first.stream) chunks.push(c);
    assert.strictEqual((await first.response).text, 'echo:hi');
    assert.ok(chunks.length > 0, 'chunks streamed through');
    await chat.send('again');
    assert.deepStrictEqual(runner.calls[1].req.init, { snapshotId: 'snap-1' });
  });

  it('runs one boxed turn per input, like the Dev UI with several inputs', async () => {
    const runner = new FakeRunner(fakeAgent());
    const coder = box(genkit({}), { runner }).defineAgent({
      name: 'coder',
      stateManagement: 'server',
    });
    const bidi = coder.streamBidi({ sessionId: 's-9' });
    bidi.send(user('a'));
    bidi.send(user('b'));
    bidi.close();
    for await (const _ of bidi.stream) {
      // drain
    }
    const output = await bidi.output;
    assert.strictEqual(output.snapshotId, 'snap-2');
    assert.deepStrictEqual(
      runner.calls.map((c) => c.req.init),
      [{ sessionId: 's-9' }, { snapshotId: 'snap-1', sessionId: 's-9' }]
    );
  });

  it('carries client-managed state between turns', async () => {
    let turns = 0;
    const runner = new FakeRunner(() => ({
      state: { messages: [], custom: { n: ++turns } },
      finishReason: 'stop',
    }));
    const coder = box(genkit({}), { runner }).defineAgent({
      name: 'coder',
      stateManagement: 'client',
    });
    const bidi = coder.streamBidi({});
    bidi.send(user('a'));
    bidi.send(user('b'));
    bidi.close();
    await bidi.output;
    assert.deepStrictEqual(runner.calls[1].req.init, {
      state: { messages: [], custom: { n: 1 } },
    });
  });

  it('stops at a failed or detached turn', async () => {
    const runner = new FakeRunner(
      fakeAgent({ finish: (t) => (t === 1 ? 'failed' : 'stop') })
    );
    const coder = box(genkit({}), { runner }).defineAgent({
      name: 'coder',
      stateManagement: 'server',
    });
    const bidi = coder.streamBidi({});
    bidi.send(user('a'));
    bidi.send(user('b'));
    bidi.close();
    const output = await bidi.output;
    assert.strictEqual(output.finishReason, 'failed');
    assert.strictEqual(runner.calls.length, 1);
  });

  it('forwards snapshot reads and aborts to the box', async () => {
    const runner = new FakeRunner(fakeAgent());
    const coder = box(genkit({}), { runner }).defineAgent({
      name: 'coder',
      stateManagement: 'server',
      abortable: true,
    });
    const snap = await coder.getSnapshot('snap-1');
    assert.strictEqual(snap?.snapshotId, 'snap-1');
    assert.strictEqual(await coder.abort('snap-1'), 'aborted');
    assert.deepStrictEqual(
      runner.calls.map((c) => [c.req.key, c.req.input]),
      [
        ['/agent-snapshot/coder', { snapshotId: 'snap-1' }],
        ['/agent-abort/coder', { snapshotId: 'snap-1' }],
      ]
    );
  });

  it('routes a session to one box, including snapshot-only resumes', async () => {
    const runner = new FakeRunner(fakeAgent());
    const coder = box(genkit({}), {
      runner,
      route: (req, ctx) =>
        String(ctx?.sessionId ?? sessionIdOf(req) ?? 'default'),
    }).defineAgent({ name: 'coder', stateManagement: 'server' });

    // What the Dev UI sends: init only, no context.
    await coder.run(user('a'), { init: { sessionId: 's-7' } });
    // A later resume by snapshot alone still lands on the session's box.
    await coder.run(user('b'), { init: { snapshotId: 'snap-1' } });
    await coder.getSnapshot('snap-2');

    assert.deepStrictEqual(runner.acquired, ['s-7', 's-7', 's-7']);
    // The routing hint stays on the host side.
    assert.deepStrictEqual(runner.calls[1].req.init, { snapshotId: 'snap-1' });
  });

  it('defineFromAgent copies metadata and calls the original in the box', async () => {
    const ai = genkit({});
    const real = ai.defineAgent({
      name: 'coder',
      model: ai.defineModel({ name: 'm' }, async () => ({
        message: { role: 'model', content: [{ text: 'x' }] },
      })),
      store: new InMemorySessionStore(),
    });
    const runner = new FakeRunner(fakeAgent());
    const myBox = box(ai, { runner });
    assert.throws(
      () => myBox.defineFromAgent(real, { name: 'coder' }),
      /must be renamed/
    );
    const boxed = myBox.defineFromAgent(real, { name: 'boxedCoder' });
    assert.deepStrictEqual(
      boxed.__action.metadata?.agent,
      real.__action.metadata?.agent
    );
    await boxed.chat().send('hi');
    assert.strictEqual(runner.calls[0].req.key, '/agent/coder');
  });
});

describe('defineAgent over the reflection API (Dev UI path)', () => {
  it('lists the boxed agent and streams a turn through /api/runAction', async () => {
    const runner = new FakeRunner(fakeAgent());
    const ai = genkit({});
    box(ai, { runner }).defineAgent({
      name: 'coder',
      stateManagement: 'server',
    });
    const port = await freePort();
    const server = new ReflectionServer(ai.registry, { port });
    await server.start();
    try {
      const base = `http://127.0.0.1:${port}`;
      const actions = (await (
        await fetch(`${base}/api/actions`)
      ).json()) as Record<string, { metadata?: { agent?: unknown } }>;
      assert.ok(
        actions['/agent/coder']?.metadata?.agent,
        'listed with metadata'
      );

      const res = await fetch(`${base}/api/runAction?stream=true`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key: '/agent/coder',
          input: user('hi'),
          init: { sessionId: 's-1' },
        }),
      });
      const lines = (await res.text()).trim().split('\n');
      const final = JSON.parse(lines.at(-1)!) as { result: AgentOutput };
      assert.ok(lines.length > 1, 'streamed at least one chunk');
      assert.strictEqual(final.result.snapshotId, 'snap-1');
      assert.ok(res.headers.get('x-genkit-trace-id'), 'early trace id header');
      assert.deepStrictEqual(runner.calls[0].req.init, { sessionId: 's-1' });
    } finally {
      await server.stop();
    }
  });
});

describe('sessionIdOf', () => {
  it('reads the session from turns, lookups and hints', () => {
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
      sessionIdOf({ key: '/agent/a', init: {}, sessionId: 's4' }),
      's4'
    );
    assert.strictEqual(sessionIdOf({ key: '/tool/t', input: {} }), undefined);
  });
});
