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
import { spawn } from 'node:child_process';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { box } from '../src/box.js';
import { execRunner } from '../src/runners/exec-runner.js';
import { childEnv } from '../src/runners/util.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const boxedEntry = path.join(here, 'fixtures', 'boxed-entry.ts');
const nestedEntry = path.join(here, 'fixtures', 'nested-entry.ts');
const TSX = path.join(here, '..', 'node_modules', '.bin', 'tsx');

describe('childEnv', () => {
  const parent = {
    PATH: '/bin',
    HOME: '/home/me',
    GENKIT_ENV: 'dev',
    GENKIT_FOO_TOKEN: 'secret-ish',
    DISCORD_TOKEN: 'secret',
    GEMINI_API_KEY: 'key',
  };

  it('inherits everything by default', () => {
    assert.deepStrictEqual(childEnv(parent, true, undefined, {}), parent);
  });

  it('keeps only the basics with inheritEnv: false', () => {
    assert.deepStrictEqual(childEnv(parent, false, undefined, {}), {
      PATH: '/bin',
      HOME: '/home/me',
      GENKIT_ENV: 'dev',
    });
  });

  it('adds named vars to the basics', () => {
    assert.deepStrictEqual(
      Object.keys(childEnv(parent, ['GEMINI_API_KEY'], undefined, {})).sort(),
      ['GEMINI_API_KEY', 'GENKIT_ENV', 'HOME', 'PATH']
    );
  });

  it('layers env over inherited vars and overrides over both', () => {
    const env = childEnv(
      parent,
      true,
      { GEMINI_API_KEY: 'other', GENKIT_RUNTIME_ID: 'mine' },
      { GENKIT_RUNTIME_ID: 'box-1' }
    );
    assert.strictEqual(env.GEMINI_API_KEY, 'other');
    assert.strictEqual(env.GENKIT_RUNTIME_ID, 'box-1');
  });
});

describe('execRunner (integration)', () => {
  const runners: Array<{ close(): Promise<void> }> = [];
  const track = <T extends { close(): Promise<void> }>(r: T): T => {
    runners.push(r);
    return r;
  };

  after(async () => {
    await Promise.all(runners.map((r) => r.close().catch(() => {})));
  });

  it('spawns a separate-entry box and runs a tool', async () => {
    const runner = track(execRunner({ cmd: `${TSX} ${boxedEntry}` }));
    const conn = await runner.acquire('singleton');
    const res = await conn.runAction<{ out: string }>({
      key: '/tool/shout',
      input: { text: 'hello' },
    });
    assert.strictEqual(res.result?.out, 'HELLO');
  });

  it('lists the actions the box registered', async () => {
    const runner = track(execRunner({ cmd: `${TSX} ${boxedEntry}` }));
    const conn = await runner.acquire('singleton');
    const actions = await conn.listActions();
    assert.ok(actions['/tool/shout'], 'expected /tool/shout');
    assert.ok(actions['/flow/countTo'], 'expected /flow/countTo');
  });

  it('streams flow chunks from the box', async () => {
    const runner = track(execRunner({ cmd: `${TSX} ${boxedEntry}` }));
    const conn = await runner.acquire('singleton');
    const chunks: Array<{ i: number }> = [];
    const res = await conn.runAction<{ total: number }>(
      { key: '/flow/countTo', input: { n: 3 } },
      { onChunk: (c) => chunks.push(c as { i: number }) }
    );
    assert.deepStrictEqual(
      chunks.map((c) => c.i),
      [1, 2, 3]
    );
    assert.strictEqual(res.result?.total, 3);
  });

  it('typed flow proxy streams typed chunks via box().flow()', async () => {
    const ai = genkit({});
    const b = box(ai, { runner: execRunner({ cmd: `${TSX} ${boxedEntry}` }) });
    runners.push(b);
    const countTo = b.flow({
      name: 'countTo',
      inputSchema: z.object({ n: z.number() }),
      outputSchema: z.object({ total: z.number() }),
      streamSchema: z.object({ i: z.number() }),
    });
    const { stream, output } = countTo.stream({ n: 3 });
    const seen: number[] = [];
    for await (const chunk of stream) {
      // `chunk` is typed as { i: number } from the streamSchema (S).
      seen.push(chunk.i);
    }
    const result = await output;
    assert.deepStrictEqual(seen, [1, 2, 3]);
    // `result` is typed as { total: number } from the outputSchema (O).
    assert.strictEqual(result.total, 3);
  });

  it('per-request routing gives a fresh box per key', async () => {
    const runner = track(execRunner({ cmd: `${TSX} ${boxedEntry}` }));
    const a = await runner.acquire('k1');
    const b = await runner.acquire('k2');
    // Two live boxes; both answer independently.
    const ra = await a.runAction<{ out: string }>({
      key: '/tool/shout',
      input: { text: 'a' },
    });
    const rb = await b.runAction<{ out: string }>({
      key: '/tool/shout',
      input: { text: 'b' },
    });
    assert.strictEqual(ra.result?.out, 'A');
    assert.strictEqual(rb.result?.out, 'B');
    await runner.release('k1');
    await runner.release('k2');
  });

  it('proxies a boxed agent (chat over the reflection host)', async () => {
    const ai = genkit({});
    const b = box(ai, { runner: execRunner({ cmd: `${TSX} ${boxedEntry}` }) });
    runners.push(b);
    const agent = b.agent({ name: 'echoAgent' });
    const chat = agent.chat();
    const res = await chat.send('hello');
    // The boxed echo model/agent replies with `echo:...`.
    const text = res.message.content.map((p) => p.text ?? '').join('');
    assert.ok(text.includes('echo:'), `unexpected agent reply: ${text}`);
  });

  it('accepts cmd as an argv array', async () => {
    const runner = track(execRunner({ cmd: [TSX, boxedEntry] }));
    const conn = await runner.acquire('singleton');
    const res = await conn.runAction<{ out: string }>({
      key: '/tool/shout',
      input: { text: 'argv' },
    });
    assert.strictEqual(res.result?.out, 'ARGV');
  });

  it('fails fast when the command cannot be spawned', async () => {
    const runner = track(execRunner({ cmd: 'definitely-not-a-real-binary' }));
    const started = Date.now();
    await assert.rejects(runner.acquire('k'), /failed to start/);
    assert.ok(Date.now() - started < 5_000, 'no readiness timeout');
  });

  it('fails fast when the box exits before it is ready', async () => {
    const runner = track(
      execRunner({ cmd: [process.execPath, '-e', 'process.exit(3)'] })
    );
    await assert.rejects(runner.acquire('k'), /exited before it was ready/);
  });

  it('overrides a reflection secret inherited from the caller', async () => {
    // Under `genkit start` the caller carries the CLI's secret; the box must
    // present its own host's secret instead, or registration is rejected.
    const saved = process.env.GENKIT_REFLECTION_SECRET_TOKEN;
    process.env.GENKIT_REFLECTION_SECRET_TOKEN = 'the-cli-secret';
    try {
      const runner = track(execRunner({ cmd: `${TSX} ${boxedEntry}` }));
      const conn = await runner.acquire('singleton');
      const res = await conn.runAction<{ out: string }>({
        key: '/tool/shout',
        input: { text: 'ok' },
      });
      assert.strictEqual(res.result?.out, 'OK');
    } finally {
      if (saved === undefined)
        delete process.env.GENKIT_REFLECTION_SECRET_TOKEN;
      else process.env.GENKIT_REFLECTION_SECRET_TOKEN = saved;
    }
  });

  /** Runs `fn` with `vars` set in this process's env, then restores it. */
  async function withEnv<T>(
    vars: Record<string, string>,
    fn: () => Promise<T>
  ): Promise<T> {
    const saved = Object.fromEntries(
      Object.keys(vars).map((k) => [k, process.env[k]])
    );
    Object.assign(process.env, vars);
    try {
      return await fn();
    } finally {
      for (const [k, v] of Object.entries(saved)) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
    }
  }

  async function readEnv(
    runner: ReturnType<typeof execRunner>,
    names: string[]
  ): Promise<Record<string, string>> {
    const conn = await runner.acquire('singleton');
    const res = await conn.runAction<Record<string, string>>({
      key: '/tool/readEnv',
      input: { names },
    });
    return res.result ?? {};
  }

  it('inheritEnv: false keeps secrets out of the box but sets env', async () => {
    await withEnv({ BOX_TEST_SECRET: 'hunter2' }, async () => {
      const runner = track(
        execRunner({
          cmd: `${TSX} ${boxedEntry}`,
          inheritEnv: false,
          env: { BOX_TEST_GIVEN: 'given' },
        })
      );
      const seen = await readEnv(runner, [
        'BOX_TEST_SECRET',
        'BOX_TEST_GIVEN',
        'PATH',
        'GENKIT_RUNTIME_ID',
      ]);
      assert.strictEqual(seen.BOX_TEST_SECRET, undefined);
      assert.strictEqual(seen.BOX_TEST_GIVEN, 'given');
      assert.ok(seen.PATH, 'basics are kept');
      assert.ok(seen.GENKIT_RUNTIME_ID, 'box vars are set');
    });
  });

  it('inheritEnv as a list inherits only the named vars', async () => {
    await withEnv({ BOX_TEST_A: 'a', BOX_TEST_B: 'b' }, async () => {
      const runner = track(
        execRunner({ cmd: `${TSX} ${boxedEntry}`, inheritEnv: ['BOX_TEST_A'] })
      );
      assert.deepStrictEqual(
        await readEnv(runner, ['BOX_TEST_A', 'BOX_TEST_B']),
        { BOX_TEST_A: 'a' }
      );
    });
  });

  it('inherits everything by default', async () => {
    await withEnv({ BOX_TEST_SECRET: 'hunter2' }, async () => {
      const runner = track(execRunner({ cmd: `${TSX} ${boxedEntry}` }));
      assert.deepStrictEqual(await readEnv(runner, ['BOX_TEST_SECRET']), {
        BOX_TEST_SECRET: 'hunter2',
      });
    });
  });

  it('honors readyTimeoutMs', async () => {
    // A process that stays up but never connects.
    const runner = track(
      execRunner({
        cmd: [process.execPath, '-e', 'setInterval(() => {}, 1000)'],
        readyTimeoutMs: 200,
      })
    );
    const started = Date.now();
    await assert.rejects(runner.acquire('k'), /Timed out after 200ms/);
    assert.ok(Date.now() - started < 5_000);
  });

  it('enables reflection in the box even when the caller disabled its own', async () => {
    // The caller's opt-out applies to its own Dev UI link, not to the box's
    // link to our host. Outside dev mode the box would otherwise stay dark.
    const saved = {
      enabled: process.env.GENKIT_REFLECTION_ENABLED,
      env: process.env.GENKIT_ENV,
    };
    process.env.GENKIT_REFLECTION_ENABLED = 'false';
    delete process.env.GENKIT_ENV;
    try {
      const runner = track(execRunner({ cmd: `${TSX} ${boxedEntry}` }));
      const conn = await runner.acquire('singleton');
      const res = await conn.runAction<{ out: string }>({
        key: '/tool/shout',
        input: { text: 'ok' },
      });
      assert.strictEqual(res.result?.out, 'OK');
    } finally {
      for (const [key, value] of [
        ['GENKIT_REFLECTION_ENABLED', saved.enabled],
        ['GENKIT_ENV', saved.env],
      ] as const) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  });

  it('nests self-mode boxes (agent-box -> tool-box)', async () => {
    // Run the nested entry directly as the top manager M. It spawns C1 (box A)
    // which spawns C2 (box B) and prints the result.
    const out = await new Promise<string>((resolve, reject) => {
      const child = spawn(TSX, [nestedEntry], {
        env: { ...process.env },
        stdio: ['ignore', 'pipe', 'inherit'],
      });
      let buf = '';
      child.stdout.on('data', (d) => (buf += d.toString()));
      // 'close', not 'exit': stdout may still be flushing when 'exit' fires.
      child.on('close', () => resolve(buf));
      child.on('error', reject);
    });
    const line = out.split('\n').find((l) => l.startsWith('NESTED_RESULT:'));
    assert.ok(line, `expected NESTED_RESULT in output:\n${out}`);
    const res = JSON.parse(line!.slice('NESTED_RESULT:'.length));
    assert.strictEqual(res.echoed, 'deep:hi');
    // Three distinct processes: M, C1 (viaA), C2 (deep).
    assert.notStrictEqual(res.viaPid, res.innerPid);
    assert.ok(res.viaPid > 0 && res.innerPid > 0);
  });
});
