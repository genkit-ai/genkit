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
import { describe, it } from 'node:test';
import { bubblewrap } from '../src/providers/bubblewrap.js';
import { localSandbox } from '../src/providers/index.js';
import { sandboxExec } from '../src/providers/sandbox-exec.js';
import { SubprocessProvider } from '../src/providers/subprocess.js';

const REFLECT = { url: 'ws://127.0.0.1:9999', port: 9999 };
const SPEC = {
  cmd: 'node',
  args: ['boxed.js'],
  env: { GENKIT_ENV: 'dev' },
};

describe('providers', () => {
  it('subprocess passes the spawn through unchanged', () => {
    const p = new SubprocessProvider().prepare(SPEC, REFLECT);
    assert.strictEqual(p.cmd, 'node');
    assert.deepStrictEqual(p.args, ['boxed.js']);
    assert.strictEqual(p.reflectUrl, REFLECT.url);
  });

  it('sandbox-exec wraps argv with a seatbelt profile (darwin only)', () => {
    if (process.platform !== 'darwin') {
      assert.throws(() => sandboxExec().prepare(SPEC, REFLECT), /only.*macOS/i);
      return;
    }
    const p = sandboxExec().prepare(SPEC, REFLECT);
    assert.strictEqual(p.cmd, 'sandbox-exec');
    assert.strictEqual(p.args[0], '-p');
    assert.ok(p.args[1].includes('(version 1)'));
    // The reflection dial-back hole must be present.
    assert.ok(p.args[1].includes('(allow network*)'));
    // Real command is preserved after the profile.
    assert.deepStrictEqual(p.args.slice(2), ['node', 'boxed.js']);
  });

  it('bubblewrap wraps argv with bwrap flags (linux only)', () => {
    if (process.platform !== 'linux') {
      assert.throws(() => bubblewrap().prepare(SPEC, REFLECT), /only.*Linux/i);
      return;
    }
    const p = bubblewrap().prepare(SPEC, REFLECT);
    assert.strictEqual(p.cmd, 'bwrap');
    assert.ok(p.args.includes('--die-with-parent'));
    // Default keeps the network namespace (loopback dial-back).
    assert.ok(!p.args.includes('--unshare-net'));
    const nodeIdx = p.args.indexOf('node');
    assert.deepStrictEqual(p.args.slice(nodeIdx), ['node', 'boxed.js']);
  });

  it('localSandbox picks per-OS or hard-errors', () => {
    if (process.platform === 'darwin') {
      assert.strictEqual(localSandbox().name, 'sandbox-exec');
    } else if (process.platform === 'linux') {
      assert.strictEqual(localSandbox().name, 'bubblewrap');
    } else {
      assert.throws(() => localSandbox(), /no local sandbox/i);
    }
  });
});
