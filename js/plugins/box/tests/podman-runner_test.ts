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
import { realpathSync } from 'node:fs';
import * as os from 'node:os';
import { describe, it } from 'node:test';
import { podmanRunner } from '../src/runners/podman-runner.js';

/** Reads the value following a repeated flag (e.g. every `-e`). */
function valuesFor(args: string[], flag: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === flag) out.push(args[i + 1]);
  }
  return out;
}

describe('podmanRunner', () => {
  it('requires exactly one of self/cmd, and an image', () => {
    assert.throws(
      () =>
        podmanRunner({ image: 'node:22-slim', self: true, cmd: 'node a.js' }),
      /mutually exclusive/
    );
    assert.throws(
      () => podmanRunner({ image: 'node:22-slim' }),
      /one of `self` or `cmd`/
    );
    assert.throws(
      () => podmanRunner({ image: '', cmd: 'node a.js' }),
      /`image` is required/
    );
  });

  it('publishes the container reflection port to host loopback', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
    }).buildRunArgs('box-1', 54321, 's3cret');

    // Host side is pinned to loopback.
    assert.deepStrictEqual(valuesFor(args, '-p'), ['127.0.0.1:54321:3100']);
    assert.ok(args.includes('--rm'));
    assert.ok(args.includes('--init'));
    assert.deepStrictEqual(valuesFor(args, '--name'), ['box-1']);
    // Strays are findable if the host dies before close().
    assert.deepStrictEqual(valuesFor(args, '--label'), ['genkit-box']);
  });

  it('sets the env the box needs to be reachable', () => {
    const env = valuesFor(
      podmanRunner({
        image: 'node:22-slim',
        cmd: 'node boxed.js',
      }).buildRunArgs('box-1', 54321, 's3cret'),
      '-e'
    );
    // Published ports land on eth0, so a loopback bind would be unreachable.
    assert.ok(env.includes('GENKIT_REFLECTION_HOST=0.0.0.0'));
    assert.ok(env.includes('GENKIT_REFLECTION_PORT=3100'));
    assert.ok(env.includes('GENKIT_REFLECTION_SECRET_TOKEN=s3cret'));
    // The pinned port starts the server; no dev mode in the box.
    assert.ok(!env.some((e) => e.startsWith('GENKIT_ENV=')));
  });

  it('does not leak host env into the box', () => {
    process.env.BOX_TEST_SECRET = 'do-not-leak';
    try {
      const env = valuesFor(
        podmanRunner({
          image: 'node:22-slim',
          cmd: 'node boxed.js',
        }).buildRunArgs('box-1', 54321, 's3cret'),
        '-e'
      );
      assert.ok(!env.some((e) => e.includes('do-not-leak')));
    } finally {
      delete process.env.BOX_TEST_SECRET;
    }
  });

  it('passes explicitly provided env through', () => {
    const env = valuesFor(
      podmanRunner({
        image: 'node:22-slim',
        cmd: 'node boxed.js',
        env: { MY_KEY: 'abc' },
      }).buildRunArgs('box-1', 54321, 's3cret'),
      '-e'
    );
    assert.ok(env.includes('MY_KEY=abc'));
  });

  it('defaults to the internal (egress-blocked) network', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.deepStrictEqual(valuesFor(args, '--network'), [
      'genkit-box-internal',
    ]);
  });

  it('omits --network when bridge is requested', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
      network: 'bridge',
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.deepStrictEqual(valuesFor(args, '--network'), []);
  });

  it('mounts the project at the same absolute path', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
      projectDir: '/Users/me/app',
    }).buildRunArgs('box-1', 54321, 's3cret');
    // Identical host/container paths keep argv and relative symlinks valid.
    assert.ok(valuesFor(args, '-v').includes('/Users/me/app:/Users/me/app'));
    assert.deepStrictEqual(valuesFor(args, '-w'), ['/Users/me/app']);
  });

  it('resolves symlinked project dirs', () => {
    // podman binds the real path; on macOS /tmp is a symlink to /private/tmp
    // and an unresolved path fails with "statfs ... no such file or directory".
    const real = realpathSync(os.tmpdir());
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
      projectDir: os.tmpdir(),
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.ok(valuesFor(args, '-v').includes(`${real}:${real}`));
  });

  it('shadows node_modules with the linux-built volume', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
      projectDir: '/Users/me/app',
      modulesVolume: 'genkit-box-modules',
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.ok(
      valuesFor(args, '-v').includes(
        'genkit-box-modules:/Users/me/app/node_modules'
      )
    );
  });

  it('formats readonly bind mounts', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
      mounts: [{ source: '/data', target: '/data', readonly: true }],
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.ok(valuesFor(args, '-v').includes('/data:/data:ro'));
  });

  it('puts image and command last, in that order', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node dist/boxed.js --flag',
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.deepStrictEqual(args.slice(-4), [
      'node:22-slim',
      'node',
      'dist/boxed.js',
      '--flag',
    ]);
  });

  it('self mode uses the image interpreter, not the host execPath', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      self: true,
    }).buildRunArgs('box-1', 54321, 's3cret');
    const imageIdx = args.indexOf('node:22-slim');
    // The host's execPath is the wrong OS/arch for the container.
    assert.notStrictEqual(args[imageIdx + 1], process.execPath);
    assert.strictEqual(args[imageIdx + 1], 'node');
    // The entry script carries over verbatim (same-path mount).
    assert.ok(args.slice(imageIdx + 1).includes(process.argv[1]));
  });

  it('self mode honors a custom interpreter', () => {
    const args = podmanRunner({
      image: 'my-python-box',
      self: true,
      interpreter: 'python3',
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.strictEqual(args[args.indexOf('my-python-box') + 1], 'python3');
  });

  it('appends extraArgs before the image', () => {
    const args = podmanRunner({
      image: 'node:22-slim',
      cmd: 'node boxed.js',
      extraArgs: ['--memory=512m', '--pids-limit=256'],
    }).buildRunArgs('box-1', 54321, 's3cret');
    assert.ok(args.indexOf('--memory=512m') < args.indexOf('node:22-slim'));
    assert.ok(args.indexOf('--pids-limit=256') < args.indexOf('node:22-slim'));
  });
});
