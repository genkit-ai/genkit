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
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import * as os from 'node:os';
import * as path from 'node:path';
import { afterEach, describe, it } from 'node:test';
import { pathToFileURL } from 'node:url';

const coreRoot =
  typeof __dirname !== 'undefined'
    ? path.resolve(__dirname, '..')
    : path.resolve(path.dirname(process.argv[1]), '..');
const coreNodeModules = path.join(coreRoot, 'node_modules');
const coreLibDir = path.join(coreRoot, 'lib');
const hasBuiltLib =
  fs.existsSync(path.join(coreLibDir, 'tracing.js')) &&
  fs.existsSync(path.join(coreLibDir, 'tracing.mjs'));
const tsxImportSpecifier = pathToFileURL(
  createRequire(path.join(coreRoot, 'package.json')).resolve('tsx')
).href;

type Mode = 'src' | 'cjs' | 'esm';

const modesToTest: Mode[] = hasBuiltLib ? ['src', 'cjs', 'esm'] : ['src'];

interface Fixture {
  root: string;
  storeDir: string;
  coreDir: string;
  appDir: string;
  otherDir: string;
}

const createdDirs: string[] = [];

function createIsolatedFixture(mode: Mode): Fixture {
  const root = fs.mkdtempSync(
    path.join(os.tmpdir(), 'genkit-firebase-monitoring-')
  );
  createdDirs.push(root);

  const storeDir = path.join(root, 'store');
  const storeNodeModules = path.join(storeDir, 'node_modules');
  const coreDir = path.join(storeNodeModules, '@genkit-ai', 'core');
  const appDir = path.join(root, 'app');
  const otherDir = path.join(root, 'other');

  fs.mkdirSync(coreDir, { recursive: true });
  fs.mkdirSync(appDir, { recursive: true });
  fs.mkdirSync(otherDir, { recursive: true });

  fs.copyFileSync(
    path.join(coreRoot, 'package.json'),
    path.join(coreDir, 'package.json')
  );
  if (mode === 'src') {
    fs.cpSync(path.join(coreRoot, 'src'), path.join(coreDir, 'src'), {
      recursive: true,
    });
  } else {
    fs.cpSync(coreLibDir, path.join(coreDir, 'lib'), { recursive: true });
  }

  // Symlink @genkit-ai/core's runtime dependencies without @genkit-ai/firebase.
  for (const entry of fs.readdirSync(coreNodeModules)) {
    if (entry === '@genkit-ai' || entry === '.bin' || entry === '.cache') {
      continue;
    }
    fs.symlinkSync(
      path.join(coreNodeModules, entry),
      path.join(storeNodeModules, entry)
    );
  }

  return { root, storeDir, coreDir, appDir, otherDir };
}

function writeFirebaseStub(
  parentDir: string,
  options: { label: string; asyncDelayMs?: number }
) {
  const firebaseDir = path.join(
    parentDir,
    'node_modules',
    '@genkit-ai',
    'firebase'
  );
  fs.mkdirSync(firebaseDir, { recursive: true });
  fs.writeFileSync(
    path.join(firebaseDir, 'package.json'),
    JSON.stringify({
      name: '@genkit-ai/firebase',
      version: '1.44.0',
      main: 'index.js',
    })
  );
  const delayMs = options.asyncDelayMs ?? 0;
  fs.writeFileSync(
    path.join(firebaseDir, 'index.js'),
    [
      'exports.enableFirebaseTelemetry = () => {',
      `  process.stdout.write(${JSON.stringify(`FIREBASE_INIT:${options.label}\n`)});`,
      `  global.__GENKIT_TELEMETRY_INSTRUMENTED = new Promise((resolve) => {`,
      `    setTimeout(() => {`,
      `      process.stdout.write(${JSON.stringify(`TELEMETRY_SETTLED:${options.label}\n`)});`,
      `      resolve();`,
      `    }, ${delayMs});`,
      `  });`,
      '};',
    ].join('\n')
  );
}

function runCheck(options: {
  mode: Mode;
  coreDir: string;
  scriptDir: string;
  cwd: string;
  envValue?: string;
  preInitializeTelemetry?: boolean;
}) {
  const { mode, coreDir, scriptDir, cwd, envValue, preInitializeTelemetry } =
    options;
  const entryPath =
    mode === 'src'
      ? path.join(coreDir, 'src', 'tracing.ts')
      : mode === 'cjs'
        ? path.join(coreDir, 'lib', 'tracing.js')
        : path.join(coreDir, 'lib', 'tracing.mjs');

  const scriptFile = path.join(
    scriptDir,
    mode === 'esm' ? 'run.mjs' : mode === 'cjs' ? 'run.cjs' : 'run.mts'
  );

  const preInitSnippet = preInitializeTelemetry
    ? 'global.__GENKIT_TELEMETRY_INSTRUMENTED = Promise.resolve();'
    : '';

  const scriptBody =
    mode === 'cjs'
      ? [
          preInitSnippet,
          `const { ensureBasicTelemetryInstrumentation } = require(${JSON.stringify(entryPath)});`,
          'ensureBasicTelemetryInstrumentation().then(() => {',
          '  process.stdout.write("ENSURE_DONE\\n");',
          '});',
        ].join('\n')
      : [
          preInitSnippet,
          `import { ensureBasicTelemetryInstrumentation } from ${JSON.stringify(pathToFileURL(entryPath).href)};`,
          'await ensureBasicTelemetryInstrumentation();',
          'process.stdout.write("ENSURE_DONE\\n");',
        ].join('\n');

  fs.writeFileSync(scriptFile, scriptBody);

  const env: NodeJS.ProcessEnv = { ...process.env };
  if (envValue === undefined) {
    delete env.ENABLE_FIREBASE_MONITORING;
  } else {
    env.ENABLE_FIREBASE_MONITORING = envValue;
  }

  const args =
    mode === 'src'
      ? ['--import', tsxImportSpecifier, scriptFile]
      : [scriptFile];

  const result = spawnSync(process.execPath, args, {
    cwd,
    env,
    encoding: 'utf8',
  });
  assert.strictEqual(
    result.status,
    0,
    `child process exited with ${result.status}\nstdout: ${result.stdout}\nstderr: ${result.stderr}`
  );
  return {
    stdout: result.stdout,
    stderr: result.stderr,
  };
}

describe('Firebase monitoring auto-initialization', () => {
  afterEach(() => {
    while (createdDirs.length > 0) {
      const dir = createdDirs.pop()!;
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  for (const mode of modesToTest) {
    describe(`mode: ${mode}`, () => {
      it('does nothing when ENABLE_FIREBASE_MONITORING is unset or false', () => {
        const fixture = createIsolatedFixture(mode);
        for (const envValue of [undefined, 'false']) {
          const { stdout, stderr } = runCheck({
            mode,
            coreDir: fixture.coreDir,
            scriptDir: fixture.appDir,
            cwd: fixture.appDir,
            envValue,
          });
          assert.strictEqual(stdout.trim(), 'ENSURE_DONE');
          assert.strictEqual(stderr.trim(), '');
        }
      });

      it('warns without throwing when ENABLE_FIREBASE_MONITORING=true and @genkit-ai/firebase is missing', () => {
        const fixture = createIsolatedFixture(mode);
        const { stdout, stderr } = runCheck({
          mode,
          coreDir: fixture.coreDir,
          scriptDir: fixture.appDir,
          cwd: fixture.appDir,
          envValue: 'true',
        });
        assert.strictEqual(stdout.trim(), 'ENSURE_DONE');
        assert.match(stderr, /npm i --save @genkit-ai\/firebase/);
      });

      it('loads a peer-adjacent @genkit-ai/firebase even when process.cwd() is elsewhere and awaits telemetry init', () => {
        const fixture = createIsolatedFixture(mode);
        writeFirebaseStub(fixture.storeDir, {
          label: 'peer',
          asyncDelayMs: 30,
        });

        const { stdout, stderr } = runCheck({
          mode,
          coreDir: fixture.coreDir,
          scriptDir: fixture.otherDir,
          cwd: fixture.otherDir,
          envValue: 'true',
        });

        assert.strictEqual(stderr.trim(), '');
        assert.deepStrictEqual(stdout.trim().split('\n'), [
          'FIREBASE_INIT:peer',
          'TELEMETRY_SETTLED:peer',
          'ENSURE_DONE',
        ]);
      });

      it('falls back to app node_modules from process.cwd() when @genkit-ai/firebase is not peer-adjacent', () => {
        const fixture = createIsolatedFixture(mode);
        writeFirebaseStub(fixture.appDir, { label: 'app-cwd' });

        const { stdout, stderr } = runCheck({
          mode,
          coreDir: fixture.coreDir,
          scriptDir: fixture.appDir,
          cwd: fixture.appDir,
          envValue: 'true',
        });

        assert.strictEqual(stderr.trim(), '');
        assert.deepStrictEqual(stdout.trim().split('\n'), [
          'FIREBASE_INIT:app-cwd',
          'TELEMETRY_SETTLED:app-cwd',
          'ENSURE_DONE',
        ]);
      });

      it('falls back to entry script node_modules when process.cwd() differs from the app root', () => {
        const fixture = createIsolatedFixture(mode);
        writeFirebaseStub(fixture.appDir, { label: 'app-entry' });

        const { stdout, stderr } = runCheck({
          mode,
          coreDir: fixture.coreDir,
          scriptDir: fixture.appDir,
          cwd: fixture.otherDir,
          envValue: 'true',
        });

        assert.strictEqual(stderr.trim(), '');
        assert.deepStrictEqual(stdout.trim().split('\n'), [
          'FIREBASE_INIT:app-entry',
          'TELEMETRY_SETTLED:app-entry',
          'ENSURE_DONE',
        ]);
      });

      it('skips auto-init when telemetry instrumentation is already configured', () => {
        const fixture = createIsolatedFixture(mode);
        const { stdout, stderr } = runCheck({
          mode,
          coreDir: fixture.coreDir,
          scriptDir: fixture.appDir,
          cwd: fixture.appDir,
          envValue: 'true',
          preInitializeTelemetry: true,
        });

        assert.strictEqual(stdout.trim(), 'ENSURE_DONE');
        assert.strictEqual(stderr.trim(), '');
      });
    });
  }
});
