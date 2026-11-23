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

import { execFileSync } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const tscBin = require.resolve('typescript/bin/tsc');
const pkgDir = realpathSync(fileURLToPath(new URL('..', import.meta.url)));

let stdout;
try {
  stdout = execFileSync(
    process.execPath,
    [tscBin, '-p', 'tsconfig.public-types.json', '--noEmit', '--listFiles'],
    {
      cwd: pkgDir,
      encoding: 'utf8',
      maxBuffer: 10 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );
} catch (err) {
  if (err && typeof err === 'object') {
    if ('stdout' in err && err.stdout) {
      process.stdout.write(err.stdout);
    }
    if ('stderr' in err && err.stderr) {
      process.stderr.write(err.stderr);
    }
    if ('status' in err && typeof err.status === 'number') {
      process.exit(err.status);
    }
  }
  throw err;
}

const loadedFiles = new Set(
  stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((filePath) => resolve(filePath))
);

const expectedDeclarations = ['lib/index.d.mts', 'lib/index.d.ts'];
const missing = expectedDeclarations.filter(
  (relPath) => !loadedFiles.has(resolve(pkgDir, relPath))
);

if (missing.length > 0) {
  console.error(
    `Expected TypeScript to load both ESM and CJS declaration entry points, but missing: ${missing.join(', ')}`
  );
  process.exit(1);
}
