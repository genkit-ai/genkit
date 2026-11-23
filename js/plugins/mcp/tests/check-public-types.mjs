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
const ts = require('typescript');
const tscBin = require.resolve('typescript/bin/tsc');
const pkgDir = realpathSync(fileURLToPath(new URL('..', import.meta.url)));
const configPath = resolve(pkgDir, 'tsconfig.public-types.json');

function main() {
  let stdout;
  try {
    stdout = execFileSync(
      process.execPath,
      [tscBin, '-p', configPath, '--noEmit', '--listFiles'],
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
        process.exitCode = err.status;
        return;
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

  const readResult = ts.readConfigFile(configPath, ts.sys.readFile);
  if (readResult.error) {
    console.error(
      ts.flattenDiagnosticMessageText(readResult.error.messageText, '\n')
    );
    process.exitCode = 1;
    return;
  }
  const parsed = ts.parseJsonConfigFileContent(
    readResult.config,
    ts.sys,
    pkgDir
  );
  if (parsed.errors.length > 0) {
    for (const diagnostic of parsed.errors) {
      console.error(
        ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')
      );
    }
    process.exitCode = 1;
    return;
  }

  const checks = [
    {
      fixture: 'tests/public-types.mts',
      expectedDts: 'lib/index.d.mts',
      mode: ts.ModuleKind.ESNext,
    },
    {
      fixture: 'tests/public-types.cts',
      expectedDts: 'lib/index.d.ts',
      mode: ts.ModuleKind.CommonJS,
    },
  ];

  const failures = [];
  for (const { fixture, expectedDts, mode } of checks) {
    const expectedPath = resolve(pkgDir, expectedDts);
    if (!loadedFiles.has(expectedPath)) {
      failures.push(`program did not load ${expectedDts}`);
    }
    const resolution = ts.resolveModuleName(
      '@genkit-ai/mcp',
      resolve(pkgDir, fixture),
      parsed.options,
      ts.sys,
      undefined,
      undefined,
      mode
    );
    const resolvedFile = resolution.resolvedModule?.resolvedFileName
      ? resolve(resolution.resolvedModule.resolvedFileName)
      : null;
    if (resolvedFile !== expectedPath) {
      failures.push(
        `${fixture} resolved '@genkit-ai/mcp' to ${resolvedFile ?? '<unresolved>'} instead of ${expectedPath}`
      );
    }
  }

  if (failures.length > 0) {
    console.error(
      `Public declaration verification failed:\n- ${failures.join('\n- ')}`
    );
    process.exitCode = 1;
  }
}

main();
