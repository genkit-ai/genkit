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

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

const checker = join(__dirname, 'copyright.ts');
const runner = require.resolve('tsx/cli');
const license = readFileSync(checker, 'utf8').split(' */')[0] + ' */\n\n';
const body = 'console.log("fixture");\n';

function run(command: string, args: string[], cwd: string) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: 'utf8',
    timeout: 30_000,
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null, result.stderr);
  return result;
}

const cases = [
  ...[
    ['LF', '\n'],
    ['CRLF', '\r\n'],
    ['CR', '\r'],
    ['line separator', '\u2028'],
    ['paragraph separator', '\u2029'],
  ].map(([name, newline]) => ({
    name: `hashbang with ${name}`,
    prefix: `#!/usr/bin/env node${newline}`,
    source: `#!/usr/bin/env node${newline}${body}`,
    remainder: body,
    output: 'fixture',
    licensed: false,
  })),
  {
    name: 'hashbang without a final newline',
    prefix: '#!/usr/bin/env node\n',
    source: '#!/usr/bin/env node',
    remainder: '',
    output: '',
    licensed: false,
  },
  {
    name: 'already licensed hashbang',
    prefix: '#!/usr/bin/env node\r\n',
    source: `#!/usr/bin/env node\r\n${license}${body}`,
    remainder: body,
    output: 'fixture',
    licensed: true,
  },
  {
    name: 'ordinary source containing a non-leading hashbang',
    prefix: '',
    source: 'console.log("#! is ordinary text");\n',
    remainder: 'console.log("#! is ordinary text");\n',
    output: '#! is ordinary text',
    licensed: false,
  },
  {
    name: 'empty source',
    prefix: '',
    source: '',
    remainder: '',
    output: '',
    licensed: false,
  },
];

for (const extension of ['js', 'mjs', 'cjs']) {
  for (const fixture of cases) {
    test(`${extension}: ${fixture.name}`, (t) => {
      const directory = mkdtempSync(join(tmpdir(), 'genkit-copyright-'));
      t.after(() => rmSync(directory, { recursive: true, force: true }));
      const git = run('git', ['init', '--quiet'], directory);
      assert.equal(git.status, 0, git.stderr);
      const filename = `fixture.${extension}`;
      const file = join(directory, filename);
      writeFileSync(file, fixture.source);

      function executable() {
        const syntax = run(process.execPath, ['--check', filename], directory);
        assert.equal(syntax.status, 0, syntax.stderr);
        const executed = run(process.execPath, [filename], directory);
        assert.equal(executed.status, 0, executed.stderr);
        assert.equal(executed.stdout.trim(), fixture.output);
      }

      function check() {
        return run(process.execPath, [runner, checker, '--check'], directory);
      }

      function update() {
        const result = run(process.execPath, [runner, checker], directory);
        assert.equal(result.status, 0, result.stderr);
      }

      executable();
      const before = check();
      assert.equal(before.status, fixture.licensed ? 0 : 1, before.stderr);
      if (!fixture.licensed) {
        assert.match(before.stderr, /Copyright header missing in fixture\./);
      }
      assert.equal(readFileSync(file, 'utf8'), fixture.source);

      update();
      executable();
      const updated = readFileSync(file, 'utf8');
      if (fixture.licensed) {
        assert.equal(updated, fixture.source);
      } else {
        assert.ok(updated.startsWith(fixture.prefix + '/**\n'));
        assert.equal(updated.match(/Copyright \d{4} Google LLC/g)?.length, 1);
        assert.equal(updated.split(' */\n\n')[1], fixture.remainder);
      }
      const after = check();
      assert.equal(after.status, 0, after.stderr);
      assert.equal(readFileSync(file, 'utf8'), updated);
      update();
      assert.equal(readFileSync(file, 'utf8'), updated);
    });
  }
}
