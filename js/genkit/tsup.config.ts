/**
 * Copyright 2024 Google LLC
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

import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { build, defineConfig, type Options } from 'tsup';
import { defaultOptions } from '../tsup.common';

/**
 * Extracts leading block comments (license and module JSDoc) from a source file.
 */
function extractLeadingComments(source: string): string {
  const match = source.match(/^(?:\s*\/\*[\s\S]*?\*\/\s*)+/);
  return match ? `${match[0].trim()}\n\n` : '';
}

export default defineConfig({
  ...(defaultOptions as Options),
  dts: false,
  async onSuccess() {
    await build({
      ...(defaultOptions as Options),
      config: false,
      clean: false,
      dts: { only: true },
      external: ['./plugin.js', './genkit.js'],
    });

    const indexSource = await readFile('src/index.ts', 'utf8');
    const header = extractLeadingComments(indexSource);

    for (const indexDtsPath of ['lib/index.d.ts', 'lib/index.d.mts']) {
      const indexDts = await readFile(indexDtsPath, 'utf8');
      await writeFile(indexDtsPath, `${header}${indexDts}`, 'utf8');
    }

    const libEntries = await readdir('lib', { recursive: true });
    for (const entry of libEntries) {
      if (!entry.endsWith('.d.mts')) continue;
      const dtsMtsPath = join('lib', entry);
      const content = await readFile(dtsMtsPath, 'utf8');
      const rewritten = content.replace(
        /from '(\.\.?\/[^']+)\.js'/g,
        "from '$1.mjs'"
      );
      if (rewritten !== content) {
        await writeFile(dtsMtsPath, rewritten, 'utf8');
      }
    }
  },
});
