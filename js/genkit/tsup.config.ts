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

import { readFile, writeFile } from 'node:fs/promises';
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

    for (const [dtsExt, jsExt] of [
      ['d.ts', 'js'],
      ['d.mts', 'mjs'],
    ] as const) {
      const indexDtsPath = `lib/index.${dtsExt}`;
      const genkitDtsPath = `lib/genkit.${dtsExt}`;
      const [indexDts, rawGenkitDts] = await Promise.all([
        readFile(indexDtsPath, 'utf8'),
        readFile(genkitDtsPath, 'utf8'),
      ]);

      const genkitDts =
        jsExt === 'mjs'
          ? rawGenkitDts.replace(/from '(\.\.?\/[^']+)\.js'/g, "from '$1.mjs'")
          : rawGenkitDts;

      const combinedIndexDts =
        header +
        indexDts.replace(
          /export \{[^}]*\} from '\.\/genkit\.(?:js|mjs)';/,
          genkitDts.trim()
        );

      const genkitReexportDts = `${header}export { Genkit, type GenkitOptions, type PromptFn, __disableReflectionApi, genkit } from './index.${jsExt}';\n`;

      await Promise.all([
        writeFile(indexDtsPath, combinedIndexDts, 'utf8'),
        writeFile(genkitDtsPath, genkitReexportDts, 'utf8'),
      ]);
    }

    for (const dtsMtsFile of [
      'lib/plugin.d.mts',
      'lib/genkit-beta.d.mts',
      'lib/beta.d.mts',
    ]) {
      const content = await readFile(dtsMtsFile, 'utf8');
      await writeFile(
        dtsMtsFile,
        content.replace(/from '(\.\.?\/[^']+)\.js'/g, "from '$1.mjs'"),
        'utf8'
      );
    }
  },
});
