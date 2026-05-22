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

import type { Plugin } from 'esbuild';
import { readFile } from 'node:fs/promises';

/**
 * Rewrites `export * from './foo.js'` to `export * from './foo.mjs'` in
 * ESM-format builds. With `bundle: false`, esbuild transpiles each source
 * file individually and preserves its import paths as-is, so the ESM
 * output of a wildcard re-export keeps pointing at the sibling CJS `.js`
 * file. Vite's SSR module runner (e.g. Angular SSR) cannot statically
 * expand `export *` through a CJS file, leaving named bindings (`z`,
 * `genkit`, …) undefined for consumers.
 *
 * Scope is intentionally narrow: only `export *` wildcard re-exports are
 * affected. Regular `import`/`import-from` statements keep their `.js`
 * targets so existing CJS-interop paths (which tolerate extensionless
 * `require`) continue to work for plugins whose source uses extensionless
 * relative imports.
 */
const rewriteWildcardReexportsForEsm: Plugin = {
  name: 'rewrite-wildcard-reexports-for-esm',
  setup(build) {
    if (build.initialOptions.format !== 'esm') return;
    const loaders: Record<string, 'ts' | 'tsx' | 'js' | 'jsx'> = {
      ts: 'ts',
      tsx: 'tsx',
      mts: 'ts',
      cts: 'ts',
      js: 'js',
      jsx: 'jsx',
      mjs: 'js',
      cjs: 'js',
    };
    build.onLoad(
      { filter: /\.(ts|tsx|mts|cts|js|jsx|mjs|cjs)$/ },
      async (args) => {
        const ext = args.path.match(/\.([^.]+)$/)?.[1] ?? '';
        const loader = loaders[ext];
        if (!loader) return null;
        const source = await readFile(args.path, 'utf8');
        const rewritten = source.replace(
          /(export\s*\*\s*from\s*)(['"`])(\.\.?\/[^'"`]+?)\.js\2/g,
          '$1$2$3.mjs$2'
        );
        if (rewritten === source) return null;
        return { contents: rewritten, loader };
      }
    );
  },
};

export const defaultOptions = {
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  clean: true,
  shims: true,
  outDir: 'lib',
  entry: ['src/**/*.ts'],
  bundle: false,
  treeshake: false,
  esbuildPlugins: [rewriteWildcardReexportsForEsm],
};

/**
 *
 */
export function fromPackageJson(packageJson: {
  exports?: { [key: string]: { import: string } };
}): string[] {
  if (!packageJson.exports) return ['./src/index.ts'];
  const out: string[] = [];
  for (const key in packageJson.exports) {
    if (Object.prototype.hasOwnProperty.call(packageJson.exports, key)) {
      const importFile = packageJson.exports[key].import;
      out.push(importFile.replace('./lib', './src').replace('.mjs', '.ts'));
    }
  }
  return out;
}
