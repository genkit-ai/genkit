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
 * Rewrites all relative `.js` imports and exports to their `.mjs`
 * equivalents in ESM-format builds.  With `bundle: false`, esbuild
 * transpiles each source file individually and preserves its import paths
 * as-is, so the ESM output keeps pointing at the sibling CJS `.js` files.
 *
 * This causes two classes of problems:
 *
 * 1. **Bundler failures** - Webpack (Next.js) injects `import.meta`-based
 *    HMR code into every module it processes; when the module is a CJS
 *    `.js` file reached from an ESM entry, the parse fails.  Vite's SSR
 *    runner cannot statically expand `export *` through CJS, leaving
 *    named bindings undefined.
 *
 * 2. **Dual-instance / `instanceof` breakage** - If only *re-exports* are
 *    rewritten but regular `import … from` statements are left pointing
 *    at `.js`, Node (and bundlers) load the same logical module twice:
 *    once as ESM (`.mjs`) via the re-export and once as CJS (`.js`) via
 *    the internal import.  Classes such as `GenkitError` then exist as
 *    two distinct constructors, so `instanceof` checks across the
 *    boundary fail.
 *
 * To avoid both issues every relative `.js` path - whether it appears in
 * an `import`, `export`, `export *`, or side-effect `import` - is
 * rewritten to `.mjs` in ESM output.
 */
const rewriteRelativeImportsForEsm: Plugin = {
  name: 'rewrite-relative-imports-for-esm',
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
        const rewritten = source
          // Rewrite `from './foo.js'` in both import and export statements.
          //   import { Foo } from './foo.js'    → import { Foo } from './foo.mjs'
          //   export { Bar } from '../bar.js'   → export { Bar } from '../bar.mjs'
          //   export * from './utils/index.js'  → export * from './utils/index.mjs'
          .replace(/(from\s*)(['"`])(\.\.?\/[^'"`]+?)\.js\2/g, '$1$2$3.mjs$2')
          // Rewrite side-effect and dynamic imports:
          //   import './polyfills.js'             → import './polyfills.mjs'
          //   await import('./reflection-v2.js')  → await import('./reflection-v2.mjs')
          .replace(
            /(import\s*\(?\s*)(['"`])(\.\.?\/[^'"`]+?)\.js\2/g,
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
  esbuildPlugins: [rewriteRelativeImportsForEsm],
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
