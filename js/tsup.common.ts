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
import { statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

/**
 * Rewrites relative import/export paths to `.mjs` in ESM-format builds.
 *
 * With `bundle: false`, esbuild transpiles each source file individually and
 * preserves import paths as-is. This plugin ensures all relative paths in ESM
 * output point at the `.mjs` files rather than the CJS `.js` siblings.
 *
 * Without this fix two classes of problems occur:
 *
 * 1. **Bundler failures** – Webpack (Next.js) injects `import.meta`-based HMR
 *    code into every module; when the module is a CJS `.js` file reached from
 *    an ESM entry the parse fails. Vite cannot statically expand `export *`
 *    through CJS.
 *
 * 2. **Dual-instance / `instanceof` breakage** – If some paths point at `.js`
 *    and others at `.mjs`, Node loads the same module twice (CJS + ESM).
 *    Classes like `GenkitError` then have two distinct constructors and
 *    `instanceof` checks fail across the boundary.
 *
 * The plugin handles two cases:
 *
 *   from './foo.js'    → from './foo.mjs'         (explicit .js extension)
 *   import('./bar.js') → import('./bar.mjs')      (dynamic import with .js)
 *   from '../util'     → from '../util/index.mjs' (bare directory import)
 *
 * Bare file imports (e.g. `from './error'`) are **not** supported and will
 * cause a build error. All relative imports must use explicit `.js` extensions
 * (TypeScript resolves `./error.js` → `./error.ts` with `moduleResolution:
 * NodeNext`).
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
        const fileDir = dirname(args.path);
        const errors: { text: string }[] = [];

        // Single regex matches all relative paths in from/import clauses.
        const rewritten = source.replace(
          /((?:from|import)\s*\(?\s*)(['"`])(\.\.?\/[^'"`]+?)\2/g,
          (match, prefix, quote, importPath) => {
            // Already .mjs — nothing to do
            if (importPath.endsWith('.mjs')) return match;

            // Explicit .js — rewrite to .mjs
            if (importPath.endsWith('.js')) {
              return `${prefix}${quote}${importPath.slice(0, -3)}.mjs${quote}`;
            }

            // Has some other extension (e.g. .json, .css) — leave as-is
            const lastSegment = importPath.split('/').pop() || '';
            if (lastSegment.includes('.')) return match;

            // No extension — only directories are allowed (they get /index.mjs)
            const resolved = resolve(fileDir, importPath);
            try {
              if (statSync(resolved).isDirectory()) {
                return `${prefix}${quote}${importPath}/index.mjs${quote}`;
              }
            } catch {
              // Not a directory
            }

            // Bare file import — this is not allowed. Use explicit .js extension.
            errors.push({
              text:
                `Bare relative import "${importPath}" in ${args.path}. ` +
                `Add an explicit .js extension (e.g. "${importPath}.js"). ` +
                `TypeScript with moduleResolution:NodeNext resolves .js → .ts.`,
            });
            return match;
          }
        );

        if (errors.length > 0) {
          return { errors };
        }

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
