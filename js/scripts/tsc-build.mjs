/**
 * Copyright 2025 Google LLC
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

/**
 * Dual CJS/ESM library build driver based on `tsc` (TypeScript 7).
 *
 * Replaces tsup for per-file (non-bundled) library builds. Run it from a
 * package directory (it uses `process.cwd()`), e.g. via a `compile` script:
 *
 *   "compile": "node ../scripts/tsc-build.mjs"        (core/ai/genkit)
 *   "compile": "node ../../scripts/tsc-build.mjs"     (plugins/*)
 *
 * What it does:
 *
 *   1. Generates two throwaway tsconfigs in the package dir, each extending the
 *      package's own `tsconfig.json` (for `include`, `skipLibCheck`, `lib`,
 *      etc.) plus a shared build base (`js/tsconfig.build.{cjs,esm}.json`) that
 *      flips on emit and sets the module format. Nothing is copy-pasted per
 *      package.
 *   2. Runs `tsc` twice (CJS + ESM) into temp output dirs.
 *   3. Assembles the final `lib/` dir: CJS output copied verbatim as
 *      `.js`/`.js.map`/`.d.ts`; ESM output copied as `.mjs`/`.mjs.map`/`.d.mts`
 *      with relative import specifiers rewritten from `./foo.js` to
 *      `./foo.mjs` (and bare directory imports to `./dir/index.mjs`).
 *
 * The `.mjs` rewrite reproduces the old tsup esbuild plugin
 * (`rewriteRelativeImportsForEsm`). All relative imports must use explicit
 * `.js` extensions (enforced by `moduleResolution: node16/nodenext`).
 *
 * Flags:
 *   --out <dir>       final output dir (default: lib)
 *   --root <dir>      source root dir (default: src)
 *   --watch           rebuild on source changes
 */

import { spawnSync } from 'node:child_process';
import {
  cpSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  watch,
  writeFileSync,
} from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const jsRoot = resolve(scriptDir, '..');

const args = process.argv.slice(2);
function flag(name, fallback) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
}

const pkgDir = process.cwd();
const outDir = flag('--out', 'lib');
const rootDir = flag('--root', 'src');

const cjsTmp = '.tsc-cjs';
const esmTmp = '.tsc-esm';
const cjsProject = '.tsconfig.build.cjs.json';
const esmProject = '.tsconfig.build.esm.json';

const tscBin = resolve(
  pkgDir,
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'tsc.cmd' : 'tsc'
);

/**
 * Writes a throwaway tsconfig extending the package config + a shared build
 * base, then returns its path.
 */
function writeProject(file, sharedBase, tmpOut) {
  const base = relative(pkgDir, join(jsRoot, sharedBase)).split('\\').join('/');
  const config = {
    extends: ['./tsconfig.json', base],
    compilerOptions: {
      rootDir: `./${rootDir}`,
      outDir: `./${tmpOut}`,
    },
  };
  writeFileSync(join(pkgDir, file), JSON.stringify(config, null, 2) + '\n');
}

function runTsc(project) {
  const res = spawnSync(tscBin, ['-p', project], {
    stdio: 'inherit',
    cwd: pkgDir,
  });
  if (res.status !== 0) process.exit(res.status ?? 1);
}

/**
 * Rewrites relative import/export specifiers in ESM source from `.js` to `.mjs`
 * (and bare directory imports to `/index.mjs`).
 */
function rewriteEsmImports(source, fileDir) {
  return source.replace(
    /((?:from|import)\s*\(?\s*)(['"`])(\.\.?\/[^'"`]+?)\2/g,
    (match, prefix, quote, importPath) => {
      if (importPath.endsWith('.mjs')) return match;
      if (importPath.endsWith('.js')) {
        return `${prefix}${quote}${importPath.slice(0, -3)}.mjs${quote}`;
      }
      const lastSegment = importPath.split('/').pop() || '';
      if (lastSegment.includes('.')) return match;
      const resolved = resolve(fileDir, importPath);
      try {
        if (statSync(resolved).isDirectory()) {
          return `${prefix}${quote}${importPath}/index.mjs${quote}`;
        }
      } catch {
        // not a directory, fall through
      }
      throw new Error(
        `Bare relative import "${importPath}" in ${fileDir}. ` +
          `Add an explicit .js extension (e.g. "${importPath}.js").`
      );
    }
  );
}

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else out.push(full);
  }
  return out;
}

/** Copies CJS output verbatim into the final output dir. */
function emitCjs() {
  for (const file of walk(cjsTmp)) {
    const rel = relative(cjsTmp, file);
    const dest = join(outDir, rel);
    mkdirSync(dirname(dest), { recursive: true });
    cpSync(file, dest);
  }
}

/** Copies ESM output into the final dir, remapping extensions and imports. */
function emitEsm() {
  for (const file of walk(esmTmp)) {
    const rel = relative(esmTmp, file);
    const fileDir = dirname(file);

    let dest;
    if (rel.endsWith('.d.ts')) {
      dest = join(outDir, rel.slice(0, -'.d.ts'.length) + '.d.mts');
    } else if (rel.endsWith('.js')) {
      dest = join(outDir, rel.slice(0, -'.js'.length) + '.mjs');
    } else if (rel.endsWith('.js.map')) {
      dest = join(outDir, rel.slice(0, -'.js.map'.length) + '.mjs.map');
    } else if (rel.endsWith('.d.ts.map')) {
      dest = join(outDir, rel.slice(0, -'.d.ts.map'.length) + '.d.mts.map');
    } else {
      dest = join(outDir, rel);
    }

    mkdirSync(dirname(dest), { recursive: true });

    if (rel.endsWith('.js') || rel.endsWith('.d.ts')) {
      let content = readFileSync(file, 'utf8');
      content = rewriteEsmImports(content, fileDir);
      // Point sourcemap comment at the .mjs.map sibling.
      content = content.replace(
        /\/\/# sourceMappingURL=(.+)\.js\.map/g,
        '//# sourceMappingURL=$1.mjs.map'
      );
      writeFileSync(dest, content);
    } else if (rel.endsWith('.js.map') || rel.endsWith('.d.ts.map')) {
      const map = JSON.parse(readFileSync(file, 'utf8'));
      if (typeof map.file === 'string') {
        map.file = map.file
          .replace(/\.d\.ts$/, '.d.mts')
          .replace(/\.js$/, '.mjs');
      }
      writeFileSync(dest, JSON.stringify(map));
    } else {
      cpSync(file, dest);
    }
  }
}

function cleanup() {
  for (const p of [cjsTmp, esmTmp, cjsProject, esmProject]) {
    rmSync(join(pkgDir, p), { recursive: true, force: true });
  }
}

function build() {
  cleanup();
  writeProject(cjsProject, 'tsconfig.build.cjs.json', cjsTmp);
  writeProject(esmProject, 'tsconfig.build.esm.json', esmTmp);
  try {
    runTsc(cjsProject);
    runTsc(esmProject);
    mkdirSync(outDir, { recursive: true });
    emitCjs();
    emitEsm();
  } finally {
    cleanup();
  }
  console.log(`tsc-build: wrote dual CJS/ESM output to ${outDir}/`);
}

function safeBuild() {
  try {
    build();
  } catch (err) {
    console.error(err instanceof Error ? err.message : err);
  }
}

if (args.includes('--watch')) {
  safeBuild();
  console.log(`tsc-build: watching ${rootDir}/ for changes...`);
  let timer = null;
  watch(join(pkgDir, rootDir), { recursive: true }, () => {
    clearTimeout(timer);
    timer = setTimeout(safeBuild, 100);
  });
} else {
  build();
}
