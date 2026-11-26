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

/**
 * Keeps each published package's `typesVersions` in sync with its `exports`.
 *
 * `exports` is the source of truth. `typesVersions` exists only so projects on
 * TypeScript < 6 with `moduleResolution: node10` (the old default for
 * `module: commonjs`) can resolve subpath types like `genkit/beta`: node10
 * ignores `exports`. Newer resolution modes ignore `typesVersions` when
 * `exports` is present, so the two never conflict.
 *
 *   pnpm types-versions          # rewrite package.json files
 *   pnpm types-versions --check  # fail if out of sync or a .d.ts is missing (run after build)
 */

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';

// `null` is valid in exports (blocks a subpath).
type ExportEntry = string | null | { types?: string; default?: string };

function typesOf(entry: ExportEntry): string | undefined {
  return entry && typeof entry === 'object' ? entry.types : undefined;
}
interface PackageJson {
  name: string;
  private?: boolean;
  exports?: Record<string, ExportEntry>;
  typesVersions?: Record<string, Record<string, string[]>>;
  [key: string]: unknown;
}

const jsDir = join(import.meta.dirname, '..');
const check = process.argv.includes('--check');

function packageDirs(): string[] {
  const plugins = readdirSync(join(jsDir, 'plugins')).map((d) =>
    join(jsDir, 'plugins', d)
  );
  return [
    join(jsDir, 'core'),
    join(jsDir, 'ai'),
    join(jsDir, 'genkit'),
    ...plugins,
  ].filter((d) => existsSync(join(d, 'package.json')));
}

/** `{ "./beta": { types: "./lib/beta.d.ts" } }` -> `{ "beta": ["lib/beta.d.ts"] }` */
function expectedTypesVersions(
  pkg: PackageJson
): PackageJson['typesVersions'] | undefined {
  const paths: Record<string, string[]> = {};
  for (const [key, entry] of Object.entries(pkg.exports ?? {})) {
    // The root entry is covered by the top-level `types` field.
    const types = typesOf(entry);
    if (key === '.' || !types) continue;
    paths[key.replace(/^\.\//, '')] = [types.replace(/^\.\//, '')];
  }
  return Object.keys(paths).length ? { '*': paths } : undefined;
}

/** Returns a copy with `typesVersions` set (placed right after `exports`) or removed. */
function withTypesVersions(
  pkg: PackageJson,
  typesVersions: PackageJson['typesVersions'] | undefined
): PackageJson {
  const out: PackageJson = { name: pkg.name };
  for (const [k, v] of Object.entries(pkg)) {
    if (k === 'typesVersions') continue;
    out[k] = v;
    if (k === 'exports' && typesVersions) out.typesVersions = typesVersions;
  }
  return out;
}

const problems: string[] = [];
let updated = 0;
for (const dir of packageDirs()) {
  const file = join(dir, 'package.json');
  const raw = readFileSync(file, 'utf8');
  const pkg = JSON.parse(raw) as PackageJson;
  if (pkg.private) continue;
  const rel = relative(jsDir, file);
  const expected = expectedTypesVersions(pkg);

  if (check) {
    if (JSON.stringify(pkg.typesVersions) !== JSON.stringify(expected)) {
      problems.push(`${rel}: typesVersions out of sync with exports`);
    }
    for (const [key, entry] of Object.entries(pkg.exports ?? {})) {
      const types = typesOf(entry);
      if (types && !existsSync(join(dir, types))) {
        problems.push(`${rel}: exports["${key}"].types missing: ${types}`);
      }
    }
    continue;
  }

  const next = JSON.stringify(withTypesVersions(pkg, expected), null, 2) + '\n';
  if (next !== raw) {
    writeFileSync(file, next);
    updated++;
    console.log(`updated ${rel}`);
  }
}

if (check) {
  if (problems.length) {
    console.error(problems.join('\n'));
    console.error('\nRun `pnpm types-versions` in js/ (and build) to fix.');
    process.exit(1);
  }
  console.log('typesVersions in sync with exports');
} else {
  console.log(`${updated} package.json file(s) updated`);
}
