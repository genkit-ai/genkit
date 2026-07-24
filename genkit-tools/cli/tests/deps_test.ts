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

import { describe, expect, it } from '@jest/globals';
import { readFileSync, readdirSync } from 'fs';
import { join, relative } from 'path';

const cliRoot = join(__dirname, '..');
const srcRoot = join(cliRoot, 'src');
const zodImportPattern = /\bfrom\s+['"]zod['"]|require\(['"]zod['"]\)/;

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      return sourceFiles(entryPath);
    }
    return entryPath.endsWith('.ts') ? [entryPath] : [];
  });
}

describe('package dependencies', () => {
  it('declares zod when CLI source imports it directly', () => {
    const packageJson = JSON.parse(
      readFileSync(join(cliRoot, 'package.json'), 'utf8')
    ) as { dependencies?: Record<string, string> };

    const zodImporters = sourceFiles(srcRoot)
      .filter((file) => zodImportPattern.test(readFileSync(file, 'utf8')))
      .map((file) => relative(cliRoot, file))
      .sort();

    expect(zodImporters.length).toBeGreaterThan(0);
    expect(packageJson.dependencies?.zod).toBe('^3.22.4');
  });
});
