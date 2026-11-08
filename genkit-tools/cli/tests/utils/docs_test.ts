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

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Doc, detectProjectLanguage, searchDocs } from '../../src/utils/docs';

describe('docs utils', () => {
  describe('searchDocs', () => {
    const sampleDocs: Record<string, Doc> = {
      'js/app-frameworks/angular.md': {
        title: 'Angular',
        description: 'Add Genkit to an Angular app with SSR server routes.',
        text: '# Angular guide',
        lang: 'js',
        headers: 'Setup\nServer Routes',
      },
      'go/app-frameworks/angular.md': {
        title: 'Angular',
        description: 'Connect an Angular UI to your Genkit Go backend.',
        text: '# Go Angular guide',
        lang: 'go',
        headers: 'Setup\nGo Backend',
      },
      'js/flows.md': {
        title: 'Defining Flows',
        description: 'Learn how to define and run Genkit workflow functions.',
        text: '# Flows',
        lang: 'js',
        headers: 'Basic Flows\nStreaming\nSSR',
      },
      'js/minimal.md': {
        title: 'Minimal Doc',
        text: '# Minimal',
        lang: 'js',
        headers: '',
      },
    };

    it('filters search results to the requested language', () => {
      const jsResults = searchDocs(sampleDocs, 'angular', 'js');
      expect(jsResults).toHaveLength(1);
      expect(jsResults[0].file).toBe('js/app-frameworks/angular.md');

      const goResults = searchDocs(sampleDocs, 'angular', 'go');
      expect(goResults).toHaveLength(1);
      expect(goResults[0].file).toBe('go/app-frameworks/angular.md');
    });

    it('ranks title and description matches higher than header-only matches', () => {
      const results = searchDocs(sampleDocs, 'SSR', 'js');
      expect(results.map((r) => r.file)).toEqual([
        'js/app-frameworks/angular.md',
        'js/flows.md',
      ]);
      expect(results[0].score).toBeGreaterThan(results[1].score);
    });

    it('ignores stop words and two-character terms', () => {
      expect(searchDocs(sampleDocs, 'and the for to js', 'js')).toEqual([]);
    });

    it('handles documents with missing optional description', () => {
      const results = searchDocs(sampleDocs, 'minimal', 'js');
      expect(results).toHaveLength(1);
      expect(results[0].file).toBe('js/minimal.md');
    });
  });

  describe('detectProjectLanguage', () => {
    let tmpDir: string;

    beforeEach(() => {
      tmpDir = fs.realpathSync(
        fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-docs-lang-'))
      );
    });

    afterEach(() => {
      vi.restoreAllMocks();
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    it.each([
      { file: 'go.mod', expected: 'go' },
      { file: 'main.go', expected: 'go' },
      { file: 'pyproject.toml', expected: 'python' },
      { file: 'requirements.txt', expected: 'python' },
      { file: 'pubspec.yaml', expected: 'dart' },
      { file: 'package.json', expected: 'js' },
    ])(
      'detects "$expected" when "$file" is present in projectRoot',
      ({ file, expected }) => {
        const dir = path.join(tmpDir, `proj-${file}`);
        fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(path.join(dir, file), '');

        expect(detectProjectLanguage(dir)).toBe(expected);
      }
    );

    it('falls back to "js" when no marker files exist', () => {
      const emptyDir = path.join(tmpDir, 'empty');
      fs.mkdirSync(emptyDir, { recursive: true });

      expect(detectProjectLanguage(emptyDir)).toBe('js');
    });

    it('falls back to "js" when projectRoot does not exist', () => {
      expect(detectProjectLanguage(path.join(tmpDir, 'does-not-exist'))).toBe(
        'js'
      );
    });

    it('uses process.cwd() when projectRoot is omitted', () => {
      const pyDir = path.join(tmpDir, 'cwd-py');
      fs.mkdirSync(pyDir, { recursive: true });
      fs.writeFileSync(path.join(pyDir, 'pyproject.toml'), '[project]');

      vi.spyOn(process, 'cwd').mockReturnValue(pyDir);
      expect(detectProjectLanguage()).toBe('python');
    });
  });
});
