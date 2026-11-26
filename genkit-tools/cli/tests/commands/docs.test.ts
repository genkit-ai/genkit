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

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { docsList, docsSearch } from '../../src/commands/docs.js';
import * as docsUtils from '../../src/utils/docs.js';
import * as outputUtils from '../../src/utils/output.js';
import { SAMPLE_DOCS } from '../utils/docs.js';

describe('docs CLI commands', () => {
  let stdoutLines: string[];
  let stderrLines: string[];

  beforeEach(() => {
    stdoutLines = [];
    stderrLines = [];
    vi.spyOn(docsUtils, 'loadDocs').mockResolvedValue(SAMPLE_DOCS);
    vi.spyOn(outputUtils, 'writeStdout').mockImplementation((text = '') => {
      stdoutLines.push(text);
    });
    vi.spyOn(outputUtils, 'writeStderr').mockImplementation((text = '') => {
      stderrLines.push(text);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('docs:list', () => {
    it('auto-detects project language when [language] is omitted', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('python');

      await docsList.parseAsync(['node', 'docs:list']);
      const output = stdoutLines.join('\n');
      expect(docsUtils.detectProjectLanguage).toHaveBeenCalled();
      expect(output).toContain('Path: python/flows.md');
      expect(stderrLines).toEqual([
        'Genkit Documentation Index (python):\n',
        "Use 'genkit docs:read <path>' to read a document.",
      ]);
    });

    it('respects an explicit [language] argument over the detected project language', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      await docsList.parseAsync(['node', 'docs:list', 'dart']);
      const output = stdoutLines.join('\n');
      expect(output).toContain('Path: dart/flows.md');
      expect(stderrLines).toEqual([
        'Genkit Documentation Index (dart):\n',
        "Use 'genkit docs:read <path>' to read a document.",
      ]);
    });
  });

  describe('docs:search', () => {
    it('auto-detects project language when [language] is omitted', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      await docsSearch.parseAsync(['node', 'docs:search', 'Flows']);
      const output = stdoutLines.join('\n');
      expect(docsUtils.detectProjectLanguage).toHaveBeenCalled();
      expect(output).toContain('Path: go/flows.md');
      expect(stderrLines).toEqual([
        expect.stringContaining('matching documents for "Flows":'),
      ]);
    });

    it('respects an explicit [language] argument over the detected project language', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      await docsSearch.parseAsync(['node', 'docs:search', 'Flows', 'python']);
      const output = stdoutLines.join('\n');
      expect(output).toContain('Path: python/flows.md');
    });
  });
});
