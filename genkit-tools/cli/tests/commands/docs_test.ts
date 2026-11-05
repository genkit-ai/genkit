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

import { logger } from '@genkit-ai/tools-common/utils';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { docsList, docsSearch } from '../../src/commands/docs';
import * as docsUtils from '../../src/utils/docs';
import { SAMPLE_DOCS } from '../utils/docs';

describe('docs CLI commands', () => {
  let loggedLines: string[];

  beforeEach(() => {
    loggedLines = [];
    vi.spyOn(docsUtils, 'loadDocs').mockResolvedValue(SAMPLE_DOCS);
    vi.spyOn(logger, 'info').mockImplementation(((...args: unknown[]) => {
      loggedLines.push(args.map(String).join(' '));
    }) as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('docs:list', () => {
    it('auto-detects project language when [language] is omitted', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('python');

      await docsList.parseAsync(['node', 'docs:list']);
      const output = loggedLines.join('\n');
      expect(docsUtils.detectProjectLanguage).toHaveBeenCalled();
      expect(output).toContain('Genkit Documentation Index (python):');
      expect(output).toContain('Path: python/flows.md');
    });

    it('respects an explicit [language] argument over the detected project language', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      await docsList.parseAsync(['node', 'docs:list', 'dart']);
      const output = loggedLines.join('\n');
      expect(output).toContain('Genkit Documentation Index (dart):');
      expect(output).toContain('Path: dart/flows.md');
    });
  });

  describe('docs:search', () => {
    it('auto-detects project language when [language] is omitted', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      await docsSearch.parseAsync(['node', 'docs:search', 'Flows']);
      const output = loggedLines.join('\n');
      expect(docsUtils.detectProjectLanguage).toHaveBeenCalled();
      expect(output).toContain('Path: go/flows.md');
    });

    it('respects an explicit [language] argument over the detected project language', async () => {
      vi.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      await docsSearch.parseAsync(['node', 'docs:search', 'Flows', 'python']);
      const output = loggedLines.join('\n');
      expect(output).toContain('Path: python/flows.md');
    });
  });
});
