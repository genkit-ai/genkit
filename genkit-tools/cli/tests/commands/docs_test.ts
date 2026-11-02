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
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { docsList, docsSearch } from '../../src/commands/docs';
import * as docsUtils from '../../src/utils/docs';
import { SAMPLE_DOCS, WORKSPACE_MARKERS } from '../utils/docs';

describe('docs CLI commands', () => {
  let tmpDir: string;
  let loggedLines: string[];

  beforeEach(() => {
    tmpDir = fs.realpathSync(
      fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-cli-docs-'))
    );
    loggedLines = [];
    jest.spyOn(docsUtils, 'loadDocs').mockResolvedValue(SAMPLE_DOCS);
    jest.spyOn(logger, 'info').mockImplementation(((...args: unknown[]) => {
      loggedLines.push(args.map(String).join(' '));
    }) as any);
  });

  afterEach(() => {
    jest.restoreAllMocks();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('docs:list', () => {
    it.each(WORKSPACE_MARKERS)(
      'auto-detects $expectedLang when [language] is omitted ($marker)',
      async ({ marker, expectedLang, expectedPath }) => {
        const projDir = path.join(tmpDir, `list-${marker}`);
        fs.mkdirSync(projDir, { recursive: true });
        fs.writeFileSync(path.join(projDir, marker), '');
        jest.spyOn(process, 'cwd').mockReturnValue(projDir);

        await docsList.parseAsync(['node', 'docs:list']);
        const output = loggedLines.join('\n');
        expect(output).toContain(
          `Genkit Documentation Index (${expectedLang}):`
        );
        expect(output).toContain(`Path: ${expectedPath}`);
      }
    );

    it('respects an explicit [language] argument over the detected project language', async () => {
      const goDir = path.join(tmpDir, 'list-go-override');
      fs.mkdirSync(goDir, { recursive: true });
      fs.writeFileSync(path.join(goDir, 'go.mod'), 'module example.com/app');
      jest.spyOn(process, 'cwd').mockReturnValue(goDir);

      await docsList.parseAsync(['node', 'docs:list', 'dart']);
      const output = loggedLines.join('\n');
      expect(output).toContain('Genkit Documentation Index (dart):');
      expect(output).toContain('Path: dart/flows.md');
    });
  });

  describe('docs:search', () => {
    it.each(WORKSPACE_MARKERS)(
      'auto-detects $expectedLang when [language] is omitted ($marker)',
      async ({ marker, expectedPath }) => {
        const projDir = path.join(tmpDir, `search-${marker}`);
        fs.mkdirSync(projDir, { recursive: true });
        fs.writeFileSync(path.join(projDir, marker), '');
        jest.spyOn(process, 'cwd').mockReturnValue(projDir);

        await docsSearch.parseAsync(['node', 'docs:search', 'Flows']);
        const output = loggedLines.join('\n');
        expect(output).toContain(`Path: ${expectedPath}`);
      }
    );

    it('respects an explicit [language] argument over the detected project language', async () => {
      const goDir = path.join(tmpDir, 'search-go-override');
      fs.mkdirSync(goDir, { recursive: true });
      fs.writeFileSync(path.join(goDir, 'go.mod'), 'module example.com/app');
      jest.spyOn(process, 'cwd').mockReturnValue(goDir);

      await docsSearch.parseAsync(['node', 'docs:search', 'Flows', 'python']);
      const output = loggedLines.join('\n');
      expect(output).toContain('Path: python/flows.md');
    });
  });
});
