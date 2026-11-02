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

import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
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
import type { Doc } from '../../src/utils/docs';
import * as docsUtils from '../../src/utils/docs';

jest.mock('@genkit-ai/tools-common/utils', () => {
  const actual = jest.requireActual('@genkit-ai/tools-common/utils') as Record<
    string,
    unknown
  >;
  return {
    ...actual,
    findProjectRoot: jest.fn(),
  };
});

const SAMPLE_DOCS: Record<string, Doc> = {
  'js/flows.md': {
    title: 'Defining Flows (JS)',
    description: 'Define flows in JavaScript with defineFlow.',
    text: '# JS Flows',
    lang: 'js',
    headers: 'defineFlow',
  },
  'go/flows.md': {
    title: 'Defining Flows (Go)',
    description: 'Define flows in Go with genkit.DefineFlow.',
    text: '# Go Flows',
    lang: 'go',
    headers: 'DefineFlow',
  },
  'python/flows.md': {
    title: 'Defining Flows (Python)',
    description: 'Define flows in Python with @ai.flow.',
    text: '# Python Flows',
    lang: 'python',
    headers: 'flow decorator',
  },
  'dart/flows.md': {
    title: 'Defining Flows (Dart)',
    description: 'Define flows in Dart with ai.defineFlow.',
    text: '# Dart Flows',
    lang: 'dart',
    headers: 'defineFlow schemantic',
  },
};

describe('docs:list and docs:search CLI commands', () => {
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

  it.each([
    { marker: 'go.mod', expectedLang: 'go', expectedPath: 'go/flows.md' },
    {
      marker: 'pyproject.toml',
      expectedLang: 'python',
      expectedPath: 'python/flows.md',
    },
    {
      marker: 'pubspec.yaml',
      expectedLang: 'dart',
      expectedPath: 'dart/flows.md',
    },
    {
      marker: 'package.json',
      expectedLang: 'js',
      expectedPath: 'js/flows.md',
    },
  ])(
    'auto-detects $expectedLang in docs:list and docs:search when [language] is omitted ($marker)',
    async ({ marker, expectedLang, expectedPath }) => {
      const projDir = path.join(tmpDir, `cli-${marker}`);
      fs.mkdirSync(projDir, { recursive: true });
      fs.writeFileSync(path.join(projDir, marker), '');
      (findProjectRoot as jest.Mock<any>).mockResolvedValue(projDir);

      loggedLines = [];
      await docsList.parseAsync(['node', 'docs:list']);
      const listOutput = loggedLines.join('\n');
      expect(listOutput).toContain(
        `Genkit Documentation Index (${expectedLang}):`
      );
      expect(listOutput).toContain(`Path: ${expectedPath}`);

      loggedLines = [];
      await docsSearch.parseAsync(['node', 'docs:search', 'Flows']);
      const searchOutput = loggedLines.join('\n');
      expect(searchOutput).toContain(`Path: ${expectedPath}`);
    }
  );

  it('respects explicit [language] argument in docs:list and docs:search', async () => {
    const goDir = path.join(tmpDir, 'cli-go-override');
    fs.mkdirSync(goDir, { recursive: true });
    fs.writeFileSync(path.join(goDir, 'go.mod'), 'module example.com/app');
    (findProjectRoot as jest.Mock<any>).mockResolvedValue(goDir);

    loggedLines = [];
    await docsList.parseAsync(['node', 'docs:list', 'dart']);
    const listOutput = loggedLines.join('\n');
    expect(listOutput).toContain('Genkit Documentation Index (dart):');
    expect(listOutput).toContain('Path: dart/flows.md');

    loggedLines = [];
    await docsSearch.parseAsync(['node', 'docs:search', 'Flows', 'python']);
    const searchOutput = loggedLines.join('\n');
    expect(searchOutput).toContain('Path: python/flows.md');
  });
});
