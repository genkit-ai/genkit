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
import { z } from 'zod';
import { defineDocsTool } from '../../src/mcp/docs';
import { defineUsageGuideTool } from '../../src/mcp/usage';
import { detectProjectLanguage } from '../../src/mcp/utils';
import type { Doc } from '../../src/utils/docs';
import * as docsUtils from '../../src/utils/docs';

jest.mock('@genkit-ai/tools-common/utils', () => {
  const actual = jest.requireActual('@genkit-ai/tools-common/utils') as Record<
    string,
    unknown
  >;
  return {
    ...actual,
    record: jest.fn(() => Promise.resolve()),
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

describe('MCP language auto-detection, docs, and usage guide tools', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.realpathSync(
      fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-mcp-lang-'))
    );
    jest.spyOn(docsUtils, 'loadDocs').mockResolvedValue(SAMPLE_DOCS);
  });

  afterEach(() => {
    jest.restoreAllMocks();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('detectProjectLanguage', () => {
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

    it('prioritizes non-JS project markers when package.json is also present', () => {
      const goDir = path.join(tmpDir, 'go-with-pkg');
      fs.mkdirSync(goDir, { recursive: true });
      fs.writeFileSync(path.join(goDir, 'go.mod'), 'module example.com/app');
      fs.writeFileSync(path.join(goDir, 'package.json'), '{}');
      expect(detectProjectLanguage(goDir)).toBe('go');

      const standaloneGoDir = path.join(tmpDir, 'standalone-go-with-pkg');
      fs.mkdirSync(standaloneGoDir, { recursive: true });
      fs.writeFileSync(path.join(standaloneGoDir, 'main.go'), 'package main');
      fs.writeFileSync(path.join(standaloneGoDir, 'package.json'), '{}');
      expect(detectProjectLanguage(standaloneGoDir)).toBe('go');

      const pyDir = path.join(tmpDir, 'py-with-pkg');
      fs.mkdirSync(pyDir, { recursive: true });
      fs.writeFileSync(path.join(pyDir, 'pyproject.toml'), '[project]');
      fs.writeFileSync(path.join(pyDir, 'package.json'), '{}');
      expect(detectProjectLanguage(pyDir)).toBe('python');

      const dartDir = path.join(tmpDir, 'dart-with-pkg');
      fs.mkdirSync(dartDir, { recursive: true });
      fs.writeFileSync(path.join(dartDir, 'pubspec.yaml'), 'name: app');
      fs.writeFileSync(path.join(dartDir, 'package.json'), '{}');
      expect(detectProjectLanguage(dartDir)).toBe('dart');
    });

    it('uses process.cwd() when projectRoot is omitted', () => {
      const pyDir = path.join(tmpDir, 'cwd-py');
      fs.mkdirSync(pyDir, { recursive: true });
      fs.writeFileSync(path.join(pyDir, 'pyproject.toml'), '[project]');

      const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(pyDir);
      try {
        expect(detectProjectLanguage()).toBe('python');
      } finally {
        cwdSpy.mockRestore();
      }
    });
  });

  describe('defineDocsTool', () => {
    async function setupDocsServer(projectRoot: string) {
      const registeredTools: Record<string, (opts: any) => Promise<any>> = {};
      const registeredConfigs: Record<string, any> = {};
      const mockServer: any = {
        registerTool: jest.fn((name: string, config: any, handler: any) => {
          registeredConfigs[name] = config;
          registeredTools[name] = async (rawArgs: any) => {
            const schema = z.object(config.inputSchema ?? {});
            const parsed = await schema.parseAsync(rawArgs);
            return handler(parsed);
          };
        }),
      };
      await defineDocsTool(mockServer, projectRoot);
      return { registeredTools, registeredConfigs };
    }

    it.each([
      {
        marker: 'go.mod',
        expectedLang: 'go',
        expectedPath: 'go/flows.md',
      },
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
      'auto-detects $expectedLang in list_genkit_docs and search_genkit_docs when language is omitted ($marker)',
      async ({ marker, expectedLang, expectedPath }) => {
        const projDir = path.join(tmpDir, `docs-${marker}`);
        fs.mkdirSync(projDir, { recursive: true });
        fs.writeFileSync(path.join(projDir, marker), '');

        const { registeredTools } = await setupDocsServer(projDir);

        const listRes = await registeredTools['list_genkit_docs']({});
        expect(listRes.content[0].text).toContain(
          `Genkit Documentation Index (${expectedLang}):`
        );
        expect(listRes.content[0].text).toContain(`FilePath: ${expectedPath}`);

        const searchRes = await registeredTools['search_genkit_docs']({
          query: 'Flows',
        });
        expect(searchRes.content[0].text).toContain(
          `**FilePath**: ${expectedPath}`
        );
      }
    );

    it('respects explicit language override in list_genkit_docs and search_genkit_docs', async () => {
      const goDir = path.join(tmpDir, 'go-override');
      fs.mkdirSync(goDir, { recursive: true });
      fs.writeFileSync(path.join(goDir, 'go.mod'), 'module example.com/app');

      const { registeredTools } = await setupDocsServer(goDir);

      const listRes = await registeredTools['list_genkit_docs']({
        language: 'dart',
      });
      expect(listRes.content[0].text).toContain(
        'Genkit Documentation Index (dart):'
      );
      expect(listRes.content[0].text).toContain('FilePath: dart/flows.md');

      const searchRes = await registeredTools['search_genkit_docs']({
        query: 'Flows',
        language: 'python',
      });
      expect(searchRes.content[0].text).toContain(
        '**FilePath**: python/flows.md'
      );
    });
  });

  describe('defineUsageGuideTool', () => {
    async function setupUsageServer(projectRoot: string) {
      const registeredTools: Record<string, (opts: any) => Promise<any>> = {};
      const mockServer: any = {
        registerTool: jest.fn((name: string, config: any, handler: any) => {
          registeredTools[name] = async (rawArgs: any) => {
            const schema = z.object(config.inputSchema ?? {});
            const parsed = await schema.parseAsync(rawArgs);
            return handler(parsed);
          };
        }),
      };
      await defineUsageGuideTool(mockServer, projectRoot);
      return { registeredTools };
    }

    it.each([
      {
        marker: 'package.json',
        expectedHeading: '# Genkit Node.js API Rules',
      },
      {
        marker: 'go.mod',
        expectedHeading: '# Genkit Go API Rules',
      },
    ])(
      'auto-detects project language for get_usage_guide when language is omitted ($marker)',
      async ({ marker, expectedHeading }) => {
        const projDir = path.join(tmpDir, `usage-${marker}`);
        fs.mkdirSync(projDir, { recursive: true });
        fs.writeFileSync(path.join(projDir, marker), '');

        const { registeredTools } = await setupUsageServer(projDir);
        const res = await registeredTools['get_usage_guide']({});
        expect(res.content[0].text).toContain(expectedHeading);
      }
    );

    it.each([
      { marker: 'pyproject.toml', expectedLang: 'python' },
      { marker: 'pubspec.yaml', expectedLang: 'dart' },
    ])(
      'directs caller to docs tools when language is omitted in a $expectedLang workspace ($marker)',
      async ({ marker, expectedLang }) => {
        const projDir = path.join(tmpDir, `usage-${marker}`);
        fs.mkdirSync(projDir, { recursive: true });
        fs.writeFileSync(path.join(projDir, marker), '');

        const { registeredTools } = await setupUsageServer(projDir);
        const res = await registeredTools['get_usage_guide']({});
        expect(res.content[0].text).toContain(
          `No standalone usage guide is bundled for '${expectedLang}' projects.`
        );
        expect(res.content[0].text).toContain('list_genkit_docs');
        expect(res.content[0].text).toContain('search_genkit_docs');
        expect(res.content[0].text).not.toContain('# Genkit Node.js API Rules');
      }
    );

    it.each([
      { language: 'js', expectedHeading: '# Genkit Node.js API Rules' },
      { language: 'go', expectedHeading: '# Genkit Go API Rules' },
    ])(
      'returns the $language usage guide when explicitly requested',
      async ({ language, expectedHeading }) => {
        const emptyDir = path.join(tmpDir, `explicit-${language}`);
        fs.mkdirSync(emptyDir, { recursive: true });

        const { registeredTools } = await setupUsageServer(emptyDir);
        const res = await registeredTools['get_usage_guide']({ language });
        expect(res.content[0].text).toContain(expectedHeading);
      }
    );
  });
});
