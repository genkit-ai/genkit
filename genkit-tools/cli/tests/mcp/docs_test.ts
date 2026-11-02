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
import type { Doc } from '../../src/utils/docs';
import * as docsUtils from '../../src/utils/docs';

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

const WORKSPACE_MARKERS = [
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
] as const;

describe('MCP docs and usage guide tools', () => {
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

  async function setupDocsServer(projectRoot: string) {
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
    await defineDocsTool(mockServer, projectRoot);
    return { registeredTools };
  }

  describe('list_genkit_docs', () => {
    it.each(WORKSPACE_MARKERS)(
      'auto-detects $expectedLang when language is omitted ($marker)',
      async ({ marker, expectedLang, expectedPath }) => {
        const projDir = path.join(tmpDir, `list-docs-${marker}`);
        fs.mkdirSync(projDir, { recursive: true });
        fs.writeFileSync(path.join(projDir, marker), '');

        const { registeredTools } = await setupDocsServer(projDir);
        const listRes = await registeredTools['list_genkit_docs']({});
        expect(listRes.content[0].text).toContain(
          `Genkit Documentation Index (${expectedLang}):`
        );
        expect(listRes.content[0].text).toContain(`FilePath: ${expectedPath}`);
      }
    );

    it('respects an explicit language override', async () => {
      const goDir = path.join(tmpDir, 'list-go-override');
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
    });
  });

  describe('search_genkit_docs', () => {
    it.each(WORKSPACE_MARKERS)(
      'auto-detects $expectedLang when language is omitted ($marker)',
      async ({ marker, expectedPath }) => {
        const projDir = path.join(tmpDir, `search-docs-${marker}`);
        fs.mkdirSync(projDir, { recursive: true });
        fs.writeFileSync(path.join(projDir, marker), '');

        const { registeredTools } = await setupDocsServer(projDir);
        const searchRes = await registeredTools['search_genkit_docs']({
          query: 'Flows',
        });
        expect(searchRes.content[0].text).toContain(
          `**FilePath**: ${expectedPath}`
        );
      }
    );

    it('respects an explicit language override', async () => {
      const goDir = path.join(tmpDir, 'search-go-override');
      fs.mkdirSync(goDir, { recursive: true });
      fs.writeFileSync(path.join(goDir, 'go.mod'), 'module example.com/app');

      const { registeredTools } = await setupDocsServer(goDir);
      const searchRes = await registeredTools['search_genkit_docs']({
        query: 'Flows',
        language: 'python',
      });
      expect(searchRes.content[0].text).toContain(
        '**FilePath**: python/flows.md'
      );
    });
  });

  describe('get_usage_guide', () => {
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
      'auto-detects project language when language is omitted ($marker)',
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
