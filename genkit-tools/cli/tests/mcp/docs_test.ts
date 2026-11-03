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
import { defineDocsTool } from '../../src/mcp/docs';
import { defineUsageGuideTool } from '../../src/mcp/usage';
import * as docsUtils from '../../src/utils/docs';
import { SAMPLE_DOCS } from '../utils/docs';

describe('MCP docs and usage guide tools', () => {
  let registeredTools: Record<string, (opts: any) => Promise<any>>;

  beforeEach(async () => {
    registeredTools = {};
    jest.spyOn(docsUtils, 'loadDocs').mockResolvedValue(SAMPLE_DOCS);

    const mockServer: any = {
      registerTool: jest.fn((name: string, _config: any, handler: any) => {
        registeredTools[name] = handler;
      }),
    };
    await defineDocsTool(mockServer, '/workspace');
    await defineUsageGuideTool(mockServer, '/workspace');
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('list_genkit_docs', () => {
    it('auto-detects project language when language is omitted', async () => {
      jest.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('python');

      const res = await registeredTools['list_genkit_docs']({});
      expect(docsUtils.detectProjectLanguage).toHaveBeenCalledWith(
        '/workspace'
      );
      expect(res.content[0].text).toContain(
        'Genkit Documentation Index (python):'
      );
      expect(res.content[0].text).toContain('FilePath: python/flows.md');
    });

    it('respects an explicit language override', async () => {
      jest.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      const res = await registeredTools['list_genkit_docs']({
        language: 'dart',
      });
      expect(res.content[0].text).toContain(
        'Genkit Documentation Index (dart):'
      );
      expect(res.content[0].text).toContain('FilePath: dart/flows.md');
    });
  });

  describe('search_genkit_docs', () => {
    it('auto-detects project language when language is omitted', async () => {
      jest.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      const res = await registeredTools['search_genkit_docs']({
        query: 'Flows',
      });
      expect(docsUtils.detectProjectLanguage).toHaveBeenCalledWith(
        '/workspace'
      );
      expect(res.content[0].text).toContain('**FilePath**: go/flows.md');
    });

    it('respects an explicit language override', async () => {
      jest.spyOn(docsUtils, 'detectProjectLanguage').mockReturnValue('go');

      const res = await registeredTools['search_genkit_docs']({
        query: 'Flows',
        language: 'python',
      });
      expect(res.content[0].text).toContain('**FilePath**: python/flows.md');
    });
  });

  describe('get_usage_guide', () => {
    it.each([
      {
        detectedLang: 'js' as const,
        expectedHeading: '# Genkit Node.js API Rules',
      },
      {
        detectedLang: 'go' as const,
        expectedHeading: '# Genkit Go API Rules',
      },
    ])(
      'auto-detects $detectedLang usage guide when language is omitted',
      async ({ detectedLang, expectedHeading }) => {
        jest
          .spyOn(docsUtils, 'detectProjectLanguage')
          .mockReturnValue(detectedLang);

        const res = await registeredTools['get_usage_guide']({});
        expect(docsUtils.detectProjectLanguage).toHaveBeenCalledWith(
          '/workspace'
        );
        expect(res.content[0].text).toContain(expectedHeading);
      }
    );

    it.each(['python' as const, 'dart' as const])(
      'directs caller to docs tools when language is omitted in a %s workspace',
      async (detectedLang) => {
        jest
          .spyOn(docsUtils, 'detectProjectLanguage')
          .mockReturnValue(detectedLang);

        const res = await registeredTools['get_usage_guide']({});
        expect(res.content[0].text).toContain(
          `No standalone usage guide is bundled for '${detectedLang}' projects.`
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
        jest
          .spyOn(docsUtils, 'detectProjectLanguage')
          .mockReturnValue('python');

        const res = await registeredTools['get_usage_guide']({ language });
        expect(res.content[0].text).toContain(expectedHeading);
      }
    );
  });
});
