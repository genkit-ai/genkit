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
import * as docsUtils from '../../src/utils/docs';
import { SAMPLE_DOCS } from '../utils/docs';

describe('MCP docs tools', () => {
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
});
