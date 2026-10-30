/**
 * Copyright 2025 Google LLC
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

import { record } from '@genkit-ai/tools-common/utils';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { ContentBlock } from '@modelcontextprotocol/sdk/types.js';
import z from 'zod';
import { GENKIT_CONTEXT as DartContext } from '../commands/init-ai-tools/context/dart';
import { GENKIT_CONTEXT as GoContext } from '../commands/init-ai-tools/context/go';
import { GENKIT_CONTEXT as JsContext } from '../commands/init-ai-tools/context/nodejs';
import { GENKIT_CONTEXT as PythonContext } from '../commands/init-ai-tools/context/python';
import { McpRunToolEvent } from './analytics';
import {
  SUPPORTED_DOC_LANGUAGES,
  SupportedDocLanguage,
  detectProjectLanguage,
} from './utils';

const USAGE_GUIDES: Record<SupportedDocLanguage, string> = {
  js: JsContext,
  go: GoContext,
  python: PythonContext,
  dart: DartContext,
};

export async function defineUsageGuideTool(
  server: McpServer,
  projectRoot?: string
) {
  server.registerTool(
    'get_usage_guide',
    {
      title: 'Genkit Instructions',
      description:
        'Use this tool to look up the official Genkit usage guide, including project setup instructions and API best practices. ALWAYS call this before implementing Genkit features.',
      inputSchema: {
        language: z
          .enum(SUPPORTED_DOC_LANGUAGES)
          .describe(
            'which language this usage guide is for (js, go, python, dart; defaults to auto-detected project language); type: string'
          )
          .optional(),
      },
    },
    async ({ language }) => {
      await record(new McpRunToolEvent('get_usage_guide'));

      const lang = language || detectProjectLanguage(projectRoot);
      const content = [
        {
          type: 'text',
          text: USAGE_GUIDES[lang] ?? JsContext,
        },
      ] as ContentBlock[];

      return { content };
    }
  );
}
