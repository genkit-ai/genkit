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
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp';
import { ContentBlock } from '@modelcontextprotocol/sdk/types';
import z from 'zod';
import { McpRunToolEvent } from './analytics.js';

import { GENKIT_CONTEXT as GoContext } from '../commands/init-ai-tools/context/go.js';
import { GENKIT_CONTEXT as JsContext } from '../commands/init-ai-tools/context/nodejs.js';

export async function defineUsageGuideTool(server: McpServer) {
  server.registerTool(
    'get_usage_guide',
    {
      title: 'Genkit Instructions',
      description:
        'Use this tool to look up the official Genkit usage guide, including project setup instructions and API best practices. ALWAYS call this before implementing Genkit features.',
      inputSchema: {
        language: z
          .enum(['js', 'go'])
          .describe('which language this usage guide is for; type: string')
          .default('js')
          .optional(),
      },
    },
    async ({ language }) => {
      await record(new McpRunToolEvent('get_usage_guide'));

      const content = [] as ContentBlock[];
      if (!language) {
        language = 'js';
      }
      let text = JsContext;
      if (language === 'go') {
        text = GoContext;
      }
      content.push({
        type: 'text',
        text,
      });

      return { content };
    }
  );
}
