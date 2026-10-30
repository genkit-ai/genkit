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
import { GENKIT_CONTEXT as GoContext } from '../commands/init-ai-tools/context/go';
import { GENKIT_CONTEXT as JsContext } from '../commands/init-ai-tools/context/nodejs';
import { McpRunToolEvent } from './analytics';
import { detectProjectLanguage } from './utils';

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
          .enum(['js', 'go'])
          .describe(
            'which language this usage guide is for (js, go; defaults to auto-detected project language); type: string'
          )
          .optional(),
      },
    },
    async ({ language }) => {
      await record(new McpRunToolEvent('get_usage_guide'));

      const content = [] as ContentBlock[];
      const targetLanguage = language ?? detectProjectLanguage(projectRoot);
      if (targetLanguage !== 'js' && targetLanguage !== 'go') {
        content.push({
          type: 'text',
          text: `No standalone usage guide is bundled for '${targetLanguage}' projects. Use 'list_genkit_docs' and 'search_genkit_docs' (which default to '${targetLanguage}' in this workspace) followed by 'read_genkit_docs' to look up Genkit ${targetLanguage} documentation and setup instructions.`,
        });
        return { content };
      }

      content.push({
        type: 'text',
        text: targetLanguage === 'go' ? GoContext : JsContext,
      });

      return { content };
    }
  );
}
