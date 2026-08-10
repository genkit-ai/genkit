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
import { loadDocs, searchDocs } from '../utils/docs';
import { McpRunToolEvent } from './analytics.js';

export async function defineDocsTool(server: McpServer) {
  const documents = await loadDocs();

  server.registerTool(
    'list_genkit_docs',
    {
      title: 'List Genkit Docs',
      description:
        'Use this to see a list of available Genkit documentation files. Returns `filePaths` that can be passed to `read_genkit_docs`.',
      inputSchema: {
        language: z
          .enum(['js', 'go', 'python'])
          .describe(
            'Which language to list docs for (default js); type: string.'
          )
          .default('js'),
      },
    },
    async ({ language }) => {
      await record(new McpRunToolEvent('list_genkit_docs'));
      const lang = language || 'js';

      const fileList = Object.keys(documents)
        .filter((file) => file.startsWith(lang))
        .sort();

      const output =
        `Genkit Documentation Index (${lang}):\n\n` +
        fileList
          .map((file) => {
            const doc = documents[file];
            let summary = ` - FilePath: ${file}\n   Title: ${doc.title}\n`;
            if (doc.description) {
              summary += `   Description: ${doc.description}\n`;
            }
            if (doc.headers) {
              summary += `   Headers:\n     ${doc.headers.split('\n').join('\n     ')}\n`;
            }
            return summary;
          })
          .join('\n') +
        `\n\nUse 'search_genkit_docs' to find specific topics. Copy the 'FilePath' values to 'read_genkit_docs' to read content.`;

      return {
        content: [{ type: 'text', text: output }],
      };
    }
  );

  server.registerTool(
    'search_genkit_docs',
    {
      title: 'Search Genkit Docs',
      description:
        'Use this to search the Genkit documentation using keywords. Returns ranked results with `filePaths` for `read_genkit_docs`. Warning: Generic terms (e.g. "the", "and") may return false positives; use specific technical terms (e.g. "rag", "firebase", "context").',
      inputSchema: {
        query: z
          .string()
          .describe('Keywords to search for in documentation; type: string.'),
        language: z
          .enum(['js', 'go', 'python'])
          .describe(
            'Which language to search docs for (default js); type: string.'
          )
          .default('js'),
      },
    },
    async ({ query, language }) => {
      await record(new McpRunToolEvent('search_genkit_docs'));
      const lang = language || 'js';

      const results = searchDocs(documents, query, lang).slice(0, 10); // Top 10

      if (results.length === 0) {
        return {
          content: [
            {
              type: 'text',
              text: `No results found for "${query}" in ${lang} docs. Try broader keywords or use 'list_genkit_docs' to see all files.`,
            },
          ],
        };
      }

      const output =
        `Found ${results.length} matching documents for "${query}":\n\n` +
        results
          .map((r) => {
            let summary = `### ${r.doc.title}\n**FilePath**: ${r.file}\n`;
            if (r.doc.description)
              summary += `**Description**: ${r.doc.description}\n`;
            return summary;
          })
          .join('\n') +
        `\n\nCopy the 'FilePath' values to 'read_genkit_docs' to read content.`;

      return {
        content: [{ type: 'text', text: output }],
      };
    }
  );

  server.registerTool(
    'read_genkit_docs',
    {
      title: 'Read Genkit Docs',
      description:
        'Use this to read the full content of specific Genkit documentation files. You must provide `filePaths` from the list/search tools.',
      inputSchema: {
        filePaths: z
          .array(z.string())
          .describe(
            'The `filePaths` of the docs to read. Obtain these exactly from `list_genkit_docs` or `search_genkit_docs` (e.g. "js/overview.md"); type: string[].'
          ),
      },
    },
    async ({ filePaths }) => {
      await record(new McpRunToolEvent('read_genkit_docs'));

      const content = [] as ContentBlock[];

      // filePaths is required by Zod schema, but check length just in case
      if (!filePaths || !filePaths.length) {
        return {
          content: [
            {
              type: 'text',
              text: 'No filePaths provided. Please provide an array of file paths.',
            },
          ],
          isError: true,
        };
      }

      for (const file of filePaths) {
        if (documents[file]) {
          content.push({ type: 'text', text: documents[file].text });
        } else {
          content.push({
            type: 'text',
            text: `Document not found: ${file}. Please check the path using list_genkit_docs.`,
          });
        }
      }

      return { content };
    }
  );
}
