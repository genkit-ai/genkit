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

import * as fs from 'fs/promises';
import { ToolAction, z } from 'genkit';
import { tool } from 'genkit/beta';

export function defineSearchAndReplaceTool(
  resolvePath: (requestedPath: string) => string,
  prefix?: string
): ToolAction {
  return tool(
    {
      name: `${prefix || ''}search_and_replace`,
      description:
        'Replaces text in a file using search and replace blocks. Use this tool to edit existing files.',
      inputSchema: z.object({
        filePath: z.string().describe('File path relative to root.'),
        edits: z.array(
          z
            .string()
            .describe(
              'A search and replace block string in the format:\n' +
                '<<<<<<< SEARCH\n[search content]\n=======\n[replace content]\n>>>>>>> REPLACE'
            )
        ),
      }),
      outputSchema: z.string(),
    },
    async (input) => {
      const targetFile = resolvePath(input.filePath);
      let content = await fs.readFile(targetFile, 'utf8');

      for (const editBlock of input.edits) {
        const startMarker = '<<<<<<< SEARCH\n';
        const endMarker = '\n>>>>>>> REPLACE';
        const separator = '\n=======\n';

        if (
          !editBlock.startsWith(startMarker) ||
          !editBlock.endsWith(endMarker)
        ) {
          throw new Error(
            'Invalid edit block format. Block must start with "<<<<<<< SEARCH\\n" and end with "\\n>>>>>>> REPLACE"'
          );
        }

        const innerContent = editBlock.substring(
          startMarker.length,
          editBlock.length - endMarker.length
        );

        // Find all possible separator positions
        const separatorIndices: number[] = [];
        let pos = innerContent.indexOf(separator);
        while (pos !== -1) {
          separatorIndices.push(pos);
          pos = innerContent.indexOf(separator, pos + 1);
        }

        if (separatorIndices.length === 0) {
          throw new Error(
            'Invalid edit block format. Missing separator "\\n=======\\n"'
          );
        }

        // Try to find a split that matches the content
        let match: { search: string; replace: string } | null = null;
        let matchCount = 0;

        for (const splitIndex of separatorIndices) {
          const search = innerContent.substring(0, splitIndex);
          const replace = innerContent.substring(splitIndex + separator.length);

          if (content.includes(search)) {
            // If we already have a match, only replace it if this one is longer (more specific)
            if (!match || search.length > match.search.length) {
              match = { search, replace };
            }
            matchCount++;
          }
        }

        if (matchCount === 0) {
          throw new Error(
            `Search content not found in file ${input.filePath}. ` +
              `Make sure the search block matches the file content exactly, ` +
              `including whitespace and indentation.`
          );
        }

        if (matchCount > 1) {
          // If multiple splits match, prefer the one with the longest search string.
          // This assumes that a longer match is more specific and likely what the user intended
          // if they included markers in their search block.
        }

        // Apply the replacement (first occurrence only)
        if (match) {
          content = content.replace(match.search, () => match!.replace);
        }
      }

      await fs.writeFile(targetFile, content, 'utf8');
      return `Successfully applied ${input.edits.length} edit(s) to ${input.filePath}.`;
    }
  );
}
