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
import * as path from 'path';

export function defineListFileTool(
  resolvePath: (requestedPath: string) => string,
  prefix?: string
): ToolAction {
  return tool(
    {
      name: `${prefix || ''}list_files`,
      description:
        'Lists files and directories in a given path. Returns a list of objects with path and type.',
      inputSchema: z.object({
        dirPath: z
          .string()
          .describe('Directory path relative to root.')
          .default(''),
        recursive: z
          .boolean()
          .describe('Whether to list files recursively.')
          .default(false),
      }),
      outputSchema: z.array(
        z.object({ path: z.string(), isDirectory: z.boolean() })
      ),
    },
    async (input) => {
      const targetDir = resolvePath(input.dirPath);

      async function list(dir: string, recursive: boolean, base: string = '') {
        const results: { path: string; isDirectory: boolean }[] = [];
        const entries = await fs.readdir(dir, { withFileTypes: true });
        for (const entry of entries) {
          const relativePath = path.join(base, entry.name);
          results.push({
            path: relativePath,
            isDirectory: entry.isDirectory(),
          });
          if (entry.isDirectory() && recursive) {
            const subResults = await list(
              path.join(dir, entry.name),
              true,
              relativePath
            );
            results.push(...subResults);
          }
        }
        return results;
      }
      return await list(targetDir, input.recursive);
    }
  );
}
