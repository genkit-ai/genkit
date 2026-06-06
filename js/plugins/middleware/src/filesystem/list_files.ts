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

import type { AgentBackend } from '@genkit-ai/ai/backends';
import { ToolAction, z } from 'genkit';
import { tool } from 'genkit/beta';

export function defineListFileTool(
  getBackend: () => AgentBackend,
  resolveBackendPath: (requestedPath: string) => string,
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
      const result = await getBackend().ls(resolveBackendPath(input.dirPath), {
        recursive: input.recursive,
      });
      if (result.error) {
        throw new Error(result.error);
      }
      return result.files ?? [];
    }
  );
}
