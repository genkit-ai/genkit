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

export function defineWriteFileTool(
  resolvePath: (requestedPath: string) => string,
  prefix?: string
): ToolAction {
  return tool(
    {
      name: `${prefix || ''}write_file`,
      description:
        'Writes content to a file, overwriting it if it exists. Use this tool to create new files.',
      inputSchema: z.object({
        filePath: z.string().describe('File path relative to root.'),
        content: z.string().describe('Content to write to the file.'),
      }),
      outputSchema: z.string(),
    },
    async (input) => {
      const targetFile = resolvePath(input.filePath);
      await fs.mkdir(path.dirname(targetFile), { recursive: true });
      await fs.writeFile(targetFile, input.content, 'utf8');
      return `File ${input.filePath} written successfully.`;
    }
  );
}
