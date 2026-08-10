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
import { MessageData, Part, ToolAction, z } from 'genkit';
import { tool } from 'genkit/beta';
import * as mime from 'mime-types';
import * as path from 'path';

export function defineReadFileTool(
  messageQueue: MessageData[],
  resolvePath: (requestedPath: string) => string,
  prefix?: string
): ToolAction {
  const toolName = `${prefix || ''}read_file`;
  return tool(
    {
      name: toolName,
      description: 'Reads the contents of a file',
      inputSchema: z.object({
        filePath: z.string().describe('File path relative to root.'),
      }),
      outputSchema: z.string(),
    },
    async (input) => {
      const targetFile = resolvePath(input.filePath);
      const ext = path.extname(targetFile).toLowerCase();
      const mimeType = mime.lookup(ext);
      const isImage = mimeType != false && mimeType?.startsWith('image/');

      const parts: Part[] = [];

      if (isImage && mimeType) {
        const buffer = await fs.readFile(targetFile);
        const base64 = buffer.toString('base64');

        parts.push({
          text: `\n\nread_file result ${mimeType} ${input.filePath}`,
        });
        parts.push({
          media: {
            url: `data:${mimeType};base64,${base64}`,
            contentType: mimeType,
          },
        });
      } else {
        const content = await fs.readFile(targetFile, 'utf8');
        parts.push({
          text: `<read_file path="${input.filePath}">\n${content}\n</read_file>`,
          metadata: {
            filePath: input.filePath,
            filesystemMiddlewareTool: toolName,
          },
        });
      }

      if (
        messageQueue.length > 0 &&
        messageQueue[messageQueue.length - 1].role === 'user'
      ) {
        messageQueue[messageQueue.length - 1].content.push(...parts);
      } else {
        messageQueue.push({
          role: 'user',
          content: parts,
          metadata: {
            filesystemMiddlewareTool: toolName,
          },
        });
      }

      return `File ${input.filePath} read successfully, see contents below`;
    }
  );
}
