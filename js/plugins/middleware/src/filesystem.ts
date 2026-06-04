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
  generateMiddleware,
  MessageData,
  z,
  type GenerateMiddleware,
} from 'genkit';
import {
  requireCurrentBackend,
  type AgentBackend,
} from '@genkit-ai/ai/backends';
import * as path from 'path';
import { defineListFileTool } from './filesystem/list_files.js';
import { defineReadFileTool } from './filesystem/read_file.js';
import { defineSearchAndReplaceTool } from './filesystem/search_and_replace.js';
import { defineWriteFileTool } from './filesystem/write_file.js';

export const FilesystemOptionsSchema = z.object({
  rootDirectory: z
    .string()
    .describe(
      'The root directory to which all filesystem operations are restricted.'
    ),
  allowWriteAccess: z
    .boolean()
    .optional()
    .describe('If true, allows write access to the filesystem.'),
  toolNamePrefix: z
    .string()
    .optional()
    .describe('Prefix to add to the name of the injected tools.'),
});

export type FilesystemOptions = z.infer<typeof FilesystemOptionsSchema>;

/**
 * Creates a middleware that grants the LLM access to the filesystem.
 * Injects `list_files`, `read_file`, `write_file`, and `search_and_replace` tools restricted to the provided `rootDirectory`.
 */
export const filesystem: GenerateMiddleware<typeof FilesystemOptionsSchema> =
  generateMiddleware(
    {
      name: 'filesystem',
      description:
        'Injects tools for reading, writing, and searching files in a directory.',
      configSchema: FilesystemOptionsSchema,
    },
    ({ config, ai }) => {
      if (!config?.rootDirectory) {
        throw new Error(
          'filesystem middleware requires a rootDirectory option'
        );
      }
      const getBackend = (): AgentBackend => requireCurrentBackend(ai.registry);
      const rootDirectory =
        config.rootDirectory.replace(/^[/\\]+/, '') || '.';
      const resolveBackendPath = (requestedPath: string) => {
        const normalizedPath = requestedPath.replace(/^[/\\]+/, '');
        const backendPath = path.normalize(
          path.join(rootDirectory, normalizedPath)
        );
        const relativeToRoot = path.relative(rootDirectory, backendPath);
        if (
          relativeToRoot === '..' ||
          relativeToRoot.startsWith(`..${path.sep}`) ||
          path.isAbsolute(relativeToRoot)
        ) {
          throw new Error('Access denied: Path is outside of root directory.');
        }
        return backendPath;
      };

      // Middleware is instantiated once per top generate call, so it's ok (by design) to keep state here.
      const messageQueue: MessageData[] = [];

      const listFilesTool = defineListFileTool(
        getBackend,
        resolveBackendPath,
        config.toolNamePrefix
      );
      const readFileTool = defineReadFileTool(
        messageQueue,
        getBackend,
        resolveBackendPath,
        config.toolNamePrefix
      );

      const filesystemTools = [listFilesTool, readFileTool];
      if (config.allowWriteAccess) {
        const writeFileTool = defineWriteFileTool(
          getBackend,
          resolveBackendPath,
          config.toolNamePrefix
        );
        const searchAndReplaceTool = defineSearchAndReplaceTool(
          getBackend,
          resolveBackendPath,
          config.toolNamePrefix
        );
        filesystemTools.push(writeFileTool, searchAndReplaceTool);
      }
      const filesystemToolNames = filesystemTools.map((t) => t.__action.name);

      return {
        tools: filesystemTools,
        tool: async (req, ctx, next) => {
          try {
            return await next(req, ctx);
          } catch (e: any) {
            // Don't catch ToolInterruptError — let it propagate for interrupt handling
            // (e.g. from toolApproval middleware further down the chain).
            if (e.name === 'ToolInterruptError') throw e;

            if (filesystemToolNames.includes(req.toolRequest.name)) {
              // Return a tool response with the error text so the LLM can
              // see what went wrong and retry.  This is provider-agnostic
              // (Anthropic requires a proper tool response after every tool
              // call — injecting a user-role message breaks its protocol).
              return {
                toolResponse: {
                  name: req.toolRequest.name,
                  ref: req.toolRequest.ref,
                  output: `Tool '${req.toolRequest.name}' failed: ${
                    e.message || String(e)
                  }`,
                },
              };
            }
            throw e;
          }
        },
        generate: async (envelope, ctx, next) => {
          const { request } = envelope;
          let { messageIndex } = envelope;
          if (messageQueue.length > 0) {
            if (ctx.onChunk) {
              for (const msg of messageQueue) {
                ctx.onChunk({
                  role: msg.role,
                  index: messageIndex++,
                  content: msg.content,
                });
              }
            }
            request.messages.push(...messageQueue);
            messageQueue.length = 0;
          }
          return await next({ ...envelope, request, messageIndex }, ctx);
        },
      };
    }
  );
