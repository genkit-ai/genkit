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
import { z } from 'zod';
import { waitForRuntime } from '../utils/manager-utils.js';
import { McpRunToolEvent } from './analytics.js';
import {
  McpToolOptions,
  formatMcpError,
  getCommonSchema,
  resolveProjectRoot,
} from './utils.js';

export function defineRuntimeTools(server: McpServer, options: McpToolOptions) {
  server.registerTool(
    'start_runtime',
    {
      title: 'Starts a Genkit runtime process',
      description: `Use this to start a Genkit runtime process (typically the entry point to the user's app). A running runtime is required before calling \`list_flows\` or \`run_flow\`, and will also be picked up by \`genkit start\` to power the Developer UI.

      Examples:
        {command: "npx", args: ["tsx", "src/index.ts"]}
        {command: "uv", args: ["run", "python", "main.py"]}
        {command: "go", args: ["run", "."]}
        {command: "dart", args: ["run", "bin/main.dart"]}
        {command: "npm", args: ["run", "dev"]}
        {command: "npm", args: ["run", "dev"], projectRoot: "path/to/project"}`,
      inputSchema: getCommonSchema(options.explicitProjectRoot, {
        command: z
          .string()
          .describe(
            'The executable to run (pass arguments separately in `args`). Eg: `"npx"`, `"uv"`, `"go"`, `"dart"`, `"npm"`; type: string.'
          ),
        args: z
          .array(z.string())
          .describe(
            'The array of string args for the command to run. Eg: `["tsx", "src/index.ts"]`; type: string[].'
          ),
      }),
    },
    async (opts) => {
      await record(new McpRunToolEvent('start_runtime'));
      const rootOrError = resolveProjectRoot(
        options.explicitProjectRoot,
        opts,
        options.projectRoot
      );
      if (typeof rootOrError !== 'string') return rootOrError;

      try {
        await options.manager.getManagerWithDevProcess({
          projectRoot: rootOrError,
          command: opts.command,
          args: opts.args,
          explicitProjectRoot: options.explicitProjectRoot,
          timeout: options.timeout,
        });
      } catch (err) {
        return {
          isError: true,
          content: [
            {
              type: 'text',
              text: `Error: ${formatMcpError(err)}`,
            },
          ],
        };
      }

      return {
        content: [
          {
            type: 'text',
            text: `Done.`,
          },
        ],
      };
    }
  );

  const registerControlTool = (
    name: string,
    title: string,
    action: 'kill' | 'restart',
    description: string
  ) => {
    server.registerTool(
      name,
      {
        title,
        description,
        inputSchema: getCommonSchema(options.explicitProjectRoot),
      },
      async (opts) => {
        await record(new McpRunToolEvent(name));
        const rootOrError = resolveProjectRoot(
          options.explicitProjectRoot,
          opts,
          options.projectRoot
        );
        if (typeof rootOrError !== 'string') return rootOrError;

        const runtimeManager = await options.manager.getManager(rootOrError);
        if (!runtimeManager.processManager) {
          return {
            isError: true,
            content: [
              { type: 'text', text: `No runtime process currently running.` },
            ],
          };
        }

        try {
          if (action === 'restart') {
            const existingRuntimeIds = new Set(
              runtimeManager.listRuntimes().map((r) => r.id)
            );
            await runtimeManager.processManager.restart();
            const processPromise =
              runtimeManager.processManager.waitForProcess?.();
            if (options.timeout !== 0 && processPromise) {
              await waitForRuntime(
                runtimeManager,
                processPromise,
                options.timeout,
                existingRuntimeIds
              );
            }
          } else {
            await runtimeManager.processManager.kill();
          }
        } catch (err) {
          return {
            isError: true,
            content: [
              {
                type: 'text',
                text: `Error: ${formatMcpError(err)}`,
              },
            ],
          };
        }
        return {
          content: [{ type: 'text', text: `Done.` }],
        };
      }
    );
  };

  registerControlTool(
    'kill_runtime',
    'Kills any existing Genkit runtime process',
    'kill',
    'Use this to kill an existing runtime that was started using the `start_runtime` tool.'
  );
  registerControlTool(
    'restart_runtime',
    'Restarts any existing Genkit runtime process',
    'restart',
    'Use this to restart an existing runtime that was started using the `start_runtime` tool (e.g., after editing source files when the runtime command does not automatically hot-reload).'
  );
}
