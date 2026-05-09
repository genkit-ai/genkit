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

import { GenkitToolsError } from '@genkit-ai/tools-common/manager';
import { record } from '@genkit-ai/tools-common/utils';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp';
import z from 'zod';
import { McpRunToolEvent } from './analytics.js';
import {
  McpToolOptions,
  getCommonSchema,
  resolveProjectRoot,
} from './utils.js';

export function defineFlowTools(server: McpServer, options: McpToolOptions) {
  server.registerTool(
    'list_flows',
    {
      title: 'List Genkit Flows',
      description:
        'Use this to discover available Genkit flows or inspect the input schema of Genkit flows to know how to successfully call them.',
      inputSchema: getCommonSchema(options.explicitProjectRoot),
    },
    async (opts) => {
      await record(new McpRunToolEvent('list_flows'));
      const rootOrError = resolveProjectRoot(
        options.explicitProjectRoot,
        opts,
        options.projectRoot
      );
      if (typeof rootOrError !== 'string') return rootOrError;

      const runtimeManager = await options.manager.getManager(rootOrError);
      const actions = await runtimeManager.listActions();

      let flows = '';
      for (const key of Object.keys(actions)) {
        if (key.startsWith('/flow/')) {
          flows += ' - Flow name: ' + key.substring('/flow/'.length) + '\n';
          if (actions[key].description) {
            flows += '   Description: ' + actions[key].description + '\n';
          }
          flows +=
            '   Input schema: ' +
            JSON.stringify(actions[key].inputSchema, undefined, 2) +
            '\n\n';
        }
      }

      return { content: [{ type: 'text', text: flows }] };
    }
  );

  server.registerTool(
    'run_flow',
    {
      title: 'Run Flow',
      description: 'Runs the flow with the provided input',
      inputSchema: getCommonSchema(options.explicitProjectRoot, {
        flowName: z.string().describe('name of the flow; type: string'),
        input: z
          .string()
          .describe(
            'Flow input as JSON object encoded as string (it will be passed through `JSON.parse`). Must conform to the schema; type: string.'
          )
          .optional(),
      }),
    },
    async (opts) => {
      await record(new McpRunToolEvent('run_flow'));
      const rootOrError = resolveProjectRoot(
        options.explicitProjectRoot,
        opts,
        options.projectRoot
      );
      if (typeof rootOrError !== 'string') return rootOrError;
      const { flowName, input } = opts;

      try {
        const runtimeManager = await options.manager.getManager(rootOrError);
        const response = await runtimeManager.runAction({
          key: `/flow/${flowName}`,
          input: input !== undefined ? JSON.parse(input) : undefined,
        });
        return {
          content: [
            { type: 'text', text: JSON.stringify(response, undefined, 2) },
          ],
        };
      } catch (e) {
        const errStr =
          e instanceof GenkitToolsError ? e.formatError() : JSON.stringify(e);
        return {
          isError: true,
          content: [
            {
              type: 'text',
              text: `Error: ${errStr}`,
            },
          ],
        };
      }
    }
  );
}
