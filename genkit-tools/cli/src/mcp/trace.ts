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

export function defineTraceTools(server: McpServer, options: McpToolOptions) {
  server.registerTool(
    'get_trace',
    {
      title: 'Get Genkit Trace',
      description: 'Returns the trace details.',
      inputSchema: getCommonSchema(options.explicitProjectRoot, {
        traceId: z
          .string()
          .describe(
            'trace id (typically returned after running a flow or other actions); type: string'
          ),
      }),
    },
    async (opts) => {
      await record(new McpRunToolEvent('get_trace'));
      const rootOrError = resolveProjectRoot(
        options.explicitProjectRoot,
        opts,
        options.projectRoot
      );
      if (typeof rootOrError !== 'string') return rootOrError;
      const { traceId } = opts;

      try {
        const runtimeManager = await options.manager.getManager(rootOrError);
        const response = await runtimeManager.getTrace({ traceId });
        return {
          content: [
            // TODO: render the trace insetad of of dumping it as is.
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
