/**
 * Copyright 2024 Google LLC
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

import { logger } from '@genkit-ai/tools-common/utils';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio';
import { defineDocsTool } from '../mcp/docs';
import { defineFlowTools } from './flows';
import { defineInitPrompt } from './prompts/init';
import { defineRuntimeTools } from './runtime';
import { defineTraceTools } from './trace';
import { defineUsageGuideTool } from './usage';
import { McpRuntimeManager, McpToolOptions } from './utils';

export async function startMcpServer(params: {
  projectRoot: string;
  explicitProjectRoot: boolean;
  timeout?: number;
  /** `--experimental-auth`. */
  auth?: boolean;
}) {
  const { projectRoot, explicitProjectRoot, timeout, auth } = params;
  logger.info(`Starting MCP server in: ${projectRoot}`);

  const server = new McpServer(
    {
      name: 'Genkit MCP',
      version: '0.0.2',
      title: 'Genkit MCP Server',
      description:
        'Provides Genkit documentation, usage guides, runtime process management, flow discovery/execution, and execution trace inspection.',
    },
    {
      instructions:
        'Use this server to look up Genkit documentation and API guides (`get_usage_guide`, `search_genkit_docs`, `list_genkit_docs`, `read_genkit_docs`), manage the local Genkit runtime (`start_runtime`, `restart_runtime`, `kill_runtime`), discover and run Genkit flows (`list_flows`, `run_flow`), and inspect execution traces (`get_trace`). Always check `get_usage_guide` or search Genkit docs before implementing Genkit features.',
    }
  );

  await defineDocsTool(server);
  await defineUsageGuideTool(server);
  defineInitPrompt(server);

  const manager = new McpRuntimeManager(auth);
  const options: McpToolOptions = {
    projectRoot,
    explicitProjectRoot,
    timeout,
    manager,
  };

  defineFlowTools(server, options);
  defineTraceTools(server, options);
  defineRuntimeTools(server, options);

  return new Promise(async (resolve) => {
    const transport = new StdioServerTransport();
    const cleanup = async () => {
      try {
        await manager.kill();
      } catch (e) {
        // ignore
      }
      resolve(undefined);
      process.exit(0);
    };
    transport.onclose = async () => {
      try {
        await manager.kill();
      } catch (e) {
        // ignore
      }
      resolve(undefined);
      process.exit(0);
    };
    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    await server.connect(transport);
    logger.info('Genkit MCP Server running on stdio');
  });
}
