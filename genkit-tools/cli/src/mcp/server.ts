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
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
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
}) {
  const { projectRoot, explicitProjectRoot, timeout } = params;
  logger.info(`Starting MCP server in: ${projectRoot}`);

  const server = new McpServer({
    name: 'Genkit MCP',
    version: '0.0.2',
  });

  await defineDocsTool(server);
  await defineUsageGuideTool(server);
  defineInitPrompt(server);

  const manager = new McpRuntimeManager();
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
