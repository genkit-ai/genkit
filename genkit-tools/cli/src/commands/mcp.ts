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

import {
  debugToFile,
  findProjectRoot,
  forceStderr,
} from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import { startMcpServer } from '../mcp/server';

interface McpOptions {
  projectRoot?: string;
  debug?: boolean | string;
  explicitProjectRoot?: boolean;
  timeout?: string;
}

/** Command to run MCP server. */
export const mcp = new Command('mcp')
  .option('--project-root [projectRoot]', 'Project root')
  .option('-d, --debug [path]', 'debug to file')
  .option(
    '--timeout [timeout]',
    'Timeout for runtime to start (ms). Default 30000.'
  )
  .option(
    '--explicitProjectRoot',
    'Whether runtime dependent tools need projectRoot specified. Needed for use with Google Antigravity',
    false
  )
  .description('run MCP stdio server (EXPERIMENTAL, subject to change)')
  .action(async (options: McpOptions) => {
    forceStderr();
    if (options.debug) {
      debugToFile(
        typeof options.debug === 'string' ? options.debug : undefined
      );
    }
    await startMcpServer({
      projectRoot: options.projectRoot ?? (await findProjectRoot()),
      explicitProjectRoot: options.explicitProjectRoot ?? false,
      timeout: options.timeout ? parseInt(options.timeout, 10) : undefined,
    });
  });
