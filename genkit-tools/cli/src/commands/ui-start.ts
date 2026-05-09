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
import {
  findProjectRoot,
  findServersDir,
  isValidDevToolsInfo,
  logger,
  waitUntilHealthy,
  type DevToolsInfo,
} from '@genkit-ai/tools-common/utils';
import axios from 'axios';
import { spawn, type ChildProcess } from 'child_process';
import * as clc from 'colorette';
import { Command } from 'commander';
import fs from 'fs/promises';
import getPort, { makeRange } from 'get-port';
import open from 'open';
import path from 'path';
import { detectCLIRuntime } from '../utils/runtime-detector';
import {
  buildServerHarnessSpawnConfig,
  validateExecutablePath,
} from '../utils/spawn-config';

interface StartOptions {
  port?: string;
  open?: boolean;
}

/** Command to start the Genkit Developer UI. */
export const uiStart = new Command('ui:start')
  .description(
    'start the Developer UI which connects to runtimes in the same directory'
  )
  .option('-p, --port <number>', 'Port to serve on (defaults to 4000)')
  .option('-o, --open', 'Open the browser on UI start up')
  .action(async (options: StartOptions) => {
    let port: number;
    if (options.port) {
      port = Number(options.port);
      if (isNaN(port) || port < 0) {
        logger.error(`"${options.port}" is not a valid port number`);
        return;
      }
    } else {
      port = await getPort({ port: makeRange(4000, 4099) });
    }
    const serversDir = await findServersDir(await findProjectRoot());
    const toolsJsonPath = path.join(serversDir, 'tools.json');
    try {
      const toolsJsonContent = await fs.readFile(toolsJsonPath, 'utf-8');
      const serverInfo = JSON.parse(toolsJsonContent) as DevToolsInfo;
      if (isValidDevToolsInfo(serverInfo)) {
        try {
          await axios.get(`${serverInfo.url}/api/__health`);
          logger.info(
            clc.green(
              `\n  Genkit Developer UI is already running at: ${serverInfo.url}`
            )
          );
          logger.info(`  To stop the UI, run \`genkit ui:stop\`.\n`);
          return;
        } catch (error) {
          logger.debug(
            'Found UI server metadata but server is not healthy. Starting a new one...'
          );
        }
      }
    } catch (error) {
      logger.debug('No UI running. Starting a new one...');
    }
    logger.info('Starting...');
    try {
      await startAndWaitUntilHealthy(port, serversDir);
    } catch (error) {
      logger.error(`Failed to start Genkit Developer UI: ${error}`);
      return;
    }
    try {
      await fs.mkdir(serversDir, { recursive: true });
      await fs.writeFile(
        toolsJsonPath,
        JSON.stringify(
          {
            url: `http://localhost:${port}`,
            timestamp: new Date().toISOString(),
          },
          null,
          2
        )
      );
    } catch (error) {
      logger.error(
        `Failed to write UI server metadata. UI server will continue to run.`
      );
    }
    logger.info(
      `\n  ${clc.green(`Genkit Developer UI started at: http://localhost:${port}`)}`
    );
    logger.info(`  To stop the UI, run \`genkit ui:stop\`.\n`);
    try {
      await axios.get(`http://localhost:${port}/api/trpc/listActions`);
    } catch (error) {
      logger.info(
        'Set env variable `GENKIT_ENV` to `dev` and start your app code to interact with it in the UI.'
      );
    }
    if (options.open) {
      open(`http://localhost:${port}`);
    }
  });

/**
 * Starts the UI server in a child process and waits until it is healthy. Once it's healthy, the child process is detached.
 */
async function startAndWaitUntilHealthy(
  port: number,
  serversDir: string
): Promise<ChildProcess> {
  // Detect runtime environment
  const cliRuntime = detectCLIRuntime();
  logger.debug(
    `Detected CLI runtime: ${cliRuntime.type} at ${cliRuntime.execPath}`
  );
  if (cliRuntime.scriptPath) {
    logger.debug(`Script path: ${cliRuntime.scriptPath}`);
  }

  // Build spawn configuration
  const logPath = path.join(serversDir, 'devui.log');
  const spawnConfig = buildServerHarnessSpawnConfig(cliRuntime, port, logPath);

  // Validate executable path
  const isExecutable = await validateExecutablePath(spawnConfig.command);
  if (!isExecutable) {
    throw new GenkitToolsError(
      `Unable to execute command: ${spawnConfig.command}. ` +
        `The file does not exist or is not executable.`
    );
  }

  logger.debug(
    `Spawning: ${spawnConfig.command} ${spawnConfig.args.join(' ')}`
  );
  const child = spawn(
    spawnConfig.command,
    spawnConfig.args,
    spawnConfig.options
  );

  // Wait for the process to be ready
  return new Promise<ChildProcess>((resolve, reject) => {
    // Handle process events
    child.on('error', (error) => {
      logger.error(`Failed to start UI process: ${error.message}`);
      reject(
        new GenkitToolsError(`Failed to start UI process: ${error.message}`, {
          cause: error,
        })
      );
    });

    child.on('exit', (code) => {
      const msg = `UI process exited unexpectedly with code ${code}`;
      logger.error(msg);
      reject(new GenkitToolsError(msg));
    });

    // Wait for the UI to become healthy
    waitUntilHealthy(`http://localhost:${port}`, 10000 /* 10 seconds */)
      .then((isHealthy) => {
        if (isHealthy) {
          child.unref();
          resolve(child);
        } else {
          const msg =
            'Timed out while waiting for UI to become healthy. ' +
            'To view full logs, set DEBUG environment variable.';
          logger.error(msg);
          reject(new GenkitToolsError(msg));
        }
      })
      .catch((error) => {
        logger.error(`Health check failed: ${error.message}`);
        reject(
          new GenkitToolsError(`Health check failed: ${error.message}`, {
            cause: error,
          })
        );
      });
  });
}
