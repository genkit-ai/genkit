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

import type { RuntimeManager } from '@genkit-ai/tools-common/manager';
import { startServer } from '@genkit-ai/tools-common/server';
import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import getPort, { makeRange } from 'get-port';
import open from 'open';
import { startDevProcessManager, startManager } from '../utils/manager-utils';

interface RunOptions {
  noui?: boolean;
  port?: string;
  open?: boolean;
  disableRealtimeTelemetry?: boolean;
  corsOrigin?: string;
}

/** Command to run code in dev mode and/or the Dev UI. */
export const start = new Command('start')
  .description('runs a command in Genkit dev mode')
  .option('-n, --noui', 'do not start the Dev UI', false)
  .option('-p, --port <port>', 'port for the Dev UI')
  .option('-o, --open', 'Open the browser on UI start up')
  .option(
    '--disable-realtime-telemetry',
    'Disable real-time telemetry streaming'
  )
  .option(
    '--cors-origin <origin>',
    'specify the allowed origin for CORS requests'
  )
  .action(async (options: RunOptions) => {
    const projectRoot = await findProjectRoot();
    if (projectRoot.includes('/.Trash/')) {
      logger.warn(
        'It appears your current project root is in the trash folder. ' +
          'Please make sure that you current working directory is correct.'
      );
    }
    // Always start the manager.
    let manager: RuntimeManager;
    let processPromise: Promise<void> | undefined;
    if (start.args.length > 0) {
      const result = await startDevProcessManager(
        projectRoot,
        start.args[0],
        start.args.slice(1),
        {
          disableRealtimeTelemetry: options.disableRealtimeTelemetry,
          corsOrigin: options.corsOrigin,
        }
      );
      manager = result.manager;
      processPromise = result.processPromise;
    } else {
      manager = await startManager({
        projectRoot,
        manageHealth: true,
        corsOrigin: options.corsOrigin,
      });
      processPromise = new Promise(() => {});
    }
    if (!options.noui) {
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
      startServer(manager, port);
      if (options.open) {
        open(`http://localhost:${port}`);
      }
    }
    await processPromise;
  });
