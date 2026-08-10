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

import { startServer } from '@genkit-ai/tools-common/server';
import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import getPort, { makeRange } from 'get-port';
import open from 'open';
import { getDevEnvVars, startDevProcessManager } from '../utils/manager-utils';

interface FlutterRunOptions {
  port?: string;
  open?: boolean;
  corsOrigin?: string;
  noui?: boolean;
}

/** Command to run a Flutter app in dev mode and/or the Dev UI. */
export const startFlutter = new Command('start:flutter')
  .description('runs a flutter app in Genkit dev mode')
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
  .action(async (options: FlutterRunOptions) => {
    const projectRoot = await findProjectRoot();
    if (projectRoot.includes('/.Trash/')) {
      logger.warn(
        'It appears your current project root is in the trash folder. ' +
          'Please make sure that you current working directory is correct.'
      );
    }

    const { envVars, reflectionV2Port, telemetryServerUrl } =
      await getDevEnvVars(projectRoot, {
        ...options,
        experimentalReflectionV2: true,
      });

    const dartDefines = Object.entries(envVars).map(
      ([key, value]) => `--dart-define=${key}=${value}`
    );

    const flutterArgs = ['run', ...dartDefines, ...startFlutter.args];

    const { manager, processPromise } = await startDevProcessManager(
      projectRoot,
      'flutter',
      flutterArgs,
      {
        ...options,
        experimentalReflectionV2: true,
        envVars,
        reflectionV2Port,
        telemetryServerUrl,
      }
    );

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
