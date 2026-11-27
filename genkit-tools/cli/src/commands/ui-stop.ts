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
  findProjectRoot,
  findServersDir,
  isValidDevToolsInfo,
  logger,
  waitUntilUnresponsive,
  type DevToolsInfo,
} from '@genkit-ai/tools-common/utils';
import axios from 'axios';
import * as clc from 'colorette';
import { Command } from 'commander';
import fs from 'fs/promises';
import path from 'path';

/** Command to stop the Genkit Developer UI. */
export const uiStop = new Command('ui:stop')
  .description(
    'stops any running Genkit Developer UI in this directory (deprecated: use `genkit start` instead)'
  )
  .action(async () => {
    logger.warn('`genkit ui:stop` is deprecated. Use `genkit start` instead.');
    const serversDir = await findServersDir(await findProjectRoot());
    const toolsJsonPath = path.join(serversDir, 'tools.json');
    try {
      const toolsJsonContent = await fs.readFile(toolsJsonPath, 'utf-8');
      const serverInfo = JSON.parse(toolsJsonContent) as DevToolsInfo;
      if (isValidDevToolsInfo(serverInfo)) {
        // Check if server is healthy. If it's not, metadata file is stale.
        try {
          await axios.get(`${serverInfo.url}/api/__health`);
        } catch {
          await fs.unlink(toolsJsonPath);
          logger.info('Genkit Developer UI is not running in this directory.');
          return;
        }
        // Kill the server and wait until it no longer responds as healthy then clean up metadata file.
        try {
          logger.info('Stopping...');
          await axios.post(`${serverInfo.url}/api/__quitquitquit`);
          if (await waitUntilUnresponsive(serverInfo.url)) {
            await fs.unlink(toolsJsonPath);
            logger.info(clc.green('\n  Genkit Developer UI is stopped.'));
            logger.info('  To start the UI, run `genkit start`.\n');
          } else {
            logger.info('Failed to stop running UI before timing out.');
          }
        } catch {
          logger.info('Failed to stop Genkit Developer UI.');
        }
      } else {
        logger.debug('tools.json is malformed.');
        await fs.unlink(toolsJsonPath);
        logger.info('Genkit Developer UI is not running in this directory.');
      }
    } catch {
      logger.info('Genkit Developer UI is not running in this directory.');
    }
  });
