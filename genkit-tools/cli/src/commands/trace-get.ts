/**
 * Copyright 2026 Google LLC
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

import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import { runWithManager } from '../utils/manager-utils';

/** Command to get a trace. */
export const traceGet = new Command('trace:get')
  .description('get a trace by id')
  .argument('<traceId>', 'id of the trace to get')
  .action(async (traceId: string) => {
    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      try {
        const response = await manager.getTrace({ traceId });
        if (!response) {
          logger.error(`Trace with ID '${traceId}' not found.`);
          return;
        }
        console.log(JSON.stringify(response, undefined, 2));
      } catch (e) {
        logger.error(`Error retrieving trace: ${e}`);
      }
    };

    await runWithManager(projectRoot, runAction);
  });
