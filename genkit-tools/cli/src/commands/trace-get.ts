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
import {
  findProjectRoot,
  forceStderr,
  logger,
} from '@genkit-ai/tools-common/utils';
import { yellow } from 'colorette';
import { Command, Option } from 'commander';
import { runWithManager } from '../utils/manager-utils';
import { cleanTraceJson, formatTraceTree } from '../utils/trace-formatter';

/**
 * Options for the `trace:get` CLI command.
 */
export interface TraceGetOptions {
  /** Output format: 'tree' (default execution tree) or 'json' (parsed nested telemetry JSON). */
  format: 'tree' | 'json';
  /** If true, preserves raw base64 data instead of sanitizing them with placeholders. */
  keepBase64?: boolean;
}

/** Command to get a trace. */
export const traceGet = new Command('trace:get')
  .description('get a trace by id')
  .argument('<traceId>', 'id of the trace to get')
  .addOption(
    new Option('-f, --format <format>', 'output format')
      .choices(['tree', 'json'])
      .default('tree')
  )
  .option('--keep-base64', 'do not strip base64 data URLs in output', false)
  .action(async (traceId: string, options: TraceGetOptions) => {
    // Redirect logging to stdout for clean JSON
    forceStderr();
    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      try {
        const response = await manager.getTrace({ traceId });
        if (!response) {
          logger.error(`Trace with ID '${traceId}' not found.`);
          return;
        }

        const keepBase64 = !!options.keepBase64;
        const processedTrace = cleanTraceJson(response, keepBase64);

        if (options.format === 'json') {
          console.log(JSON.stringify(processedTrace, undefined, 2));
        } else {
          console.log(
            yellow(
              'Hint: pass `--format json` flag to get trace data in JSON format\n'
            )
          );
          console.log(formatTraceTree(processedTrace));
        }
      } catch (e) {
        logger.error(`Error retrieving trace: ${e}`);
      }
    };

    await runWithManager(projectRoot, runAction);
  });
