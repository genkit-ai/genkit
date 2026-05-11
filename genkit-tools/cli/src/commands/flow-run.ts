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

import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
import * as clc from 'colorette';
import { Command } from 'commander';
import { writeFile } from 'fs/promises';
import { runWithManager } from '../utils/manager-utils';

interface FlowRunOptions {
  wait?: boolean;
  output?: string;
  stream?: boolean;
  context?: string;
}

/** Command to run a flow. */
export const flowRun = new Command('flow:run')
  .description('run a flow using provided data as input')
  .argument('<flowName>', 'name of the flow to run')
  .argument('[data]', 'JSON data to use to start the flow')
  .option('-w, --wait', 'Wait for the flow to complete', false)
  .option('-s, --stream', 'Stream output', false)
  .option('-c, --context <JSON>', 'JSON object passed to context', '')
  .option(
    '--output <filename>',
    'name of the output file to store the extracted data'
  )
  .action(async (flowName: string, data: string, options: FlowRunOptions) => {
    const dashDashIndex = process.argv.indexOf('--');
    let runtimeCommand: string[] | undefined;
    let actualData: string | undefined = data;

    // Commander removes the '--' separator from flowRun.args.
    // We find '--' in process.argv to determine which arguments belong to the runtime command
    // and which belong to the command itself (like the optional [data] argument).
    if (dashDashIndex !== -1) {
      const numArgsAfterDashDash = process.argv.length - dashDashIndex - 1;
      runtimeCommand = flowRun.args.slice(-numArgsAfterDashDash);
      const commandArgs = flowRun.args.slice(
        0,
        flowRun.args.length - numArgsAfterDashDash
      );
      if (commandArgs.length > 1) {
        actualData = commandArgs[1];
      } else {
        actualData = undefined;
      }
    }

    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      let traceId: string | undefined;
      const response = await manager.runAction(
        {
          key: `/flow/${flowName}`,
          input: actualData ? JSON.parse(actualData) : undefined,
          context: options.context ? JSON.parse(options.context) : undefined,
        },
        options.stream
          ? (chunk) => console.log(JSON.stringify(chunk, undefined, '  '))
          : undefined,
        (tid) => {
          traceId = tid;
        }
      );

      const result = response.result;

      logger.info(clc.green('Result:'));
      const resultOutput =
        typeof result === 'string'
          ? result
          : JSON.stringify(result, undefined, '  ');
      logger.info(resultOutput);
      if (traceId) {
        logger.info(`${clc.cyan('Trace ID:')} ${traceId}`);
      }

      if (options.output && result) {
        await writeFile(options.output, JSON.stringify(result, undefined, ' '));
      }
    };

    await runWithManager(projectRoot, runAction, { runtimeCommand });
  });
