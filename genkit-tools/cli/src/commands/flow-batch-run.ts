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
import { readFile, writeFile } from 'fs/promises';
import { runWithManager } from '../utils/manager-utils';

interface FlowBatchRunOptions {
  wait?: boolean;
  output?: string;
  label?: string;
  context?: string;
}

/** Command to run flows with batch input. */
export const flowBatchRun = new Command('flow:batchRun')
  .description(
    'batch run a flow using provided set of data from a file as input'
  )
  .argument('<flowName>', 'name of the flow to run')
  .argument('<inputFileName>', 'JSON batch data to use to run the flow')
  .option('-w, --wait', 'Wait for the flow to complete', false)
  .option('-c, --context <JSON>', 'JSON object passed to context', '')
  .option('--output <filename>', 'name of the output file to store the output')
  .option('--label [label]', 'label flow run in this batch')
  .action(
    async (
      flowName: string,
      fileName: string,
      options: FlowBatchRunOptions
    ) => {
      const dashDashIndex = process.argv.indexOf('--');
      let runtimeCommand: string[] | undefined;
      let actualFileName: string | undefined = fileName;

      // Commander removes the '--' separator from flowBatchRun.args.
      // We find '--' in process.argv to determine which arguments belong to the runtime command.
      if (dashDashIndex !== -1) {
        const numArgsAfterDashDash = process.argv.length - dashDashIndex - 1;
        runtimeCommand = flowBatchRun.args.slice(-numArgsAfterDashDash);
        const commandArgs = flowBatchRun.args.slice(
          0,
          flowBatchRun.args.length - numArgsAfterDashDash
        );
        actualFileName = commandArgs[1];
      }

      const projectRoot = await findProjectRoot();

      const runAction = async (manager: BaseRuntimeManager) => {
        if (!actualFileName) {
          throw new Error('Missing required argument <inputFileName>');
        }
        const inputData = JSON.parse(
          await readFile(actualFileName, 'utf8')
        ) as any[];
        let input = inputData;
        if (inputData.length === 0) {
          throw new Error('batch input data must be a non-empty array');
        }
        if (Object.hasOwn(inputData[0], 'input')) {
          // If object has "input" field, use that instead.
          input = inputData.map((d) => d.input);
        }

        const outputValues = [] as { input: any; output: any }[];
        for (const data of input) {
          logger.debug(`Running '/flow/${flowName}'...`);
          const response = await manager.runAction({
            key: `/flow/${flowName}`,
            input: data,
            context: options.context ? JSON.parse(options.context) : undefined,
            telemetryLabels: options.label
              ? { batchRun: options.label }
              : undefined,
          });

          logger.info(clc.green('Result:'));
          const resultOutput =
            typeof response.result === 'string'
              ? response.result
              : JSON.stringify(response.result, undefined, '  ');
          logger.info(resultOutput);
          if (response.telemetry?.traceId) {
            logger.info(
              `${clc.cyan('Trace ID:')} ${response.telemetry.traceId}`
            );
          }

          outputValues.push({
            input: data,
            output: response.result,
          });
        }

        if (options.output) {
          await writeFile(
            options.output,
            JSON.stringify(outputValues, undefined, ' ')
          );
        }
      };

      await runWithManager(projectRoot, runAction, { runtimeCommand });
    }
  );
