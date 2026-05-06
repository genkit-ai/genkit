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

import type { Action, EvalInputDataset } from '@genkit-ai/tools-common';
import {
  getAllEvaluatorActions,
  getExporterForString,
  getMatchingEvaluatorActions,
  runEvaluation,
  type EvalExporter,
} from '@genkit-ai/tools-common/eval';
import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import {
  confirmLlmUse,
  findProjectRoot,
  loadEvaluationDatasetFile,
  logger,
} from '@genkit-ai/tools-common/utils';
import * as clc from 'colorette';
import { Command } from 'commander';
import { runWithManager } from '../utils/manager-utils';

interface EvalRunCliOptions {
  output?: string;
  evaluators?: string;
  force?: boolean;
  batchSize?: number;
  outputFormat: string;
}

/** Command to run evaluation on a dataset. */
export const evalRun = new Command('eval:run')
  .description('evaluate provided dataset against configured evaluators')
  .argument(
    '<dataset>',
    'Dataset to evaluate on (currently only supports JSON)'
  )
  .option(
    '--output <filename>',
    'name of the output file to write evaluation results. Defaults to json output.'
  )
  .option(
    '--output-format <format>',
    'The output file format (csv, json)',
    'json'
  )
  .option(
    '--evaluators <evaluators>',
    'comma separated list of evaluators to use (by default uses all)'
  )
  .option(
    '--batchSize <batchSize>',
    'batch size to use for parallel evals (default to 1, no parallelization)',
    Number.parseInt
  )
  .option('--force', 'Automatically accept all interactive prompts')
  .action(async (dataset: string, options: EvalRunCliOptions) => {
    const dashDashIndex = process.argv.indexOf('--');
    let runtimeCommand: string[] | undefined;
    let actualDataset: string | undefined = dataset;

    // Commander removes the '--' separator from evalRun.args.
    // We find '--' in process.argv to determine which arguments belong to the runtime command.
    if (dashDashIndex !== -1) {
      const numArgsAfterDashDash = process.argv.length - dashDashIndex - 1;
      runtimeCommand = evalRun.args.slice(-numArgsAfterDashDash);
      const commandArgs = evalRun.args.slice(
        0,
        evalRun.args.length - numArgsAfterDashDash
      );
      actualDataset = commandArgs[0];
    }

    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      if (!actualDataset) {
        throw new Error('Missing required argument <dataset>');
      }

      let evaluatorActions: Action[];
      if (!options.evaluators) {
        evaluatorActions = await getAllEvaluatorActions(manager);
      } else {
        const evalActionKeys = options.evaluators
          .split(',')
          .map((k) => `/evaluator/${k}`);
        evaluatorActions = await getMatchingEvaluatorActions(
          manager,
          evalActionKeys
        );
      }
      if (!evaluatorActions.length) {
        throw new Error(
          options.evaluators
            ? `No matching evaluators found for '${options.evaluators}'`
            : `No evaluators found in your app`
        );
      }
      logger.info(
        `Using evaluators: ${evaluatorActions.map((action) => action.name).join(',')}`
      );

      if (!options.force) {
        const confirmed = await confirmLlmUse(evaluatorActions);
        if (!confirmed) {
          throw new Error('User declined using billed evaluators.');
        }
      }

      const evalDataset: EvalInputDataset =
        await loadEvaluationDatasetFile(actualDataset);
      const evalRun = await runEvaluation({
        manager,
        evaluatorActions,
        evalDataset,
        batchSize: options.batchSize,
      });

      if (options.output) {
        const exportFn: EvalExporter = getExporterForString(
          options.outputFormat
        );
        await exportFn(evalRun, options.output);
      }

      const toolsInfo = manager.getMostRecentDevUI();
      if (toolsInfo) {
        logger.info(
          clc.green(
            `\nView the evaluation results at: ${toolsInfo.url}/evaluate/${evalRun.key.evalRunId}`
          )
        );
      } else {
        logger.info(`${clc.cyan('Evaluation ID:')} ${evalRun.key.evalRunId}`);
      }
    };

    await runWithManager(projectRoot, runAction, { runtimeCommand });
  });
