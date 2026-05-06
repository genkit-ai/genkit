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
  DatasetSchema,
  type Action,
  type Dataset,
  type DatasetMetadata,
} from '@genkit-ai/tools-common';
import {
  getAllEvaluatorActions,
  getDatasetStore,
  getExporterForString,
  getMatchingEvaluatorActions,
  runEvaluation,
  runInference,
  type EvalExporter,
} from '@genkit-ai/tools-common/eval';
import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import {
  confirmLlmUse,
  findProjectRoot,
  hasAction,
  loadInferenceDatasetFile,
  logger,
} from '@genkit-ai/tools-common/utils';
import * as clc from 'colorette';
import { Command } from 'commander';
import { runWithManager } from '../utils/manager-utils';

interface EvalFlowRunCliOptions {
  input?: string;
  output?: string;
  context?: string;
  evaluators?: string;
  force?: boolean;
  batchSize?: number;
  outputFormat: string;
}

const EVAL_FLOW_SCHEMA = 'Array<{input: any; reference?: any;}>';
enum SourceType {
  DATA = 'data',
  FILE = 'file',
  DATASET = 'dataset',
}

/** Command to run a flow and evaluate the output */
export const evalFlow = new Command('eval:flow')
  .description(
    'evaluate a flow against configured evaluators using provided data as input'
  )
  .argument('<flowName>', 'Name of the flow to run')
  .argument('[data]', 'JSON data to use to start the flow')
  .option(
    '--input <input>',
    'Input dataset ID or JSON file to be used for evaluation'
  )
  .option('-c, --context <JSON>', 'JSON object passed to context', '')
  .option(
    '-o, --output <filename>',
    'Name of the output file to write evaluation results. Defaults to json output.'
  )
  // TODO: Figure out why passing a new Option with choices doesn't work
  .option(
    '--output-format <format>',
    'The output file format (csv, json)',
    'json'
  )
  .option(
    '-e, --evaluators <evaluators>',
    'comma separated list of evaluators to use (by default uses all)'
  )
  .option(
    '--batchSize <batchSize>',
    'batch size to use for parallel evals (default to 1, no parallelization)',
    Number.parseInt
  )
  .option('-f, --force', 'Automatically accept all interactive prompts')
  .action(
    async (flowName: string, data: string, options: EvalFlowRunCliOptions) => {
      const dashDashIndex = process.argv.indexOf('--');
      let runtimeCommand: string[] | undefined;
      let actualData: string | undefined = data;

      // Commander removes the '--' separator from evalFlow.args.
      // We find '--' in process.argv to determine which arguments belong to the runtime command
      // and which belong to the command itself (like the optional [data] argument).
      if (dashDashIndex !== -1) {
        const numArgsAfterDashDash = process.argv.length - dashDashIndex - 1;
        runtimeCommand = evalFlow.args.slice(-numArgsAfterDashDash);
        const commandArgs = evalFlow.args.slice(
          0,
          evalFlow.args.length - numArgsAfterDashDash
        );
        if (commandArgs.length > 1) {
          actualData = commandArgs[1];
        } else {
          actualData = undefined;
        }
      }

      const projectRoot = await findProjectRoot();

      const runAction = async (manager: BaseRuntimeManager) => {
        const actionRef = `/flow/${flowName}`;
        if (!actualData && !options.input) {
          throw new Error(
            'No input data passed. Specify input data using [data] argument or --input <filename> option'
          );
        }

        const hasTargetAction = await hasAction({ manager, actionRef });
        if (!hasTargetAction) {
          throw new Error(`Cannot find action ${actionRef}.`);
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
        logger.debug(
          `Using evaluators: ${evaluatorActions.map((action) => action.name).join(',')}`
        );

        if (!options.force) {
          const confirmed = await confirmLlmUse(evaluatorActions);
          if (!confirmed) {
            throw new Error('User declined using billed evaluators.');
          }
        }

        const sourceType = getSourceType(actualData, options.input);
        let targetDatasetMetadata;
        if (sourceType === SourceType.DATASET) {
          const datasetStore = await getDatasetStore();
          const datasetMetadatas = await datasetStore.listDatasets();
          targetDatasetMetadata = datasetMetadatas.find(
            (d: DatasetMetadata) => d.datasetId === options.input
          );
        }

        const inferenceDataset = await readInputs(
          sourceType,
          actualData,
          options.input
        );
        const evalDataset = await runInference({
          manager,
          actionRef,
          inferenceDataset,
          context: options.context,
        });

        const evalRun = await runEvaluation({
          manager,
          evaluatorActions,
          evalDataset,
          batchSize: options.batchSize,
          augments: {
            actionRef: `/flow/${flowName}`,
            datasetId:
              sourceType === SourceType.DATASET ? options.input : undefined,
            datasetVersion: targetDatasetMetadata?.version,
          },
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
    }
  );

/**
 * Reads EvalFlowInput dataset from data string or input identified.
 * Only one of these parameters is expected to be provided.
 **/
async function readInputs(
  sourceType: SourceType,
  dataField?: string,
  input?: string
): Promise<Dataset> {
  let parsedData;
  switch (sourceType) {
    case SourceType.DATA:
      parsedData = JSON.parse(dataField!);
      break;
    case SourceType.FILE:
      try {
        return await loadInferenceDatasetFile(input!);
      } catch (e) {
        throw new Error(`Error parsing the input from file. Error: ${e}`);
      }
    case SourceType.DATASET:
      const datasetStore = await getDatasetStore();
      const data = await datasetStore.getDataset(input!);
      parsedData = data;
      break;
  }

  try {
    return DatasetSchema.parse(parsedData);
  } catch (e) {
    throw new Error(
      `Error parsing the input. Please provide an array of inputs for the flow or a ${EVAL_FLOW_SCHEMA} object. Error: ${e}`
    );
  }
}

function getSourceType(data?: string, input?: string): SourceType {
  if (input) {
    if (data) {
      logger.warn('Both [data] and input provided, ignoring [data]...');
    }
    return input.endsWith('.json') || input.endsWith('.jsonl')
      ? SourceType.FILE
      : SourceType.DATASET;
  } else if (data) {
    return SourceType.DATA;
  }
  throw new Error('Must provide either data or input');
}
