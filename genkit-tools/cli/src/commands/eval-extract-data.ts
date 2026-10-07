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

import type {
  EvalInput,
  EvalInputDataset,
  TraceData,
} from '@genkit-ai/tools-common';
import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import {
  findProjectRoot,
  generateTestCaseId,
  getEvalExtractors,
  logger,
} from '@genkit-ai/tools-common/utils';
import * as clc from 'colorette';
import { Command, Option } from 'commander';
import { writeFile } from 'fs/promises';
import {
  EXPERIMENTAL_AUTH_OPTION_HELP,
  runWithManager,
} from '../utils/manager-utils';
import { parsePositiveInt } from '../utils/option-parsers';

interface EvalDatasetOptions {
  /** --experimental-auth. */
  experimentalAuth?: boolean;
  output?: string;
  maxRows: number;
  label?: string;
}

/** Command to extract evaluation data. */
export const evalExtractData = new Command('eval:extract-data')
  .alias('eval:extractData')
  .description('extract evaluation data for a given flow from the trace store')
  .argument('<flowName>', 'name of the flow to extract data for')
  .option(
    '--output <filename>',
    'name of the output file to store the extracted data'
  )
  .option(
    '--max-rows <maxRows>',
    'maximum number of rows',
    parsePositiveInt,
    100
  )
  .addOption(
    new Option('--maxRows <maxRows>', 'maximum number of rows')
      .argParser(parsePositiveInt)
      .hideHelp()
  )
  .option('--label <label>', 'only extract traces with this batchRun label')
  .option('--experimental-auth', EXPERIMENTAL_AUTH_OPTION_HELP)
  .action(async (flowName: string, options: EvalDatasetOptions) => {
    const { maxRows } = options;
    const dashDashIndex = process.argv.indexOf('--');
    let runtimeCommand: string[] | undefined;
    if (dashDashIndex !== -1) {
      runtimeCommand = process.argv.slice(dashDashIndex + 1);
    }

    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      const extractors = await getEvalExtractors(`/flow/${flowName}`);

      logger.debug(`Extracting trace data '/flow/${flowName}'...`);
      let dataset: EvalInputDataset = [];
      let continuationToken = undefined;
      while (dataset.length < maxRows) {
        const response = await manager.listTraces({
          limit: maxRows,
          continuationToken,
        });
        continuationToken = response.continuationToken;
        const traces = response.traces;
        const batch: EvalInput[] = traces
          .map((t) => {
            const rootSpan = Object.values(t.spans).find(
              (s) =>
                s.attributes['genkit:metadata:subtype'] === 'flow' &&
                (!options.label ||
                  s.attributes['batchRun'] === options.label) &&
                s.attributes['genkit:name'] === flowName
            );
            if (!rootSpan) {
              return undefined;
            }
            return t;
          })
          .filter((t): t is TraceData => !!t)
          .map((trace) => {
            return {
              testCaseId: generateTestCaseId(),
              input: extractors.input(trace),
              output: extractors.output(trace),
              context: toArray(extractors.context(trace)),
              // The trace (t) does not contain the traceId, so we have to pull it out of the
              // spans, de- dupe, and turn it back into an array.
              traceIds: Array.from(
                new Set(Object.values(trace.spans).map((span) => span.traceId))
              ),
            } as EvalInput;
          })
          .filter((result): result is EvalInput => !!result);
        batch.forEach((d) => dataset.push(d));
        if (dataset.length > maxRows) {
          dataset = dataset.splice(0, maxRows);
          break;
        }
        if (!continuationToken) {
          break;
        }
      }

      if (options.output) {
        logger.debug(`Writing data to '${options.output}'...`);
        await writeFile(
          options.output,
          JSON.stringify(dataset, undefined, '  ')
        );
      } else {
        logger.debug(`Results will not be written to file.`);
        logger.info(clc.green('Results:'));
        logger.info(JSON.stringify(dataset, undefined, '  '));
      }
    };

    await runWithManager(projectRoot, runAction, {
      runtimeCommand,
      auth: options.experimentalAuth,
    });
  });

function toArray(input: any) {
  return Array.isArray(input) ? input : [input];
}
