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

import { writeFile } from 'fs/promises';
import { json2csv } from 'json-2-csv';
import type { EvalRun } from '../types/eval.js';
import { logger } from '../utils/logger.js';

const SUPPORTED_FORMATS: Record<string, EvalExporter> = {
  csv: toCsv,
  json: toJson,
};

/**
 * Interface for exporting evaluation to file.
 */
export type EvalExporter = (
  evalRun: EvalRun,
  filePath: string
) => Promise<void>;

/**
 * Export an evalRun to csv.
 */
export async function toCsv(evalRun: EvalRun, filePath: string) {
  // manually unpack the metrics into columns because we don't always want to pull nested fields into columns.
  const unpackedCases = unpackMetricsToColumns(evalRun);
  const csvRecords = json2csv(unpackedCases, {
    emptyFieldValue: '',
    expandNestedObjects: false,
  });
  logger.info(`Writing csv results to '${filePath}'...`);
  await writeFile(filePath, csvRecords);
}

/**
 * Export an evalRun to json.
 */
export async function toJson(evalRun: EvalRun, filePath: string) {
  logger.info(`Writing json results to '${filePath}'...`);
  await writeFile(filePath, JSON.stringify(evalRun.results, undefined, '  '));
}

function unpackMetricsToColumns(evalRun: EvalRun): Record<string, any>[] {
  return evalRun.results.map((result) => {
    const record: Record<string, any> = {
      ...result,
    };
    // remove metrics so that we can unnest them
    delete record['metrics'];
    result.metrics?.forEach((metric) => {
      record[`${metric.evaluator}_score`] = metric.score;
      record[`${metric.evaluator}_rationale`] = metric.rationale;
      record[`${metric.evaluator}_error`] = metric.error;
      record[`${metric.evaluator}_traceId`] = metric.traceId;
      record[`${metric.evaluator}_spanId`] = metric.spanId;
    });
    return record;
  });
}

export function getExporterForString(outputFormat: string): EvalExporter {
  if (!(outputFormat in SUPPORTED_FORMATS)) {
    logger.info(
      `Encountered unrecognized output format ${outputFormat}. Defaulting to json.`
    );
    return toJson;
  }
  return SUPPORTED_FORMATS[outputFormat];
}
