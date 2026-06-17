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

import { confirm } from '@inquirer/prompts';
import { randomUUID } from 'crypto';
import { createReadStream } from 'fs';
import { readFile } from 'fs/promises';
import { createInterface } from 'readline';
import type { BaseRuntimeManager } from '../manager';
import {
  findToolsConfig,
  isEvalField,
  type EvalField,
  type EvaluationExtractor,
  type InputStepSelector,
  type OutputStepSelector,
  type StepSelector,
} from '../plugin';
import {
  DatasetSchema,
  EvalInputDatasetSchema,
  EvaluationDatasetSchema,
  EvaluationSampleSchema,
  GenerateRequestSchema,
  InferenceDatasetSchema,
  InferenceSampleSchema,
  type Action,
  type Dataset,
  type DocumentData,
  type EvalInputDataset,
  type EvaluationSample,
  type GenerateRequest,
  type InferenceSample,
  type MessageData,
  type NestedSpanData,
  type RetrieverResponse,
  type TraceData,
} from '../types';
import { logger } from './logger';
import { stackTraceSpans } from './trace';

export type EvalExtractorFn = (t: TraceData) => any;

export const EVALUATOR_ACTION_PREFIX = '/evaluator';

// Update js/ai/src/evaluators.ts if you change this value
export const EVALUATOR_METADATA_KEY_DISPLAY_NAME = 'evaluatorDisplayName';
export const EVALUATOR_METADATA_KEY_DEFINITION = 'evaluatorDefinition';
export const EVALUATOR_METADATA_KEY_IS_BILLED = 'evaluatorIsBilled';

export function evaluatorName(action: Action) {
  return `${EVALUATOR_ACTION_PREFIX}/${action.name}`;
}

export function isEvaluator(key: string) {
  return key.startsWith(EVALUATOR_ACTION_PREFIX);
}

export async function confirmLlmUse(
  evaluatorActions: Action[]
): Promise<boolean> {
  const isBilled = evaluatorActions.some(
    (action) =>
      action.metadata && action.metadata[EVALUATOR_METADATA_KEY_IS_BILLED]
  );

  if (!isBilled) {
    return true;
  }

  const confirmed = await confirm({
    message:
      'For each example, the evaluation makes calls to APIs that may result in being charged. Do you wish to proceed?',
    default: false,
  });

  return confirmed;
}

function getRootSpan(trace: TraceData): NestedSpanData | undefined {
  return stackTraceSpans(trace);
}

function safeParse(value?: string) {
  if (value) {
    try {
      return JSON.parse(value);
    } catch (e) {
      return '';
    }
  }
  return '';
}

const DEFAULT_INPUT_EXTRACTOR: EvalExtractorFn = (trace: TraceData) => {
  const rootSpan = getRootSpan(trace);
  return safeParse(rootSpan?.attributes['genkit:input'] as string);
};
const DEFAULT_OUTPUT_EXTRACTOR: EvalExtractorFn = (trace: TraceData) => {
  const rootSpan = getRootSpan(trace);
  return safeParse(rootSpan?.attributes['genkit:output'] as string);
};
const DEFAULT_CONTEXT_EXTRACTOR: EvalExtractorFn = (trace: TraceData) => {
  return Object.values(trace.spans)
    .filter((s) => s.attributes['genkit:metadata:subtype'] === 'retriever')
    .flatMap((s) => {
      const output: RetrieverResponse = safeParse(
        s.attributes['genkit:output'] as string
      );
      if (!output) {
        return [];
      }
      return output.documents.flatMap((d: DocumentData) =>
        d.content.map((c) => c.text).filter((text): text is string => !!text)
      );
    });
};

const DEFAULT_FLOW_EXTRACTORS: Record<EvalField, EvalExtractorFn> = {
  input: DEFAULT_INPUT_EXTRACTOR,
  output: DEFAULT_OUTPUT_EXTRACTOR,
  context: DEFAULT_CONTEXT_EXTRACTOR,
};

const DEFAULT_MODEL_EXTRACTORS: Record<EvalField, EvalExtractorFn> = {
  input: DEFAULT_INPUT_EXTRACTOR,
  output: DEFAULT_OUTPUT_EXTRACTOR,
  context: () => [],
};

function getStepAttribute(
  trace: TraceData,
  stepName: string,
  attributeName?: string
) {
  // Default to output
  const attr = attributeName ?? 'genkit:output';
  const values = Object.values(trace.spans)
    .filter((step) => step.displayName === stepName)
    .flatMap((step) => {
      return safeParse(step.attributes[attr] as string);
    });
  if (values.length === 0) {
    return '';
  }
  if (values.length === 1) {
    return values[0];
  }
  // Return array if multiple steps have the same name
  return values;
}

function getExtractorFromStepName(stepName: string): EvalExtractorFn {
  return (trace: TraceData) => {
    return getStepAttribute(trace, stepName);
  };
}

function getExtractorFromStepSelector(
  stepSelector: StepSelector
): EvalExtractorFn {
  return (trace: TraceData) => {
    let stepName: string | undefined = undefined;
    let selectedAttribute = 'genkit:output'; // default

    if (Object.hasOwn(stepSelector, 'inputOf')) {
      stepName = (stepSelector as InputStepSelector).inputOf;
      selectedAttribute = 'genkit:input';
    } else {
      stepName = (stepSelector as OutputStepSelector).outputOf;
      selectedAttribute = 'genkit:output';
    }
    if (!stepName) {
      return '';
    } else {
      return getStepAttribute(trace, stepName, selectedAttribute);
    }
  };
}

function getExtractorMap(extractor: EvaluationExtractor) {
  const extractorMap: Record<EvalField, EvalExtractorFn> = {} as Record<
    EvalField,
    EvalExtractorFn
  >;
  for (const [key, value] of Object.entries(extractor)) {
    if (isEvalField(key)) {
      if (typeof value === 'string') {
        extractorMap[key] = getExtractorFromStepName(value);
      } else if (typeof value === 'object') {
        extractorMap[key] = getExtractorFromStepSelector(value);
      } else if (typeof value === 'function') {
        extractorMap[key] = value;
      }
    }
  }
  return extractorMap;
}

export async function getEvalExtractors(
  actionRef: string
): Promise<Record<string, EvalExtractorFn>> {
  if (actionRef.startsWith('/model')) {
    // Always use defaults for model extraction.
    logger.debug(
      'getEvalExtractors - modelRef provided, using default extractors'
    );
    return Promise.resolve(DEFAULT_MODEL_EXTRACTORS);
  }
  const config = await findToolsConfig();
  const extractors = config?.evaluators
    ?.filter((e) => e.actionRef === actionRef)
    .map((e) => e.extractors);
  if (!extractors) {
    return Promise.resolve(DEFAULT_FLOW_EXTRACTORS);
  }
  let composedExtractors = DEFAULT_FLOW_EXTRACTORS;
  for (const extractor of extractors) {
    const extractorFunction = getExtractorMap(extractor);
    composedExtractors = { ...composedExtractors, ...extractorFunction };
  }
  return Promise.resolve(composedExtractors);
}

/**Global function to generate testCaseId */
export function generateTestCaseId() {
  return randomUUID();
}

/** Load a {@link Dataset} file. Supports JSON / JSONL */
export async function loadInferenceDatasetFile(
  fileName: string
): Promise<Dataset> {
  const isJsonl = fileName.endsWith('.jsonl');

  if (isJsonl) {
    return await readJsonlForInference(fileName);
  } else {
    const parsedData = JSON.parse(await readFile(fileName, 'utf8'));
    let dataset = InferenceDatasetSchema.parse(parsedData);
    dataset = dataset.map((sample: InferenceSample) => ({
      ...sample,
      testCaseId: sample.testCaseId ?? generateTestCaseId(),
    }));
    return DatasetSchema.parse(dataset);
  }
}

/** Load a {@link EvalInputDataset} file. Supports JSON / JSONL */
export async function loadEvaluationDatasetFile(
  fileName: string
): Promise<EvalInputDataset> {
  const isJsonl = fileName.endsWith('.jsonl');

  if (isJsonl) {
    return await readJsonlForEvaluation(fileName);
  } else {
    const parsedData = JSON.parse(await readFile(fileName, 'utf8'));
    let evaluationInput = EvaluationDatasetSchema.parse(parsedData);
    evaluationInput = evaluationInput.map((evalSample: EvaluationSample) => ({
      ...evalSample,
      testCaseId: evalSample.testCaseId ?? generateTestCaseId(),
      traceIds: evalSample.traceIds ?? [],
    }));
    return EvalInputDatasetSchema.parse(evaluationInput);
  }
}

async function readJsonlForInference(fileName: string): Promise<Dataset> {
  const lines = await readLines(fileName);
  const samples: Dataset = [];
  for (const line of lines) {
    const parsedSample = InferenceSampleSchema.parse(JSON.parse(line));
    samples.push({
      ...parsedSample,
      testCaseId: parsedSample.testCaseId ?? generateTestCaseId(),
    });
  }
  return samples;
}

async function readJsonlForEvaluation(
  fileName: string
): Promise<EvalInputDataset> {
  const lines = await readLines(fileName);
  const inputs: EvalInputDataset = [];
  for (const line of lines) {
    const parsedSample = EvaluationSampleSchema.parse(JSON.parse(line));
    inputs.push({
      ...parsedSample,
      testCaseId: parsedSample.testCaseId ?? generateTestCaseId(),
      traceIds: parsedSample.traceIds ?? [],
    });
  }
  return inputs;
}

async function readLines(fileName: string): Promise<string[]> {
  const lines: string[] = [];
  const fileStream = createReadStream(fileName);
  const rl = createInterface({
    input: fileStream,
    crlfDelay: Number.POSITIVE_INFINITY,
  });

  for await (const line of rl) {
    lines.push(line);
  }
  return lines;
}

export async function hasAction(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
}): Promise<boolean> {
  const { manager, actionRef } = { ...params };
  const actionsRecord = await manager.listActions();

  return actionsRecord.hasOwnProperty(actionRef);
}

export async function getAction(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
}): Promise<Action | undefined> {
  const { manager, actionRef } = { ...params };
  const allActions = await manager.listActions();

  return Object.values(allActions).find((action) => action.key === actionRef);
}

/** Helper function that maps string data to GenerateRequest */
export function getModelInput(data: any, modelConfig: any): GenerateRequest {
  let message: MessageData;
  if (typeof data === 'string') {
    message = {
      role: 'user',
      content: [
        {
          text: data,
        },
      ],
    } as MessageData;
    return {
      messages: message ? [message] : [],
      config: modelConfig,
    };
  } else {
    const maybeRequest = GenerateRequestSchema.safeParse(data);
    if (maybeRequest.success) {
      return { ...maybeRequest.data, config: modelConfig };
    } else {
      throw new Error(
        `Unable to parse model input as MessageSchema. Details: ${maybeRequest.error}`
      );
    }
  }
}

/**
 * Helper method to groupBy an array of objects, replaces lodash equivalent.
 */
export function groupBy(
  arr: any[],
  criteria: ((i: any) => any) | string
): Record<string, any[]> {
  return arr.reduce((obj, item) => {
    const key =
      typeof criteria === 'function' ? criteria(item) : item[criteria];

    if (!obj.hasOwnProperty(key)) {
      obj[key] = [];
    }
    obj[key].push(item);

    return obj;
  }, {});
}

/**
 * Helper method to countBy an array of objects, replaces lodash equivalent.
 */
export function countBy(
  arr: any[],
  criteria: ((i: any) => any) | string
): Record<string, number> {
  return arr.reduce((acc, item) => {
    const key =
      typeof criteria === 'function' ? criteria(item) : item[criteria];
    acc[key] = (acc[key] || 0) + 1;

    return acc;
  }, {});
}

/**
 * Helper method to meanBy an array of objects, replaces lodash equivalent.
 */
export function meanBy(
  arr: any[],
  criteria: ((i: any) => any) | string
): number | undefined {
  if (!arr || arr.length === 0) {
    return undefined;
  }

  let sum = 0;
  for (const item of arr) {
    const value =
      typeof criteria === 'function' ? criteria(item) : item[criteria];
    sum += value;
  }

  return sum / arr.length;
}
