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

import { randomUUID } from 'crypto';
import { z } from 'zod';
import { getDatasetStore, getEvalStore } from '.';
import type { BaseRuntimeManager } from '../manager/manager';
import {
  GenerateActionOptions,
  GenerateResponseData,
  type Action,
  type CandidateData,
  type Dataset,
  type EvalInput,
  type EvalKeyAugments,
  type EvalRun,
  type EvalRunKey,
  type RunNewEvaluationRequest,
  type SpanData,
} from '../types';
import {
  evaluatorName,
  generateTestCaseId,
  getEvalExtractors,
  getModelInput,
  hasAction,
  isEvaluator,
  logger,
  stackTraceSpans,
} from '../utils';
import {
  enrichResultsWithScoring,
  extractMetricSummaries,
  extractMetricsMetadata,
} from './parser';

interface InferenceRunState {
  testCaseId: string;
  input: any;
  reference?: any;
  traceIds: string[];
  response?: any;
  evalError?: string;
}

interface FullInferenceSample {
  testCaseId: string;
  input: any;
  reference?: any;
}

const SUPPORTED_ACTION_TYPES = ['flow', 'model', 'executable-prompt'] as const;
type SupportedActionType = (typeof SUPPORTED_ACTION_TYPES)[number];
const GENERATE_ACTION_UTIL = '/util/generate';

/**
 * Starts a new evaluation run. Intended to be used via the reflection API.
 */
export async function runNewEvaluation(
  manager: BaseRuntimeManager,
  request: RunNewEvaluationRequest
): Promise<EvalRunKey> {
  const { dataSource, actionRef, evaluators } = request;
  const { datasetId, data } = dataSource;
  if (!datasetId && !data) {
    throw new Error(`Either 'data' or 'datasetId' must be provided`);
  }

  const hasTargetAction = await hasAction({ manager, actionRef });
  if (!hasTargetAction) {
    throw new Error(`Cannot find action ${actionRef}.`);
  }

  let inferenceDataset: Dataset;
  let metadata = {};

  if (datasetId) {
    const datasetStore = await getDatasetStore();
    logger.info(`Fetching dataset ${datasetId}...`);
    const dataset = await datasetStore.getDataset(datasetId);
    if (dataset.length === 0) {
      throw new Error(`Dataset ${datasetId} is empty`);
    }
    inferenceDataset = dataset as Dataset;

    const datasetMetadatas = await datasetStore.listDatasets();
    const targetDatasetMetadata = datasetMetadatas.find(
      (d) => d.datasetId === datasetId
    );
    const datasetVersion = targetDatasetMetadata?.version;
    metadata = { datasetId, datasetVersion };
  } else {
    const rawData = data!.map((sample) => ({
      ...sample,
      testCaseId: sample.testCaseId ?? generateTestCaseId(),
    }));
    inferenceDataset = rawData as Dataset;
  }

  logger.info('Running inference...');
  const evalDataset = await runInference({
    manager,
    actionRef,
    inferenceDataset,
    context: request.options?.context,
    actionConfig: request.options?.actionConfig,
  });
  const evaluatorActions = await getMatchingEvaluatorActions(
    manager,
    evaluators
  );

  const evalRun = await runEvaluation({
    manager,
    evaluatorActions,
    evalDataset,
    batchSize: request.options?.batchSize,
    augments: {
      ...metadata,
      actionRef,
      actionConfig: request.options?.actionConfig,
    },
  });
  return evalRun.key;
}

/** Handles the Inference part of Inference-Evaluation cycle */
export async function runInference(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
  inferenceDataset: Dataset;
  context?: string;
  actionConfig?: any;
}): Promise<EvalInput[]> {
  const { manager, actionRef, inferenceDataset, context, actionConfig } =
    params;
  if (!isSupportedActionRef(actionRef)) {
    throw new Error('Inference is only supported on flows and models');
  }

  const evalDataset: EvalInput[] = await bulkRunAction({
    manager,
    actionRef,
    inferenceDataset,
    context,
    actionConfig,
  });
  return evalDataset;
}

/** Handles the Evaluation part of Inference-Evaluation cycle */
export async function runEvaluation(params: {
  manager: BaseRuntimeManager;
  evaluatorActions: Action[];
  evalDataset: EvalInput[];
  augments?: EvalKeyAugments;
  batchSize?: number;
}): Promise<EvalRun> {
  const { manager, evaluatorActions, evalDataset, augments, batchSize } =
    params;
  if (evalDataset.length === 0) {
    throw new Error('Cannot run evaluation, no data provided');
  }
  const evalRunId = randomUUID();
  const scores: Record<string, any> = {};
  logger.info('Running evaluation...');

  const runtime = manager.getMostRecentRuntime();
  const isNodeRuntime = runtime?.genkitVersion?.startsWith('nodejs') ?? false;

  for (const action of evaluatorActions) {
    const name = evaluatorName(action);
    const response = await manager.runAction({
      key: name,
      input: {
        dataset: evalDataset.filter((row) => !row.error),
        evalRunId,
        batchSize: isNodeRuntime ? batchSize : undefined,
      },
    });
    scores[name] = response.result;
    logger.info(
      `Finished evaluator '${action.name}'. Trace ID: ${response.telemetry?.traceId}`
    );
  }

  const scoredResults = enrichResultsWithScoring(scores, evalDataset);
  const metadata = extractMetricsMetadata(evaluatorActions);
  const metricSummaries = extractMetricSummaries(scores);

  const evalRun = {
    key: {
      evalRunId,
      createdAt: new Date().toISOString(),
      metricSummaries,
      metricsMetadata: metadata,
      ...augments,
    },
    results: scoredResults,
  };

  logger.info('Finished evaluation, writing key...');
  const evalStore = await getEvalStore();
  await evalStore.save(evalRun);
  return evalRun;
}

export async function getAllEvaluatorActions(
  manager: BaseRuntimeManager
): Promise<Action[]> {
  const allActions = await manager.listActions();
  const allEvaluatorActions = [];
  for (const key in allActions) {
    if (isEvaluator(key)) {
      allEvaluatorActions.push(allActions[key]);
    }
  }
  return allEvaluatorActions;
}

export async function getMatchingEvaluatorActions(
  manager: BaseRuntimeManager,
  evaluators?: string[]
): Promise<Action[]> {
  if (!evaluators) {
    return [];
  }
  const allEvaluatorActions = await getAllEvaluatorActions(manager);
  const filteredEvaluatorActions = allEvaluatorActions.filter((action) =>
    evaluators.includes(action.key)
  );
  if (filteredEvaluatorActions.length === 0) {
    if (allEvaluatorActions.length == 0) {
      throw new Error('No evaluators installed');
    }
  }
  return filteredEvaluatorActions;
}

async function bulkRunAction(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
  inferenceDataset: Dataset;
  context?: string;
  actionConfig?: any;
}): Promise<EvalInput[]> {
  const { manager, actionRef, inferenceDataset, context, actionConfig } =
    params;
  const actionType = getSupportedActionType(actionRef);
  if (inferenceDataset.length === 0) {
    throw new Error('Cannot run inference, no data provided');
  }

  // Convert to satisfy TS checks. `input` is required in `Dataset` type, but
  // ZodAny also includes `undefined` in TS checks. This explcit conversion
  // works around this.
  const fullInferenceDataset = inferenceDataset as FullInferenceSample[];

  const states: InferenceRunState[] = [];
  const evalInputs: EvalInput[] = [];
  for (const sample of fullInferenceDataset) {
    logger.info(`Running inference '${actionRef}' ...`);
    if (actionType === 'model') {
      states.push(
        await runModelAction({
          manager,
          actionRef,
          sample,
          modelConfig: actionConfig,
        })
      );
    } else if (actionType === 'flow') {
      states.push(
        await runFlowAction({
          manager,
          actionRef,
          sample,
          context,
        })
      );
    } else {
      // executable-prompt action
      states.push(
        await runPromptAction({
          manager,
          actionRef,
          sample,
          context,
          promptConfig: actionConfig,
        })
      );
    }
  }

  logger.info(`Gathering evalInputs...`);
  for (const state of states) {
    evalInputs.push(await gatherEvalInput({ manager, actionRef, state }));
  }
  return evalInputs;
}

async function runFlowAction(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
  sample: FullInferenceSample;
  context?: any;
}): Promise<InferenceRunState> {
  const { manager, actionRef, sample, context } = { ...params };
  let state: InferenceRunState;
  try {
    const runActionResponse = await manager.runAction({
      key: actionRef,
      input: sample.input,
      context: context ? JSON.parse(context) : undefined,
    });
    state = {
      ...sample,
      traceIds: runActionResponse.telemetry?.traceId
        ? [runActionResponse.telemetry?.traceId]
        : [],
      response: runActionResponse.result,
    };
  } catch (e: any) {
    const traceId = e?.data?.details?.traceId;
    state = {
      ...sample,
      traceIds: traceId ? [traceId] : [],
      evalError: `Error when running inference. Details: ${e?.message ?? e}`,
    };
  }
  return state;
}

async function runModelAction(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
  sample: FullInferenceSample;
  modelConfig?: any;
}): Promise<InferenceRunState> {
  const { manager, actionRef, modelConfig, sample } = { ...params };
  let state: InferenceRunState;
  try {
    const modelInput = getModelInput(sample.input, modelConfig);
    const runActionResponse = await manager.runAction({
      key: actionRef,
      input: modelInput,
    });
    state = {
      ...sample,
      traceIds: runActionResponse.telemetry?.traceId
        ? [runActionResponse.telemetry?.traceId]
        : [],
      response: runActionResponse.result,
    };
  } catch (e: any) {
    const traceId = e?.data?.details?.traceId;
    state = {
      ...sample,
      traceIds: traceId ? [traceId] : [],
      evalError: `Error when running inference. Details: ${e?.message ?? e}`,
    };
  }
  return state;
}

async function runPromptAction(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
  sample: FullInferenceSample;
  context?: any;
  promptConfig?: any;
}): Promise<InferenceRunState> {
  const { manager, actionRef, sample, context, promptConfig } = { ...params };
  const { model: modelFromConfig, ...restOfConfig } = promptConfig;
  if (!modelFromConfig) {
    throw new Error(
      'Missing model: Please specific model for prompt evaluation'
    );
  }
  const model = modelFromConfig.split('/model/').pop();
  if (!model) {
    throw new Error(`Improper model provided: ${modelFromConfig}`);
  }
  let state: InferenceRunState;
  let renderedPrompt: {
    result: GenerateActionOptions;
    traceId: string;
  };
  // Step 1. Render the prompt with inputs
  try {
    const runActionResponse = await manager.runAction({
      key: actionRef,
      input: sample.input,
      context: context ? JSON.parse(context) : undefined,
    });

    renderedPrompt = {
      traceId: runActionResponse.telemetry?.traceId!,
      result: runActionResponse.result as GenerateActionOptions,
    };
  } catch (e: any) {
    if (e instanceof z.ZodError) {
      state = {
        ...sample,
        traceIds: [],
        evalError: `Error parsing prompt response. Details: ${JSON.stringify(e.format())}`,
      };
    } else {
      const traceId = e?.data?.details?.traceId;
      state = {
        ...sample,
        traceIds: traceId ? [traceId] : [],
        evalError: `Error when rendering prompt. Details: ${e?.message ?? e}`,
      };
    }
    return state;
  }
  // Step 2. Run rendered prompt on the model
  try {
    let modelInput = renderedPrompt.result;
    // Override with runtime specific config
    if (Object.keys(restOfConfig ?? {}).length > 0) {
      modelInput = { ...modelInput, model, config: restOfConfig };
    } else {
      modelInput = { ...modelInput, model };
    }
    const runActionResponse = await manager.runAction({
      key: GENERATE_ACTION_UTIL,
      input: modelInput,
    });
    const traceIds = runActionResponse.telemetry?.traceId
      ? [renderedPrompt.traceId, runActionResponse.telemetry?.traceId]
      : [renderedPrompt.traceId];
    state = {
      ...sample,
      traceIds: traceIds,
      response: runActionResponse.result,
    };
  } catch (e: any) {
    const traceId = e?.data?.details?.traceId;
    const traceIds = traceId
      ? [renderedPrompt.traceId, traceId]
      : [renderedPrompt.traceId];
    state = {
      ...sample,
      traceIds: traceIds.filter((t): t is string => !!t),
      evalError: `Error when running inference. Details: ${e?.message ?? e}`,
    };
  }
  return state;
}

async function gatherEvalInput(params: {
  manager: BaseRuntimeManager;
  actionRef: string;
  state: InferenceRunState;
}): Promise<EvalInput> {
  const { manager, actionRef, state } = params;

  const actionType = getSupportedActionType(actionRef);
  const extractors = await getEvalExtractors(actionRef);
  const traceIds = state.traceIds;

  if (
    traceIds.length === 0 ||
    (actionType === 'executable-prompt' && traceIds.length < 2)
  ) {
    logger.warn('No valid traceId available...');
    return {
      ...state,
      error: state.evalError,
      testCaseId: state.testCaseId,
      traceIds: traceIds,
    };
  }

  // Only the last collected trace to be used for evaluation.
  const traceId = traceIds.at(-1)!;
  // Sleep to let traces persist.
  await new Promise((resolve) => setTimeout(resolve, 2000));
  const trace = await manager.getTrace({
    traceId,
  });

  // Always use original input for models and prompts.
  const input = actionType === 'flow' ? extractors.input(trace) : state.input;
  let custom = undefined;
  if (actionType === 'executable-prompt') {
    const promptTrace = await manager.getTrace({
      traceId: traceIds[0],
    });
    custom = { renderedPrompt: extractors.output(promptTrace) };
  }

  const nestedSpan = stackTraceSpans(trace);
  if (!nestedSpan) {
    return {
      testCaseId: state.testCaseId,
      input,
      error: `Unable to extract any spans from trace ${traceId}`,
      reference: state.reference,
      custom,
      traceIds: traceIds,
    };
  }

  if (nestedSpan.attributes['genkit:state'] === 'error') {
    return {
      testCaseId: state.testCaseId,
      input,
      error:
        getSpanErrorMessage(nestedSpan) ?? `Unknown error in trace ${traceId}`,
      reference: state.reference,
      custom,
      traceIds: traceIds,
    };
  }

  const output = extractors.output(trace);
  const context = extractors.context(trace);
  const error =
    actionType === 'model' ? getErrorFromModelResponse(output) : undefined;

  return {
    // TODO Replace this with unified trace class
    testCaseId: state.testCaseId,
    input,
    output,
    error,
    context: Array.isArray(context) ? context : [context],
    reference: state.reference,
    custom,
    traceIds: traceIds,
  };
}

function getSpanErrorMessage(span: SpanData): string | undefined {
  if (span && span.status?.code === 2 /* SpanStatusCode.ERROR */) {
    // It's possible for a trace to have multiple exception events,
    // however we currently only expect and display the first one.
    const event = span.timeEvents?.timeEvent
      ?.filter((e) => e.annotation.description === 'exception')
      .shift();
    return (
      (event?.annotation?.attributes['exception.message'] as string) ?? 'Error'
    );
  }
}

function getErrorFromModelResponse(obj: any): string | undefined {
  const response = obj as GenerateResponseData;

  // Legacy response is present
  const hasLegacyResponse =
    !!response.candidates && response.candidates.length > 0;
  // New (non-deprecated) response is present
  const hasNewResponse = !!response.message;

  if (!response || (!hasLegacyResponse && !hasNewResponse)) {
    return `No response was extracted from the output. '${JSON.stringify(obj)}'`;
  }

  if (hasLegacyResponse) {
    // We currently only support the first candidate
    const candidate = response.candidates![0] as CandidateData;
    if (candidate.finishReason === 'blocked') {
      return candidate.finishMessage || `Generation was blocked by the model.`;
    }
  }

  if (hasNewResponse) {
    if (response.finishReason === 'blocked') {
      return response.finishMessage || `Generation was blocked by the model.`;
    }
  }
}

function isSupportedActionRef(actionRef: string) {
  return SUPPORTED_ACTION_TYPES.some((supportedType) =>
    actionRef.startsWith(`/${supportedType}`)
  );
}

function getSupportedActionType(actionRef: string): SupportedActionType {
  if (actionRef.startsWith('/model')) {
    return 'model';
  }
  if (actionRef.startsWith('/flow')) {
    return 'flow';
  }
  if (actionRef.startsWith('/executable-prompt')) {
    return 'executable-prompt';
  }
  throw new Error(`Unsupported action type: ${actionRef}`);
}
