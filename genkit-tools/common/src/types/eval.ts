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

import type { JSONSchema7 } from 'json-schema';
import { z } from 'zod';
import zodToJsonSchema from 'zod-to-json-schema';
import type {
  CreateDatasetRequest,
  ListEvalKeysRequest,
  ListEvalKeysResponse,
  UpdateDatasetRequest,
} from './apis';
import { GenerateActionOptionsSchema, GenerateRequestSchema } from './model';

/**
 * This file defines schema and types that are used by the Eval store.
 */

/**
 * Supported datatype for model datasets
 */
export const ModelInferenceInputSchema = z.union([
  z.string(),
  GenerateRequestSchema,
]);
export type ModelInferenceInput = z.infer<typeof ModelInferenceInputSchema>;
export const ModelInferenceInputJSONSchema = zodToJsonSchema(
  ModelInferenceInputSchema,
  {
    $refStrategy: 'none',
    removeAdditionalStrategy: 'strict',
  }
) as JSONSchema7;

/**
 * GenerateRequest JSON schema to support eval-inference using models
 */
export const GenerateRequestJSONSchema = zodToJsonSchema(
  GenerateRequestSchema,
  {
    $refStrategy: 'none',
    removeAdditionalStrategy: 'strict',
  }
) as JSONSchema7;

/**
 * Combined GenerateInput JSON schema to support eval-inference using models
 */
export const GenerateInputJSONSchema = zodToJsonSchema(
  z.union([GenerateRequestSchema, GenerateActionOptionsSchema]),
  {
    $refStrategy: 'none',
    removeAdditionalStrategy: 'strict',
  }
) as JSONSchema7;

/**
 * A single sample to be used for inference.
 **/
export const InferenceSampleSchema = z.object({
  testCaseId: z.string().optional(),
  input: z.any(),
  reference: z.any().optional(),
});
export type InferenceSample = z.infer<typeof InferenceSampleSchema>;

/**
 * A set of samples that is ready for inference.
 *
 * This should be used in user-facing surfaces (CLI/API inputs) to accommodate various user input formats. For internal wire-transfer/storage, prefer {@link Dataset}.
 */
export const InferenceDatasetSchema = z.array(InferenceSampleSchema);
export type InferenceDataset = z.infer<typeof InferenceDatasetSchema>;

/**
 * Represents a Dataset, to be used for bulk-inference / evaluation. This is a more optionated form of InferenceDataset, which guarantees testCaseId for each test sample.
 */
export const DatasetSchema = z.array(
  z.object({
    testCaseId: z.string(),
    input: z.any(),
    reference: z.any().optional(),
  })
);
export type Dataset = z.infer<typeof DatasetSchema>;

/**
 * A record that is ready for evaluation.
 *
 * This should be used in user-facing surfaces (CLI/API inputs) to accommodate various user input formats. For use in EvaluatorAction, use {@link EvalInput}.
 */
export const EvaluationSampleSchema = z.object({
  testCaseId: z.string().optional(),
  input: z.any(),
  output: z.any(),
  error: z.string().optional(),
  context: z.array(z.any()).optional(),
  reference: z.any().optional(),
  traceIds: z.array(z.string()).optional(),
});
export type EvaluationSample = z.infer<typeof EvaluationSampleSchema>;

/**
 * A set of samples that is ready for evaluation.
 *
 * This should be used in user-facing surfaces (CLI/API inputs) to accommodate various user input formats. For use in EvaluatorAction, use {@link EvalInput}.
 */
export const EvaluationDatasetSchema = z.array(EvaluationSampleSchema);
export type EvaluationDataset = z.infer<typeof EvaluationDatasetSchema>;

/**
 * A fully-valid record to be used for evaluation.
 *
 * This is not user facing. Required fields that are missing from
 * {@link EvaluationSampleSchema} must be auto-popoulated.
 */
export const EvalInputSchema = z.object({
  testCaseId: z.string(),
  input: z.any(),
  output: z.any(),
  error: z.string().optional(),
  context: z.array(z.any()).optional(),
  reference: z.any().optional(),
  custom: z.record(z.string(), z.any()).optional(),
  traceIds: z.array(z.string()),
});
export type EvalInput = z.infer<typeof EvalInputSchema>;

export const EvalInputDatasetSchema = z.array(EvalInputSchema);
export type EvalInputDataset = z.infer<typeof EvalInputDatasetSchema>;

const EvalStatusEnumSchema = z.enum(['UNKNOWN', 'PASS', 'FAIL']);
/** Enum that indicates if an evaluation has passed or failed */
export enum EvalStatusEnum {
  UNKNOWN = 'UNKNOWN',
  PASS = 'PASS',
  FAIL = 'FAIL',
}

export const EvalMetricSchema = z.object({
  evaluator: z.string(),
  scoreId: z.string().optional(),
  score: z.union([z.number(), z.string(), z.boolean()]).optional(),
  status: EvalStatusEnumSchema.optional(),
  rationale: z.string().optional(),
  error: z.string().optional(),
  traceId: z.string().optional(),
  spanId: z.string().optional(),
});
export type EvalMetric = z.infer<typeof EvalMetricSchema>;

/**
 * The scored result of an evaluation of an individual inference.
 *
 * This should be an extension of the input with the evaluation metrics.
 */
export const EvalResultSchema = EvalInputSchema.extend({
  metrics: z.array(EvalMetricSchema).optional(),
});
export type EvalResult = z.infer<typeof EvalResultSchema>;

/**
 * A unique identifier for an Evaluation Run.
 */
export const EvalRunKeySchema = z.object({
  actionRef: z.string().optional(),
  datasetId: z.string().optional(),
  datasetVersion: z.number().optional(),
  evalRunId: z.string(),
  createdAt: z.string(),
  actionConfig: z.any().optional(),
  metricSummaries: z.array(z.record(z.string(), z.any())).optional(),
  metricsMetadata: z
    .record(
      z.string(),
      z.object({
        displayName: z.string(),
        definition: z.string(),
      })
    )
    .optional(),
});
export type EvalRunKey = z.infer<typeof EvalRunKeySchema>;
export const EvalKeyAugmentsSchema = EvalRunKeySchema.pick({
  datasetId: true,
  datasetVersion: true,
  actionRef: true,
  actionConfig: true,
});
export type EvalKeyAugments = z.infer<typeof EvalKeyAugmentsSchema>;

/**
 * A container for the results of evaluation over a batch of test cases.
 */
export const EvalRunSchema = z.object({
  key: EvalRunKeySchema,
  results: z.array(EvalResultSchema),
  /* deprecated, use key.metricsMetadata */
  metricsMetadata: z
    .record(
      z.string(),
      z.object({
        displayName: z.string(),
        definition: z.string(),
      })
    )
    .optional(),
});
export type EvalRun = z.infer<typeof EvalRunSchema>;

/**
 * Eval store persistence interface.
 */
export interface EvalStore {
  /**
   * Persist an EvalRun
   * @param evalRun the result of evaluating a collection of test cases.
   */
  save(evalRun: EvalRun): Promise<void>;

  /**
   * Load a single EvalRun from storage
   * @param evalRunId the ID of the EvalRun
   */
  load(evalRunId: string): Promise<EvalRun | undefined>;

  /**
   * List the keys of all EvalRuns from storage
   * @param query (optional) filter criteria for the result list
   */
  list(query?: ListEvalKeysRequest): Promise<ListEvalKeysResponse>;

  /**
   * Delete EvalRun by ID
   * @param evalRunId the ID of the EvalRun
   */
  delete(evalRunId: string): Promise<void>;
}

export const DatasetSchemaSchema = z.object({
  inputSchema: z
    .record(z.any())
    .describe('Valid JSON Schema for the `input` field of dataset entry.')
    .optional(),
  referenceSchema: z
    .record(z.any())
    .describe('Valid JSON Schema for the `reference` field of dataset entry.')
    .optional(),
});

/** Type of dataset, useful for UI niceties. */
export const DatasetTypeSchema = z.enum([
  'UNKNOWN',
  'FLOW',
  'MODEL',
  'EXECUTABLE_PROMPT',
]);
export type DatasetType = z.infer<typeof DatasetTypeSchema>;

/**
 * Metadata for Dataset objects containing version, create and update time, etc.
 */
export const DatasetMetadataSchema = z.object({
  /** unique. user-provided or auto-generated */
  datasetId: z.string(),
  size: z.number(),
  schema: DatasetSchemaSchema.optional(),
  datasetType: DatasetTypeSchema,
  targetAction: z.string().optional(),
  metricRefs: z.array(z.string()).default([]),
  /** 1 for v1, 2 for v2, etc */
  version: z.number(),
  createTime: z.string(),
  updateTime: z.string(),
});
export type DatasetMetadata = z.infer<typeof DatasetMetadataSchema>;

/**
 * Eval dataset store persistence interface.
 */
export interface DatasetStore {
  /**
   * Create new dataset with the given data
   * @param req create requeest with the data
   * @returns dataset metadata
   */
  createDataset(req: CreateDatasetRequest): Promise<DatasetMetadata>;

  /**
   * Update dataset
   * @param req update requeest with new data
   * @returns updated dataset metadata
   */
  updateDataset(req: UpdateDatasetRequest): Promise<DatasetMetadata>;

  /**
   * Get existing dataset
   * @param datasetId the ID of the dataset
   * @returns dataset ready for inference
   */
  getDataset(datasetId: string): Promise<Dataset>;

  /**
   * List all existing datasets
   * @returns array of dataset metadata objects
   */
  listDatasets(): Promise<DatasetMetadata[]>;

  /**
   * Delete existing dataset
   * @param datasetId the ID of the dataset
   */
  deleteDataset(datasetId: string): Promise<void>;
}
