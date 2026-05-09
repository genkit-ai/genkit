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

import { action, z, type Action } from '@genkit-ai/core';
import { logger } from '@genkit-ai/core/logging';
import type { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import { SpanMetadata, runInNewSpan } from '@genkit-ai/core/tracing';
import { randomUUID } from 'crypto';

export const ATTR_PREFIX = 'genkit';
export const SPAN_STATE_ATTR = ATTR_PREFIX + ':state';

export const BaseDataPointSchema = z.object({
  input: z.unknown(),
  output: z.unknown().optional(),
  context: z.array(z.unknown()).optional(),
  reference: z.unknown().optional(),
  testCaseId: z.string().optional(),
  traceIds: z.array(z.string()).optional(),
});

// DataPoint that is to be used for actions. This needs testCaseId to be present.
export const BaseEvalDataPointSchema = BaseDataPointSchema.extend({
  testCaseId: z.string(),
});
export type BaseEvalDataPoint = z.infer<typeof BaseEvalDataPointSchema>;

const EvalStatusEnumSchema = z.enum(['UNKNOWN', 'PASS', 'FAIL']);

/** Enum that indicates if an evaluation has passed or failed */
export enum EvalStatusEnum {
  UNKNOWN = 'UNKNOWN',
  PASS = 'PASS',
  FAIL = 'FAIL',
}

export const ScoreSchema = z.object({
  id: z
    .string()
    .describe(
      'Optional ID to differentiate different scores if applying in a single evaluation'
    )
    .optional(),
  score: z.union([z.number(), z.string(), z.boolean()]).optional(),
  status: EvalStatusEnumSchema.optional(),
  error: z.string().optional(),
  details: z
    .object({
      reasoning: z.string().optional(),
    })
    .passthrough()
    .optional(),
});

// Update genkit-tools/src/utils/evals.ts if you change this value
export const EVALUATOR_METADATA_KEY_DISPLAY_NAME = 'evaluatorDisplayName';
export const EVALUATOR_METADATA_KEY_DEFINITION = 'evaluatorDefinition';
export const EVALUATOR_METADATA_KEY_IS_BILLED = 'evaluatorIsBilled';

export type Score = z.infer<typeof ScoreSchema>;
export type BaseDataPoint = z.infer<typeof BaseDataPointSchema>;
export type Dataset<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
> = Array<z.infer<DataPoint>>;

export const EvalResponseSchema = z.object({
  sampleIndex: z.number().optional(),
  testCaseId: z.string(),
  traceId: z.string().optional(),
  spanId: z.string().optional(),
  evaluation: z.union([ScoreSchema, z.array(ScoreSchema)]),
});
export type EvalResponse = z.infer<typeof EvalResponseSchema>;

export const EvalResponsesSchema = z.array(EvalResponseSchema);
export type EvalResponses = z.infer<typeof EvalResponsesSchema>;

export type EvaluatorFn<
  EvalDataPoint extends
    typeof BaseEvalDataPointSchema = typeof BaseEvalDataPointSchema,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = (
  input: z.infer<EvalDataPoint>,
  evaluatorOptions?: z.infer<CustomOptions>
) => Promise<EvalResponse>;

export type EvaluatorAction<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = Action<typeof EvalRequestSchema, typeof EvalResponsesSchema> & {
  __dataPointType?: DataPoint;
  __configSchema?: CustomOptions;
};

function withMetadata<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  evaluator: Action<typeof EvalRequestSchema, typeof EvalResponsesSchema>,
  dataPointType?: DataPoint,
  configSchema?: CustomOptions
): EvaluatorAction<DataPoint, CustomOptions> {
  const withMeta = evaluator as EvaluatorAction<DataPoint, CustomOptions>;
  withMeta.__dataPointType = dataPointType;
  withMeta.__configSchema = configSchema;
  return withMeta;
}

const EvalRequestSchema = z.object({
  dataset: z.array(BaseDataPointSchema),
  evalRunId: z.string(),
  options: z.unknown(),
});

export interface EvaluatorParams<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  evaluator: EvaluatorArgument<DataPoint, CustomOptions>;
  dataset: Dataset<DataPoint>;
  evalRunId?: string;
  options?: z.infer<CustomOptions>;
}

export interface EvaluatorOptions<
  DataPoint extends typeof BaseDataPointSchema,
  EvaluatorOpts extends z.ZodTypeAny,
> {
  name: string;
  displayName: string;
  definition: string;
  dataPointType?: DataPoint;
  configSchema?: EvaluatorOpts;
  isBilled?: boolean;
}

/**
 * Creates evaluator action for the provided {@link EvaluatorFn} implementation.
 */
export function defineEvaluator<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
  EvalDataPoint extends
    typeof BaseEvalDataPointSchema = typeof BaseEvalDataPointSchema,
  EvaluatorOpts extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: EvaluatorOptions<DataPoint, EvaluatorOpts>,
  runner: EvaluatorFn<EvalDataPoint, EvaluatorOpts>
): EvaluatorAction {
  const e = evaluator(options, runner);

  registry.registerAction('evaluator', e);

  return e;
}

/**
 * Creates evaluator action for the provided {@link EvaluatorFn} implementation.
 */
export function evaluator<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
  EvalDataPoint extends
    typeof BaseEvalDataPointSchema = typeof BaseEvalDataPointSchema,
  EvaluatorOpts extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: EvaluatorOptions<DataPoint, EvaluatorOpts>,
  runner: EvaluatorFn<EvalDataPoint, EvaluatorOpts>
): EvaluatorAction {
  const evalMetadata = {};
  evalMetadata[EVALUATOR_METADATA_KEY_IS_BILLED] =
    options.isBilled == undefined ? true : options.isBilled;
  evalMetadata[EVALUATOR_METADATA_KEY_DISPLAY_NAME] = options.displayName;
  evalMetadata[EVALUATOR_METADATA_KEY_DEFINITION] = options.definition;
  if (options.configSchema) {
    evalMetadata['customOptions'] = toJsonSchema({
      schema: options.configSchema,
    });
  }
  const evaluator = action(
    {
      actionType: 'evaluator',
      name: options.name,
      inputSchema: EvalRequestSchema.extend({
        dataset: options.dataPointType
          ? z.array(options.dataPointType)
          : z.array(BaseDataPointSchema),
        options: options.configSchema ?? z.unknown(),
        evalRunId: z.string(),
        batchSize: z.number().optional(),
      }),
      outputSchema: EvalResponsesSchema,
      metadata: {
        type: 'evaluator',
        evaluator: evalMetadata,
      },
    },
    async (i) => {
      const evalResponses: EvalResponses = [];
      // This also populates missing testCaseIds
      const batches = getBatchedArray(i.dataset, i.batchSize);

      for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
        const batch = batches[batchIndex];
        try {
          if (batch.length === 1) {
            const results = await runBatch(
              runner,
              batch,
              i.batchSize,
              batchIndex,
              i.evalRunId,
              i.options
            );
            evalResponses.push(...results);
          } else {
            await runInNewSpan(
              {
                metadata: {
                  name: `Batch ${batchIndex}`,
                  metadata: { 'evaluator:evalRunId': i.evalRunId },
                },
              },
              async (metadata, otSpan) => {
                const results = await runBatch(
                  runner,
                  batch,
                  i.batchSize,
                  batchIndex,
                  i.evalRunId,
                  i.options,
                  metadata
                );
                evalResponses.push(...results);
              }
            );
          }
        } catch (e) {
          logger.error(
            `Evaluation of batch ${batchIndex} failed: \n${(e as Error).stack}`
          );
          continue;
        }
      }
      return evalResponses;
    }
  );
  const ewm = withMetadata(
    evaluator as any as Action<
      typeof EvalRequestSchema,
      typeof EvalResponsesSchema
    >,
    options.dataPointType,
    options.configSchema
  );
  return ewm;
}

export type EvaluatorArgument<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> =
  | string
  | EvaluatorAction<DataPoint, CustomOptions>
  | EvaluatorReference<CustomOptions>;

/**
 * A veneer for interacting with evaluators.
 */
export async function evaluate<
  DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  params: EvaluatorParams<DataPoint, CustomOptions>
): Promise<EvalResponses> {
  let evaluator: EvaluatorAction<DataPoint, CustomOptions>;
  if (typeof params.evaluator === 'string') {
    evaluator = await registry.lookupAction(`/evaluator/${params.evaluator}`);
  } else if (Object.hasOwnProperty.call(params.evaluator, 'info')) {
    evaluator = await registry.lookupAction(
      `/evaluator/${params.evaluator.name}`
    );
  } else {
    evaluator = params.evaluator as EvaluatorAction<DataPoint, CustomOptions>;
  }
  if (!evaluator) {
    throw new Error('Unable to utilize the provided evaluator');
  }
  return (await evaluator({
    dataset: params.dataset,
    options: params.options,
    evalRunId: params.evalRunId ?? randomUUID(),
  })) as EvalResponses;
}

export const EvaluatorInfoSchema = z.object({
  /** Friendly label for this evaluator */
  label: z.string().optional(),
  metrics: z.array(z.string()),
});
export type EvaluatorInfo = z.infer<typeof EvaluatorInfoSchema>;

export interface EvaluatorReference<CustomOptions extends z.ZodTypeAny> {
  name: string;
  configSchema?: CustomOptions;
  info?: EvaluatorInfo;
}

/**
 * Helper method to configure a {@link EvaluatorReference} to a plugin.
 */
export function evaluatorRef<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: EvaluatorReference<CustomOptionsSchema>
): EvaluatorReference<CustomOptionsSchema> {
  return { ...options };
}

/**
 * Helper method to generated batched array. Also ensures each testCase has a
 * testCaseId
 */
function getBatchedArray<T extends { testCaseId?: string }>(
  arr: T[],
  batchSize?: number
): T[][] {
  let size: number;
  if (!batchSize) {
    size = 1;
  } else {
    size = batchSize;
  }

  const batches: T[][] = [];
  for (var i = 0; i < arr.length; i += size) {
    batches.push(
      arr.slice(i, i + size).map((d) => ({
        ...d,
        testCaseId: d.testCaseId ?? randomUUID(),
      }))
    );
  }

  return batches;
}

async function runBatch<
  EvalDataPoint extends
    typeof BaseEvalDataPointSchema = typeof BaseEvalDataPointSchema,
  EvaluatorOpts extends z.ZodTypeAny = z.ZodTypeAny,
>(
  runner: EvaluatorFn<EvalDataPoint, EvaluatorOpts>,
  batch: BaseDataPoint[],
  batchSize: number | undefined,
  batchIndex: number,
  evalRunId: string,
  options: any,
  batchMetadata?: SpanMetadata
): Promise<EvalResponses> {
  if (batchMetadata) {
    batchMetadata.input = batch;
  }
  const evalRunPromises = batch.map((d, index) => {
    const sampleIndex = batchSize ? batchSize * batchIndex + index : batchIndex;
    const datapoint = d as BaseEvalDataPoint;
    return runInNewSpan(
      {
        metadata: {
          name: `Test Case ${datapoint.testCaseId}`,
          metadata: { 'evaluator:evalRunId': evalRunId },
        },
      },
      async (metadata, otSpan) => {
        const spanId = otSpan.spanContext().spanId;
        const traceId = otSpan.spanContext().traceId;
        metadata.input = datapoint;
        try {
          const result = await runner(datapoint, options);
          metadata.output = result;
          return {
            ...result,
            traceId,
            spanId,
            sampleIndex,
          };
        } catch (error) {
          return {
            sampleIndex,
            spanId,
            traceId,
            testCaseId: datapoint.testCaseId,
            evaluation: {
              error: `Evaluation of test case ${datapoint.testCaseId} failed: \n${error}`,
            },
          };
        }
      }
    );
  });

  const allResults = await Promise.all(evalRunPromises);
  if (batchMetadata) {
    batchMetadata.output = allResults;
  }
  return allResults;
}
