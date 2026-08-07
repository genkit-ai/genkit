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

import type { protos } from '@google-cloud/aiplatform';
import { z, type Action, type Genkit } from 'genkit';
import type { GoogleAuth } from 'google-auth-library';
import { EvaluatorFactory } from './evaluator_factory.js';
import type { VertexAIEvaluationMetricConfig } from './types';
import {
  VertexAIEvaluationMetric,
  VertexAIEvaluationMetricType,
} from './types';

function stringify(input: unknown) {
  return typeof input === 'string' ? input : JSON.stringify(input);
}

export function vertexEvaluators(
  ai: Genkit,
  auth: GoogleAuth,
  metrics: VertexAIEvaluationMetric[],
  projectId: string,
  location: string
): Action[] {
  const factory = new EvaluatorFactory(auth, location, projectId);
  return metrics.map((metric) => {
    const { type: metricType, metricSpec } = isConfig(metric)
      ? metric
      : { type: metric, metricSpec: {} };

    switch (metricType) {
      case VertexAIEvaluationMetricType.BLEU: {
        return createBleuEvaluator(ai, factory, metricSpec);
      }
      case VertexAIEvaluationMetricType.ROUGE: {
        return createRougeEvaluator(ai, factory, metricSpec);
      }
      case VertexAIEvaluationMetricType.FLUENCY: {
        return createFluencyEvaluator(ai, factory, metricSpec);
      }
      case VertexAIEvaluationMetricType.SAFETY: {
        return createSafetyEvaluator(ai, factory, metricSpec);
      }
      case VertexAIEvaluationMetricType.GROUNDEDNESS: {
        return createGroundednessEvaluator(ai, factory, metricSpec);
      }
      case VertexAIEvaluationMetricType.SUMMARIZATION_QUALITY: {
        return createSummarizationQualityEvaluator(ai, factory, metricSpec);
      }
      case VertexAIEvaluationMetricType.SUMMARIZATION_HELPFULNESS: {
        return createSummarizationHelpfulnessEvaluator(ai, factory, metricSpec);
      }
      case VertexAIEvaluationMetricType.SUMMARIZATION_VERBOSITY: {
        return createSummarizationVerbosityEvaluator(ai, factory, metricSpec);
      }
    }
  });
}

function isConfig(
  config: VertexAIEvaluationMetric
): config is VertexAIEvaluationMetricConfig {
  return (config as VertexAIEvaluationMetricConfig).type !== undefined;
}

const BleuResponseSchema = z.object({
  bleuResults: z.object({
    bleuMetricValues: z.array(z.object({ score: z.number() })),
  }),
});

// TODO: Add support for batch inputs
function createBleuEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.IBleuSpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.BLEU,
      displayName: 'BLEU',
      definition:
        'Computes the BLEU score by comparing the output against the ground truth',
      responseSchema: BleuResponseSchema,
    },
    (datapoint) => {
      return {
        bleuInput: {
          metricSpec,
          instances: [
            {
              prediction: stringify(datapoint.output),
              reference: datapoint.reference as string,
            },
          ],
        },
      };
    },
    (response) => {
      return {
        score: response.bleuResults.bleuMetricValues[0].score,
      };
    }
  );
}

const RougeResponseSchema = z.object({
  rougeResults: z.object({
    rougeMetricValues: z.array(z.object({ score: z.number() })),
  }),
});

// TODO: Add support for batch inputs
function createRougeEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.IRougeSpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.ROUGE,
      displayName: 'ROUGE',
      definition:
        'Computes the ROUGE score by comparing the output against the ground truth',
      responseSchema: RougeResponseSchema,
    },
    (datapoint) => {
      return {
        rougeInput: {
          metricSpec,
          instances: [
            {
              prediction: stringify(datapoint.output),
              reference: datapoint.reference as string,
            },
          ],
        },
      };
    },
    (response) => {
      return {
        score: response.rougeResults.rougeMetricValues[0].score,
      };
    }
  );
}

const FluencyResponseSchema = z.object({
  fluencyResult: z.object({
    score: z.number(),
    explanation: z.string(),
    confidence: z.number(),
  }),
});

function createFluencyEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.IFluencySpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.FLUENCY,
      displayName: 'Fluency',
      definition: 'Assesses the language mastery of an output',
      responseSchema: FluencyResponseSchema,
    },
    (datapoint) => {
      return {
        fluencyInput: {
          metricSpec,
          instance: {
            prediction: stringify(datapoint.output),
          },
        },
      };
    },
    (response) => {
      return {
        score: response.fluencyResult.score,
        details: {
          reasoning: response.fluencyResult.explanation,
        },
      };
    }
  );
}

const SafetyResponseSchema = z.object({
  safetyResult: z.object({
    score: z.number(),
    explanation: z.string(),
    confidence: z.number(),
  }),
});

function createSafetyEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.ISafetySpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.SAFETY,
      displayName: 'Safety',
      definition: 'Assesses the level of safety of an output',
      responseSchema: SafetyResponseSchema,
    },
    (datapoint) => {
      return {
        safetyInput: {
          metricSpec,
          instance: {
            prediction: stringify(datapoint.output),
          },
        },
      };
    },
    (response) => {
      return {
        score: response.safetyResult.score,
        details: {
          reasoning: response.safetyResult.explanation,
        },
      };
    }
  );
}

const GroundednessResponseSchema = z.object({
  groundednessResult: z.object({
    score: z.number(),
    explanation: z.string(),
    confidence: z.number(),
  }),
});

function createGroundednessEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.IGroundednessSpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.GROUNDEDNESS,
      displayName: 'Groundedness',
      definition:
        'Assesses the ability to provide or reference information included only in the context',
      responseSchema: GroundednessResponseSchema,
    },
    (datapoint) => {
      return {
        groundednessInput: {
          metricSpec,
          instance: {
            prediction: stringify(datapoint.output),
            context: datapoint.context?.join('. '),
          },
        },
      };
    },
    (response) => {
      return {
        score: response.groundednessResult.score,
        details: {
          reasoning: response.groundednessResult.explanation,
        },
      };
    }
  );
}

const SummarizationQualityResponseSchema = z.object({
  summarizationQualityResult: z.object({
    score: z.number(),
    explanation: z.string(),
    confidence: z.number(),
  }),
});

function createSummarizationQualityEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.ISummarizationQualitySpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.SUMMARIZATION_QUALITY,
      displayName: 'Summarization quality',
      definition: 'Assesses the overall ability to summarize text',
      responseSchema: SummarizationQualityResponseSchema,
    },
    (datapoint) => {
      return {
        summarizationQualityInput: {
          metricSpec,
          instance: {
            prediction: stringify(datapoint.output),
            instruction: stringify(datapoint.input),
            context: datapoint.context?.join('. '),
          },
        },
      };
    },
    (response) => {
      return {
        score: response.summarizationQualityResult.score,
        details: {
          reasoning: response.summarizationQualityResult.explanation,
        },
      };
    }
  );
}

const SummarizationHelpfulnessResponseSchema = z.object({
  summarizationHelpfulnessResult: z.object({
    score: z.number(),
    explanation: z.string(),
    confidence: z.number(),
  }),
});

function createSummarizationHelpfulnessEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.ISummarizationHelpfulnessSpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.SUMMARIZATION_HELPFULNESS,
      displayName: 'Summarization helpfulness',
      definition:
        'Assesses the ability to provide a summarization, which contains the details necessary to substitute the original text',
      responseSchema: SummarizationHelpfulnessResponseSchema,
    },
    (datapoint) => {
      return {
        summarizationHelpfulnessInput: {
          metricSpec,
          instance: {
            prediction: stringify(datapoint.output),
            instruction: stringify(datapoint.input),
            context: datapoint.context?.join('. '),
          },
        },
      };
    },
    (response) => {
      return {
        score: response.summarizationHelpfulnessResult.score,
        details: {
          reasoning: response.summarizationHelpfulnessResult.explanation,
        },
      };
    }
  );
}

const SummarizationVerbositySchema = z.object({
  summarizationVerbosityResult: z.object({
    score: z.number(),
    explanation: z.string(),
    confidence: z.number(),
  }),
});

function createSummarizationVerbosityEvaluator(
  ai: Genkit,
  factory: EvaluatorFactory,
  metricSpec: protos.google.cloud.aiplatform.v1.ISummarizationVerbositySpec
): Action {
  return factory.create(
    ai,
    {
      metric: VertexAIEvaluationMetricType.SUMMARIZATION_VERBOSITY,
      displayName: 'Summarization verbosity',
      definition: 'Aassess the ability to provide a succinct summarization',
      responseSchema: SummarizationVerbositySchema,
    },
    (datapoint) => {
      return {
        summarizationVerbosityInput: {
          metricSpec,
          instance: {
            prediction: stringify(datapoint.output),
            instruction: stringify(datapoint.input),
            context: datapoint.context?.join('. '),
          },
        },
      };
    },
    (response) => {
      return {
        score: response.summarizationVerbosityResult.score,
        details: {
          reasoning: response.summarizationVerbosityResult.explanation,
        },
      };
    }
  );
}
