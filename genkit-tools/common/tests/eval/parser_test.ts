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

import { describe, expect, it } from 'vitest';
import {
  enrichResultsWithScoring,
  extractMetricSummaries,
} from '../../src/eval';
import {
  EvalMetricSchema,
  EvalStatusEnum,
  type EvalResult,
} from '../../src/types/eval';
import type { EvalFnResponse, EvalResponse } from '../../src/types/evaluator';

describe('parser', () => {
  const evalRunResults: EvalResult[] = [
    {
      testCaseId: 'case1',
      input: 'Who is bob best friend?',
      output: 'Patrick',
      context: [
        "Bob's best friend is Patrick.",
        'Bob has a friend named Patrick.',
        'Bob has a friend named Sandy.',
      ],
      metrics: [],
      traceIds: ['trace2'],
    },
    {
      testCaseId: 'case2',
      input: 'How many friends does Bob have?',
      output: '2',
      context: [
        "Bob's best friend is Patrick.",
        'Bob has a friend named Patrick.',
        'Bob has a friend named Sandy.',
      ],
      metrics: [],
      traceIds: ['trace2'],
    },
  ];

  const evaluatorOutput: Record<string, EvalResponse> = {
    '/evaluator/genkit/faithfulness': [
      {
        testCaseId: 'case1',
        sampleIndex: 0,
        evaluation: {
          score: 1,
          details: {
            reasoning: 'It looks good to me!',
          },
        },
      },
      {
        testCaseId: 'case2',
        sampleIndex: 1,
        evaluation: {
          score: 1,
          details: {
            reasoning: 'I thought the LLM did a very nice job',
          },
        },
      },
    ],
    '/evaluator/genkit/context_relevancy': [
      {
        testCaseId: 'case1',
        sampleIndex: 0,
        evaluation: {
          score: 1,
          details: {
            reasoning: 'Context was utilized.',
          },
        },
      },
      {
        testCaseId: 'case2',
        sampleIndex: 0,
        evaluation: {
          score: 1,
          details: {
            reasoning: 'Context was utilized.',
          },
        },
      },
    ],
  };

  describe('enrichResultsWithScoring', () => {
    it('Adds scoring data to eval results', () => {
      const results = enrichResultsWithScoring(evaluatorOutput, evalRunResults);
      expect(results).toHaveLength(2);
      results.forEach((result) => {
        expect(result.metrics).toMatchObject([
          EvalMetricSchema.parse({
            evaluator: '/evaluator/genkit/faithfulness',
            score: 1,
          }),
          EvalMetricSchema.parse({
            evaluator: '/evaluator/genkit/context_relevancy',
            score: 1,
          }),
        ]);
        expect(result.traceIds).toHaveLength(1);
      });
    });
  });

  describe('extractMetricSummaries', () => {
    const simpleEvalOutput: Record<string, EvalResponse> = {
      '/evaluator/genkit/context_relevancy': [
        {
          testCaseId: 'case1',
          evaluation: {
            score: 7,
          },
        },
        {
          testCaseId: 'case2',
          evaluation: {
            score: 10,
          },
        },
        {
          testCaseId: 'case3',
          evaluation: {
            score: 5,
          },
        },
      ],
    };

    describe('simpler scenarios', () => {
      it('mean for simple numeric scores', () => {
        const results = extractMetricSummaries(simpleEvalOutput);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 0,
          statusDistribution: { undefined: 3 },
          // 7 + 10 + 5
          averageScore: 22.0 / 3,
        });
      });

      it('scoreDistribution for simple boolean scores', () => {
        const booleanScores = mockScores(simpleEvalOutput, [
          {
            score: true,
          },
          { score: false },
          { score: true },
        ]);
        const results = extractMetricSummaries(booleanScores);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 0,
          statusDistribution: { undefined: 3 },
          // True, False, True
          scoreDistribution: { true: 2, false: 1 },
        });
      });

      it('scoreDistribution for simple string scores (under 5)', () => {
        const stringScores = mockScores(simpleEvalOutput, [
          { score: 'TYPE_0' },
          { score: 'TYPE_1' },
          { score: 'TYPE_0' },
        ]);
        const results = extractMetricSummaries(stringScores);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 0,
          statusDistribution: { undefined: 3 },
          // TYPE_0, TYPE_1, TYPE_0
          scoreDistribution: { TYPE_0: 2, TYPE_1: 1 },
        });
      });

      it('scoreDistribution for simple string scores (over 5)', () => {
        const extendedSimpleEvalOutput: Record<string, EvalResponse> = {};
        // 2x the simpleEvalOutput to get 6 samples.
        extendedSimpleEvalOutput['/evaluator/genkit/context_relevancy'] = Array(
          2
        )
          .fill(simpleEvalOutput['/evaluator/genkit/context_relevancy'])
          .flat();

        const stringScores = mockScores(extendedSimpleEvalOutput, [
          { score: 'TYPE_0' },
          { score: 'TYPE_1' },
          { score: 'TYPE_2' },
          { score: 'TYPE_3' },
          { score: 'TYPE_4' },
          { score: 'TYPE_5' },
        ]);
        const results = extractMetricSummaries(stringScores);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 6,
          errorCount: 0,
          scoreUndefinedCount: 0,
          statusDistribution: { undefined: 6 },
        });
      });

      it('status distribution for simple numeric scores', () => {
        const scores = [
          {
            score: 0,
            status: EvalStatusEnum.PASS,
          },
          {
            score: 1,
            status: EvalStatusEnum.FAIL,
          },
          {
            score: 2,
          },
        ];
        const withStatus = mockScores(simpleEvalOutput, scores);
        const results = extractMetricSummaries(withStatus);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 0,
          statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          // avg(0, 1, 2)
          averageScore: 3.0 / 3,
        });
      });
    });

    describe('edge cases', () => {
      it('metrics if scores are undefined but status available', () => {
        const scores = [
          {
            status: EvalStatusEnum.PASS,
          },
          {
            status: EvalStatusEnum.FAIL,
          },
          {},
        ];
        const undefinedScores = mockScores(simpleEvalOutput, scores);
        const results = extractMetricSummaries(undefinedScores);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 3,
          // PASS, FAIL, undefined
          statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
        });
      });

      it('metrics if some scores are undefined with status available', () => {
        const scores = [
          {
            score: 0,
            status: EvalStatusEnum.PASS,
          },
          {
            score: 1,
            status: EvalStatusEnum.FAIL,
          },
          {},
        ];
        const someDefinedScores = mockScores(simpleEvalOutput, scores);
        const results = extractMetricSummaries(someDefinedScores);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 1,
          // avg(0, 1)
          averageScore: 1 / 2.0,
          // PASS, FAIL, undefined
          statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
        });
      });

      it('metrics if some scores are undefined, some errors and with status available', () => {
        const scores = [
          {
            score: undefined,
            error: 'some error',
            status: EvalStatusEnum.PASS,
          },
          {
            score: 1,
            status: EvalStatusEnum.FAIL,
          },
          { error: 'some error' },
        ];
        const someDefinedScores = mockScores(simpleEvalOutput, scores);
        const results = extractMetricSummaries(someDefinedScores);

        expect(results).toHaveLength(1);

        const result = results[0];
        expect(result).toEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 2,
          scoreUndefinedCount: 2,
          // avg(1)
          averageScore: 1.0,
          // PASS, FAIL, undefined
          statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
        });
      });
    });

    describe('multiple evaluators grouped', () => {
      const multiOutput: Record<string, EvalResponse> = {
        '/evaluator/genkit/faithfulness': [
          {
            testCaseId: 'case1',
            evaluation: {
              score: 7,
            },
          },
          {
            testCaseId: 'case2',
            evaluation: {
              score: 10,
            },
          },
          {
            testCaseId: 'case3',
            evaluation: {
              score: 5,
            },
          },
        ],
        '/evaluator/genkit/context_relevancy': [
          {
            testCaseId: 'case1',
            evaluation: {
              score: true,
            },
          },
          {
            testCaseId: 'case2',
            evaluation: {
              score: false,
            },
          },
          {
            testCaseId: 'case3',
            evaluation: {
              score: true,
            },
          },
        ],
      };

      it('treats each evaluator separately', () => {
        const results = extractMetricSummaries(multiOutput);

        expect(results).toHaveLength(2);
        expect(results).toContainEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 0,
          statusDistribution: { undefined: 3 },
          // true, false, true
          scoreDistribution: { true: 2, false: 1 },
        });
        expect(results).toContainEqual({
          evaluator: '/evaluator/genkit/faithfulness',
          testCaseCount: 3,
          errorCount: 0,
          scoreUndefinedCount: 0,
          statusDistribution: { undefined: 3 },
          // avg(7, 10, 5)
          averageScore: 22.0 / 3,
        });
      });

      it('treats each evaluator separately, with errors, status, undefined scores', () => {
        const mockFaithfulness = [
          {
            status: EvalStatusEnum.PASS,
            error: 'some error',
          },
          {
            score: 10,
            status: EvalStatusEnum.FAIL,
          },
          {
            error: 'some error',
          },
        ];
        const mockContextRel = [
          {
            score: 'alpha',
            status: EvalStatusEnum.PASS,
          },
          {
            status: EvalStatusEnum.FAIL,
            error: 'some error',
          },
          {
            score: 'gamma',
          },
        ];
        const someDefinedScores = reMapScores(
          multiOutput,
          (response, i, evaluator) => {
            if (evaluator === '/evaluator/genkit/faithfulness') {
              return {
                testCaseId: response.testCaseId,
                evaluation: mockFaithfulness[i],
              };
            } else {
              return {
                testCaseId: response.testCaseId,
                evaluation: mockContextRel[i],
              };
            }
          }
        );
        const results = extractMetricSummaries(someDefinedScores);
        expect(results).toHaveLength(2);
        expect(results).toContainEqual({
          evaluator: '/evaluator/genkit/faithfulness',
          testCaseCount: 3,
          errorCount: 2,
          scoreUndefinedCount: 2,
          statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          // avg(10)
          averageScore: 10.0,
        });
        expect(results).toContainEqual({
          evaluator: '/evaluator/genkit/context_relevancy',
          testCaseCount: 3,
          errorCount: 1,
          scoreUndefinedCount: 1,
          statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          // alpha, gamma
          scoreDistribution: { alpha: 1, gamma: 1 },
        });
      });

      describe('multi-scores', () => {
        it('mix of scores', () => {
          const mockEvaluations = [
            {
              score: 1,
              status: EvalStatusEnum.PASS,
            },
            [
              {
                score: 1,
                status: EvalStatusEnum.FAIL,
              },
              {
                score: 2,
                status: EvalStatusEnum.PASS,
              },
            ],
            {
              score: undefined,
            },
          ];
          const mixedScores = reMapScores(simpleEvalOutput, (response, i) => ({
            testCaseId: response.testCaseId,
            evaluation: mockEvaluations[i],
          }));
          const results = extractMetricSummaries(mixedScores);

          expect(results).toHaveLength(1);

          const result = results[0];
          expect(result).toEqual({
            evaluator: '/evaluator/genkit/context_relevancy',
            testCaseCount: 3,
            errorCount: 0,
            scoreUndefinedCount: 1,
            // avg(1, 1, 2)
            averageScore: 4.0 / 3,
            // PASS, FAIL, PASS, undefined
            statusDistribution: { undefined: 1, PASS: 2, FAIL: 1 },
          });
        });

        it('scores with IDs', () => {
          const mockEvaluations = [
            [
              {
                score: 5,
                id: 'numeric',
                status: EvalStatusEnum.PASS,
              },
              {
                score: 'YES',
                id: 'enum',
                status: EvalStatusEnum.FAIL,
              },
            ],
            [
              {
                score: 7,
                id: 'numeric',
                status: EvalStatusEnum.FAIL,
              },
              {
                score: 'NO',
                id: 'enum',
                status: EvalStatusEnum.PASS,
              },
            ],
            [
              {
                score: undefined,
                id: 'numeric',
                error: 'somer error',
              },
              {
                score: undefined,
                id: 'enum',
              },
            ],
          ];
          const mixedScores = reMapScores(simpleEvalOutput, (response, i) => ({
            testCaseId: response.testCaseId,
            evaluation: mockEvaluations[i],
          }));
          const results = extractMetricSummaries(mixedScores);

          expect(results).toHaveLength(2);
          expect(results).toContainEqual({
            evaluator: '/evaluator/genkit/context_relevancy/numeric',
            testCaseCount: 3,
            errorCount: 1,
            scoreUndefinedCount: 1,
            // avg(5, 7)
            averageScore: 12.0 / 2,
            // PASS, FAIL, undefined
            statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          });
          expect(results).toContainEqual({
            evaluator: '/evaluator/genkit/context_relevancy/enum',
            testCaseCount: 3,
            errorCount: 0,
            scoreUndefinedCount: 1,
            // YES, NO
            scoreDistribution: { YES: 1, NO: 1 },
            // FAIL, PASS, undefined
            statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          });
        });

        it('multi-scores with IDs', () => {
          const mockEvaluations = [
            [
              {
                score: 5,
                id: 'numeric',
                status: EvalStatusEnum.PASS,
              },
              {
                score: 'YES',
                id: 'enum',
                status: EvalStatusEnum.FAIL,
              },
            ],
            [
              {
                score: 7,
                id: 'numeric',
                status: EvalStatusEnum.FAIL,
              },
              {
                score: 'NO',
                id: 'enum',
                status: EvalStatusEnum.PASS,
              },
            ],
            [
              {
                score: undefined,
                id: 'numeric',
                error: 'somer error',
              },
              {
                score: undefined,
                id: 'enum',
              },
            ],
          ];
          const mixedScores = reMapScores(multiOutput, (response, i) => ({
            testCaseId: response.testCaseId,
            evaluation: mockEvaluations[i],
          }));
          const results = extractMetricSummaries(mixedScores);

          expect(results).toHaveLength(4);
          expect(results).toContainEqual({
            evaluator: '/evaluator/genkit/context_relevancy/numeric',
            testCaseCount: 3,
            errorCount: 1,
            scoreUndefinedCount: 1,
            // avg(5, 7)
            averageScore: 12.0 / 2,
            // PASS, FAIL, undefined
            statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          });
          expect(results).toContainEqual({
            evaluator: '/evaluator/genkit/context_relevancy/enum',
            testCaseCount: 3,
            errorCount: 0,
            scoreUndefinedCount: 1,
            // YES, NO
            scoreDistribution: { YES: 1, NO: 1 },
            // FAIL, PASS, undefined
            statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          });
          expect(results).toContainEqual({
            evaluator: '/evaluator/genkit/faithfulness/numeric',
            testCaseCount: 3,
            errorCount: 1,
            scoreUndefinedCount: 1,
            // avg(5, 7)
            averageScore: 12.0 / 2,
            // PASS, FAIL, undefined
            statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          });
          expect(results).toContainEqual({
            evaluator: '/evaluator/genkit/faithfulness/enum',
            testCaseCount: 3,
            errorCount: 0,
            scoreUndefinedCount: 1,
            // YES, NO
            scoreDistribution: { YES: 1, NO: 1 },
            // FAIL, PASS, undefined
            statusDistribution: { undefined: 1, PASS: 1, FAIL: 1 },
          });
        });
      });
    });
  });
});

function reMapScores(
  scoresMap: Record<string, EvalResponse>,
  fn: (
    score: EvalFnResponse,
    index: number,
    evaluator?: string
  ) => EvalFnResponse
): Record<string, EvalResponse> {
  const remapped: Record<string, EvalResponse> = {};

  for (const [evaluator, scores] of Object.entries(scoresMap)) {
    remapped[evaluator] = scores.map((score, index) =>
      fn(score, index, evaluator)
    );
  }
  return remapped;
}

function mockScores(
  scoresMap: Record<string, EvalResponse>,
  mockedScores: any[] | Record<string, any[]>
): Record<string, EvalResponse> {
  const remapped: Record<string, EvalResponse> = {};

  for (const [evaluator, scores] of Object.entries(scoresMap)) {
    remapped[evaluator] = scores.map((score, index) => {
      const evaluation = Array.isArray(mockedScores)
        ? { ...mockedScores[index] }
        : mockedScores[evaluator][index];
      return {
        testCaseId: score.testCaseId,
        evaluation,
      };
    });
  }
  return remapped;
}
