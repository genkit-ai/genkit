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

import fs from 'fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { toCsv, toJson } from '../../src/eval/exporter';
import type { EvalResult, EvalRun, EvalRunKey } from '../../src/types/eval';

vi.mock('fs/promises', () => ({
  get writeFile() {
    return fs.promises.writeFile;
  },
}));

vi.mock('crypto', () => {
  return {
    createHash: vi.fn().mockReturnThis(),
    update: vi.fn().mockReturnThis(),
    digest: vi.fn(() => 'store-root'),
  };
});

const EVAL_RESULTS: EvalResult[] = [
  {
    testCaseId: 'alakjdshfalsdkjh',
    input: { subject: 'something', style: 'structured' },
    output: { output: 'output', other: 'other' },
    context: ['context1', 'context2'],
    metrics: [
      {
        evaluator: 'faithfulness',
        score: 0.5,
        rationale: 'somewhat faithful',
        traceId: '123',
        spanId: '456',
      },
      {
        evaluator: 'answer_relevancy',
        error: 'errored\nwith\nreturns\n',
      },
    ],
    traceIds: ['abc123', 'defhij'],
    reference: { structured: 'structured', output: 'output', other: 'other' },
  },
  {
    testCaseId: 'poqiweurqwepru',
    input: 'This is just a string',
    output: 'This is also just a string',
    context: [],
    metrics: [
      {
        evaluator: 'faithfulness',
        score: 0,
        rationale:
          'The provided context does not mention typical cat behaviors, so I cannot answer this question from the provided context.',
        traceId: '789',
        spanId: '101',
      },
      {
        evaluator: 'answer_relevancy',
        error: 'errored, with a comma',
      },
    ],
    traceIds: [],
  },
];

const EVAL_RUN_KEY: EvalRunKey = {
  actionRef: 'flow/myAwesomeFlow',
  evalRunId: 'abc1234',
  createdAt: new Date().toISOString(),
};

const EVAL_RUN: EvalRun = {
  key: EVAL_RUN_KEY,
  results: EVAL_RESULTS,
};

const CSV_OUTPUT_FILE = '/tmp/myAwesomeOutput.csv';
const JSON_OUTPUT_FILE = '/tmp/myAwesomeOutput.json';

describe('exporter', () => {
  beforeEach(() => {
    fs.promises.writeFile = vi.fn(async () => Promise.resolve(undefined));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('toCsv', () => {
    it('should handle vanilla strings', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: 'input',
          output: 'output',
          context: [],
          metrics: [
            {
              evaluator: 'faithfulness',
              score: 0.5,
              rationale: 'faithful',
              traceId: '123',
              spanId: '456',
            },
          ],
          traceIds: [],
        },
      ];
      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId';
      const expectedRecord = `testCase1,input,output,[],[],0.5,faithful,,123,456`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });

    it('should unnest metrics', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: 'input',
          output: 'output',
          context: [],
          metrics: [
            {
              evaluator: 'faithfulness',
              score: 0.5,
              rationale: 'faithful',
              traceId: '123',
              spanId: '456',
            },
            {
              evaluator: 'answer_relevancy',
              score: 1.0,
              rationale: 'relevant',
              traceId: '789',
              spanId: '101',
            },
          ],
          traceIds: [],
        },
      ];

      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId,answer_relevancy_score,answer_relevancy_rationale,answer_relevancy_error,answer_relevancy_traceId,answer_relevancy_spanId';
      const expectedRecord = `testCase1,input,output,[],[],0.5,faithful,,123,456,1,relevant,,789,101`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });

    it('should handle errors in metrics', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: 'input',
          output: 'output',
          context: [],
          metrics: [
            {
              evaluator: 'faithfulness',
              error: 'This is an error!',
            },
          ],
          traceIds: [],
        },
      ];

      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId';
      const expectedRecord = `testCase1,input,output,[],[],,,This is an error!,,`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });

    it('should stringify structured input', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: { subject: 'something', style: 'structured' },
          output: 'output',
          context: [],
          metrics: [
            {
              evaluator: 'faithfulness',
              score: 0.5,
              rationale: 'faithful',
              traceId: '123',
              spanId: '456',
            },
          ],
          traceIds: [],
        },
      ];

      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId';
      const expectedRecord = `testCase1,\"{\"\"subject\"\":\"\"something\"\",\"\"style\"\":\"\"structured\"\"}\",output,[],[],0.5,faithful,,123,456`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });

    it('should handle carriage returns', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: 'input',
          output: 'output',
          context: [],
          metrics: [
            {
              evaluator: 'faithfulness',
              error: 'errored\nwith\na\ncarriage return',
            },
          ],
          traceIds: [],
        },
      ];

      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId';
      const expectedRecord = `testCase1,input,output,[],[],,,\"errored
with
a
carriage return\",,`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });

    it('should handle context and trace arrays', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: 'input',
          output: 'output',
          context: ['context1', 'context2'],
          metrics: [
            {
              evaluator: 'faithfulness',
              score: 0.5,
              rationale: 'faithful',
              traceId: '123',
              spanId: '456',
            },
          ],
          traceIds: ['trace1', 'trace2'],
        },
      ];

      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId';
      const expectedRecord = `testCase1,input,output,\"[\"\"context1\"\",\"\"context2\"\"]\",\"[\"\"trace1\"\",\"\"trace2\"\"]\",0.5,faithful,,123,456`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });

    it('should handle commas in strings', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: 'input, with, extra, commas',
          output: 'output',
          context: [],
          metrics: [
            {
              evaluator: 'faithfulness',
              score: 0.5,
              rationale: 'faithful',
              traceId: '123',
              spanId: '456',
            },
          ],
          traceIds: [],
        },
      ];

      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId';
      const expectedRecord = `testCase1,\"input, with, extra, commas\",output,[],[],0.5,faithful,,123,456`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });

    it('should include ground truth', () => {
      const evalResults: EvalResult[] = [
        {
          testCaseId: 'testCase1',
          input: 'input',
          output: 'output',
          context: [],
          metrics: [
            {
              evaluator: 'faithfulness',
              score: 0.5,
              rationale: 'faithful',
              traceId: '123',
              spanId: '456',
            },
          ],
          traceIds: [],
          reference: 'This is the honest truth',
        },
      ];

      toCsv({ key: EVAL_RUN_KEY, results: evalResults }, CSV_OUTPUT_FILE);

      const expectedHeader =
        'testCaseId,input,output,context,traceIds,reference,faithfulness_score,faithfulness_rationale,faithfulness_error,faithfulness_traceId,faithfulness_spanId';
      const expectedRecord = `testCase1,input,output,[],[],This is the honest truth,0.5,faithful,,123,456`;
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        CSV_OUTPUT_FILE,
        `${expectedHeader}\n${expectedRecord}`
      );
    });
  });

  describe('toJson', () => {
    it('should write json string', () => {
      toJson(EVAL_RUN, JSON_OUTPUT_FILE);
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        JSON_OUTPUT_FILE,
        JSON.stringify(EVAL_RESULTS, undefined, '  ')
      );
    });
  });
});
