/**
 * Copyright 2026 Google LLC
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
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import Ajv from 'ajv';
import type { JSONSchema7 } from 'json-schema';
import { runEvaluation } from '../../src/eval/evaluate';
import { LocalFileEvalStore } from '../../src/eval/localFileEvalStore';
import type { BaseRuntimeManager } from '../../src/manager/manager';
import type { Action } from '../../src/types';

const dataset = [
  { testCaseId: 'case-1', input: 'input', output: 'output', traceIds: [] },
];

function evaluatorSchema(options: JSONSchema7, required = true): JSONSchema7 {
  return {
    type: 'object',
    properties: {
      dataset: { type: 'array' },
      evalRunId: { type: 'string' },
      options,
      batchSize: { type: 'number' },
    },
    required: ['dataset', 'evalRunId', ...(required ? ['options'] : [])],
  };
}

function evaluation(inputSchema?: JSONSchema7) {
  const validate = new Ajv().compile(inputSchema ?? {});
  const runAction = jest.fn<BaseRuntimeManager['runAction']>(
    async (request) => {
      if (!validate(request.input)) {
        throw new Error(JSON.stringify(validate.errors));
      }
      return {
        result: [{ testCaseId: 'case-1', evaluation: { score: 1 } }],
        telemetry: { traceId: 'trace-1' },
      };
    }
  );
  const manager = {
    getMostRecentRuntime: () => ({ genkitVersion: 'nodejs/1.43.0' }),
    runAction,
  } as unknown as BaseRuntimeManager;
  const action: Action = {
    key: '/evaluator/test',
    name: 'test',
    inputSchema,
    metadata: { evaluator: {} },
  };
  return {
    run: () =>
      runEvaluation({
        manager,
        evaluatorActions: [action],
        evalDataset: dataset,
      }),
    input: () => runAction.mock.calls[0][0].input,
  };
}

describe('runEvaluation evaluator options', () => {
  beforeEach(() => {
    jest.spyOn(LocalFileEvalStore, 'getEvalStore').mockResolvedValue({
      save: jest.fn(async () => undefined),
    } as unknown as LocalFileEvalStore);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('runs an evaluator whose required object options have only optional fields', async () => {
    const evaluator = evaluation(
      evaluatorSchema({
        type: 'object',
        properties: { judgeModel: { type: 'string' } },
        additionalProperties: false,
      })
    );

    const result = await evaluator.run();

    expect(evaluator.input()).toHaveProperty('options', {});
    expect(result.results[0].metrics?.[0].score).toBe(1);
  });

  it('still rejects missing required configuration fields', async () => {
    const evaluator = evaluation(
      evaluatorSchema({
        type: 'object',
        properties: { judgeModel: { type: 'string' } },
        required: ['judgeModel'],
      })
    );

    await expect(evaluator.run()).rejects.toThrow('"instancePath":"/options"');
    expect(evaluator.input()).toHaveProperty('options', {});
  });

  it.each<[string, JSONSchema7 | undefined]>([
    ['no input schema', undefined],
    ['no configuration schema', evaluatorSchema({}, false)],
    ['optional object options', evaluatorSchema({ type: 'object' }, false)],
    [
      'a top-level object default',
      evaluatorSchema(
        {
          type: 'object',
          properties: { judgeModel: { type: 'string' } },
          default: { judgeModel: 'default-model' },
        },
        false
      ),
    ],
  ])('keeps options omitted for %s', async (_name, inputSchema) => {
    const evaluator = evaluation(inputSchema);

    await evaluator.run();

    expect(Object.hasOwn(evaluator.input(), 'options')).toBe(false);
  });

  it.each<[string, JSONSchema7]>([
    ['scalar', { type: 'string' }],
    ['array', { type: 'array', items: { type: 'string' } }],
  ])('does not invent %s configuration', async (_name, options) => {
    const evaluator = evaluation(evaluatorSchema(options));

    await expect(evaluator.run()).rejects.toThrow(
      '"missingProperty":"options"'
    );

    expect(Object.hasOwn(evaluator.input(), 'options')).toBe(false);
  });

  it.each([
    [
      'non-array required',
      { required: {}, properties: { options: { type: 'object' } } },
    ],
    [
      'null options schema',
      { required: ['options'], properties: { options: null } },
    ],
  ])(
    'defers malformed reflection metadata with %s to the runtime',
    async (_name, inputSchema) => {
      const runtimeError = new Error('Runtime rejected the evaluator request');
      const runAction = jest
        .fn<BaseRuntimeManager['runAction']>()
        .mockRejectedValue(runtimeError);
      const manager = {
        getMostRecentRuntime: () => ({ genkitVersion: 'nodejs/1.43.0' }),
        runAction,
      } as unknown as BaseRuntimeManager;
      const action: Action = {
        key: '/evaluator/test',
        name: 'test',
        inputSchema,
        metadata: { evaluator: {} },
      };

      await expect(
        runEvaluation({
          manager,
          evaluatorActions: [action],
          evalDataset: dataset,
        })
      ).rejects.toBe(runtimeError);

      expect(runAction).toHaveBeenCalledTimes(1);
      expect(Object.hasOwn(runAction.mock.calls[0][0].input, 'options')).toBe(
        false
      );
    }
  );
});
