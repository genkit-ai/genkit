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

import { describe, expect, it, jest } from '@jest/globals';

// Mock utils used inside evaluate.ts to avoid touching real traces/config.
jest.mock('../../src/utils', () => {
  const logger = {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  };
  return {
    evaluatorName: (action: any) => `/evaluator/${action.name}`,
    generateTestCaseId: () => 'test-case',
    getEvalExtractors: jest.fn(async () => ({
      input: (trace: any) => trace.mockInput,
      output: (trace: any) => trace.mockOutput,
      context: () => [],
    })),
    getModelInput: (data: any) => data,
    hasAction: jest.fn(() => Promise.resolve(true)),
    isEvaluator: (key: string) => key.startsWith('/evaluator'),
    logger,
    stackTraceSpans: jest.fn(() => ({ attributes: {}, spans: [] })),
  };
});

jest.mock('../../src/eval', () => ({
  getEvalStore: jest.fn(() =>
    Promise.resolve({ save: jest.fn(() => Promise.resolve()) })
  ),
  getDatasetStore: jest.fn(),
}));

jest.mock('../../src/eval/parser', () => ({
  enrichResultsWithScoring: (_scores: any, dataset: any) => dataset,
  extractMetricSummaries: () => ({}),
  extractMetricsMetadata: () => ({}),
}));

import type { Action, EvalInput } from '../../src/types';
import { bulkRunAction, runEvaluation } from '../../src/eval/evaluate';

function createMockManager() {
  return {
    runAction: jest.fn((..._args: any[]) => Promise.resolve({})),
    getTrace: jest.fn((..._args: any[]) => Promise.resolve({})),
    getMostRecentRuntime: jest.fn(() => ({ genkitVersion: 'nodejs-1.0' })),
  };
}

function createAction(name: string): Action {
  return {
    key: `/evaluator/${name}`,
    name,
    description: '',
    inputSchema: null,
    outputSchema: null,
    metadata: null,
  };
}

describe('bulkRunAction', () => {
  it('runs samples in batches respecting batchSize', async () => {
    const manager = createMockManager();
    const delayMs = 40;
    manager.runAction.mockImplementation(async (_req: any) => {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      return {
        result: 'ok',
        telemetry: { traceId: 'trace' },
      };
    });
    manager.getTrace.mockResolvedValue({
      spans: {},
      mockInput: 'input',
      mockOutput: 'output',
    });

    const dataset = Array.from({ length: 4 }, (_, i) => ({
      testCaseId: `case-${i}`,
      input: { value: i },
    }));

    const start = Date.now();
    const results: EvalInput[] = await bulkRunAction({
      manager: manager as any,
      actionRef: '/flow/test',
      inferenceDataset: dataset as any,
      batchSize: 2,
    });
    const duration = Date.now() - start;

    expect(results).toHaveLength(4);
    expect(manager.runAction).toHaveBeenCalledTimes(4);
  }, 15000);

  it('continues processing after an error', async () => {
    const manager = createMockManager();
    manager.runAction
      .mockImplementationOnce(async () => {
        throw new Error('boom');
      })
      .mockImplementation(async () => ({
        result: 'ok',
        telemetry: { traceId: 'trace' },
      }));
    manager.getTrace.mockResolvedValue({
      spans: {},
      mockInput: 'input',
      mockOutput: 'output',
    });

    const dataset = [
      { testCaseId: 'case-1', input: {} },
      { testCaseId: 'case-2', input: {} },
      { testCaseId: 'case-3', input: {} },
    ];

    const results: EvalInput[] = await bulkRunAction({
      manager: manager as any,
      actionRef: '/flow/test',
      inferenceDataset: dataset as any,
      batchSize: 2,
    });

    expect(results).toHaveLength(3);
    expect(results.some((r) => r.error)).toBe(true);
    expect(manager.runAction).toHaveBeenCalledTimes(3);
  }, 10000);

  it('maintains constant concurrency with worker pool', async () => {
    const manager = createMockManager();
    let activeCount = 0;
    let maxActiveCount = 0;
    const delays = [100, 10, 10, 100];
    let callIndex = 0;
    manager.runAction.mockImplementation(async () => {
      activeCount++;
      maxActiveCount = Math.max(maxActiveCount, activeCount);
      const delay = delays[callIndex++] ?? 10;
      await new Promise((resolve) => setTimeout(resolve, delay));
      activeCount--;
      return {
        result: 'ok',
        telemetry: { traceId: 'trace' },
      };
    });
    manager.getTrace.mockResolvedValue({
      spans: {},
      mockInput: 'input',
      mockOutput: 'output',
    });

    const dataset = Array.from({ length: 4 }, (_, i) => ({
      testCaseId: `case-${i}`,
      input: { value: i },
    }));

    const results = await bulkRunAction({
      manager: manager as any,
      actionRef: '/flow/test',
      inferenceDataset: dataset as any,
      batchSize: 2,
    });

    expect(results).toHaveLength(4);
    expect(maxActiveCount).toBeLessThanOrEqual(2);
    expect(manager.runAction).toHaveBeenCalledTimes(4);
  }, 15000);
});

describe('runEvaluation', () => {
  it('executes evaluator actions in parallel', async () => {
    const manager = createMockManager();
    let started = 0;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    manager.runAction.mockImplementation(async () => {
      started++;
      if (started === 2) {
        release();
      }
      await gate;
      return { result: { ok: true }, telemetry: { traceId: 'trace' } };
    });

    const actions = [createAction('a'), createAction('b')];
    const evalDataset: EvalInput[] = [
      { testCaseId: 't1', input: 'in', output: 'out', traceIds: ['trace'] },
    ];

    const evalPromise = runEvaluation({
      manager: manager as any,
      evaluatorActions: actions,
      evalDataset,
    });

    // Give both runAction calls a moment to start and block on the gate.
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(manager.runAction).toHaveBeenCalledTimes(2);

    // Unblock both and finish.
    release();
    await evalPromise;
  });

  it('filters errored samples once before passing to evaluators', async () => {
    const manager = createMockManager();
    const capturedDatasets: any[] = [];
    manager.runAction.mockImplementation(async (req: any) => {
      capturedDatasets.push(req.input.dataset);
      return { result: { ok: true }, telemetry: { traceId: 'trace' } };
    });

    const actions = [createAction('a'), createAction('b')];
    const evalDataset: EvalInput[] = [
      { testCaseId: 't1', input: 'in', output: 'out', traceIds: ['trace'] },
      { testCaseId: 't2', input: 'in', output: 'out', error: 'bad', traceIds: ['trace'] },
      { testCaseId: 't3', input: 'in', output: 'out', traceIds: ['trace'] },
    ];

    await runEvaluation({
      manager: manager as any,
      evaluatorActions: actions,
      evalDataset,
    });

    expect(capturedDatasets).toHaveLength(2);
    const [first, second] = capturedDatasets;
    expect(first).toBe(second);
    expect(first).toHaveLength(2);
    expect(first.every((r: EvalInput) => !r.error)).toBe(true);
  });
});
