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

import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import { flowBatchRun } from '../../src/commands/flow-batch-run';
import { flowRun } from '../../src/commands/flow-run';
import { runWithManager } from '../../src/utils/manager-utils';

jest.mock('@genkit-ai/tools-common/utils');
jest.mock('../../src/utils/manager-utils');

describe('flow:run and flow:batchRun validation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    process.exitCode = undefined;
    (findProjectRoot as jest.Mock<any>).mockResolvedValue('/mock/project/root');
    jest.spyOn(logger, 'error').mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    process.exitCode = undefined;
  });

  it('rejects invalid --context JSON in flow:run before starting manager', async () => {
    const cmd = flowRun
      .exitOverride()
      .configureOutput({ writeOut: () => {}, writeErr: () => {} });
    await expect(
      cmd.parseAsync(['node', 'flow:run', 'myFlow', '-c', '{not-json}'])
    ).rejects.toThrow(
      /option '-c, --context <JSON>' argument '\{not-json\}' is invalid/
    );
    expect(runWithManager).not.toHaveBeenCalled();
  });

  it('rejects invalid [data] JSON in flow:run before starting manager', async () => {
    await flowRun.parseAsync(['node', 'flow:run', 'myFlow', '{not-json}']);
    expect(logger.error).toHaveBeenCalledWith(
      expect.stringContaining('Invalid JSON in [data]:')
    );
    expect(process.exitCode).toBe(1);
    expect(runWithManager).not.toHaveBeenCalled();
  });

  it('does not treat "-- <command>" as [data] when [data] is omitted', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'genkit',
      'flow:run',
      'myFlow',
      '--',
      'npm',
      'start',
    ];
    try {
      await flowRun.parseAsync([
        'node',
        'flow:run',
        'myFlow',
        '--',
        'npm',
        'start',
      ]);
      expect(logger.error).not.toHaveBeenCalled();
      expect(runWithManager).toHaveBeenCalledWith(
        '/mock/project/root',
        expect.any(Function),
        expect.objectContaining({
          runtimeCommand: ['npm', 'start'],
        })
      );
    } finally {
      process.argv = origArgv;
    }
  });

  it('rejects invalid --context JSON in flow:batchRun before starting manager', async () => {
    const cmd = flowBatchRun
      .exitOverride()
      .configureOutput({ writeOut: () => {}, writeErr: () => {} });
    await expect(
      cmd.parseAsync([
        'node',
        'flow:batchRun',
        'myFlow',
        'input.json',
        '-c',
        '{not-json}',
      ])
    ).rejects.toThrow(
      /option '-c, --context <JSON>' argument '\{not-json\}' is invalid/
    );
    expect(runWithManager).not.toHaveBeenCalled();
  });
});
