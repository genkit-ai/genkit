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
  findProjectRoot,
  logger,
  stackTraceSpans,
} from '@genkit-ai/tools-common/utils';
import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { traceList } from '../../src/commands/trace-list';
import { runWithManager } from '../../src/utils/manager-utils';

jest.mock('@genkit-ai/tools-common/utils');
jest.mock('../../src/utils/manager-utils');

describe('trace:list command', () => {
  let mockManager: any;

  beforeEach(() => {
    jest.clearAllMocks();
    traceList.setOptionValue('limit', 15);
    traceList.setOptionValue('status', undefined);
    traceList.setOptionValue('type', undefined);
    traceList.setOptionValue('name', undefined);
    traceList.setOptionValue('continuationToken', undefined);

    mockManager = {
      listTraces: jest.fn(),
    };

    (findProjectRoot as jest.Mock<any>).mockResolvedValue('/mock/project/root');

    (runWithManager as jest.Mock<any>).mockImplementation(
      async (projectRoot: any, action: any) => {
        await action(mockManager);
      }
    );

    jest.spyOn(logger, 'info').mockImplementation((() => {}) as any);
    jest.spyOn(logger, 'error').mockImplementation((() => {}) as any);
    jest.spyOn(console, 'log').mockImplementation(() => {});
  });

  it('should list traces with default limit', async () => {
    mockManager.listTraces.mockResolvedValue({
      traces: [
        {
          traceId: 'trace-1',
          displayName: 'Test Trace 1',
          startTime: 1000,
          endTime: 2000,
        },
      ],
    });

    (stackTraceSpans as jest.Mock<any>).mockReturnValue({
      attributes: {
        'genkit:state': 'success',
        'genkit:input': 'my input string',
        'genkit:output': 'my output string',
      },
    });

    await traceList.parseAsync(['node', 'trace:list']);

    expect(findProjectRoot).toHaveBeenCalled();
    expect(runWithManager).toHaveBeenCalled();
    expect(mockManager.listTraces).toHaveBeenCalledWith({
      limit: 15,
      continuationToken: undefined,
      filter: {
        eq: undefined,
        neq: { 'genkitx:ignore-trace': ['true'] },
      },
    });
    expect(console.log).toHaveBeenCalledWith('Found 1 trace:\n');
    expect(console.log).toHaveBeenCalledWith('ID:       trace-1');
    expect(console.log).toHaveBeenCalledWith('Status:   success');
    expect(console.log).toHaveBeenCalledWith('Input:    my input string');
    expect(console.log).toHaveBeenCalledWith('Output:   my output string');
  });

  it('should list traces with filters and custom limit', async () => {
    mockManager.listTraces.mockResolvedValue({
      traces: [
        {
          traceId: 'trace-2',
          displayName: 'Error Trace',
          startTime: 1000,
        },
      ],
    });

    await traceList.parseAsync([
      'node',
      'trace:list',
      '--limit',
      '5',
      '--status',
      'error',
      '--type',
      'Flow',
      '--name',
      'myFlow',
    ]);

    expect(mockManager.listTraces).toHaveBeenCalledWith({
      limit: 5,
      continuationToken: undefined,
      filter: {
        eq: {
          status: [2],
          type: ['Flow'],
          name: ['myFlow'],
        },
        neq: { 'genkitx:ignore-trace': ['true'] },
      },
    });
  });

  it('should handle pagination with continuation token', async () => {
    mockManager.listTraces.mockResolvedValue({
      traces: [
        {
          traceId: 'trace-3',
        },
      ],
      continuationToken: 'next-page-token',
    });

    await traceList.parseAsync([
      'node',
      'trace:list',
      '--continuation-token',
      'some-token',
    ]);

    expect(mockManager.listTraces).toHaveBeenCalledWith({
      limit: 15,
      continuationToken: 'some-token',
      filter: {
        eq: undefined,
        neq: { 'genkitx:ignore-trace': ['true'] },
      },
    });
    expect(console.log).toHaveBeenCalledWith(
      '\nTo get the next page, use: --continuation-token next-page-token'
    );
  });

  it('should log info when no traces are found', async () => {
    mockManager.listTraces.mockResolvedValue({
      traces: [],
    });

    await traceList.parseAsync(['node', 'trace:list']);

    expect(logger.info).toHaveBeenCalledWith('No traces found.');
  });

  it('should handle and log errors', async () => {
    mockManager.listTraces.mockRejectedValue(new Error('API failure'));

    await traceList.parseAsync(['node', 'trace:list']);

    expect(logger.error).toHaveBeenCalledWith(
      expect.stringContaining('Error listing traces: Error: API failure')
    );
  });

  it.each(['0', '-1', '1.5', '10abc'])(
    'should reject invalid limit "%s" before starting manager',
    async (limit) => {
      const cmd = traceList
        .exitOverride()
        .configureOutput({ writeOut: () => {}, writeErr: () => {} });
      await expect(
        cmd.parseAsync(['node', 'trace:list', '--limit', limit])
      ).rejects.toThrow(
        /option '-l, --limit <number>' argument '.*' is invalid/
      );
      expect(runWithManager).not.toHaveBeenCalled();
    }
  );

  it.each(['invalid-status', '1.5', ' ', '-1'])(
    'should reject invalid status "%s" before starting manager',
    async (status) => {
      const cmd = traceList
        .exitOverride()
        .configureOutput({ writeOut: () => {}, writeErr: () => {} });
      await expect(
        cmd.parseAsync(['node', 'trace:list', '--status', status])
      ).rejects.toThrow(/option '--status <status>' argument '.*' is invalid/);
      expect(runWithManager).not.toHaveBeenCalled();
    }
  );
});
