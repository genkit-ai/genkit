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
import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { logList } from '../../src/commands/log-list';
import { runWithManager } from '../../src/utils/manager-utils';

jest.mock('@genkit-ai/tools-common/utils');
jest.mock('../../src/utils/manager-utils');

describe('log:list command', () => {
  let mockManager: any;

  beforeEach(() => {
    jest.clearAllMocks();

    mockManager = {
      listLogs: jest.fn(),
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

  it('should list logs with default limit', async () => {
    mockManager.listLogs.mockResolvedValue({
      logs: [
        {
          timestamp: 1718302195000,
          severityText: 'INFO',
          body: 'Hello World',
          attributes: {
            'genkit:name': 'test-log',
          },
        },
      ],
    });

    await logList.parseAsync(['node', 'log:list']);

    expect(findProjectRoot).toHaveBeenCalled();
    expect(runWithManager).toHaveBeenCalled();
    expect(mockManager.listLogs).toHaveBeenCalledWith({
      limit: 15,
      continuationToken: undefined,
      filter: undefined,
    });
    expect(console.log).toHaveBeenCalledWith('Found 1 log:\n');
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining('INFO'));
    expect(console.log).toHaveBeenCalledWith(
      expect.stringContaining('Hello World')
    );
    expect(console.log).toHaveBeenCalledWith(
      expect.stringContaining('Attrs:    genkit:name=test-log')
    );
  });

  it('should handle pagination with continuation token', async () => {
    mockManager.listLogs.mockResolvedValue({
      logs: [
        {
          timestamp: 1718302195000,
          body: 'Another log',
        },
      ],
      continuationToken: 'next-page-token',
    });

    await logList.parseAsync([
      'node',
      'log:list',
      '--limit',
      '15',
      '--severity',
      '',
      '--trace-id',
      '',
      '--span-id',
      '',
      '--continuation-token',
      'some-token',
    ]);

    expect(mockManager.listLogs).toHaveBeenCalledWith({
      limit: 15,
      continuationToken: 'some-token',
      filter: undefined,
    });
    expect(console.log).toHaveBeenCalledWith(
      '\nTo get the next page, use: --continuation-token next-page-token'
    );
  });

  it('should list logs with filters and custom limit', async () => {
    mockManager.listLogs.mockResolvedValue({
      logs: [
        {
          timestamp: 1718302195000,
          severityText: 'ERROR',
          body: 'An error occurred',
        },
      ],
    });

    await logList.parseAsync([
      'node',
      'log:list',
      '--limit',
      '5',
      '--severity',
      'ERROR',
      '--trace-id',
      'trace-1',
      '--span-id',
      'span-1',
      '--continuation-token',
      '',
    ]);

    expect(mockManager.listLogs).toHaveBeenCalledWith({
      limit: 5,
      continuationToken: '',
      filter: {
        severityNumber: 17,
        traceId: 'trace-1',
        spanId: 'span-1',
      },
    });
  });

  it('should output logs in text format by default', async () => {
    mockManager.listLogs.mockResolvedValue({
      logs: [
        {
          logId: 'log-123',
          traceId: 'trace-123',
          spanId: 'span-123',
          timestamp: 1718302195000,
          severityText: 'WARN',
          body: 'A'.repeat(110),
          attributes: { key: 'value' },
        },
      ],
    });

    await logList.parseAsync([
      'node',
      'log:list',
      '--limit',
      '15',
      '--severity',
      '',
      '--trace-id',
      '',
      '--span-id',
      '',
      '--continuation-token',
      '',
    ]);

    expect(console.log).toHaveBeenCalledWith('ID:       log-123');
    expect(console.log).toHaveBeenCalledWith('Trace ID: trace-123');
    expect(console.log).toHaveBeenCalledWith('Span ID:  span-123');
    expect(console.log).toHaveBeenCalledWith('Severity: WARN');
    expect(console.log).toHaveBeenCalledWith(
      `Time:     ${new Date(1718302195000).toLocaleString()}`
    );
    expect(console.log).toHaveBeenCalledWith(`Message:  ${'A'.repeat(100)}...`);
    expect(console.log).toHaveBeenCalledWith('Attrs:    key=value');
  });

  it('should output logs in jsonl format', async () => {
    mockManager.listLogs.mockResolvedValue({
      logs: [
        {
          timestamp: 1718302195000,
          body: 'JSONL log',
        },
      ],
    });

    await logList.parseAsync([
      'node',
      'log:list',
      '--format',
      'jsonl',
      '--limit',
      '15',
      '--severity',
      '',
      '--trace-id',
      '',
      '--span-id',
      '',
      '--continuation-token',
      '',
    ]);

    expect(mockManager.listLogs).toHaveBeenCalledWith({
      limit: 15,
      continuationToken: '',
      filter: undefined,
    });

    expect(console.log).toHaveBeenCalledWith(
      JSON.stringify({ timestamp: 1718302195000, body: 'JSONL log' })
    );
  });

  it('should log info when no logs are found', async () => {
    mockManager.listLogs.mockResolvedValue({
      logs: [],
    });

    await logList.parseAsync(['node', 'log:list']);

    expect(logger.info).toHaveBeenCalledWith('No logs found.');
  });

  it('should handle and log errors', async () => {
    mockManager.listLogs.mockRejectedValue(new Error('API failure'));

    await logList.parseAsync(['node', 'log:list']);

    expect(logger.error).toHaveBeenCalledWith(
      expect.stringContaining('Error listing logs: Error: API failure')
    );
  });
});
