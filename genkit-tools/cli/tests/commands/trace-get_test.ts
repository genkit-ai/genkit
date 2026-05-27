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
import { traceGet } from '../../src/commands/trace-get';
import { runWithManager } from '../../src/utils/manager-utils';

jest.mock('@genkit-ai/tools-common/utils');
jest.mock('../../src/utils/manager-utils');

const mockedFindProjectRoot = findProjectRoot as jest.MockedFunction<
  typeof findProjectRoot
>;
const mockedLogger = logger as jest.Mocked<typeof logger>;
const mockedRunWithManager = runWithManager as jest.MockedFunction<
  typeof runWithManager
>;

describe('trace:get', () => {
  const createCommand = () =>
    traceGet.exitOverride().configureOutput({
      writeOut: () => {},
      writeErr: () => {},
    });

  const mockProjectRoot = '/mock/project/root';

  beforeEach(() => {
    jest.clearAllMocks();
    mockedFindProjectRoot.mockResolvedValue(mockProjectRoot);

    // Mock console.log to avoid spamming test output
    jest.spyOn(console, 'log').mockImplementation(() => {});

    // Provide a default implementation for runWithManager
    mockedRunWithManager.mockImplementation(async (projectRoot, fn) => {
      // Simulate calling the action with a mocked manager
      const mockManager = {
        getTrace: jest
          .fn<any>()
          .mockResolvedValue({ traceId: 'test-id', details: 'mock-trace' }),
      };
      await fn(mockManager as any);
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should get and print trace details', async () => {
    await createCommand().parseAsync(['node', 'trace:get', 'test-trace-id']);

    expect(mockedFindProjectRoot).toHaveBeenCalled();
    expect(mockedRunWithManager).toHaveBeenCalled();

    // Verify console.log was called with stringified trace
    expect(console.log).toHaveBeenCalledWith(
      JSON.stringify(
        { traceId: 'test-id', details: 'mock-trace' },
        undefined,
        2
      )
    );
  });

  it('should handle trace not found', async () => {
    mockedRunWithManager.mockImplementation(async (projectRoot, fn) => {
      const mockManager = {
        getTrace: jest.fn<any>().mockResolvedValue(undefined), // Simulate trace not found
      };
      await fn(mockManager as any);
    });

    await createCommand().parseAsync(['node', 'trace:get', 'missing-trace-id']);

    expect(mockedLogger.error).toHaveBeenCalledWith(
      "Trace with ID 'missing-trace-id' not found."
    );
  });

  it('should handle errors thrown by getTrace', async () => {
    const errorMsg = 'Failed to connect to telemetry server';
    mockedRunWithManager.mockImplementation(async (projectRoot, fn) => {
      const mockManager = {
        getTrace: jest.fn<any>().mockRejectedValue(new Error(errorMsg)),
      };
      await fn(mockManager as any);
    });

    await createCommand().parseAsync(['node', 'trace:get', 'error-trace-id']);

    expect(mockedLogger.error).toHaveBeenCalledWith(
      `Error retrieving trace: Error: ${errorMsg}`
    );
  });
});
