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
  extractTrajectoryFromTrace,
  findProjectRoot,
  forceStderr,
  logger,
  mergeSessionTrajectories,
  parseAndSanitizeJson,
  sanitizeBase64DataUrls,
  stackTraceSpans,
} from '@genkit-ai/tools-common/utils';
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
const mockedForceStderr = forceStderr as jest.MockedFunction<
  typeof forceStderr
>;
const mockedRunWithManager = runWithManager as jest.MockedFunction<
  typeof runWithManager
>;
const {
  stackTraceSpans: actualStackTraceSpans,
  parseAndSanitizeJson: actualParseAndSanitizeJson,
  sanitizeBase64DataUrls: actualSanitizeBase64DataUrls,
  extractTrajectoryFromTrace: actualExtractTrajectoryFromTrace,
  mergeSessionTrajectories: actualMergeSessionTrajectories,
} = jest.requireActual('@genkit-ai/tools-common/utils') as any;

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
    mockedForceStderr.mockImplementation(() => {});
    (stackTraceSpans as jest.MockedFunction<any>).mockImplementation(
      actualStackTraceSpans
    );
    (parseAndSanitizeJson as jest.MockedFunction<any>).mockImplementation(
      actualParseAndSanitizeJson
    );
    (sanitizeBase64DataUrls as jest.MockedFunction<any>).mockImplementation(
      actualSanitizeBase64DataUrls
    );
    (extractTrajectoryFromTrace as jest.MockedFunction<any>).mockImplementation(
      actualExtractTrajectoryFromTrace
    );
    (mergeSessionTrajectories as jest.MockedFunction<any>).mockImplementation(
      actualMergeSessionTrajectories
    );

    // Reset option values on the commander object
    traceGet.setOptionValue('format', undefined);
    traceGet.setOptionValue('keepBase64', undefined);
    traceGet.setOptionValue('trajectory', undefined);
    traceGet.setOptionValue('session', undefined);

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
    process.exitCode = undefined;
    jest.restoreAllMocks();
  });

  it('should get and print trace details in tree format by default', async () => {
    await createCommand().parseAsync(['node', 'trace:get', 'test-trace-id']);

    expect(mockedForceStderr).toHaveBeenCalled();
    expect(mockedFindProjectRoot).toHaveBeenCalled();
    expect(mockedRunWithManager).toHaveBeenCalled();

    expect(console.log).toHaveBeenCalledWith(
      expect.stringContaining('Trace ID: test-id')
    );
  });

  it('should print trace in json format when -f json is specified', async () => {
    mockedRunWithManager.mockImplementation(async (projectRoot, fn) => {
      const mockManager = {
        getTrace: jest.fn<any>().mockResolvedValue({
          traceId: 'test-id',
          spans: {
            s1: {
              spanId: 's1',
              attributes: {
                'genkit:input': '{"query":"hello"}',
                img: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
              },
            },
          },
        }),
      };
      await fn(mockManager as any);
    });

    await createCommand().parseAsync([
      'node',
      'trace:get',
      'test-trace-id',
      '-f',
      'json',
    ]);

    const loggedStr = (console.log as jest.Mock).mock.calls[0][0] as string;
    const parsed = JSON.parse(loggedStr);
    expect(parsed.spans.s1.attributes['genkit:input']).toEqual({
      query: 'hello',
    });
    expect(parsed.spans.s1.attributes.img).toContain('base64 data');
  });

  it('should preserve base64 data when --keep-base64 is specified', async () => {
    const rawImage =
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
    mockedRunWithManager.mockImplementation(async (projectRoot, fn) => {
      const mockManager = {
        getTrace: jest.fn<any>().mockResolvedValue({
          traceId: 'test-id',
          spans: {
            s1: {
              spanId: 's1',
              attributes: {
                img: rawImage,
              },
            },
          },
        }),
      };
      await fn(mockManager as any);
    });

    await createCommand().parseAsync([
      'node',
      'trace:get',
      'test-trace-id',
      '-f',
      'json',
      '--keep-base64',
    ]);

    const loggedStr = (console.log as jest.Mock).mock.calls[0][0] as string;
    const parsed = JSON.parse(loggedStr);
    expect(parsed.spans.s1.attributes.img).toEqual(rawImage);
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
    expect(process.exitCode).toBe(1);
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
    expect(process.exitCode).toBe(1);
  });

  it('should output JSON trajectory bundle when --trajectory is passed', async () => {
    mockedRunWithManager.mockImplementation(async (_projectRoot, fn) => {
      const mockManager = {
        getTrace: jest.fn<any>().mockResolvedValue({
          traceId: 'traj-trace-1',
          startTime: 100,
          endTime: 200,
          spans: {
            m1: {
              traceId: 'traj-trace-1',
              spanId: 'm1',
              displayName: '/model/gemini-2.5-pro',
              startTime: 100,
              endTime: 200,
              instrumentationLibrary: { name: 'genkit' },
              spanKind: 'INTERNAL',
              attributes: {
                'genkit:metadata:subtype': 'model',
                'genkit:input': JSON.stringify({
                  messages: [{ role: 'user', content: [{ text: 'Hi agent' }] }],
                }),
                'genkit:output': JSON.stringify({
                  message: {
                    role: 'model',
                    content: [{ text: 'Hello user' }],
                  },
                  finishReason: 'stop',
                }),
              },
            },
          },
        }),
      };
      await fn(mockManager as any);
    });

    await createCommand().parseAsync([
      'node',
      'trace:get',
      'traj-trace-1',
      '--trajectory',
    ]);

    const loggedStr = (console.log as jest.Mock).mock.calls[0][0] as string;
    const bundle = JSON.parse(loggedStr);
    expect(bundle.rootTrajectoryId).toBe('traj-trace-1');
    expect(bundle.trajectories['traj-trace-1'].steps).toHaveLength(2);
  });

  it('should merge session traces when --session and -f json are passed', async () => {
    const trace1 = {
      traceId: 't1',
      startTime: 100,
      endTime: 200,
      spans: {
        m1: {
          traceId: 't1',
          spanId: 'm1',
          displayName: '/model/gemini',
          startTime: 100,
          endTime: 200,
          instrumentationLibrary: { name: 'genkit' },
          spanKind: 'INTERNAL',
          attributes: {
            'genkit:metadata:subtype': 'model',
            'genkit:metadata:agent:sessionId': 'sess-99',
            'genkit:input': JSON.stringify({
              messages: [{ role: 'user', content: [{ text: 'Turn 1' }] }],
            }),
            'genkit:output': JSON.stringify({
              message: { role: 'model', content: [{ text: 'Reply 1' }] },
              finishReason: 'stop',
            }),
          },
        },
      },
    };
    const trace2 = {
      traceId: 't2',
      startTime: 300,
      endTime: 400,
      spans: {
        m2: {
          traceId: 't2',
          spanId: 'm2',
          displayName: '/model/gemini',
          startTime: 300,
          endTime: 400,
          instrumentationLibrary: { name: 'genkit' },
          spanKind: 'INTERNAL',
          attributes: {
            'genkit:metadata:subtype': 'model',
            'genkit:metadata:agent:sessionId': 'sess-99',
            'genkit:input': JSON.stringify({
              messages: [
                { role: 'user', content: [{ text: 'Turn 1' }] },
                { role: 'model', content: [{ text: 'Reply 1' }] },
                { role: 'user', content: [{ text: 'Turn 2' }] },
              ],
            }),
            'genkit:output': JSON.stringify({
              message: { role: 'model', content: [{ text: 'Reply 2' }] },
              finishReason: 'stop',
            }),
          },
        },
      },
    };

    mockedRunWithManager.mockImplementation(async (_projectRoot, fn) => {
      const mockManager = {
        getTrace: jest
          .fn<any>()
          .mockImplementation(async ({ traceId }: { traceId: string }) => {
            return traceId === 't1' ? trace1 : trace2;
          }),
        listTraces: jest.fn<any>().mockResolvedValue({
          traces: [{ traceId: 't2' }, { traceId: 't1' }],
        }),
      };
      await fn(mockManager as any);
    });

    await createCommand().parseAsync([
      'node',
      'trace:get',
      't1',
      '--trajectory',
      '--session',
      '-f',
      'json',
    ]);

    const loggedStr = (console.log as jest.Mock).mock.calls[0][0] as string;
    const bundle = JSON.parse(loggedStr);
    expect(bundle.sessionId).toBe('sess-99');
    expect(bundle.trajectories.t1.steps).toHaveLength(4);
  });
});

