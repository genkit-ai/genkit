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

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import axios from 'axios';
import getPort from 'get-port';
import { BaseRuntimeManager } from '../src/manager/manager';
import { startServer } from '../src/server/server';

describe('Tools Server', () => {
  let port: number;
  let serverPromise: Promise<void>;
  let mockManager: any;

  beforeEach(async () => {
    port = await getPort();
    mockManager = {
      projectRoot: './',
      disableRealtimeTelemetry: false,
      runAction: jest.fn(),
      streamTrace: jest.fn(),
      listActions: jest.fn(),
      listTraces: jest.fn(),
      getTrace: jest.fn(),
      getMostRecentRuntime: jest.fn(),
      listRuntimes: jest.fn(),
      onRuntimeEvent: jest.fn(),
      cancelAction: jest.fn(),
    };
    serverPromise = startServer(mockManager as BaseRuntimeManager, port);
  });

  afterEach(async () => {
    const exitSpy = jest
      .spyOn(process, 'exit')
      .mockImplementation((code?: any) => {
        return undefined as never;
      });
    try {
      await axios.post(`http://localhost:${port}/api/__quitquitquit`);
    } catch (e) {
      // Ignore
    }
    await serverPromise;
    exitSpy.mockRestore();
  });

  it('should handle runAction', async () => {
    mockManager.runAction.mockResolvedValue({ result: 'bar' });

    let response;
    try {
      response = await axios.post(`http://localhost:${port}/api/runAction`, {
        key: 'foo',
        input: 'bar',
      });
    } catch (e: any) {
      throw new Error(`runAction failed: ${e.message}`);
    }

    expect(response.data.result).toBe('bar');
    expect(mockManager.runAction).toHaveBeenCalledWith(
      expect.objectContaining({ key: 'foo' }),
      undefined,
      expect.any(Function)
    );
  });

  it('should handle bidi streaming', async () => {
    let inputStream: AsyncIterable<any> | undefined;
    let finishAction: (() => void) | undefined;

    mockManager.runAction.mockImplementation(
      async (input: any, cb: any, trace: any, stream: any) => {
        inputStream = stream;
        await new Promise<void>((resolve) => {
          finishAction = resolve;
        });
        return { result: 'done' };
      }
    );

    const responsePromise = axios
      .post(
        `http://localhost:${port}/api/streamAction?bidi=true`,
        { key: 'bidi' },
        { responseType: 'stream' }
      )
      .catch((e) => {
        throw new Error(`Stream action failed: ${e.message}`);
      });

    // Wait for runAction to be called
    while (!inputStream) {
      await new Promise((r) => setTimeout(r, 10));
    }

    const traceId = 'test-trace-id';
    // Get the onTraceId callback from the mock call args
    const [inputArg, cb, onTraceIdCallback] =
      mockManager.runAction.mock.calls[0];
    onTraceIdCallback(traceId);

    // Collect input chunks in background
    const chunks: any[] = [];
    const collectPromise = (async () => {
      for await (const chunk of inputStream!) {
        chunks.push(chunk);
      }
    })();

    // Now send input
    try {
      await axios.post(`http://localhost:${port}/api/sendBidiInput`, {
        traceId,
        chunk: 'input1',
      });

      await axios.post(`http://localhost:${port}/api/endBidiInput`, {
        traceId,
      });
    } catch (e: any) {
      throw new Error(`send/end input failed: ${e.message}`);
    }

    await collectPromise;
    expect(chunks).toEqual(['input1']);

    // Emit output chunk
    if (cb) cb({ result: 'chunk1' });

    // Finish action
    finishAction!();

    const response = await responsePromise;
    const stream = response.data;
    const outputChunks: string[] = [];
    for await (const chunk of stream) {
      outputChunks.push(chunk.toString());
    }
    const output = outputChunks.join('');
    expect(output).toContain('chunk1');
    expect(output).toContain('done');
  });
});
