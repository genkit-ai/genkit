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
import * as analytics from '../src/utils/analytics';

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

  describe('analytics events', () => {
    let recordSpy: jest.Spied<typeof analytics.recordRequestEvent>;

    beforeEach(() => {
      recordSpy = jest
        .spyOn(analytics, 'recordRequestEvent')
        .mockImplementation(() => {});
      jest.spyOn(analytics, 'record').mockImplementation(async () => {});
    });

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('should record analytics event for runAction with valid string key', async () => {
      mockManager.runAction.mockResolvedValue({ result: 'bar' });

      await axios.post(`http://localhost:${port}/api/runAction`, {
        key: '/flow/foo',
        input: 'bar',
      });

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'runAction',
            action: 'flow',
            status: 'success',
          }),
        })
      );
    });

    it('should record analytics event for runAction with non-string key as unknown action', async () => {
      mockManager.runAction.mockResolvedValue({ result: 'bar' });

      await axios.post(`http://localhost:${port}/api/runAction`, {
        key: 123,
        input: 'bar',
      });

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'runAction',
            action: 'unknown',
            status: 'success',
          }),
        })
      );
    });

    it('should record analytics event with full path specifically for /util/generate while splitting other util actions', async () => {
      mockManager.runAction.mockResolvedValue({ result: 'bar' });

      await axios.post(`http://localhost:${port}/api/runAction`, {
        key: '/util/generate',
        input: 'bar',
      });

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'runAction',
            action: '/util/generate',
            status: 'success',
          }),
        })
      );

      await axios.post(`http://localhost:${port}/api/runAction`, {
        key: '/util/other',
        input: 'bar',
      });

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'runAction',
            action: 'util',
            status: 'success',
          }),
        })
      );
    });

    it('should record analytics event for streamTrace', async () => {
      mockManager.streamTrace.mockImplementation(
        async (_opts: any, cb: any) => {
          cb({ traceId: 'test-trace' });
        }
      );
      await axios.post(`http://localhost:${port}/api/streamTrace`, {
        traceId: 'test-trace',
      });
      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'streamTrace',
            status: 'success',
          }),
        })
      );
    });

    it('should record failure analytics event for sendBidiInput when stream is not found', async () => {
      await axios.post(
        `http://localhost:${port}/api/sendBidiInput`,
        {
          traceId: 'non-existent',
          chunk: 'test',
        },
        { validateStatus: () => true }
      );
      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'sendBidiInput',
            status: 'failure',
          }),
        })
      );
    });

    it('should record failure analytics event for endBidiInput when stream is not found', async () => {
      await axios.post(
        `http://localhost:${port}/api/endBidiInput`,
        {
          traceId: 'non-existent',
        },
        { validateStatus: () => true }
      );
      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'endBidiInput',
            status: 'failure',
          }),
        })
      );
    });

    it('should record failure analytics event for runAction when action fails', async () => {
      mockManager.runAction.mockRejectedValue({
        data: { message: 'Action failed' },
      });

      await axios.post(
        `http://localhost:${port}/api/runAction`,
        { key: '/flow/foo', input: 'bar' },
        { validateStatus: () => true }
      );

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'runAction',
            action: 'flow',
            status: 'failure',
          }),
        })
      );
    });

    it('should record success analytics event for streamAction', async () => {
      mockManager.runAction.mockImplementation(async (_input: any, cb: any) => {
        if (cb) cb({ result: 'chunk' });
        return { result: 'done' };
      });

      await axios.post(
        `http://localhost:${port}/api/streamAction`,
        { key: '/flow/foo', input: 'bar' },
        { responseType: 'stream' }
      );

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'streamAction',
            action: 'flow',
            status: 'success',
          }),
        })
      );
    });

    it('should record failure analytics event for streamAction on error', async () => {
      mockManager.runAction.mockRejectedValue({
        data: { message: 'Stream error' },
      });

      await axios.post(
        `http://localhost:${port}/api/streamAction`,
        { key: '/flow/foo', input: 'bar' },
        { validateStatus: () => true }
      );

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'streamAction',
            action: 'flow',
            status: 'failure',
          }),
        })
      );
    });

    it('should record success analytics event for sendBidiInput when stream exists', async () => {
      let inputStream: AsyncIterable<any> | undefined;
      let finishAction: (() => void) | undefined;

      mockManager.runAction.mockImplementation(
        async (_input: any, _cb: any, onTraceId: any, stream: any) => {
          inputStream = stream;
          if (onTraceId) onTraceId('bidi-send-trace-id');
          await new Promise<void>((resolve) => {
            finishAction = resolve;
          });
          return { result: 'done' };
        }
      );

      const streamActionPromise = axios.post(
        `http://localhost:${port}/api/streamAction?bidi=true`,
        { key: '/flow/bidiFlow' },
        { responseType: 'stream' }
      );

      while (!inputStream) {
        await new Promise((r) => setTimeout(r, 10));
      }

      await axios.post(`http://localhost:${port}/api/sendBidiInput`, {
        traceId: 'bidi-send-trace-id',
        chunk: 'hello',
      });

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'sendBidiInput',
            status: 'success',
          }),
        })
      );

      await axios
        .post(`http://localhost:${port}/api/endBidiInput`, {
          traceId: 'bidi-send-trace-id',
        })
        .catch(() => {});
      if (finishAction) finishAction();
      await streamActionPromise.catch(() => {});
    });

    it('should record success analytics event for endBidiInput when stream exists', async () => {
      let inputStream: AsyncIterable<any> | undefined;
      let finishAction: (() => void) | undefined;

      mockManager.runAction.mockImplementation(
        async (_input: any, _cb: any, onTraceId: any, stream: any) => {
          inputStream = stream;
          if (onTraceId) onTraceId('bidi-end-trace-id');
          await new Promise<void>((resolve) => {
            finishAction = resolve;
          });
          return { result: 'done' };
        }
      );

      const streamActionPromise = axios.post(
        `http://localhost:${port}/api/streamAction?bidi=true`,
        { key: '/flow/bidiFlow' },
        { responseType: 'stream' }
      );

      while (!inputStream) {
        await new Promise((r) => setTimeout(r, 10));
      }

      await axios.post(`http://localhost:${port}/api/endBidiInput`, {
        traceId: 'bidi-end-trace-id',
      });

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'endBidiInput',
            status: 'success',
          }),
        })
      );

      if (finishAction) finishAction();
      await streamActionPromise.catch(() => {});
    });

    it('should record failure analytics event for streamTrace when traceId is missing or stream throws', async () => {
      await axios.post(
        `http://localhost:${port}/api/streamTrace`,
        {},
        { validateStatus: () => true }
      );

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'streamTrace',
            status: 'failure',
          }),
        })
      );

      mockManager.streamTrace.mockRejectedValue({
        data: { message: 'Trace not found' },
      });
      await axios.post(
        `http://localhost:${port}/api/streamTrace`,
        { traceId: 'bad-trace' },
        { validateStatus: () => true }
      );

      expect(recordSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'tools_request',
          parameters: expect.objectContaining({
            route: 'streamTrace',
            status: 'failure',
          }),
        })
      );
    });

    it('should record analytics event for sendPageView', async () => {
      const recordDirectSpy = jest.spyOn(analytics, 'record');

      await axios.get(
        `http://localhost:${port}/api/sendPageView?input=${encodeURIComponent(
          JSON.stringify({ pageTitle: 'test-page' })
        )}`
      );

      expect(recordDirectSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'page_view',
          parameters: { page_title: 'test-page' },
        })
      );
    });

    it('should record analytics event for sendSelectContent', async () => {
      const recordDirectSpy = jest.spyOn(analytics, 'record');

      await axios.get(
        `http://localhost:${port}/api/sendSelectContent?input=${encodeURIComponent(
          JSON.stringify({
            contentType: 'button',
            contentId: 'run',
            pageTitle: 'test-page',
          })
        )}`
      );

      expect(recordDirectSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'select_content',
          parameters: {
            content_type: 'button',
            content_id: 'run',
            page_title: 'test-page',
          },
        })
      );
    });
  });
});
