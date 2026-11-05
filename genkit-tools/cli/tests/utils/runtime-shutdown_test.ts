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

import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { waitForRuntimeShutdown } from '../../src/utils/runtime-shutdown';

describe('waitForRuntimeShutdown', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('stops once on normal completion and removes signal listeners', async () => {
    const stopOnce = jest.fn(async () => {});
    const manager = { stopOnce } as unknown as BaseRuntimeManager;
    const before = process.listenerCount('SIGTERM');

    await waitForRuntimeShutdown(manager, Promise.resolve());

    expect(stopOnce).toHaveBeenCalledTimes(1);
    expect(process.listenerCount('SIGTERM')).toBe(before);
  });

  it('waits for one cleanup on repeated termination signals', async () => {
    let finishProcess!: () => void;
    const processPromise = new Promise<void>((resolve) => {
      finishProcess = resolve;
    });
    let finishStop!: () => void;
    const stopOnce = jest.fn(
      () =>
        new Promise<void>((resolve) => {
          finishStop = resolve;
        })
    );
    const manager = { stopOnce } as unknown as BaseRuntimeManager;
    const exit = jest
      .spyOn(process, 'exit')
      .mockImplementation(() => undefined as never);
    const before = new Set(process.listeners('SIGTERM'));
    const waiting = waitForRuntimeShutdown(manager, processPromise);
    const handler = process
      .listeners('SIGTERM')
      .find((listener) => !before.has(listener));
    expect(handler).toBeDefined();

    handler!('SIGTERM');
    handler!('SIGTERM');
    expect(stopOnce).toHaveBeenCalledTimes(1);
    expect(exit).not.toHaveBeenCalled();

    finishStop();
    await new Promise((resolve) => setImmediate(resolve));
    expect(exit).toHaveBeenCalledWith(0);
    finishProcess();
    await waiting;
    expect(stopOnce).toHaveBeenCalledTimes(1);
  });

  it('removes signal listeners when cleanup rejects', async () => {
    const manager = {
      stopOnce: jest.fn(async () => {
        throw new Error('cleanup failed');
      }),
    } as unknown as BaseRuntimeManager;
    const before = process.listenerCount('SIGTERM');
    await expect(
      waitForRuntimeShutdown(manager, Promise.resolve())
    ).rejects.toThrow('cleanup failed');
    expect(process.listenerCount('SIGTERM')).toBe(before);
  });

  it('preserves the process failure when runtime cleanup also fails', async () => {
    const processError = new Error('process failed');
    const stopOnce = jest.fn(async () => {
      throw new Error('cleanup failed');
    });
    const manager = { stopOnce } as unknown as BaseRuntimeManager;
    const before = process.listenerCount('SIGTERM');

    await expect(
      waitForRuntimeShutdown(manager, Promise.reject(processError))
    ).rejects.toBe(processError);

    expect(stopOnce).toHaveBeenCalledTimes(1);
    expect(process.listenerCount('SIGTERM')).toBe(before);
  });

  it('exits with failure after a signal if runtime cleanup fails', async () => {
    let finishProcess!: () => void;
    const processPromise = new Promise<void>((resolve) => {
      finishProcess = resolve;
    });
    const manager = {
      stopOnce: jest.fn(async () => {
        throw new Error('cleanup failed');
      }),
    } as unknown as BaseRuntimeManager;
    const exit = jest
      .spyOn(process, 'exit')
      .mockImplementation(() => undefined as never);
    const before = new Set(process.listeners('SIGTERM'));
    const waiting = waitForRuntimeShutdown(manager, processPromise);
    const handler = process
      .listeners('SIGTERM')
      .find((listener) => !before.has(listener));

    handler!('SIGTERM');
    await new Promise((resolve) => setImmediate(resolve));
    expect(exit).toHaveBeenCalledWith(1);
    finishProcess();
    await expect(waiting).rejects.toThrow('cleanup failed');
  });
});
