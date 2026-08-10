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

import { RuntimeEvent } from '@genkit-ai/tools-common/manager';
import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import {
  waitForActionKeys,
  waitForRuntime,
} from '../../src/utils/manager-utils';

describe('waitForRuntime', () => {
  let mockManager: any;
  let mockProcessPromise: Promise<void>;
  let processReject: (reason?: any) => void;

  beforeEach(() => {
    mockManager = {
      listRuntimes: jest.fn(),
      onRuntimeEvent: jest.fn(),
    };
    mockProcessPromise = new Promise((_, reject) => {
      processReject = reject;
    });
  });

  it('should resolve immediately if runtime is already present', async () => {
    mockManager.listRuntimes.mockReturnValue([{}]);
    await expect(
      waitForRuntime(mockManager, mockProcessPromise)
    ).resolves.toBeUndefined();
  });

  it('should wait for runtime event and resolve', async () => {
    mockManager.listRuntimes.mockReturnValue([]);
    let eventCallback: (event: RuntimeEvent, runtime: any) => void;

    mockManager.onRuntimeEvent.mockImplementation((cb: any) => {
      eventCallback = cb;
      return jest.fn(); // unsubscribe
    });

    const waitPromise = waitForRuntime(mockManager, mockProcessPromise);

    // Simulate event
    setTimeout(() => {
      eventCallback(RuntimeEvent.ADD, { id: 'any-id' });
    }, 10);

    await expect(waitPromise).resolves.toBeUndefined();
  });

  it('should reject if process exits early', async () => {
    mockManager.listRuntimes.mockReturnValue([]);
    mockManager.onRuntimeEvent.mockReturnValue(jest.fn());

    const waitPromise = waitForRuntime(mockManager, mockProcessPromise);

    // Simulate process exit
    processReject(new Error('Process exited'));

    await expect(waitPromise).rejects.toThrow('Process exited');
  });

  it('should timeout if runtime never appears', async () => {
    jest.useFakeTimers();
    mockManager.listRuntimes.mockReturnValue([]);
    mockManager.onRuntimeEvent.mockReturnValue(jest.fn());

    const waitPromise = waitForRuntime(mockManager, mockProcessPromise);

    jest.advanceTimersByTime(30000);

    await expect(waitPromise).rejects.toThrow(
      'Timeout waiting for runtime to be ready'
    );
    jest.useRealTimers();
  });
});

describe('waitForActionKeys', () => {
  let mockManager: any;

  beforeEach(() => {
    mockManager = {
      listActions: jest.fn(),
      listRuntimes: jest.fn().mockReturnValue([{}]),
    };
  });

  it('resolves immediately when no keys are required', async () => {
    await expect(waitForActionKeys(mockManager, [])).resolves.toBeUndefined();
    expect(mockManager.listActions).not.toHaveBeenCalled();
  });

  it('resolves once all required actions are registered', async () => {
    // First poll: action missing. Second poll: action present.
    mockManager.listActions
      .mockResolvedValueOnce({})
      .mockResolvedValueOnce({ '/flow/testFlow': {} });

    await expect(
      waitForActionKeys(mockManager, ['/flow/testFlow'], {
        pollIntervalMs: 1,
      })
    ).resolves.toBeUndefined();
    expect(mockManager.listActions).toHaveBeenCalledTimes(2);
  });

  it('keeps polling while listActions rejects, then resolves', async () => {
    mockManager.listActions
      .mockRejectedValueOnce(new Error('not ready'))
      .mockResolvedValueOnce({ '/flow/testFlow': {} });

    await expect(
      waitForActionKeys(mockManager, ['/flow/testFlow'], {
        pollIntervalMs: 1,
      })
    ).resolves.toBeUndefined();
    expect(mockManager.listActions).toHaveBeenCalledTimes(2);
  });

  it('stops early once the action list stabilizes without the target', async () => {
    // The action list never contains the target and stops changing, so we
    // should give up after the stability window rather than the full timeout.
    mockManager.listActions.mockResolvedValue({ '/flow/other': {} });

    const start = Date.now();
    await expect(
      waitForActionKeys(mockManager, ['/flow/missing'], {
        pollIntervalMs: 1,
        stableForMs: 20,
        timeoutMs: 30000,
      })
    ).resolves.toBeUndefined();
    // Should return around the stability window, well before the timeout.
    expect(Date.now() - start).toBeLessThan(5000);
  });

  it('keeps waiting while the action list is still growing', async () => {
    // The list keeps changing (still registering), so the stability window
    // should keep resetting until the target finally appears.
    mockManager.listActions
      .mockResolvedValueOnce({ a: {} })
      .mockResolvedValueOnce({ a: {}, b: {} })
      .mockResolvedValueOnce({ a: {}, b: {}, c: {} })
      .mockResolvedValue({ a: {}, b: {}, c: {}, '/flow/testFlow': {} });

    await expect(
      waitForActionKeys(mockManager, ['/flow/testFlow'], {
        pollIntervalMs: 1,
        stableForMs: 20,
      })
    ).resolves.toBeUndefined();
    expect(mockManager.listActions).toHaveBeenCalledTimes(4);
  });

  it('stops early (does not wait for timeout) when the runtime disconnects', async () => {
    // Runtime is present initially, then disconnects. The action is never
    // registered. With a long timeout, the function should still return
    // promptly instead of polling until the deadline.
    mockManager.listRuntimes
      .mockReturnValueOnce([{}]) // initial hasSeenRuntime check
      .mockReturnValueOnce([{}]) // first loop iteration
      .mockReturnValue([]); // subsequent iterations: disconnected
    mockManager.listActions.mockResolvedValue({});

    const start = Date.now();
    await expect(
      waitForActionKeys(mockManager, ['/flow/testFlow'], {
        pollIntervalMs: 1,
        timeoutMs: 30000,
      })
    ).resolves.toBeUndefined();
    // Should return well before the 30s deadline.
    expect(Date.now() - start).toBeLessThan(5000);
  });
});
