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
import { waitForRuntime } from '../../src/utils/manager-utils';

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
