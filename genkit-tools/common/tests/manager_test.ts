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

import { describe, expect, it, jest } from '@jest/globals';
import { RuntimeManager } from '../src/manager/manager';
import { RuntimeEvent } from '../src/manager/types';

jest.mock('chokidar', () => ({
  watch: jest.fn().mockReturnValue({
    on: jest.fn(),
    close: jest.fn(),
  }),
}));

describe('RuntimeManager', () => {
  it('should allow unsubscribing from runtime events', async () => {
    const manager = await RuntimeManager.create({ projectRoot: '.' });
    const listener = jest.fn();

    // Subscribe
    const unsubscribe = manager.onRuntimeEvent(listener);

    // Simulate event
    (manager as any).eventEmitter.emit(RuntimeEvent.ADD, { id: '1' });
    expect(listener).toHaveBeenCalledTimes(1);

    // Unsubscribe
    unsubscribe();

    // Simulate event again
    (manager as any).eventEmitter.emit(RuntimeEvent.ADD, { id: '2' });
    expect(listener).toHaveBeenCalledTimes(1); // Should not have increased
  });
});
