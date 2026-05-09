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

import { describe, expect, it } from '@jest/globals';
import { ProcessManager } from '../src/manager/process-manager';

describe('ProcessManager', () => {
  it('should start and stop a process successfully', async () => {
    const manager = new ProcessManager('node', [
      '-e',
      'console.log("test"); process.exit(0);',
    ]);
    expect(manager.status()).toEqual({ status: 'stopped' });
    await manager.start({ nonInteractive: true });
    expect(manager.status()).toEqual({ status: 'stopped' });
  });

  it('should reject on non-zero exit code', async () => {
    const manager = new ProcessManager('node', ['-e', 'process.exit(1);']);
    await expect(manager.start({ nonInteractive: true })).rejects.toThrow(
      'app process exited with code 1'
    );
    expect(manager.status()).toEqual({ status: 'stopped' });
  });

  it('should be able to kill a process', async () => {
    // A process that runs for a long time
    const manager = new ProcessManager('node', [
      '-e',
      'setTimeout(() => {}, 10000)',
    ]);
    expect(manager.status()).toEqual({ status: 'stopped' });

    const startPromise = manager.start({ nonInteractive: true });
    // Give it a moment to start
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(manager.status()).toEqual({ status: 'running' });

    await manager.kill();
    expect(manager.status()).toEqual({ status: 'stopped' });

    // The start promise should resolve when the process is killed.
    await startPromise;
  });

  it('should be able to restart a process', async () => {
    const manager = new ProcessManager('node', [
      '-e',
      'console.log("test"); process.exit(0);',
    ]);
    expect(manager.status()).toEqual({ status: 'stopped' });

    // Start, let it finish
    await manager.start({ nonInteractive: true });
    expect(manager.status()).toEqual({ status: 'stopped' });

    // Restart
    await manager.restart({ nonInteractive: true });
    expect(manager.status()).toEqual({ status: 'running' });
  });

  it('kill should be idempotent', async () => {
    const manager = new ProcessManager('node', ['-e', '']);
    await manager.kill();
    await manager.kill();
    expect(manager.status()).toEqual({ status: 'stopped' });
  });
});
