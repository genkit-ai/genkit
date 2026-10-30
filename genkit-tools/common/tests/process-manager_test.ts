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
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { ProcessManager } from '../src/manager/process-manager';
import { logger } from '../src/utils';

type TerminationHook = {
  terminateProcess(pid: number, callback: (error: Error | null) => void): void;
};

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
    await manager.waitForProcess();
  });

  it('should preserve nonInteractive and cwd options when restart() is called without arguments', async () => {
    const tmpDir = fs.realpathSync(
      fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-pm-test-'))
    );
    const capturedDebug: string[] = [];
    const origDebug = logger.debug.bind(logger);
    logger.debug = ((...args: unknown[]) => {
      capturedDebug.push(args.map(String).join(' '));
    }) as typeof logger.debug;

    try {
      const manager = new ProcessManager('node', [
        '-e',
        'console.log("> simulated npm banner from " + process.cwd()); process.exit(0);',
      ]);

      await manager.start({ nonInteractive: true, cwd: tmpDir });
      capturedDebug.length = 0;

      // Restart with no options (mirrors MCP restart_runtime and Dev UI restartAppProcess)
      await manager.restart();
      await manager.waitForProcess();

      expect(
        capturedDebug.some(
          (line) =>
            line.includes('[ProcessManager Stdout]') &&
            line.includes(`> simulated npm banner from ${tmpDir}`)
        )
      ).toBe(true);
    } finally {
      logger.debug = origDebug;
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('kill should be idempotent', async () => {
    const manager = new ProcessManager('node', ['-e', '']);
    await manager.kill();
    await manager.kill();
    expect(manager.status()).toEqual({ status: 'stopped' });
  });

  it('accepts a termination error if the child has already exited', async () => {
    const manager = new ProcessManager('node', []);
    const child = Object.assign(new EventEmitter(), {
      pid: 99_999_999,
      killed: false,
      exitCode: null,
      signalCode: null,
    });
    (manager as any).appProcess = child;
    const terminateSpy = jest
      .spyOn(manager as unknown as TerminationHook, 'terminateProcess')
      .mockImplementation(
        (_pid: number, callback: (error: Error | null) => void) => {
          callback(new Error('process not found'));
        }
      );
    try {
      await expect(manager.kill()).resolves.toBeUndefined();
    } finally {
      terminateSpy.mockRestore();
    }
  });

  it('accepts an exited child even if its PID has been reused', async () => {
    const manager = new ProcessManager('node', []);
    const child = Object.assign(new EventEmitter(), {
      pid: process.pid,
      killed: false,
      exitCode: 0,
      signalCode: null,
    });
    (manager as any).appProcess = child;
    const terminateSpy = jest
      .spyOn(manager as unknown as TerminationHook, 'terminateProcess')
      .mockImplementation(
        (_pid: number, callback: (error: Error | null) => void) => {
          callback(new Error('process not found'));
        }
      );
    try {
      await expect(manager.kill()).resolves.toBeUndefined();
    } finally {
      terminateSpy.mockRestore();
    }
  });

  it('rejects a termination error while the child is still running', async () => {
    const manager = new ProcessManager('node', []);
    const child = Object.assign(new EventEmitter(), {
      pid: process.pid,
      killed: false,
      exitCode: null,
      signalCode: null,
    });
    (manager as any).appProcess = child;
    const terminateSpy = jest
      .spyOn(manager as unknown as TerminationHook, 'terminateProcess')
      .mockImplementation(
        (_pid: number, callback: (error: Error | null) => void) => {
          callback(new Error('access denied'));
        }
      );
    try {
      await expect(manager.kill()).rejects.toThrow('access denied');
    } finally {
      terminateSpy.mockRestore();
    }
  });

  it('kills a child process and its grandchild', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-process-tree-'));
    const script = path.join(dir, 'child.cjs');
    const pidsFile = path.join(dir, 'pids.json');
    fs.writeFileSync(
      script,
      `const { spawn } = require('child_process');
       const fs = require('fs');
       const grandchild = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' });
       fs.writeFileSync(process.argv[2], JSON.stringify({ child: process.pid, grandchild: grandchild.pid }));
       setInterval(() => {}, 1000);`
    );
    const manager = new ProcessManager('node', [script, pidsFile]);
    let pids: { child: number; grandchild: number } | undefined;
    const startPromise = manager.start({ nonInteractive: true });
    try {
      for (let i = 0; i < 100 && !fs.existsSync(pidsFile); i++) {
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      expect(fs.existsSync(pidsFile)).toBe(true);
      pids = JSON.parse(fs.readFileSync(pidsFile, 'utf8'));
      await manager.kill();
      await startPromise;
      for (let i = 0; i < 100 && isAlive(pids!.grandchild); i++) {
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      expect(isAlive(pids!.child)).toBe(false);
      expect(isAlive(pids!.grandchild)).toBe(false);
    } finally {
      await manager.kill();
      for (const pid of [pids?.child, pids?.grandchild]) {
        if (pid && isAlive(pid)) process.kill(pid, 'SIGTERM');
      }
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }, 15_000);
});

function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
