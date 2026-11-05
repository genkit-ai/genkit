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

import { ChildProcess, execFile, spawn } from 'child_process';
import terminate from 'terminate';
import { logger } from '../utils';

export type ProcessStatus = 'running' | 'stopped' | 'unconfigured';

export interface AppProcessStatus {
  status: ProcessStatus;
}

export interface ProcessManagerStartOptions {
  cwd?: string;
  nonInteractive?: boolean;
}

/**
 * Manages a child process.
 */
export class ProcessManager {
  private appProcess?: ChildProcess;
  private startOptions?: ProcessManagerStartOptions;
  private currentProcessPromise?: Promise<void>;

  private _status: ProcessStatus = 'stopped';
  private manualRestart = false;
  private stopping = false;

  constructor(
    private readonly command: string,
    private readonly args: string[],
    private readonly env: NodeJS.ProcessEnv = {}
  ) {}

  /**
   * Starts the process.
   */
  start(options?: ProcessManagerStartOptions): Promise<void> {
    const resolvedOptions = options
      ? { ...this.startOptions, ...options }
      : this.startOptions;
    this.startOptions = resolvedOptions;
    logger.debug(`Starting process: ${this.command} ${this.args.join(' ')}`);
    this.currentProcessPromise = new Promise((resolve, reject) => {
      this._status = 'running';
      this.appProcess = spawn(this.command, this.args, {
        cwd: resolvedOptions?.cwd,
        env: {
          ...process.env,
          ...this.env,
        },
        shell: process.platform === 'win32',
        stdio: resolvedOptions?.nonInteractive ? 'pipe' : 'inherit',
      });

      if (!resolvedOptions?.nonInteractive) {
        // In interactive mode, stdio: inherit handles the piping.
      } else {
        this.appProcess.stderr?.on('data', (data) => {
          logger.debug(`[ProcessManager Stderr] ${data.toString()}`);
        });
        this.appProcess.stdout?.on('data', (data) => {
          logger.debug(`[ProcessManager Stdout] ${data.toString()}`);
        });
      }

      this.appProcess.on('error', (error): void => {
        logger.error(`Error in app process: ${error}`);
        this.cleanup();
        reject(error);
      });

      this.appProcess.on('exit', (code, signal) => {
        this.cleanup();
        if (this.manualRestart) {
          this.manualRestart = false;
          resolve();
          return;
        }
        if (this.stopping) {
          this.stopping = false;
          resolve();
          return;
        }
        // If the process was killed by a signal, it's not an error in this context.
        if (code === 0 || signal) {
          resolve();
        } else {
          reject(new Error(`app process exited with code ${code}`));
        }
      });
    });
    return this.currentProcessPromise;
  }

  /**
   * Kills the currently-running process and starts a new one.
   */
  async restart(options?: ProcessManagerStartOptions): Promise<void> {
    this.manualRestart = true;
    await this.kill();
    this.manualRestart = false;
    this.start(options).catch(() => {});
  }

  /**
   * Returns the promise for the currently running or most recently started process.
   */
  waitForProcess(): Promise<void> | undefined {
    return this.currentProcessPromise;
  }

  /**
   * Kills the currently-running process.
   */
  kill(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.appProcess || !this.appProcess.pid || this.appProcess.killed) {
        this._status = 'stopped';
        resolve();
        return;
      }

      // The 'exit' listener is set up in start() and will handle cleanup.
      this.appProcess.on('exit', () => {
        resolve();
      });

      const pid = this.appProcess.pid;
      this.stopping = true;
      const onTerminated = (err: Error | null) => {
        if (err) {
          logger.debug(`Error during process termination: ${err.message}`);
          // A child may exit before Node delivers its 'exit' event. Only fail
          // shutdown if the target PID is still running after the kill error.
          if (
            this.appProcess &&
            this.appProcess.exitCode === null &&
            this.appProcess.signalCode === null &&
            isProcessAlive(pid)
          ) {
            this.stopping = false;
            reject(err);
            return;
          }
        }
        resolve();
      };
      this.terminateProcess(pid, onTerminated);
    });
  }

  private terminateProcess(
    pid: number,
    callback: (error: Error | null) => void
  ): void {
    if (process.platform === 'win32') {
      // terminate relies on wmic.exe, which is absent on current Windows.
      // taskkill targets only the process tree rooted at our own child PID.
      execFile('taskkill.exe', ['/PID', String(pid), '/T', '/F'], callback);
    } else {
      terminate(pid, 'SIGTERM', callback);
    }
  }

  status(): AppProcessStatus {
    return {
      status: this._status,
    };
  }

  private cleanup() {
    if (this.appProcess) {
      this.appProcess.stdout?.removeAllListeners();
      this.appProcess.stderr?.removeAllListeners();
    }
    this.appProcess = undefined;
    this._status = 'stopped';
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== 'ESRCH';
  }
}
