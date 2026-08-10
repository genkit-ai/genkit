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

import { ChildProcess, spawn } from 'child_process';
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

  private _status: ProcessStatus = 'stopped';
  private manualRestart = false;

  constructor(
    private readonly command: string,
    private readonly args: string[],
    private readonly env: NodeJS.ProcessEnv = {}
  ) {}

  /**
   * Starts the process.
   */
  start(options?: ProcessManagerStartOptions): Promise<void> {
    logger.debug(`Starting process: ${this.command} ${this.args.join(' ')}`);
    return new Promise((resolve, reject) => {
      this._status = 'running';
      this.appProcess = spawn(this.command, this.args, {
        cwd: options?.cwd,
        env: {
          ...process.env,
          ...this.env,
        },
        shell: process.platform === 'win32',
        stdio: options?.nonInteractive ? 'pipe' : 'inherit',
      });

      if (!options?.nonInteractive) {
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
  }

  /**
   * Kills the currently-running process and starts a new one.
   */
  async restart(options?: ProcessManagerStartOptions): Promise<void> {
    this.manualRestart = true;
    await this.kill();
    this.start(options).catch(() => {});
  }

  /**
   * Kills the currently-running process.
   */
  kill(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.appProcess || !this.appProcess.pid || this.appProcess.killed) {
        this._status = 'stopped';
        resolve();
        return;
      }

      // The 'exit' listener is set up in start() and will handle cleanup.
      this.appProcess.on('exit', () => {
        resolve();
      });

      terminate(this.appProcess.pid, 'SIGTERM', (err) => {
        if (err) {
          // This can happen if the process is already gone, which is fine.
          logger.debug(`Error during process termination: ${err.message}`);
        }
        resolve();
      });
    });
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
