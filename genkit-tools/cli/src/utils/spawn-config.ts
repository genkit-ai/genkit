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

import { type SpawnOptions } from 'child_process';
import { access, constants } from 'fs/promises';
import { SERVER_HARNESS_COMMAND } from '../commands/server-harness';
import { type CLIRuntimeInfo } from './runtime-detector';

/**
 * Configuration for spawning a child process
 */
export interface SpawnConfig {
  /** Executable to run */
  command: string;
  /** Arguments array */
  args: string[];
  /** Spawn options */
  options: SpawnOptions;
}

/**
 * Validates that a path exists and is executable
 * @param path - Path to validate
 * @returns true if the path exists and is executable
 */
export async function validateExecutablePath(path: string): Promise<boolean> {
  try {
    // Remove surrounding quotes if present (handle quotation)
    const normalizedPath =
      path.startsWith('"') && path.endsWith('"') ? path.slice(1, -1) : path;
    await access(normalizedPath, constants.F_OK | constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validates that a port number is valid (integer between 0 and 65535)
 * @param port - Port number to validate
 * @returns true if the port is valid
 */
function isValidPort(port: number): boolean {
  return Number.isInteger(port) && port >= 0 && port <= 65535;
}

/**
 * Builds spawn configuration for the server harness based on runtime info
 *
 * @param cliRuntime - CLI runtime information from detector
 * @param port - Port number for the server (must be valid port 0-65535)
 * @param logPath - Path to the log file
 * @returns Spawn configuration for child_process.spawn
 * @throws Error if port is invalid or runtime info is missing required fields
 *
 * @example
 * ```typescript
 * const cliRuntime = detectRuntime();
 * const config = buildServerHarnessSpawnConfig(cliRuntime, 4000, '/path/to/log');
 * const child = spawn(config.command, config.args, config.options);
 * ```
 */
export function buildServerHarnessSpawnConfig(
  cliRuntime: CLIRuntimeInfo,
  port: number,
  logPath: string
): SpawnConfig {
  // Validate inputs
  if (!cliRuntime) {
    throw new Error('CLI runtime info is required');
  }
  if (!cliRuntime.execPath) {
    throw new Error('CLI runtime execPath is required');
  }
  if (!isValidPort(port)) {
    throw new Error(
      `Invalid port number: ${port}. Must be between 0 and 65535`
    );
  }
  if (!logPath) {
    throw new Error('Log path is required');
  }

  let command = cliRuntime.execPath;
  let args: string[];

  if (cliRuntime.type === 'compiled-binary') {
    // For compiled binaries, execute directly with arguments
    args = [SERVER_HARNESS_COMMAND, port.toString(), logPath];
  } else {
    // For interpreted runtimes (Node.js, Bun), include script path if available
    args = cliRuntime.scriptPath
      ? [
          cliRuntime.scriptPath,
          SERVER_HARNESS_COMMAND,
          port.toString(),
          logPath,
        ]
      : [SERVER_HARNESS_COMMAND, port.toString(), logPath];
  }

  // Build spawn options with platform-specific settings
  const options: SpawnOptions = {
    stdio: ['ignore', 'ignore', 'ignore'] as const,
    detached: false,
    // Use shell on Windows for better compatibility with paths containing spaces
    shell: cliRuntime.platform === 'win32',
  };

  // Handles spaces in the command and arguments on Windows
  if (cliRuntime.platform === 'win32') {
    command = `"${command}"`;
    args = args.map((arg) => `"${arg}"`);
  }

  return {
    command,
    args,
    options,
  };
}
