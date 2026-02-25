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

import * as fs from 'fs/promises';
import * as path from 'path';
import type { Runtime } from '../manager/types';
import { isConnectionRefusedError } from './errors';
import { logger } from './logger';

export interface DevToolsInfo {
  /** URL of the dev tools server. */
  url: string;
  /** Timestamp of when the dev tools server was started. */
  timestamp: string;
}

/**
 * Finds the project root by looking for a `package.json` file.
 */
export async function findProjectRoot(): Promise<string> {
  const projectMarkers = [
    'package.json',
    'go.mod',
    'pyproject.toml',
    'requirements.txt',
    'pom.xml',
    'build.gradle',
  ];

  let currentDir = process.cwd();
  while (currentDir !== path.parse(currentDir).root) {
    try {
      const checks = projectMarkers.map((file) =>
        fs
          .access(path.join(currentDir, file))
          .then(() => true)
          .catch(() => false)
      );
      const results = await Promise.all(checks);
      if (results.some((found) => found)) {
        return currentDir;
      }
    } catch {
      // Continue searching if any errors occur
    }
    currentDir = path.dirname(currentDir);
  }
  return process.cwd();
}

/**
 * Finds the Genkit hidden directory containing runtime state files.
 */
export async function findRuntimesDir(projectRoot: string): Promise<string> {
  return path.join(projectRoot, '.genkit', 'runtimes');
}

/**
 * Finds the Genkit hidden directory containing server (UI server, telemetry server, etc) state files.
 */
export async function findServersDir(projectRoot: string): Promise<string> {
  return path.join(projectRoot, '.genkit', 'servers');
}

/**
 * Extract project name (i.e. basename of the project root folder) from the
 * path to a runtime file.
 *
 * e.g.) /path/to/<basename>/.genkit/runtimes/123.json returns <basename>.
 *
 * @param filePath path to a runtime file
 * @returns project name
 */
export function projectNameFromGenkitFilePath(filePath: string): string {
  const parts = filePath.split('/');
  const basePath = parts
    .slice(
      0,
      Math.max(
        parts.findIndex((value) => value === '.genkit'),
        0
      )
    )
    .join('/');
  return basePath === '' ? 'unknown' : path.basename(basePath);
}

/**
 * Detects what runtime is used in the current directory.
 * @returns Runtime of the project directory.
 */
export async function detectRuntime(directory: string): Promise<Runtime> {
  const files = await fs.readdir(directory);
  for (const file of files) {
    const filePath = path.join(directory, file);
    const stat = await fs.stat(filePath);
    if (stat.isFile() && (path.extname(file) === '.go' || file === 'go.mod')) {
      return 'go';
    }
  }
  try {
    await fs.access(path.join(directory, 'package.json'));
    return 'nodejs';
  } catch {
    return undefined;
  }
}

/**
 * Checks the health of a server with a /api/__health endpoint.
 */
export async function checkServerHealth(
  url: string,
  id?: string
): Promise<boolean> {
  try {
    const response = await fetch(`${url}/api/__health${id ? `?id=${id}` : ''}`);
    return response.status === 200;
  } catch (error) {
    if (isConnectionRefusedError(error)) {
      return false;
    }
  }
  return true;
}

/**
 * Waits until the server is healthy or the timeout is reached.
 */
export async function waitUntilHealthy(
  url: string,
  maxTimeout = 10000
): Promise<boolean> {
  const startTime = Date.now();
  while (Date.now() - startTime < maxTimeout) {
    try {
      const response = await fetch(`${url}/api/__health`);
      if (response.status === 200) {
        return true;
      }
    } catch (error) {
      // Ignore errors and continue retrying
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

/**
 * Waits until the server becomes unresponsive or the timeout is reached.
 */
export async function waitUntilUnresponsive(
  url: string,
  maxTimeout = 10000
): Promise<boolean> {
  const startTime = Date.now();
  while (Date.now() - startTime < maxTimeout) {
    try {
      const health = await fetch(`${url}/api/__health`);
    } catch (error) {
      if (isConnectionRefusedError(error)) {
        return true;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

export async function retriable<T>(
  fn: () => Promise<T>,
  opts: { maxRetries?: number; delayMs?: number }
): Promise<T> {
  const maxRetries = opts.maxRetries ?? 3;
  const delayMs = opts.delayMs ?? 0;

  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (e) {
      if (attempt >= maxRetries - 1) {
        throw e;
      }
      if (delayMs > 0) {
        await new Promise((r) => setTimeout(r, delayMs));
      }
    }
    attempt++;
  }
}

/**
 * Checks if the provided data is a valid dev tools server state file.
 */
export function isValidDevToolsInfo(data: any): data is DevToolsInfo {
  return (
    typeof data === 'object' &&
    typeof data.url === 'string' &&
    typeof data.timestamp === 'string'
  );
}

/**
 * Writes the toolsInfo file to the project root
 */
export async function writeToolsInfoFile(url: string, projectRoot: string) {
  const serversDir = await findServersDir(projectRoot);
  const toolsJsonPath = path.join(serversDir, `tools-${process.pid}.json`);
  try {
    const serverInfo = {
      url,
      timestamp: new Date().toISOString(),
    } as DevToolsInfo;
    await fs.mkdir(serversDir, { recursive: true });
    await fs.writeFile(toolsJsonPath, JSON.stringify(serverInfo, null, 2));
    logger.debug(`Tools Info file written: ${toolsJsonPath}`);
  } catch (error) {
    logger.error('Error writing tools config', error);
  }
}

/**
 * Removes the toolsInfo file.
 */
export async function removeToolsInfoFile(
  fileName: string,
  projectRoot: string
) {
  try {
    const serversDir = await findServersDir(projectRoot);
    const filePath = path.join(serversDir, fileName);
    await fs.unlink(filePath);
    logger.debug(`Removed unhealthy toolsInfo file ${fileName} from manager.`);
  } catch (error) {
    logger.debug(`Failed to delete toolsInfo file: ${error}`);
  }
}
