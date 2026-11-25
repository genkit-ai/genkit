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

import * as clc from 'colorette';
import * as fs from 'fs';
import * as path from 'path';
import { format, stripVTControlCharacters } from 'util';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

let debugFilePath: string | undefined;

export function debugToFile(dirPath?: string) {
  debugFilePath = dirPath
    ? path.join(dirPath, 'genkit-debug.log')
    : 'genkit-debug.log';
  logger.level = 'debug';
}

function shouldLog(currentLevel: string, targetLevel: LogLevel): boolean {
  const currentPriority =
    LEVEL_PRIORITY[currentLevel as LogLevel] ?? LEVEL_PRIORITY.info;
  return currentPriority <= LEVEL_PRIORITY[targetLevel];
}

function formatArgs(args: unknown[]): string {
  if (args.length === 1 && typeof args[0] === 'string') {
    return args[0];
  }
  return format(...args);
}

function formatLine(level: LogLevel, message: string): string {
  let levelColor: clc.Color;
  switch (level) {
    case 'error':
      levelColor = clc.red;
      break;
    case 'warn':
      levelColor = clc.yellow;
      break;
    case 'debug':
    case 'info':
      levelColor = (text) => text.toString();
      break;
  }

  const label = level.charAt(0).toUpperCase() + level.slice(1);
  return `${clc.bold(levelColor(label))}: ${message}`;
}

function emit(level: LogLevel, args: unknown[]): void {
  if (!shouldLog(logger.level, level)) {
    return;
  }

  const message = formatArgs(args);
  process.stderr.write(`${formatLine(level, message)}\n`);

  if (debugFilePath) {
    try {
      const record = JSON.stringify({
        level,
        message: stripVTControlCharacters(message),
        timestamp: new Date().toISOString(),
      });
      fs.appendFileSync(debugFilePath, `${record}\n`);
    } catch {
      // Ignore file-sink errors so logging never crashes the CLI.
    }
  }
}

export interface Logger {
  level: string;
  debug(...args: unknown[]): void;
  info(...args: unknown[]): void;
  warn(...args: unknown[]): void;
  error(...args: unknown[]): void;
}

export const logger: Logger = {
  level: process.env.DEBUG ? 'debug' : 'info',
  debug(...args: unknown[]) {
    emit('debug', args);
  },
  info(...args: unknown[]) {
    emit('info', args);
  },
  warn(...args: unknown[]) {
    emit('warn', args);
  },
  error(...args: unknown[]) {
    emit('error', args);
  },
};
