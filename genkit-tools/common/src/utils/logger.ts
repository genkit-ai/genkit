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
import * as path from 'path';
import * as winston from 'winston';

const defaultTransport = new winston.transports.Console();

export function forceStderr() {
  logger.remove(defaultTransport);
  logger.add(
    // log to stderr to avoid
    new winston.transports.Stream({
      stream: process.stderr,
    })
  );
}

export function debugToFile(dirPath?: string) {
  logger.add(
    new winston.transports.File({
      filename: dirPath
        ? path.join(dirPath, 'genkit-debug.log')
        : 'genkit-debug.log',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      ),
    })
  );
  logger.level = 'debug';
}

export const logger = winston.createLogger({
  level: process.env.DEBUG ? 'debug' : 'info',
  format: winston.format.printf((log) => {
    // Anything logged at 'info' level will show as just the plain message
    if (log.level === 'info') return log.message as string;

    let levelColor: clc.Color;
    switch (log.level) {
      case 'error':
        levelColor = clc.red;
        break;
      case 'warn':
        levelColor = clc.yellow;
        break;
      default:
        // Default is nothing.
        levelColor = (text) => text.toString();
        break;
    }

    const level = log.level.charAt(0).toUpperCase() + log.level.slice(1);
    return `${clc.bold(levelColor(level))}: ${log.message}`;
  }),
  transports: [defaultTransport],
});
