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

import { context } from '@opentelemetry/api';
import { logs, SeverityNumber } from '@opentelemetry/api-logs';

const LOG_LEVELS = ['debug', 'info', 'warn', 'error'];

const loggerKey = '__genkit_logger';

const _defaultLogger = {
  shouldLog(targetLevel: string) {
    return LOG_LEVELS.indexOf(this.level) <= LOG_LEVELS.indexOf(targetLevel);
  },
  debug(...args: any) {
    this.shouldLog('debug') && console.debug(...args);
  },
  info(...args: any) {
    this.shouldLog('info') && console.info(...args);
  },
  warn(...args: any) {
    this.shouldLog('warn') && console.warn(...args);
  },
  error(...args: any) {
    this.shouldLog('error') && console.error(...args);
  },
  level: 'info',
};

function getLogger() {
  if (!global[loggerKey]) {
    global[loggerKey] = _defaultLogger;
  }
  return global[loggerKey];
}

class Logger {
  readonly defaultLogger = _defaultLogger;

  private _emitOtel(
    level: string,
    args: any[],
    explicitBody?: string,
    explicitAttributes?: Record<string, any>
  ) {
    if (process.env.GENKIT_OTEL_ENABLE_LOGS !== 'true') {
      return;
    }

    try {
      const currentLevel = getLogger().level || 'info';
      if (LOG_LEVELS.indexOf(currentLevel) > LOG_LEVELS.indexOf(level)) {
        return;
      }

      const otelLogger = logs.getLogger('genkit-logger');
      let severityNumber: SeverityNumber;
      switch (level) {
        case 'debug':
          severityNumber = SeverityNumber.DEBUG;
          break;
        case 'info':
          severityNumber = SeverityNumber.INFO;
          break;
        case 'warn':
          severityNumber = SeverityNumber.WARN;
          break;
        case 'error':
          severityNumber = SeverityNumber.ERROR;
          break;
        default:
          severityNumber = SeverityNumber.UNSPECIFIED;
          break;
      }

      let body;
      const attributes: Record<string, any> = explicitAttributes || {};
      if (explicitBody !== undefined) {
        body = explicitBody;
      } else if (args.length === 1 && typeof args[0] === 'string') {
        body = args[0];
      } else {
        const util = require('util');
        body = util.format(...args);
      }

      let activeContext;
      try {
        activeContext = context.active();
      } catch (e) {
        // No-op if @opentelemetry/api trace is uninitialized or missing right now
      }

      otelLogger.emit({
        severityNumber,
        severityText: level.toUpperCase(),
        body,
        attributes,
        ...(activeContext ? { context: activeContext } : {}),
      });
    } catch (err) {
      // safe ignore
    }
  }

  init(fn: any) {
    global[loggerKey] = fn;
  }

  info(...args: any) {
    // eslint-disable-next-line prefer-spread
    getLogger().info.apply(getLogger(), args);
    this._emitOtel('info', args);
  }
  debug(...args: any) {
    // eslint-disable-next-line prefer-spread
    getLogger().debug.apply(getLogger(), args);
    this._emitOtel('debug', args);
  }
  error(...args: any) {
    // eslint-disable-next-line prefer-spread
    getLogger().error.apply(getLogger(), args);
    this._emitOtel('error', args);
  }
  warn(...args: any) {
    // eslint-disable-next-line prefer-spread
    getLogger().warn.apply(getLogger(), args);
    this._emitOtel('warn', args);
  }

  setLogLevel(level: 'error' | 'warn' | 'info' | 'debug') {
    getLogger().level = level;
  }

  logStructured(msg: string, metadata: any) {
    getLogger().info(msg, metadata);
    this._emitOtel('info', [], msg, metadata);
  }

  logStructuredError(msg: string, metadata: any) {
    getLogger().error(msg, metadata);
    this._emitOtel('error', [], msg, metadata);
  }
}

/**
 * Genkit logger.
 *
 * ```ts
 * import { logger } from 'genkit/logging';
 *
 * logger.setLogLevel('debug');
 * ```
 */
export const logger = new Logger();
