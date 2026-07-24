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

  private _log(level: string, ...args: any[]) {
    const currentLevel = getLogger().level || 'info';
    if (LOG_LEVELS.indexOf(currentLevel) > LOG_LEVELS.indexOf(level)) {
      return;
    }

    if (args.length === 0) return;

    let msg = args[0];
    let metadata: any = {};

    if (
      typeof msg === 'object' &&
      msg !== null &&
      !(msg instanceof Error) &&
      'message' in msg
    ) {
      metadata = this._mergeErrorMetadata(msg.metadata, msg.error);
      msg = msg.message;
    } else if (msg instanceof Error) {
      metadata = this._mergeErrorMetadata(undefined, msg);
      msg = msg.message || String(msg);
    } else if (typeof msg === 'string') {
      const second = args[1];
      const third = args[2];
      if (second instanceof Error) {
        metadata = this._mergeErrorMetadata(undefined, second);
      } else if (
        second &&
        typeof second === 'object' &&
        !Array.isArray(second)
      ) {
        metadata = this._mergeErrorMetadata(
          second,
          third instanceof Error ? third : undefined
        );
      } else if (args.length > 1) {
        getLogger()[level].apply(getLogger(), args);
        this._emitOtel(level, args);
        return;
      }
    } else {
      getLogger()[level].apply(getLogger(), args);
      this._emitOtel(level, args);
      return;
    }

    if (Object.keys(metadata).length > 0) {
      getLogger()[level](msg, metadata);
    } else {
      getLogger()[level](msg);
    }
    this._emitOtel(level, [], msg, metadata);
  }

  init(fn: any) {
    global[loggerKey] = fn;
  }

  info(payload: {
    message: string;
    metadata?: Record<string, any>;
    error?: any;
  }): void;
  info(message: any, metadata?: any, error?: any): void;
  info(...args: any[]) {
    this._log('info', ...args);
  }

  debug(payload: {
    message: string;
    metadata?: Record<string, any>;
    error?: any;
  }): void;
  debug(message: any, metadata?: any, error?: any): void;
  debug(...args: any[]) {
    this._log('debug', ...args);
  }

  error(payload: {
    message: string;
    metadata?: Record<string, any>;
    error?: any;
  }): void;
  error(message: any, metadata?: any, error?: any): void;
  error(...args: any[]) {
    this._log('error', ...args);
  }

  warn(payload: {
    message: string;
    metadata?: Record<string, any>;
    error?: any;
  }): void;
  warn(message: any, metadata?: any, error?: any): void;
  warn(...args: any[]) {
    this._log('warn', ...args);
  }

  setLogLevel(level: 'error' | 'warn' | 'info' | 'debug') {
    getLogger().level = level;
  }

  private _mergeErrorMetadata(metadata: any, err?: any): any {
    const mergedMetadata = { ...metadata };
    if (err) {
      mergedMetadata['exception.type'] = err.name || 'Error';
      mergedMetadata['exception.message'] = err.message || String(err);
      if (err.stack) {
        mergedMetadata['exception.stacktrace'] = err.stack;
      }
    }
    return mergedMetadata;
  }

  /**
   * @deprecated Use `logger.info(...)` instead.
   */
  logStructured(msg: string, metadata: any, err?: any) {
    this.info(msg, metadata, err);
  }

  /**
   * @deprecated Use `logger.error(...)` instead.
   */
  logStructuredError(msg: string, metadata: any, err?: any) {
    this.error(msg, metadata, err);
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
