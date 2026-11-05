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

import { LoggingWinston } from '@google-cloud/logging-winston';
import { getCurrentEnv } from 'genkit';
import { logger } from 'genkit/logging';
import type { Writable } from 'stream';
import type { GcpTelemetryConfig } from './types.js';
import { loggingDenied, loggingDeniedHelpText } from './utils.js';

/**
 * Additional streams for writing log data to. Useful for unit testing.
 */
let additionalStream: Writable;
let useJsonFormatOverride = false;

/**
 * Provides a logger for exporting Genkit debug logs to GCP Cloud
 * logs.
 */
export class GcpLogger {
  constructor(private readonly config: GcpTelemetryConfig) {}

  async getLogger(env: string) {
    // Dynamically importing winston here more strictly controls
    // the import order relative to registering instrumentation
    // with OpenTelemetry. Incorrect import order will trigger
    // an internal OT warning and will result in logs not being
    // associated with correct spans/traces.
    const winston = await import('winston');

    const format =
      useJsonFormatOverride || this.shouldExport(env)
        ? { format: winston.format.json() }
        : {
            format: winston.format.printf((info): string => {
              return `[${info.level}] ${info.message}`;
            }),
          };

    const transports: any[] = [];
    if (this.shouldExport(env)) {
      transports.push(
        new LoggingWinston({
          labels: { module: 'genkit' },
          prefix: 'genkit',
          logName: 'genkit_log',
          projectId: this.config.projectId,
          credentials: this.config.credentials,
          autoRetry: true,
          defaultCallback: await this.getErrorHandler(),
        })
      );
    } else if (!additionalStream) {
      // When a test stream is registered it replaces the console. Tests write
      // very large log lines (truncation cases) and echoing them to stdout
      // makes CI log capture stall for minutes.
      transports.push(new winston.transports.Console());
    }
    if (additionalStream) {
      transports.push(
        new winston.transports.Stream({ stream: additionalStream })
      );
    }
    return winston.createLogger({
      transports: transports,
      ...format,
      exceptionHandlers: [new winston.transports.Console()],
    });
  }

  private async getErrorHandler(): Promise<(err: Error | null) => void> {
    // only log the first time
    let instructionsLogged = false;
    const helpInstructions = await loggingDeniedHelpText();

    return async (err: Error | null) => {
      // Use the defaultLogger so that logs don't get swallowed by
      // the open telemetry exporter
      const defaultLogger = logger.defaultLogger;
      if (err && loggingDenied(err)) {
        if (!instructionsLogged) {
          instructionsLogged = true;
          defaultLogger.error(
            `Unable to send logs to Google Cloud: ${err.message}\n\n${helpInstructions}\n`
          );
        }
      } else if (err) {
        defaultLogger.error(`Unable to send logs to Google Cloud: ${err}`);
      }

      if (err) {
        // Assume the logger is compromised, and we need a new one
        // Reinitialize the genkit logger with a new instance with the same config
        logger.init(
          await new GcpLogger(this.config).getLogger(getCurrentEnv())
        );
        defaultLogger.info('Initialized a new GcpLogger.');
      }
    };
  }

  private shouldExport(env?: string) {
    return this.config.export;
  }
}

/**
 * Routes logs to `stream` instead of the console (export transport, if
 * enabled, is unaffected).
 * @hidden
 */
export function __addTransportStreamForTesting(stream: Writable) {
  additionalStream = stream;
}

/** @hidden */
export function __useJsonFormatForTesting() {
  useJsonFormatOverride = true;
}
