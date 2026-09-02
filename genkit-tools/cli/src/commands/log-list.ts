/**
 * Copyright 2026 Google LLC
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

import type { LogQueryFilter } from '@genkit-ai/tools-common';
import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import {
  findProjectRoot,
  forceStderr,
  logger,
} from '@genkit-ai/tools-common/utils';
import { Command, Option } from 'commander';
import { runWithManager } from '../utils/manager-utils';

export interface LogListOptions {
  limit: string;
  traceId?: string;
  spanId?: string;
  severity?: string;
  continuationToken?: string;
  format: 'text' | 'jsonl';
}

/**
 * Command to list logs. By default, logs are returned in reverse
 * chronological order.
 */
export const logList = new Command('log:list')
  .description(
    'list logs, in reverse chronological order. Filtering by trace-id is highly recommended.'
  )
  .option('-l, --limit <number>', 'limit the number of returned logs', '15')
  .option('--trace-id <id>', 'filter by trace ID')
  .option('--span-id <id>', 'filter by span ID')
  .option(
    '--severity <level>',
    'filter to logs at this severity level and higher (e.g., info, warn, error)'
  )
  .addOption(
    new Option('-f, --format <format>', 'output format')
      .choices(['text', 'jsonl'])
      .default('text')
  )
  .option('--continuation-token <token>', 'continuation token for pagination')
  .action(async (options: LogListOptions) => {
    if (options.format === 'jsonl') forceStderr();
    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      try {
        const filter: LogQueryFilter = {};
        if (options.traceId) {
          filter.traceId = options.traceId;
        }
        if (options.spanId) {
          filter.spanId = options.spanId;
        }
        if (options.severity) {
          const severities: Record<string, number> = {
            trace: 1,
            debug: 5,
            info: 9,
            warn: 13,
            error: 17,
            fatal: 21,
          };
          const num = severities[options.severity.toLowerCase()];
          if (num !== undefined) {
            filter.severityNumber = num;
          } else {
            // Fallback for custom severity texts
            filter.severityText = options.severity;
          }
        }

        const limit = Number.parseInt(options.limit, 10);
        if (Number.isNaN(limit) || limit <= 0) {
          logger.error(
            `Invalid limit: "${options.limit}". It must be a positive integer.`
          );
          return;
        }

        const listRequest = {
          limit,
          continuationToken: options.continuationToken,
          filter: Object.keys(filter).length > 0 ? filter : undefined,
        };

        const response = await manager.listLogs(listRequest);

        if (!response || !response.logs || response.logs.length === 0) {
          logger.info('No logs found.');
          return;
        }

        const logs = response.logs;

        if (options.format === 'jsonl') {
          logs.forEach((log) => {
            console.log(JSON.stringify(log));
          });
        } else {
          console.log(
            `Found ${logs.length} log${logs.length === 1 ? '' : 's'}:\n`
          );
          logs.forEach((log) => {
            let time = 'unknown';
            if (
              typeof log.timestamp === 'number' &&
              Number.isFinite(log.timestamp)
            ) {
              try {
                time = new Date(log.timestamp).toLocaleString();
              } catch {
                // Fallback to 'unknown' if timestamp is out of range
              }
            }

            const id = log.logId || 'unknown';
            const severity = log.severityText || 'unknown';
            const message = formatBody(log.body);
            const attributes = formatAttributes(log.attributes);

            console.log(`ID:       ${id}`);
            if (!options.traceId && log.traceId)
              console.log(`Trace ID: ${log.traceId}`);
            if (!options.spanId && log.spanId)
              console.log(`Span ID:  ${log.spanId}`);
            console.log(`Severity: ${severity}`);
            console.log(`Time:     ${time}`);
            if (message) console.log(`Message:  ${message}`);
            if (attributes) console.log(`Attrs:    ${attributes}`);

            console.log('---');
          });
        }

        if (response.continuationToken) {
          if (options.format === 'jsonl') {
            logger.info(
              `To get the next page, use: --continuation-token ${response.continuationToken}`
            );
          } else {
            console.log(
              `\nTo get the next page, use: --continuation-token ${response.continuationToken}`
            );
          }
        }
      } catch (e) {
        logger.error(`Error listing logs: ${e}`);
      }
    };

    await runWithManager(projectRoot, runAction);
  });

function formatBody(value: unknown): string {
  if (value === undefined || value === null) return '';
  let strValue: string;
  if (typeof value === 'object') {
    try {
      strValue = JSON.stringify(value);
    } catch {
      strValue = '[Object]';
    }
  } else {
    strValue = String(value);
  }

  // If it's a long string and doesn't match patterns, limit it
  return strValue.length > 100 ? strValue.substring(0, 100) + '...' : strValue;
}

function formatAttributes(
  attributes: Record<string, unknown> | undefined
): string {
  if (!attributes || Object.keys(attributes).length === 0) return '';
  const pairs = Object.entries(attributes).map(([key, value]) => {
    let strValue: string;
    if (typeof value === 'object' && value !== null) {
      try {
        strValue = JSON.stringify(value);
      } catch {
        strValue = '[Object]';
      }
    } else {
      strValue = String(value);
    }
    const truncated =
      strValue.length > 50 ? strValue.substring(0, 50) + '...' : strValue;
    return `${key}=${truncated}`;
  });

  const fullString = pairs.join(', ');
  return fullString.length > 100
    ? fullString.substring(0, 100) + '...'
    : fullString;
}
