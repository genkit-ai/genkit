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

import type { TraceQueryFilter } from '@genkit-ai/tools-common';
import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import {
  findProjectRoot,
  logger,
  stackTraceSpans,
} from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import { runWithManager } from '../utils/manager-utils';

function formatTruncated(val: any, maxLen = 100): string {
  if (val === undefined) return '';
  const parsed = typeof val === 'string' ? val : JSON.stringify(val);
  return parsed.length > maxLen
    ? parsed.substring(0, maxLen - 3) + '...'
    : parsed;
}

export interface TraceListOptions {
  limit: string;
  status?: string;
  type?: string;
  name?: string;
  continuationToken?: string;
}

/**
 * Command to list traces. By default, traces are returned in reverse
 * chronological order.
 */
export const traceList = new Command('trace:list')
  .description('list traces')
  .option('-l, --limit <number>', 'limit the number of returned traces', '15')
  .option('--status <status>', 'filter by root span status')
  .option('--type <type>', 'filter by root span type')
  .option('--name <name>', 'filter by root span name')
  .option('--continuation-token <token>', 'continuation token for pagination')
  .action(async (options: TraceListOptions) => {
    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      try {
        const eqFilter: Record<string, any[]> = {};
        if (options.status) {
          // Status mapped exactly as dev UI enum (Success: 0, Error: 2)
          const statusVal =
            options.status.toLowerCase() === 'error'
              ? 2
              : options.status.toLowerCase() === 'success'
                ? 0
                : Number(options.status);
          eqFilter['status'] = [statusVal];
        }
        if (options.type) {
          eqFilter['type'] = [options.type];
        }
        if (options.name) {
          eqFilter['name'] = [options.name];
        }

        const filter: TraceQueryFilter = {
          eq: Object.keys(eqFilter).length > 0 ? eqFilter : undefined,
          neq: { 'genkitx:ignore-trace': ['true'] },
        };

        const listRequest = {
          limit: Number.parseInt(options.limit, 10),
          continuationToken: options.continuationToken,
          filter,
        };

        const response = await manager.listTraces(listRequest);

        if (!response || !response.traces || response.traces.length === 0) {
          logger.info('No traces found.');
          return;
        }

        console.log(`Found ${response.traces.length} traces:\n`);

        response.traces.forEach((trace) => {
          let duration = 'unknown';
          let time = 'unknown';
          if (trace.startTime) {
            time = new Date(trace.startTime).toLocaleString();
            if (trace.endTime) {
              duration = `${trace.endTime - trace.startTime}ms`;
            }
          }

          let status = 'unknown';
          let type = 'unknown';
          let inputStr = '';
          let outputStr = '';

          const rootSpan = stackTraceSpans(trace);
          if (rootSpan) {
            const attrs = rootSpan.attributes || {};
            status =
              (attrs['genkit:state'] as string) ||
              (rootSpan.status?.code !== undefined
                ? String(rootSpan.status.code)
                : 'unknown');
            type =
              (attrs['genkit:metadata:subtype'] as string) ||
              (attrs['genkit:type'] as string) ||
              'unknown';

            inputStr = formatTruncated(attrs['genkit:input']);
            outputStr = formatTruncated(attrs['genkit:output']);
          }

          console.log(`ID:       ${trace.traceId}`);
          console.log(`Type:     ${type}`);
          console.log(`Name:     ${trace.displayName || 'unknown'}`);
          console.log(`Status:   ${status}`);
          console.log(`Time:     ${time}`);
          console.log(`Duration: ${duration}`);
          if (inputStr) console.log(`Input:    ${inputStr}`);
          if (outputStr) console.log(`Output:   ${outputStr}`);

          console.log('---');
        });

        if (response.continuationToken) {
          console.log(
            `\nTo get the next page, use: --continuation-token ${response.continuationToken}`
          );
        }
      } catch (e) {
        logger.error(`Error listing traces: ${e}`);
      }
    };

    await runWithManager(projectRoot, runAction);
  });
