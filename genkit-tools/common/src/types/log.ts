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

import { extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';
import * as z from 'zod';

import { InstrumentationLibrarySchema } from './trace';

extendZodWithOpenApi(z);

/**
 * Zod schema for log record data.
 */
export const LogRecordSchema = z
  .object({
    logId: z.string(), // Server-generated if omitted
    traceId: z.string().optional(),
    spanId: z.string().optional(),
    timestamp: z
      .number()
      .describe('log recorded time in milliseconds since the epoch'),
    severityNumber: z.number().optional(),
    severityText: z.string().optional(),
    body: z.unknown().optional(), // Represents any value: string, number, boolean, array, or object
    attributes: z.record(z.string(), z.unknown()).optional(),
    instrumentationLibrary: InstrumentationLibrarySchema.optional(),
  })
  .openapi('LogRecordData');

export type LogRecordData = z.infer<typeof LogRecordSchema>;

/**
 * Log query parameters.
 */
export const LogQuerySchema = z.object({
  limit: z.coerce.number().optional(),
  continuationToken: z.string().optional(),
  traceId: z.string().optional(),
  spanId: z.string().optional(),
});

export type LogQuery = z.infer<typeof LogQuerySchema>;

export interface LogQueryResponse {
  logs: LogRecordData[];
  continuationToken?: string;
}

export interface LogStore {
  init(): Promise<void>;
  save(logs: LogRecordData[]): Promise<void>;
  list(query?: LogQuery): Promise<LogQueryResponse>;
}
