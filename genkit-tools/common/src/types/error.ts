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

import { z } from 'zod';

/**
 * Zod schema for the canonical Genkit error wire shape
 * (`{status, message, details}`). This is the form runtimes use when an
 * error travels as data inside another value (e.g. agent outputs and
 * session snapshots), matching `HttpErrorWireFormat` in the JS runtime
 * and `GenkitError`'s wire form in the Go runtime.
 *
 * Not to be confused with {@link GenkitErrorSchema} below, which is the
 * reflection API's HTTP error envelope.
 */
export const RuntimeErrorSchema = z.object({
  /** Canonical status name (e.g. `INTERNAL`, `FAILED_PRECONDITION`). */
  status: z.string().optional(),
  /** Human-readable error message. */
  message: z.string(),
  /** Optional structured details describing the failure. */
  details: z.any().optional(),
});
export type RuntimeError = z.infer<typeof RuntimeErrorSchema>;

/**
 * Zod schema for the error envelope returned by a runtime's reflection
 * API on failed HTTP requests, including debugging context (stack,
 * trace ID) that the dev UI surfaces. Despite the name, this is a
 * transport-layer shape; errors carried as data inside values use
 * {@link RuntimeErrorSchema}.
 */
export const GenkitErrorSchema = z.object({
  message: z.string(),
  stack: z.string().optional(),
  details: z.any().optional(),
  data: z
    .object({
      genkitErrorMessage: z.string().optional(),
      genkitErrorDetails: z
        .object({
          stack: z.string().optional(),
          traceId: z.string(),
        })
        .optional(),
    })
    .optional(),
});

export type GenkitError = z.infer<typeof GenkitErrorSchema>;
