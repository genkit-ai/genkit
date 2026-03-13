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
import { z } from 'zod';

extendZodWithOpenApi(z);

// `z.any()` generates `{ nullable: true }` with no type set which is not valid OpenAPI spec.
// There's no way around this without defining a custom type. `z.array()` needs an inner type which runs into the same issue.

export const SingularAnySchema = z
  .union([z.string(), z.number(), z.bigint(), z.boolean(), z.object({})])
  .openapi('SingularAny');

export const CustomAnySchema = z
  .union([SingularAnySchema, z.array(SingularAnySchema)])
  .openapi('CustomAny');

// TODO: Figure out a way to do z.custom<JSONSchema7>() and have it generate a nullable object correctly.
export const JSONSchema7Schema = z
  .object({})
  .describe(
    'A JSON Schema Draft 7 (http://json-schema.org/draft-07/schema) object.'
  )
  .nullish()
  .openapi('JSONSchema7');

export const ActionSchema = z
  .object({
    key: z.string().describe('Action key consisting of action type and ID.'),
    name: z.string().describe('Action name.'),
    description: z
      .string()
      .describe('A description of what the action does.')
      .nullish(),
    inputSchema: JSONSchema7Schema,
    outputSchema: JSONSchema7Schema,
    metadata: z
      .record(z.string(), CustomAnySchema)
      .describe('Metadata about the action (e.g. supported model features).')
      .nullish(),
  })
  .openapi('Action');

export type Action = z.infer<typeof ActionSchema>;

export const ActionMetadataSchema = z
  .object({
    actionType: z.string().optional(),
    name: z.string(),
    description: z.string().optional(),
    inputSchema: z.unknown().optional(),
    inputJsonSchema: JSONSchema7Schema.optional(),
    outputSchema: z.unknown().optional(),
    outputJsonSchema: JSONSchema7Schema.optional(),
    streamSchema: z.unknown().optional(),
    metadata: z.record(z.string(), CustomAnySchema).optional(),
  })
  .openapi('ActionMetadata');

export type ActionMetadata = z.infer<typeof ActionMetadataSchema>;

export const RunActionResponseSchema = z.object({
  result: z.unknown().optional(),
  telemetry: z
    .object({
      traceId: z.string().optional(),
    })
    .optional(),
  genkitVersion: z.string().optional(),
});
export type RunActionResponse = z.infer<typeof RunActionResponseSchema>;
