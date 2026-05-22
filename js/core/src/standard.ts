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

import type { StandardSchemaV1 } from '@standard-schema/spec';
import type { z } from 'zod';
import type z4type from 'zod/v4';

export type { StandardSchemaV1 };

/**
 * A schema accepted by Genkit at its public API boundaries. May be:
 * - A Zod v3 schema (the default `z` from `genkit`)
 * - A Zod v4 schema (the `z4` from `genkit`)
 * - Any Standard Schema v1-compliant schema (Valibot, ArkType, Effect/Schema,
 *   etc.)
 *
 * All three kinds carry inferred TypeScript input/output types and support
 * runtime validation. Only Zod schemas support automatic JSON Schema generation
 * for LLM tool descriptions and the Genkit dev UI; other Standard Schema
 * implementations should also provide an explicit `jsonSchema` when used with
 * `defineTool`, `defineFlow`, or `definePrompt`.
 */
export type GenkitSchema = z.ZodTypeAny | z4type.ZodType | StandardSchemaV1;

// ─── Runtime discriminators ───────────────────────────────────────────────────

/**
 * Returns true if `s` is a Zod v3 schema (`ZodType` with `._def` property).
 * Zod v3 schemas also implement StandardSchemaV1, but we detect them first so
 * we can use their native `.parse()` and `zod-to-json-schema` integration.
 */
export function isZodV3Schema(s: GenkitSchema): s is z.ZodTypeAny {
  return (
    s != null &&
    typeof s === 'object' &&
    '_def' in s &&
    '~standard' in s &&
    (s as any)['~standard']?.vendor === 'zod' &&
    (s as any)['~standard']?.version === 1 &&
    // Distinguish v3 from v4: v3 uses ZodType, v4 uses $ZodType (no _def in v4)
    // v4 schemas do have _def but their vendor is 'zod' too — use the parse API
    // shape to distinguish: v3 has .parse(), v4 has .parse() but also ._zod.
    !('_zod' in s)
  );
}

/**
 * Returns true if `s` is a Zod v4 schema (`$ZodType` with `._zod` property).
 * Zod v4 schemas also implement StandardSchemaV1.
 */
export function isZodV4Schema(s: GenkitSchema): s is z4type.ZodType {
  return (
    s != null &&
    typeof s === 'object' &&
    '_zod' in s &&
    '~standard' in s &&
    (s as any)['~standard']?.vendor === 'zod'
  );
}

/**
 * Returns true if `s` carries a Standard Schema `~standard` interface but is
 * NOT a Zod schema (i.e., is Valibot, ArkType, Effect/Schema, etc.).
 */
export function isNonZodStandardSchema(s: GenkitSchema): s is StandardSchemaV1 {
  return (
    s != null &&
    typeof s === 'object' &&
    '~standard' in s &&
    !isZodV3Schema(s) &&
    !isZodV4Schema(s)
  );
}

// ─── Type-level inference helpers ────────────────────────────────────────────

/**
 * Infers the **input** type from a `GenkitSchema` — i.e., the type of the
 * value a caller provides *before* validation/transformation.
 *
 * For Zod v3: equivalent to `z.input<S>`.
 * For Zod v4: equivalent to `z4.input<S>`.
 * For other Standard Schema implementations: `StandardSchemaV1.InferInput<S>`.
 */
export type InferInput<S extends GenkitSchema> = S extends z.ZodTypeAny
  ? z.input<S>
  : S extends z4type.ZodType
    ? z4type.input<S>
    : S extends StandardSchemaV1
      ? StandardSchemaV1.InferInput<S>
      : never;

/**
 * Infers the **output** type from a `GenkitSchema` — i.e., the type of the
 * value produced *after* validation/transformation (what runner callbacks
 * receive).
 *
 * For Zod v3: equivalent to `z.output<S>` (same as `z.infer<S>`).
 * For Zod v4: equivalent to `z4.output<S>`.
 * For other Standard Schema implementations: `StandardSchemaV1.InferOutput<S>`.
 */
export type InferOutput<S extends GenkitSchema> = S extends z.ZodTypeAny
  ? z.output<S>
  : S extends z4type.ZodType
    ? z4type.output<S>
    : S extends StandardSchemaV1
      ? StandardSchemaV1.InferOutput<S>
      : never;
