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

import { Validator } from '@cfworker/json-schema';
import Ajv, { type ErrorObject, type JSONSchemaType } from 'ajv';
import addFormats from 'ajv-formats';
import { z } from 'zod';
import { getGenkitRuntimeConfig } from './config.js';
import { GenkitError } from './error.js';

import type { Registry } from './registry.js';
const ajv = new Ajv();
addFormats(ajv);

export { z }; // provide a consistent zod to use throughout genkit

/**
 * JSON schema.
 */
export type JSONSchema = JSONSchemaType<any> | any;

const jsonSchemas = new WeakMap<z.ZodType, JSONSchema>();
const ajvValidators = new WeakMap<JSONSchema, ReturnType<typeof ajv.compile>>();
const cfWorkerValidators = new WeakMap<JSONSchema, Validator>();

/**
 * Annotates a Zod schema with UI-specific metadata.
 *
 * Mutates the global zod metadata registry for this exact instance (unlike
 * `.meta()`, which returns a clone), so existing references pick it up.
 *
 * NOTE: It's typically recommended to use x-genkit-* (or similar) as the prefix.
 */
export function annotateSchema<T extends z.ZodType<any, any>>(
  schema: T,
  annotations: Record<string, any>
): T {
  const current = z.globalRegistry.get(schema) ?? {};
  z.globalRegistry.add(schema, { ...current, ...annotations });
  return schema;
}

/**
 * Wrapper object for various ways schema can be provided.
 */
export interface ProvidedSchema {
  jsonSchema?: JSONSchema;
  schema?: z.ZodType;
}

/**
 * Schema validation error.
 */
export class ValidationError extends GenkitError {
  constructor({
    data,
    errors,
    schema,
  }: {
    data: any;
    errors: ValidationErrorDetail[];
    schema: JSONSchema;
  }) {
    super({
      status: 'INVALID_ARGUMENT',
      message: `Schema validation failed. Parse Errors:\n\n${errors.map((e) => `- ${e.path}: ${e.message}`).join('\n')}\n\nProvided data:\n\n${JSON.stringify(data, null, 2)}\n\nRequired JSON schema:\n\n${JSON.stringify(schema, null, 2)}`,
      detail: { errors, schema },
    });
  }
}

/**
 * Converts a Zod schema into a JSON schema, utilizing an in-memory cache for known objects.
 * @param options Provide a json schema and/or zod schema. JSON schema has priority.
 * @returns A JSON schema.
 */
export function toJsonSchema({
  jsonSchema,
  schema,
}: ProvidedSchema): JSONSchema | undefined {
  // if neither jsonSchema or schema is present return undefined
  if (!jsonSchema && !schema) return null;
  if (jsonSchema) return jsonSchema;
  if (jsonSchemas.has(schema!)) return jsonSchemas.get(schema!)!;
  // Note: the JSON schema cache is keyed by instance, so annotations added via
  // annotateSchema after the first conversion are not reflected.
  const outSchema = z.toJSONSchema(schema!, {
    target: 'draft-07',
    // JSON schema is used to validate raw (pre-parse) data and to describe
    // inputs to models/tools, so describe the input side of transforms/defaults.
    io: 'input',
    // z.bigint(), z.symbol(), etc. become {} instead of throwing.
    unrepresentable: 'any',
    override: compatOverride,
  });
  jsonSchemas.set(schema!, outSchema);
  return outSchema;
}

/**
 * Keeps zod v4 JSON schema output close to what zod-to-json-schema (zod v3)
 * produced, where the difference matters to model providers.
 */
function compatOverride(ctx: {
  zodSchema: z.core.$ZodTypes;
  jsonSchema: z.core.JSONSchema.BaseSchema;
}) {
  const { zodSchema, jsonSchema } = ctx;
  const def = zodSchema._zod.def;
  if (def.type === 'date') {
    // v4 treats dates as unrepresentable. Data crosses the wire as ISO strings.
    jsonSchema.type = 'string';
    jsonSchema.format = 'date-time';
  } else if (
    def.type === 'union' &&
    'discriminator' in def &&
    jsonSchema.oneOf
  ) {
    // Discriminated unions emit `oneOf` in v4. Several providers (Gemini,
    // OpenAI structured outputs) only support `anyOf`; the branches are
    // disjoint by construction, so `anyOf` is equivalent. Other `oneOf`
    // unions (`z.xor()`) can overlap, so they are left alone.
    jsonSchema.anyOf = jsonSchema.oneOf;
    delete jsonSchema.oneOf;
  } else if (
    def.type === 'record' &&
    typeof jsonSchema.propertyNames === 'object' &&
    jsonSchema.propertyNames.type === 'string' &&
    Object.keys(jsonSchema.propertyNames).length === 1
  ) {
    // `propertyNames: { type: 'string' }` is a no-op for JSON objects, and
    // not supported by some provider schema subsets.
    delete jsonSchema.propertyNames;
  } else if (def.type === 'number') {
    // `.int()` adds safe-integer bounds; they're noise for models.
    if (jsonSchema.minimum === Number.MIN_SAFE_INTEGER)
      delete jsonSchema.minimum;
    if (jsonSchema.maximum === Number.MAX_SAFE_INTEGER)
      delete jsonSchema.maximum;
  } else if (def.type === 'string' && typeof jsonSchema.format === 'string') {
    if (VALIDATED_FORMATS.has(jsonSchema.format)) {
      // String formats (datetime, email, uuid, ...) also emit zod's internal
      // regex as `pattern`. `format` alone is what v3 emitted, is validated by
      // ajv-formats/cfworker, and keeps validation errors readable. A
      // user-supplied `.regex()` on top of a format is emitted under `allOf`
      // and is not affected.
      delete jsonSchema.pattern;
    } else {
      // zod-specific formats (cuid, base64, starts_with, ...): `pattern` is
      // their only check, and the non-standard `format` makes strict ajv throw
      // and some providers reject the schema. v3 emitted just the pattern.
      delete jsonSchema.format;
    }
  } else if (def.type === 'nullable') {
    collapseNullablePrimitive(jsonSchema);
  }
}

const PRIMITIVE_TYPES = new Set(['string', 'number', 'integer', 'boolean']);

/** Formats validated by both ajv-formats and @cfworker/json-schema. */
const VALIDATED_FORMATS = new Set([
  'date-time',
  'date',
  'time',
  'duration',
  'email',
  'hostname',
  'ipv4',
  'ipv6',
  'uri',
  'uri-reference',
  'uri-template',
  'uuid',
  'json-pointer',
  'relative-json-pointer',
  'regex',
]);

/**
 * v4 emits `anyOf: [{ type: 'string', description }, { type: 'null' }]` for a
 * described nullable primitive, where v3 emitted
 * `{ type: ['string', 'null'], description }`. Many provider converters only
 * understand the type array form, so fold it back. Objects/enums keep `anyOf`
 * (same as v3).
 */
function collapseNullablePrimitive(jsonSchema: z.core.JSONSchema.BaseSchema) {
  const anyOf = jsonSchema.anyOf;
  if (!Array.isArray(anyOf) || anyOf.length !== 2) return;
  const nullIdx = anyOf.findIndex(
    (s) =>
      typeof s === 'object' && s.type === 'null' && Object.keys(s).length === 1
  );
  if (nullIdx === -1) return;
  const inner = anyOf[1 - nullIdx];
  if (
    typeof inner !== 'object' ||
    typeof inner.type !== 'string' ||
    !PRIMITIVE_TYPES.has(inner.type) ||
    inner.enum !== undefined ||
    inner.const !== undefined ||
    // e.g. several `.regex()` checks emit `allOf: [{ type: 'string', ... }]`,
    // which would reject null once hoisted next to `type: [X, 'null']`.
    inner.allOf !== undefined ||
    inner.anyOf !== undefined ||
    inner.oneOf !== undefined ||
    inner.not !== undefined
  ) {
    return;
  }
  // Keys on the nullable itself (e.g. `.nullable().describe()`) win over the
  // inner schema's.
  const { anyOf: _, ...outer } = jsonSchema;
  delete jsonSchema.anyOf;
  Object.assign(jsonSchema, inner, outer, { type: [inner.type, 'null'] });
}

/**
 * Schema validation error details.
 */
export interface ValidationErrorDetail {
  path: string;
  message: string;
}

function ajvErrorToValidationErrorDetail(
  error: ErrorObject
): ValidationErrorDetail {
  return {
    path: error.instancePath.substring(1).replace(/\//g, '.') || '(root)',
    message: error.message!,
  };
}

function cfWorkerErrorToValidationErrorDetail(error: {
  instanceLocation: string;
  error: string;
}): ValidationErrorDetail {
  const path = error.instanceLocation.startsWith('#/')
    ? error.instanceLocation.substring(2)
    : '';
  return {
    path: path.replace(/\//g, '.') || '(root)',
    message: error.error,
  };
}

/**
 * Validation response.
 */
export type ValidationResponse =
  | {
      valid: true;
      errors?: undefined;
      schema: JSONSchema;
    }
  | {
      valid: false;
      errors: ValidationErrorDetail[];
      schema: JSONSchema;
    };

/**
 * Validates the provided data against the provided schema.
 */
export function validateSchema(
  data: unknown,
  options: ProvidedSchema
): ValidationResponse {
  const toValidate = toJsonSchema(options);
  if (!toValidate) {
    return { valid: true, schema: toValidate };
  }
  const validationMode = getGenkitRuntimeConfig().jsonSchemaMode;

  if (validationMode === 'interpret') {
    let validator = cfWorkerValidators.get(toValidate);
    if (!validator) {
      validator = new Validator(toValidate);
      cfWorkerValidators.set(toValidate, validator);
    }

    const result = validator.validate(sanitizeForJsonSchema(data));
    if (!result.valid) {
      return {
        valid: false,
        errors: result.errors.map(cfWorkerErrorToValidationErrorDetail),
        schema: toValidate,
      };
    }

    return {
      valid: result.valid,
      schema: toValidate,
    };
  }

  let validator = ajvValidators.get(toValidate);
  if (!validator) {
    validator = ajv.compile(toValidate);
    ajvValidators.set(toValidate, validator);
  }

  const valid = validator(data) as boolean;
  if (!valid) {
    return {
      valid: false,
      errors: (validator.errors ?? []).map(ajvErrorToValidationErrorDetail),
      schema: toValidate,
    };
  }

  return { valid, schema: toValidate };
}

/**
 * Parses raw data object against the provided schema.
 */
export function parseSchema<T = unknown>(
  data: unknown,
  options: ProvidedSchema
): T {
  const result = validateSchema(data, options);
  if (!result.valid) {
    throw new ValidationError({
      data,
      errors: result.errors,
      schema: result.schema,
    });
  }
  return data as T;
}

/**
 * Registers provided schema as a named schema object in the Genkit registry.
 *
 * @hidden
 */
export function defineSchema<T extends z.ZodType<any, any>>(
  registry: Registry,
  name: string,
  schema: T
): T {
  registry.registerSchema(name, { schema });
  return schema;
}

/**
 * Registers provided JSON schema as a named schema object in the Genkit registry.
 *
 * @hidden
 */
export function defineJsonSchema(
  registry: Registry,
  name: string,
  jsonSchema: JSONSchema
) {
  registry.registerSchema(name, { jsonSchema });
  return jsonSchema;
}

function sanitizeForJsonSchema(data: any): any {
  if (Array.isArray(data)) {
    return data.map(sanitizeForJsonSchema);
  } else if (data !== null && typeof data === 'object') {
    const out: any = {};
    for (const key in data) {
      if (data[key] !== undefined) {
        out[key] = sanitizeForJsonSchema(data[key]);
      }
    }
    return out;
  }
  return data;
}
