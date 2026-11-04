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
import type {
  StandardJSONSchemaV1,
  StandardSchemaV1,
} from '@standard-schema/spec';
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

/**
 * Any schema genkit can both validate with and describe as JSON schema:
 * implements Standard Schema and Standard JSON Schema
 * (https://standardschema.dev). Zod 4.2+ and ArkType schemas qualify as is;
 * Valibot schemas via `toStandardJsonSchema()` from `@valibot/to-json-schema`.
 */
export interface GenkitSchema<Input = any, Output = Input> {
  readonly '~standard': StandardSchemaV1.Props<Input, Output> &
    StandardJSONSchemaV1.Props<Input, Output>;
}

/** The type a caller passes in (before defaults/transforms are applied). */
export type InferInput<S extends GenkitSchema> = StandardSchemaV1.InferInput<S>;

/** The type the schema produces (after defaults/transforms are applied). */
export type InferOutput<S extends GenkitSchema> =
  StandardSchemaV1.InferOutput<S>;

/** Whether the schema is a zod (v4) schema. */
export function isZodSchema(schema: GenkitSchema): schema is z.ZodType {
  return schema['~standard'].vendor === 'zod' && '_zod' in schema;
}

const zodAdapters = new WeakMap<GenkitSchema, z.ZodType>();
const foreignSchemas = new WeakMap<z.core.$ZodType, GenkitSchema>();

/**
 * Returns a zod schema for any {@link GenkitSchema}, so genkit internals
 * (composition, JSON schema conversion) can stay zod-only. Non-zod schemas are
 * wrapped; their JSON schema comes from `~standard.jsonSchema` and validation
 * from `~standard.validate`.
 */
export function toZodSchema<S extends GenkitSchema>(
  schema: S
): z.ZodType<InferOutput<S>, InferInput<S>> {
  if (isZodSchema(schema)) {
    return schema as z.ZodType<InferOutput<S>, InferInput<S>>;
  }
  const cached = zodAdapters.get(schema);
  if (cached) return cached as z.ZodType<InferOutput<S>, InferInput<S>>;
  const std = schema['~standard'];
  const toResult = (
    result: StandardSchemaV1.Result<InferOutput<S>>,
    ctx: z.core.$RefinementCtx<unknown>
  ): InferOutput<S> => {
    if (result.issues) {
      for (const issue of result.issues) {
        ctx.addIssue({
          code: 'custom',
          message: issue.message,
          path: (issue.path ?? []).map((p) =>
            p && typeof p === 'object' ? p.key : p
          ) as PropertyKey[] as (string | number)[],
        });
      }
      return z.NEVER;
    }
    return result.value;
  };
  // Stays synchronous when the library validates synchronously (zod, ArkType,
  // Valibot all do), so composed schemas still work with `.parse()`. Only
  // libraries with async validation require `.parseAsync()`.
  const adapter = z.unknown().transform((value, ctx) => {
    const result = std.validate(value) as
      | StandardSchemaV1.Result<InferOutput<S>>
      | Promise<StandardSchemaV1.Result<InferOutput<S>>>;
    return result instanceof Promise
      ? result.then((r) => toResult(r, ctx))
      : toResult(result, ctx);
  });
  foreignSchemas.set(adapter, schema);
  zodAdapters.set(schema, adapter);
  return adapter as unknown as z.ZodType<InferOutput<S>, InferInput<S>>;
}

function foreignJsonSchema(schema: GenkitSchema): JSONSchema {
  return schema['~standard'].jsonSchema.input({ target: 'draft-07' });
}

const jsonSchemas = new WeakMap<GenkitSchema, JSONSchema>();
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
export function annotateSchema<T extends z.ZodType>(
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
  schema?: GenkitSchema;
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
  if (!isZodSchema(schema!)) {
    const out = foreignJsonSchema(schema!);
    jsonSchemas.set(schema!, out);
    return out;
  }
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
  const foreign = foreignSchemas.get(zodSchema);
  if (foreign) {
    // A wrapped non-zod schema (see toZodSchema) composed into a zod schema.
    for (const key of Object.keys(jsonSchema)) delete jsonSchema[key];
    const { $schema: _, ...rest } = foreignJsonSchema(foreign);
    Object.assign(jsonSchema, rest);
    return;
  }
  const def = zodSchema._zod.def;
  if (def.type === 'date') {
    // v4 treats dates as unrepresentable. Data crosses the wire as ISO strings.
    jsonSchema.type = 'string';
    jsonSchema.format = 'date-time';
  } else if (def.type === 'union' && jsonSchema.oneOf) {
    // Discriminated unions emit `oneOf` in v4. Several providers (Gemini,
    // OpenAI structured outputs) only support `anyOf`; the branches are
    // disjoint by construction, so `anyOf` is equivalent.
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
  } else if (
    def.type === 'string' &&
    jsonSchema.format !== undefined &&
    typeof jsonSchema.pattern === 'string'
  ) {
    // String formats (datetime, email, uuid, ...) also emit zod's internal
    // regex as `pattern`. `format` alone is what v3 emitted, is validated by
    // ajv-formats/cfworker, and keeps validation errors readable. A
    // user-supplied `.regex()` on top of a format is emitted under `allOf`
    // and is not affected.
    delete jsonSchema.pattern;
  } else if (def.type === 'nullable') {
    collapseNullablePrimitive(jsonSchema);
  }
}

const PRIMITIVE_TYPES = new Set(['string', 'number', 'integer', 'boolean']);

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
    inner.const !== undefined
  ) {
    return;
  }
  delete jsonSchema.anyOf;
  Object.assign(jsonSchema, inner, { type: [inner.type, 'null'] });
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

type ParseResult<T> =
  | { success: true; data: T }
  | { success: false; errors: ValidationErrorDetail[] };

function issuesToDetails(
  issues: ReadonlyArray<StandardSchemaV1.Issue>
): ValidationErrorDetail[] {
  return issues.map((issue) => ({
    path:
      (issue.path ?? [])
        .map((p) => String(p && typeof p === 'object' ? p.key : p))
        .join('.') || '(root)',
    message: issue.message,
  }));
}

function fromStandardResult<T>(
  result: StandardSchemaV1.Result<unknown>
): ParseResult<T> {
  return result.issues
    ? { success: false, errors: issuesToDetails(result.issues) }
    : { success: true, data: result.value as T };
}

function fromZodResult<T>(
  result: z.ZodSafeParseResult<unknown>
): ParseResult<T> {
  return result.success
    ? { success: true, data: result.data as T }
    : { success: false, errors: issuesToDetails(result.error.issues) };
}

function asyncSchemaError(): GenkitError {
  return new GenkitError({
    status: 'FAILED_PRECONDITION',
    message:
      'Schema requires async validation (e.g. an async refinement), which is not supported in this synchronous context. Use parseSchemaAsync().',
  });
}

function throwOnFailure<T>(
  data: unknown,
  options: ProvidedSchema,
  result: ParseResult<T>
): T {
  if (!result.success) {
    throw new ValidationError({
      data,
      errors: result.errors,
      schema: toJsonSchema(options),
    });
  }
  return result.data;
}

/**
 * Parses data with the provided schema and returns the schema's output:
 * defaults, transforms and refinements are applied. Zod schemas are parsed
 * with zod, other Standard Schemas with `~standard.validate`. A JSON schema
 * (which takes priority, as in {@link toJsonSchema}) is only validated against
 * and the data is returned as is.
 *
 * Throws if the schema needs async validation; use {@link parseSchemaAsync}
 * where possible.
 */
export function parseSchema<T = unknown>(
  data: unknown,
  options: ProvidedSchema
): T {
  const { schema, jsonSchema } = options;
  if (jsonSchema || !schema) {
    return throwOnFailure(data, options, validateJsonSchema<T>(data, options));
  }
  if (isIgnoredValue(schema)) return data as T;
  if (isZodSchema(schema)) {
    let result: z.ZodSafeParseResult<unknown>;
    try {
      result = schema.safeParse(data);
    } catch (e) {
      if (e instanceof z.core.$ZodAsyncError) throw asyncSchemaError();
      throw e;
    }
    return throwOnFailure(data, options, fromZodResult<T>(result));
  }
  const result = schema['~standard'].validate(data);
  if (result instanceof Promise) {
    // Nobody awaits it; keep a rejection from surfacing as unhandled.
    result.catch(() => {});
    throw asyncSchemaError();
  }
  return throwOnFailure(data, options, fromStandardResult<T>(result));
}

/**
 * Async version of {@link parseSchema}. Supports async refinements and
 * Standard Schema libraries that validate asynchronously.
 */
export async function parseSchemaAsync<T = unknown>(
  data: unknown,
  options: ProvidedSchema
): Promise<T> {
  const { schema, jsonSchema } = options;
  if (jsonSchema || !schema) {
    return throwOnFailure(data, options, validateJsonSchema<T>(data, options));
  }
  if (isIgnoredValue(schema)) return data as T;
  const result = isZodSchema(schema)
    ? fromZodResult<T>(await schema.safeParseAsync(data))
    : fromStandardResult<T>(await schema['~standard'].validate(data));
  return throwOnFailure(data, options, result);
}

/**
 * A top-level `z.void()` means "no meaningful value" (like a TS `void` return),
 * not "must be undefined": e.g. an indexer whose runner returns something, or
 * a client sending `null` for a void input. Its JSON schema is `{}`, so it was
 * never enforced; keep passing the value through as is.
 */
function isIgnoredValue(schema: GenkitSchema): boolean {
  return schema instanceof z.ZodVoid;
}

function validateJsonSchema<T>(
  data: unknown,
  options: ProvidedSchema
): ParseResult<T> {
  const result = validateSchema(data, options);
  return result.valid
    ? { success: true, data: data as T }
    : { success: false, errors: result.errors };
}

/**
 * Registers provided schema as a named schema object in the Genkit registry.
 *
 * @hidden
 */
export function defineSchema<T extends GenkitSchema>(
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
