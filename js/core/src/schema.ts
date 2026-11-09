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

import {
  Validator,
  type OutputUnit,
  type SchemaDraft,
} from '@cfworker/json-schema';
import type {
  StandardJSONSchemaV1,
  StandardSchemaV1,
} from '@standard-schema/spec';
import { z } from 'zod';
import { GenkitError } from './error.js';

import type { Registry } from './registry.js';

export { z }; // provide a consistent zod to use throughout genkit

/**
 * JSON schema.
 */
// Deliberately loose: JSON schemas come from users, plugins and remote
// servers (e.g. MCP) in many drafts and shapes.
export type JSONSchema = any;

/**
 * Any schema genkit can both validate with and describe as JSON schema:
 * implements Standard Schema and Standard JSON Schema
 * (https://standardschema.dev). Zod 4.2+ and ArkType schemas qualify as is;
 * Valibot schemas via `toStandardJsonSchema()` from `@valibot/to-json-schema`.
 *
 * Non-zod schemas are meant for end-user surfaces: flows, tools and agent
 * custom state. Plugin-facing schemas (model/embedder config, etc.) should
 * stay zod.
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
  // Optional chaining: callers may pass values that aren't valid GenkitSchemas
  // at runtime (e.g. zod < 3.24 schemas from prebuilt plugins).
  return schema?.['~standard']?.vendor === 'zod' && '_zod' in schema;
}

const zodAdapters = new WeakMap<GenkitSchema, z.ZodType>();
const foreignSchemas = new WeakMap<z.core.$ZodType, GenkitSchema>();

/**
 * Returns a zod schema for any {@link GenkitSchema}, so genkit internals
 * (composition, JSON schema conversion) can stay zod-only. Non-zod schemas are
 * wrapped; their JSON schema comes from `~standard.jsonSchema` and validation
 * from `~standard.validate`.
 *
 * @param label Describes where the schema came from, used in the error if the
 *   schema isn't supported.
 */
export function toZodSchema<S extends GenkitSchema>(
  schema: S,
  label = 'schema'
): z.ZodType<InferOutput<S>, InferInput<S>> {
  assertGenkitSchema(schema, label);
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

/**
 * Throws a descriptive error unless `schema` implements Standard Schema and
 * Standard JSON Schema. Only a couple of property reads, so it's cheap enough
 * to run whenever an action is defined. The common failure is a plugin that
 * bundles its own zod v3.
 *
 * @param label describes where the schema came from, used in the error.
 */
export function assertGenkitSchema(
  schema: unknown,
  label: string
): asserts schema is GenkitSchema {
  const std = (schema as Partial<GenkitSchema> | null | undefined)?.[
    '~standard'
  ];
  if (
    typeof std?.validate === 'function' &&
    typeof std.jsonSchema?.input === 'function'
  ) {
    return;
  }
  throw unsupportedSchemaError(label, isZod3Schema(schema));
}

function isZod3Schema(schema: unknown): boolean {
  return (
    typeof schema === 'object' &&
    schema !== null &&
    '_def' in schema &&
    !('_zod' in schema)
  );
}

/**
 * Whether a zod v4 schema has zod v3 schemas nested in it, e.g.
 * `GenerationCommonConfigSchema.extend({ seed: z3.number() })`. Only called
 * after JSON schema conversion has already failed, so cost doesn't matter.
 */
function containsZod3Schema(root: unknown): boolean {
  const seen = new Set<unknown>();
  const visit = (value: unknown): boolean => {
    if (typeof value !== 'object' || value === null || seen.has(value)) {
      return false;
    }
    seen.add(value);
    if (isZod3Schema(value)) return true;
    if ('_zod' in value) {
      return visit((value as z.core.$ZodType)._zod.def);
    }
    const children = Array.isArray(value) ? value : Object.values(value);
    return children.some(visit);
  };
  return visit(root);
}

function unsupportedSchemaError(label: string, zod3: boolean): GenkitError {
  return new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: zod3
      ? `${label} is not supported: found a zod v3 schema. Genkit requires zod 4.2+ (or another Standard Schema library that implements Standard JSON Schema). If this schema comes from a plugin, the plugin needs to be upgraded to a version compatible with this version of Genkit.`
      : `${label} is not supported: it must implement Standard Schema and Standard JSON Schema (https://standardschema.dev), e.g. a zod 4.2+ or ArkType schema.`,
  });
}

const jsonSchemas = new WeakMap<GenkitSchema, JSONSchema>();
const jsonSchemaValidators = new WeakMap<JSONSchema, Validator>();

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

// One shared catchall instance marks wireObject schemas. It survives
// `.extend()`, `.pick()`, etc., so derived schemas are recognized too.
const WIRE_CATCHALL = z.unknown();

/**
 * `z.object` for framework wire types (Part, Message, GenerateRequest, ...).
 *
 * Unknown keys survive parsing, so data produced by plugins or by a newer
 * runtime isn't silently dropped when it passes through an action. Typed as a
 * plain `z.object` on purpose: `z.looseObject` would add an index signature to
 * every inferred wire type (losing excess property checks), while data with
 * extra keys is still a valid value of the closed type. The JSON schema is
 * also that of a plain `z.object` (see compatOverride).
 *
 * Not meant for user-facing or config schemas; those keep standard zod
 * semantics.
 */
export function wireObject<T extends z.core.$ZodLooseShape>(
  shape: T
): z.ZodObject<z.core.util.Writeable<T>, z.core.$strip> {
  return z.object(shape).catchall(WIRE_CATCHALL) as unknown as z.ZodObject<
    z.core.util.Writeable<T>,
    z.core.$strip
  >;
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
 * @param label Describes where the schema came from (e.g. `configSchema of
 *   model "foo"`), used in the error if the schema isn't supported.
 * @returns A JSON schema.
 */
export function toJsonSchema(
  { jsonSchema, schema }: ProvidedSchema,
  label = 'schema'
): JSONSchema | undefined {
  // if neither jsonSchema or schema is present return undefined
  if (!hasJsonSchema(jsonSchema) && !schema) return null;
  if (hasJsonSchema(jsonSchema)) return jsonSchema;
  if (jsonSchemas.has(schema!)) return jsonSchemas.get(schema!)!;
  assertGenkitSchema(schema, label);
  if (!isZodSchema(schema)) {
    const out = foreignJsonSchema(schema);
    jsonSchemas.set(schema, out);
    return out;
  }
  // Note: the JSON schema cache is keyed by instance, so annotations added via
  // annotateSchema after the first conversion are not reflected.
  let outSchema: JSONSchema;
  try {
    outSchema = z.toJSONSchema(schema, {
      target: 'draft-07',
      // JSON schema is used to validate raw (pre-parse) data and to describe
      // inputs to models/tools, so describe the input side of transforms/defaults.
      io: 'input',
      // z.bigint(), z.symbol(), etc. become {} instead of throwing.
      unrepresentable: 'any',
      override: compatOverride,
    });
  } catch (e) {
    if (containsZod3Schema(schema)) throw unsupportedSchemaError(label, true);
    throw e;
  }
  jsonSchemas.set(schema, outSchema);
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
  } else if (def.type === 'object' && def.catchall === WIRE_CATCHALL) {
    // wireObject is loose only at runtime; describe it like the plain
    // z.object it is typed as. Omitting `additionalProperties` allows extra
    // keys just the same, and keeps wire type JSON schemas free of noise.
    delete jsonSchema.additionalProperties;
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
      // the JSON schema validator, and keeps validation errors readable. A
      // user-supplied `.regex()` on top of a format is emitted under `allOf`
      // and is not affected.
      delete jsonSchema.pattern;
    } else {
      // zod-specific formats (cuid, base64, starts_with, ...): `pattern` is
      // their only check, and some providers reject non-standard formats.
      // v3 emitted just the pattern.
      delete jsonSchema.format;
    }
  } else if (def.type === 'nullable') {
    collapseNullablePrimitive(jsonSchema);
  }
}

const PRIMITIVE_TYPES = new Set(['string', 'number', 'integer', 'boolean']);

/** Formats validated by @cfworker/json-schema. */
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

// Keywords whose error only says "a subschema failed"; the subschema's own
// errors are more specific.
const WRAPPER_KEYWORDS = new Set([
  'properties',
  'patternProperties',
  'additionalProperties',
  'items',
  'prefixItems',
  'additionalItems',
  'allOf',
  '$ref',
  'dependentSchemas',
]);

// Combinators report one summary error plus one error per failed branch; the
// summary is the readable one.
const COMBINATOR_KEYWORDS = new Set(['anyOf', 'oneOf', 'not']);

/**
 * Reduces the validator's full error tree to one error per actual problem.
 */
/**
 * The validator reports `instanceLocation` as a URI fragment: JSON pointer
 * escapes plus percent-encoding (`my key` becomes `my%20key`).
 */
function decodePointerSegment(segment: string): string {
  let decoded = segment;
  try {
    decoded = decodeURIComponent(segment);
  } catch {
    // Not percent-encoded after all; use as is.
  }
  return decoded.replace(/~1/g, '/').replace(/~0/g, '~');
}

function toValidationErrorDetails(
  errors: OutputUnit[]
): ValidationErrorDetail[] {
  const under = (e: OutputUnit, parent: OutputUnit) =>
    e !== parent && e.keywordLocation.startsWith(parent.keywordLocation + '/');
  // `additionalProperties: false` also reports a "False boolean schema" error;
  // the additionalProperties error names the offending property. A top-level
  // `false` schema reports only the latter, so keep it then.
  let kept = errors.filter((e) => e.keyword !== 'false');
  if (kept.length === 0) kept = errors;
  const combinators = kept.filter((e) => COMBINATOR_KEYWORDS.has(e.keyword));
  kept = kept.filter((e) => !combinators.some((c) => under(e, c)));
  kept = kept.filter(
    (e) => !WRAPPER_KEYWORDS.has(e.keyword) || !kept.some((o) => under(o, e))
  );
  // A failed `then`/`else` is reported on the `if` keyword ("Instance does not
  // match "then" schema"), while its specific errors sit under the sibling
  // `then`/`else` location.
  kept = kept.filter((e) => {
    if (e.keyword !== 'if') return true;
    const parent = e.keywordLocation.slice(0, -'/if'.length);
    return !kept.some(
      (o) =>
        o.keywordLocation.startsWith(`${parent}/then/`) ||
        o.keywordLocation.startsWith(`${parent}/else/`)
    );
  });
  const seen = new Set<string>();
  const details: ValidationErrorDetail[] = [];
  for (const e of kept) {
    const path = e.instanceLocation
      .replace(/^#\/?/, '')
      .split('/')
      .filter((p) => p !== '')
      .map((p) => decodePointerSegment(p))
      .join('.');
    const detail = { path: path || '(root)', message: e.error };
    const key = `${detail.path}\0${detail.message}`;
    if (!seen.has(key)) {
      seen.add(key);
      details.push(detail);
    }
  }
  return details;
}

/**
 * Whether a JSON schema was provided. `false` (rejects everything) and `true`
 * are valid schemas, so this can't be a truthiness check.
 */
function hasJsonSchema(jsonSchema: JSONSchema): boolean {
  return jsonSchema !== undefined && jsonSchema !== null;
}

function schemaDraft(jsonSchema: JSONSchema): SchemaDraft {
  const declared =
    typeof jsonSchema?.$schema === 'string' ? jsonSchema.$schema : '';
  if (declared.includes('2020-12')) return '2020-12';
  if (declared.includes('2019-09')) return '2019-09';
  if (declared.includes('draft-04')) return '4';
  // What genkit emits, and the most common draft for undeclared schemas.
  return '7';
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
  if (!hasJsonSchema(toValidate)) {
    return { valid: true, schema: toValidate };
  }
  // An interpreting validator (no codegen), so it also runs where `eval` is
  // unavailable, e.g. Cloudflare Workers. Boolean schemas (`true`/`false`)
  // can't be WeakMap keys and are cheap to build, so only objects are cached.
  let validator =
    typeof toValidate === 'object'
      ? jsonSchemaValidators.get(toValidate)
      : undefined;
  if (!validator) {
    validator = new Validator(toValidate, schemaDraft(toValidate), false);
    if (typeof toValidate === 'object') {
      jsonSchemaValidators.set(toValidate, validator);
    }
  }
  const result = validator.validate(sanitizeForJsonSchema(data));
  if (!result.valid) {
    return {
      valid: false,
      errors: toValidationErrorDetails(result.errors),
      schema: toValidate,
    };
  }
  return { valid: true, schema: toValidate };
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
      schema: jsonSchemaForError(options),
    });
  }
  return result.data;
}

/**
 * The JSON schema shown in a validation error. Best effort: a Standard Schema
 * library may refuse to produce JSON schema (e.g. for the requested draft),
 * and that must not mask the actual validation error.
 */
function jsonSchemaForError(options: ProvidedSchema): JSONSchema | undefined {
  try {
    return toJsonSchema(options);
  } catch {
    return undefined;
  }
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
  if (hasJsonSchema(jsonSchema) || !schema) {
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
  if (hasJsonSchema(jsonSchema) || !schema) {
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
