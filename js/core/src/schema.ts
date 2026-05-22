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
import zodToJsonSchema from 'zod-to-json-schema';
import { toJSONSchema as z4ToJSONSchema } from 'zod/v4/core';
import { getGenkitRuntimeConfig } from './config.js';
import { GenkitError } from './error.js';
import type { Registry } from './registry.js';
import {
  isNonZodStandardSchema,
  isZodV3Schema,
  isZodV4Schema,
  type GenkitSchema,
} from './standard.js';

export { z }; // provide a consistent zod to use throughout genkit

const ajv = new Ajv();
addFormats(ajv);

/**
 * JSON schema.
 */
export type JSONSchema = JSONSchemaType<any> | any;

const jsonSchemas = new WeakMap<object, JSONSchema>();
const ajvValidators = new WeakMap<JSONSchema, ReturnType<typeof ajv.compile>>();
const cfWorkerValidators = new WeakMap<JSONSchema, Validator>();
const schemaAnnotations = new WeakMap<z.ZodTypeAny, Record<string, any>>();

/**
 * Annotates a Zod schema with UI-specific metadata.
 *
 * NOTE: It's typically recommended to use x-genkit-* (or similar) as the prefix.
 */
export function annotateSchema<T extends z.ZodTypeAny>(
  schema: T,
  annotations: Record<string, any>
): T {
  const current = schemaAnnotations.get(schema) || {};
  schemaAnnotations.set(schema, { ...current, ...annotations });
  return schema;
}

/**
 * Wrapper object for various ways schema can be provided. Accepts Zod v3, Zod
 * v4, or any Standard Schema v1-compliant schema, as well as a pre-converted
 * JSON schema.
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
 * Converts a schema to a JSON schema for use with LLM tool definitions, the
 * dev UI, and the reflection API. Uses an in-memory cache for Zod schemas.
 *
 * - Zod v3: uses `zod-to-json-schema` (cached).
 * - Zod v4: uses the built-in `z4.toJSONSchema()` (cached).
 * - Other Standard Schema: returns `{}` (no JSON Schema can be derived without
 *   library-specific tooling). Pass an explicit `jsonSchema` alongside the
 *   schema for richer LLM tool descriptions.
 *
 * @param options Provide a json schema and/or schema. JSON schema has priority.
 * @returns A JSON schema, or null if neither is provided.
 */
export function toJsonSchema({
  jsonSchema,
  schema,
}: ProvidedSchema): JSONSchema | undefined {
  // if neither jsonSchema or schema is present return undefined
  if (!jsonSchema && !schema) return null;
  if (jsonSchema) return jsonSchema;

  const s = schema!;

  if (jsonSchemas.has(s as object)) return jsonSchemas.get(s as object)!;

  let outSchema: JSONSchema;

  if (isZodV3Schema(s)) {
    // Zod v3: use the battle-tested zod-to-json-schema adapter, then apply any
    // UI-specific annotations registered via `annotateSchema()`.
    const raw = zodToJsonSchema(s, {
      removeAdditionalStrategy: 'strict',
    }) as JSONSchema;
    outSchema = applyAnnotations(s, raw);
  } else if (isZodV4Schema(s)) {
    // Zod v4: use the built-in converter
    outSchema = z4ToJSONSchema(s as any, {
      target: 'draft-7',
      unrepresentable: 'any',
    });
  } else {
    // Non-Zod Standard Schema: we cannot generically produce JSON Schema.
    // Return an empty schema so the system keeps running — callers that need
    // LLM-facing schema info should provide an explicit `jsonSchema`.
    outSchema = {};
  }

  jsonSchemas.set(s as object, outSchema);
  return outSchema;
}

/**
 * Recursively applies annotations (registered via `annotateSchema`) to a JSON
 * schema by walking the Zod tree and matching it against the JSON schema
 * structure.
 *
 * Note: This currently does not resolve JSON schema `$ref` nodes. Annotations
 * on recursive schemas (using `z.lazy`) may not be correctly applied to the
 * referenced definitions.
 *
 * Only applies to Zod v3 schemas — `annotateSchema` accepts Zod v3 only.
 */
function applyAnnotations(schema: z.ZodTypeAny, json: any): any {
  if (!json || typeof json !== 'object') return json;

  const annotationsToApply: Record<string, any>[] = [];
  let current = schema;

  // Collect all annotations in the hierarchy (outer to inner)
  while (current) {
    const ann = schemaAnnotations.get(current);
    if (ann) annotationsToApply.push(ann);

    if (
      current instanceof z.ZodOptional ||
      current instanceof z.ZodNullable ||
      current instanceof z.ZodDefault ||
      current instanceof z.ZodEffects
    ) {
      current = (current as any)._def.innerType || (current as any)._def.schema;
    } else {
      break;
    }
  }

  // Resolve annotations (outer-most last so it wins)
  const resolvedAnnotations: Record<string, any> = {};
  for (let i = annotationsToApply.length - 1; i >= 0; i--) {
    Object.assign(resolvedAnnotations, annotationsToApply[i]);
  }

  for (const key in resolvedAnnotations) {
    if (Object.prototype.hasOwnProperty.call(json, key)) {
      console.warn(
        `Annotation key "${key}" conflicts with existing JSON schema property and will be ignored.`
      );
      continue;
    }
    json[key] = resolvedAnnotations[key];
  }

  const inner = current;
  if (inner instanceof z.ZodObject && json.properties) {
    for (const key in inner.shape) {
      if (json.properties[key]) {
        applyAnnotations(inner.shape[key], json.properties[key]);
      }
    }
  } else if (inner instanceof z.ZodArray && json.items) {
    applyAnnotations(inner.element, json.items);
  } else if (inner instanceof z.ZodUnion && json.anyOf) {
    for (let i = 0; i < inner.options.length; i++) {
      applyAnnotations(inner.options[i], json.anyOf[i]);
    }
  } else if (inner instanceof z.ZodIntersection && json.allOf) {
    const schemas: z.ZodTypeAny[] = [];
    const collect = (s: z.ZodTypeAny) => {
      if (s instanceof z.ZodIntersection) {
        collect(s._def.left);
        collect(s._def.right);
      } else {
        schemas.push(s);
      }
    };
    collect(inner);
    if (schemas.length === json.allOf.length) {
      for (let i = 0; i < schemas.length; i++) {
        applyAnnotations(schemas[i], json.allOf[i]);
      }
    }
  } else if (inner instanceof z.ZodRecord && json.additionalProperties) {
    applyAnnotations(inner.valueSchema, json.additionalProperties);
  } else if (inner instanceof z.ZodTuple && Array.isArray(json.items)) {
    for (let i = 0; i < inner.items.length; i++) {
      applyAnnotations(inner.items[i], json.items[i]);
    }
  } else if (inner instanceof z.ZodDiscriminatedUnion && json.anyOf) {
    for (let i = 0; i < inner.options.length; i++) {
      applyAnnotations(inner.options[i], json.anyOf[i]);
    }
  }

  return json;
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
 * Validates the provided data using native schema validation when possible,
 * falling back to Ajv/cfworker JSON Schema validation.
 *
 * - Zod v3/v4: uses the schema's native `.parse()` for rich error messages.
 * - Standard Schema: calls `schema['~standard'].validate()`.
 * - JSON Schema only: uses Ajv or cfworker.
 */
export function validateSchema(
  data: unknown,
  options: ProvidedSchema
): ValidationResponse {
  const { schema } = options;

  // ── Zod v3 fast path ──────────────────────────────────────────────────────
  if (schema && isZodV3Schema(schema)) {
    const result = schema.safeParse(data);
    const jsonSch = toJsonSchema(options) ?? {};
    if (!result.success) {
      return {
        valid: false,
        errors: result.error.errors.map((e) => ({
          path: e.path.join('.') || '(root)',
          message: e.message,
        })),
        schema: jsonSch,
      };
    }
    return { valid: true, schema: jsonSch };
  }

  // ── Zod v4 fast path ──────────────────────────────────────────────────────
  if (schema && isZodV4Schema(schema)) {
    const result = (schema as any).safeParse(data);
    const jsonSch = toJsonSchema(options) ?? {};
    if (!result.success) {
      const issues = result.error?.issues ?? [];
      return {
        valid: false,
        errors: issues.map((e: any) => ({
          path: (e.path ?? []).join('.') || '(root)',
          message: e.message,
        })),
        schema: jsonSch,
      };
    }
    return { valid: true, schema: jsonSch };
  }

  // ── Non-Zod Standard Schema fast path ─────────────────────────────────────
  if (schema && isNonZodStandardSchema(schema)) {
    const result = schema['~standard'].validate(data);
    const jsonSch = toJsonSchema(options) ?? {};
    // Standard Schema validate() may return a Promise for async schemas.
    // We do not support async validation in this synchronous code path.
    if (result instanceof Promise) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          'Async Standard Schema validators are not supported in synchronous ' +
          'Genkit validation contexts. Use a synchronous schema or provide ' +
          'an explicit jsonSchema.',
      });
    }
    if (result.issues) {
      return {
        valid: false,
        errors: result.issues.map((issue) => ({
          path:
            (issue.path ?? [])
              .map((p) => (typeof p === 'object' ? p.key : p))
              .join('.')
              .replace(/^\./, '') || '(root)',
          message: issue.message,
        })),
        schema: jsonSch,
      };
    }
    return { valid: true, schema: jsonSch };
  }

  // ── JSON Schema fallback (Ajv / cfworker) ─────────────────────────────────
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
 * Parses raw data against the provided schema. Uses native schema parsing when
 * possible (Zod v3/v4 `.parse()`, Standard Schema `validate()`), falling back
 * to JSON Schema validation.
 *
 * Returns the validated (and potentially transformed) output value.
 */
export function parseSchema<T = unknown>(
  data: unknown,
  options: ProvidedSchema
): T {
  const { schema } = options;

  // For Zod v3/v4: use native .parse() which applies transforms and returns the
  // output type, converting ZodError into a ValidationError.
  if (schema && (isZodV3Schema(schema) || isZodV4Schema(schema))) {
    try {
      return (schema as any).parse(data) as T;
    } catch (e: any) {
      const issues: Array<{ path: (string | number)[]; message: string }> =
        e?.errors ?? e?.issues ?? [];
      const jsonSch = toJsonSchema(options) ?? {};
      throw new ValidationError({
        data,
        errors: issues.map((issue) => ({
          path: (issue.path ?? []).join('.') || '(root)',
          message: issue.message,
        })),
        schema: jsonSch,
      });
    }
  }

  // For Standard Schema and JSON Schema: use the validation path.
  const result = validateSchema(data, options);
  if (!result.valid) {
    throw new ValidationError({
      data,
      errors: result.errors,
      schema: result.schema,
    });
  }

  // For Standard Schema: return the validated output value (which may differ
  // from the input if the schema applies coercions).
  if (schema && isNonZodStandardSchema(schema)) {
    const validateResult = schema['~standard'].validate(data);
    if (!(validateResult instanceof Promise) && !validateResult.issues) {
      return validateResult.value as T;
    }
  }

  return data as T;
}

/**
 * Builds a merged JSON Schema that is the combination of a static envelope
 * schema and an optional `configSchema` override for a named property.
 *
 * This is used by retriever/indexer/reranker/evaluator actions that have a
 * fixed request envelope (e.g. `{ query, options }`) where the `options`
 * property type depends on a user-supplied `configSchema`. Rather than using
 * Zod's `.extend()` (which requires a Zod-specific schema), we:
 *
 *  1. Keep the static Zod envelope as the action's `inputSchema` (for runtime
 *     validation — `options` stays typed as `z.any()`).
 *  2. Pass this merged JSON schema as `inputJsonSchema` so the dev UI and
 *     reflection API see the fully-typed structure.
 *
 * @param envelopeSchema The static Zod (or other) envelope schema.
 * @param propertyName The property name to override (e.g. `"options"`).
 * @param configSchema The user-supplied schema for that property (optional).
 * @param extraProps Additional JSON schema property overrides to merge in.
 */
export function mergedInputJsonSchema(
  envelopeSchema: GenkitSchema,
  propertyName: string,
  configSchema?: GenkitSchema,
  extraProps?: Record<string, JSONSchema>
): JSONSchema | undefined {
  const envelopeJson = toJsonSchema({ schema: envelopeSchema });
  if (!envelopeJson) return undefined;

  const merged: JSONSchema = { ...envelopeJson };
  if (!merged.properties) merged.properties = {};

  if (configSchema) {
    const configJson = toJsonSchema({ schema: configSchema });
    merged.properties = {
      ...merged.properties,
      [propertyName]: configJson
        ? { ...configJson }
        : merged.properties[propertyName],
    };
  }

  if (extraProps) {
    merged.properties = {
      ...merged.properties,
      ...Object.fromEntries(Object.entries(extraProps).map(([k, v]) => [k, v])),
    };
  }

  return merged;
}

/**
 * Registers provided schema as a named schema object in the Genkit registry.
 * Accepts Zod v3, Zod v4, or any Standard Schema v1-compliant schema.
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
