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

const jsonSchemas = new WeakMap<z.ZodTypeAny, JSONSchema>();
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
 * Wrapper object for various ways schema can be provided.
 */
export interface ProvidedSchema {
  jsonSchema?: JSONSchema;
  schema?: z.ZodTypeAny;
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
  const outSchema = zodToJsonSchema(schema!, {
    removeAdditionalStrategy: 'strict',
  });
  const annotatedSchema = applyAnnotations(schema!, outSchema as JSONSchema);
  jsonSchemas.set(schema!, annotatedSchema);
  return annotatedSchema;
}

/**
 * Recursively applies annotations to a JSON schema by walking the Zod tree
 * and matching it against the JSON schema structure.
 *
 * Note: This currently does not resolve JSON schema `$ref` nodes. Annotations
 * on recursive schemas (using `z.lazy`) may not be correctly applied to the
 * referenced definitions.
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
export function defineSchema<T extends z.ZodTypeAny>(
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
