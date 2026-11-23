/**
 * Copyright 2025 Google LLC
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

type JsonSchema = Record<string, unknown>;

/** Keywords whose value is a single subschema. */
const SCHEMA_KEYWORDS = ['items', 'not', 'if', 'then', 'else'];
/** Keywords whose value is a list of subschemas. */
const SCHEMA_LIST_KEYWORDS = ['anyOf', 'oneOf', 'allOf', 'prefixItems'];
/** Keywords whose value maps names to subschemas. */
const SCHEMA_MAP_KEYWORDS = ['properties', '$defs', 'definitions'];

function isObjectSchema(schema: JsonSchema): boolean {
  const type = schema.type;
  return (
    type === 'object' ||
    (Array.isArray(type) && type.includes('object')) ||
    (type === undefined && schema.properties !== undefined)
  );
}

function isSchema(value: unknown): value is JsonSchema {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Returns a copy of `schema` with `additionalProperties: false` on every
 * object schema that allows additional properties by default or with `true`.
 *
 * OpenAI's strict structured output rejects any object schema whose
 * `additionalProperties` isn't `false`, and genkit's zod conversion emits
 * `true` for every `z.object`. Under strict mode such a schema always fails
 * with a 400, so closing it only changes requests that couldn't succeed.
 * A schema-valued `additionalProperties` (e.g. `z.record`) is left alone:
 * strict mode can't express it, and the host's error is clearer than a
 * silent change.
 */
export function closeObjectSchemas(schema: JsonSchema): JsonSchema {
  const out: JsonSchema = { ...schema };
  for (const key of SCHEMA_KEYWORDS) {
    if (isSchema(out[key])) out[key] = closeObjectSchemas(out[key]);
  }
  for (const key of SCHEMA_LIST_KEYWORDS) {
    const list = out[key];
    if (Array.isArray(list)) {
      out[key] = list.map((s) => (isSchema(s) ? closeObjectSchemas(s) : s));
    }
  }
  for (const key of SCHEMA_MAP_KEYWORDS) {
    const map = out[key];
    if (isSchema(map)) {
      out[key] = Object.fromEntries(
        Object.entries(map).map(([name, s]) => [
          name,
          isSchema(s) ? closeObjectSchemas(s) : s,
        ])
      );
    }
  }
  if (isSchema(out.additionalProperties)) {
    out.additionalProperties = closeObjectSchemas(out.additionalProperties);
  } else if (
    isObjectSchema(out) &&
    (out.additionalProperties === undefined ||
      out.additionalProperties === true)
  ) {
    out.additionalProperties = false;
  }
  return out;
}
